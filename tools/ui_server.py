import json
import os
import signal
import subprocess
import sys
import threading
import time
from collections import deque
from pathlib import Path
from typing import Any, Deque, Dict, List, Optional

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field, validator

REPO_ROOT = Path(__file__).resolve().parents[1]
RUNS_DIR = REPO_ROOT / "runs"
FRONTEND_DIR = REPO_ROOT / "tools" / "ui_frontend"


class RunRequest(BaseModel):
    rom: str = Field(default="PokemonRed.gb", description="Path to ROM")
    state: str = Field(default="init.state", description="Path to savestate")
    run_name: str = Field(default="ui_run", description="Run folder name")
    num_envs: int = Field(default=8, ge=1)
    batch_size: int = Field(default=256, ge=1)
    total_multiplier: int = Field(default=1000, ge=1)
    preset: Optional[str] = Field(default=None, description="GPU sizing preset")
    stream: bool = True
    wandb: bool = False
    checkpoint_freq: Optional[int] = None
    seed: Optional[int] = None
    status_interval: float = Field(default=10.0, ge=1.0)
    eval_every_steps: Optional[int] = None
    eval_episodes: int = Field(default=2, ge=1)
    eval_max_steps: Optional[int] = None
    eval_stream: bool = False
    config: str = Field(default=str(REPO_ROOT / "configs" / "train_default.json"))

    @validator("run_name")
    def run_name_not_empty(cls, v):
        if not v or v.strip() == "":
            raise ValueError("run_name must not be empty")
        return v


class TrainingProcessManager:
    def __init__(self, repo_root: Path):
        self.repo_root = repo_root
        self.process: Optional[subprocess.Popen] = None
        self.stdout_lines: Deque[str] = deque(maxlen=200)
        self.stderr_lines: Deque[str] = deque(maxlen=200)
        self.run_info: Dict[str, Any] = {}
        self._lock = threading.Lock()

    def start(self, req: RunRequest) -> Dict[str, Any]:
        with self._lock:
            if self.process and self.process.poll() is None:
                raise RuntimeError("A training run is already active. Stop it before starting another.")

            run_dir = RUNS_DIR / req.run_name
            run_dir.mkdir(parents=True, exist_ok=True)
            status_file = run_dir / "status.json"
            eval_log = run_dir / "eval.jsonl"

            cmd = [
                sys.executable,
                str(self.repo_root / "training" / "train_ppo.py"),
                "--config",
                req.config,
                "--rom",
                req.rom,
                "--state",
                req.state,
                "--run-name",
                req.run_name,
                "--output-dir",
                str(RUNS_DIR),
                "--num-envs",
                str(req.num_envs),
                "--batch-size",
                str(req.batch_size),
                "--total-multiplier",
                str(req.total_multiplier),
                "--status-file",
                str(status_file),
                "--status-interval",
                str(req.status_interval),
                "--eval-log",
                str(eval_log),
                "--eval-episodes",
                str(req.eval_episodes),
            ]
            if req.preset:
                cmd += ["--preset", req.preset]
            if req.checkpoint_freq:
                cmd += ["--checkpoint-freq", str(req.checkpoint_freq)]
            if req.eval_every_steps:
                cmd += ["--eval-every-steps", str(req.eval_every_steps)]
            if req.eval_max_steps:
                cmd += ["--eval-max-steps", str(req.eval_max_steps)]
            if req.eval_stream:
                cmd += ["--eval-stream"]
            if not req.stream:
                cmd += ["--no-stream"]
            if req.wandb:
                cmd += ["--wandb"]
            if req.seed is not None:
                cmd += ["--seed", str(req.seed)]

            creationflags = 0
            kwargs: Dict[str, Any] = {}
            if os.name == "nt":
                creationflags = subprocess.CREATE_NEW_PROCESS_GROUP
                kwargs["creationflags"] = creationflags

            stdout_log = (run_dir / "trainer_stdout.log").open("w", encoding="utf-8")
            stderr_log = (run_dir / "trainer_stderr.log").open("w", encoding="utf-8")
            self.process = subprocess.Popen(
                cmd,
                cwd=self.repo_root,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                bufsize=1,
                **kwargs,
            )
            self.run_info = {
                "run_name": req.run_name,
                "run_dir": str(run_dir),
                "status_file": str(status_file),
                "eval_log": str(eval_log),
                "started_at": time.time(),
                "pid": self.process.pid,
                "command": cmd,
            }
            threading.Thread(target=self._pump_stream, args=(self.process.stdout, stdout_log, self.stdout_lines), daemon=True).start()
            threading.Thread(target=self._pump_stream, args=(self.process.stderr, stderr_log, self.stderr_lines), daemon=True).start()
            return self.run_info

    def _pump_stream(self, stream, file_handle, buffer: Deque[str]):
        try:
            for line in iter(stream.readline, ""):
                file_handle.write(line)
                file_handle.flush()
                buffer.append(line.rstrip())
        finally:
            file_handle.close()

    def stop(self) -> Dict[str, Any]:
        with self._lock:
            if not self.process or self.process.poll() is not None:
                return {"stopped": False, "message": "No active process."}
            try:
                if os.name == "nt":
                    self.process.send_signal(signal.CTRL_BREAK_EVENT)
                else:
                    self.process.send_signal(signal.SIGINT)
                try:
                    self.process.wait(timeout=15)
                except subprocess.TimeoutExpired:
                    self.process.terminate()
            finally:
                self.process = None
            return {"stopped": True, "run": self.run_info}

    def status(self) -> Dict[str, Any]:
        with self._lock:
            running = self.process is not None and self.process.poll() is None
            info = {"running": running, "run": self.run_info}
            if running:
                info["pid"] = self.process.pid
            info["stdout_tail"] = list(self.stdout_lines)
            info["stderr_tail"] = list(self.stderr_lines)
            return info


def read_status_file(path: Path) -> Optional[Dict[str, Any]]:
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text())
    except Exception:
        return None


def read_eval_log(path: Path, limit: int = 50) -> List[Dict[str, Any]]:
    if not path.exists():
        return []
    rows: List[Dict[str, Any]] = []
    with path.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    if limit:
        rows = rows[-limit:]
    return list(reversed(rows))


def list_checkpoints(run_dir: Path) -> List[Dict[str, Any]]:
    if not run_dir.exists():
        return []
    results = []
    for ckpt in sorted(run_dir.glob("*.zip")):
        stat = ckpt.stat()
        results.append({"path": str(ckpt), "name": ckpt.name, "mtime": stat.st_mtime, "size": stat.st_size})
    return list(reversed(results))


app = FastAPI(title="Pokemon PPO Control Panel")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
manager = TrainingProcessManager(REPO_ROOT)

if FRONTEND_DIR.exists():
    app.mount("/static", StaticFiles(directory=FRONTEND_DIR), name="static")


@app.get("/")
def root():
    index_path = FRONTEND_DIR / "index.html"
    if not index_path.exists():
        raise HTTPException(status_code=404, detail="UI not built")
    return FileResponse(index_path)


@app.post("/api/run")
def start_run(req: RunRequest):
    try:
        info = manager.start(req)
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))
    return {"started": True, "run": info}


@app.post("/api/stop")
def stop_run():
    result = manager.stop()
    return result


@app.get("/api/status")
def get_status(run: Optional[str] = None):
    proc_info = manager.status()
    status_file = None
    eval_log = None
    run_dir = None
    if run:
        run_dir = RUNS_DIR / run
        status_file = run_dir / "status.json"
        eval_log = run_dir / "eval.jsonl"
    elif proc_info.get("run"):
        status_file = Path(proc_info["run"].get("status_file", ""))
        eval_log = Path(proc_info["run"].get("eval_log", ""))
        run_dir = Path(proc_info["run"].get("run_dir", ""))
    status_payload = read_status_file(status_file) if status_file else None
    evals = read_eval_log(eval_log, limit=10) if eval_log else []
    checkpoints = list_checkpoints(run_dir) if run_dir else []
    return {
        "process": proc_info,
        "status": status_payload,
        "evals": evals,
        "checkpoints": checkpoints,
    }


@app.get("/api/evals")
def get_evals(limit: int = 50, run: Optional[str] = None):
    proc_info = manager.status()
    eval_log = None
    if run:
        eval_log = RUNS_DIR / run / "eval.jsonl"
    elif proc_info.get("run"):
        eval_log = Path(proc_info["run"].get("eval_log", ""))
    if not eval_log:
        return []
    return read_eval_log(eval_log, limit=limit)


@app.get("/api/checkpoints")
def get_checkpoints(run: Optional[str] = None):
    proc_info = manager.status()
    run_dir = None
    if run:
        run_dir = RUNS_DIR / run
    elif proc_info.get("run"):
        run_dir = Path(proc_info["run"].get("run_dir", ""))
    return list_checkpoints(run_dir) if run_dir else []


def main():
    import uvicorn

    uvicorn.run("tools.ui_server:app", host="0.0.0.0", port=8000, reload=False)


if __name__ == "__main__":
    main()
