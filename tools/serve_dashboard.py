import json
import re
import sys
import threading
from http.server import HTTPServer, SimpleHTTPRequestHandler
from pathlib import Path
from typing import Dict, List, Optional

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

RUNS_DIR = Path("runs")


def parse_checkpoint_steps(path: Path) -> Optional[int]:
    m = re.search(r"_(\d+)_steps\.zip", path.name)
    if m:
        return int(m.group(1))
    return None


def collect_runs(runs_dir: Path) -> List[Dict]:
    runs = []
    if not runs_dir.exists():
        return runs
    for run in runs_dir.iterdir():
        if not run.is_dir():
            continue
        metadata_path = run / "metadata.json"
        metadata = {}
        if metadata_path.exists():
            try:
                metadata = json.loads(metadata_path.read_text())
            except json.JSONDecodeError:
                metadata = {}
        checkpoints = sorted(run.glob("*.zip"))
        latest_ckpt = checkpoints[-1] if checkpoints else None
        runs.append(
            {
                "name": run.name,
                "path": str(run),
                "latest_checkpoint": str(latest_ckpt) if latest_ckpt else None,
                "latest_steps": parse_checkpoint_steps(latest_ckpt) if latest_ckpt else None,
                "last_modified": latest_ckpt.stat().st_mtime if latest_ckpt else run.stat().st_mtime,
                "metadata": metadata,
            }
        )
    runs.sort(key=lambda r: r["last_modified"], reverse=True)
    return runs


DASHBOARD_HTML = """<!DOCTYPE html>
<html>
<head>
  <meta charset="UTF-8" />
  <title>Pokemon RL Dashboard</title>
  <style>
    body { font-family: Arial, sans-serif; margin: 24px; background: #0b1724; color: #e8eef5; }
    h1 { margin-bottom: 8px; }
    .subtitle { color: #9fb3c8; margin-bottom: 16px; }
    .runs { display: grid; grid-template-columns: repeat(auto-fill, minmax(280px, 1fr)); gap: 16px; }
    .card { background: #122235; border: 1px solid #1f3247; border-radius: 8px; padding: 16px; box-shadow: 0 4px 12px rgba(0,0,0,0.3); }
    .title { font-weight: 700; margin-bottom: 8px; }
    .meta { font-size: 13px; color: #b7c6d9; margin-bottom: 4px; }
    .badge { display: inline-block; padding: 2px 6px; border-radius: 4px; font-size: 12px; margin-right: 6px; background: #1f8ef1; color: white; }
    .compare { margin-top: 24px; background: #122235; padding: 16px; border-radius: 8px; border: 1px solid #1f3247; }
    select, button { padding: 8px; margin-right: 8px; border-radius: 4px; border: 1px solid #294261; background: #0f1e30; color: #e8eef5; }
    button { cursor: pointer; background: #1f8ef1; border: none; }
    a { color: #1f8ef1; text-decoration: none; }
  </style>
</head>
<body>
  <h1>Pokemon Red RL Dashboard</h1>
  <div class="subtitle">Runs sorted by latest activity. Select two runs below to compare checkpoints.</div>
  <div id="runs" class="runs"></div>

  <div class="compare">
    <div style="margin-bottom:8px;font-weight:700;">Compare two runs</div>
    <select id="runA"></select>
    <select id="runB"></select>
    <button onclick="compare()">Open compare command</button>
    <div id="compareCmd" class="meta" style="margin-top:8px;"></div>
  </div>

  <script>
    async function loadRuns() {
      const res = await fetch('/api/runs');
      const data = await res.json();
      const runsDiv = document.getElementById('runs');
      runsDiv.innerHTML = '';
      const selA = document.getElementById('runA');
      const selB = document.getElementById('runB');
      selA.innerHTML = ''; selB.innerHTML = '';
      data.forEach((run, idx) => {
        const card = document.createElement('div');
        card.className = 'card';
        card.innerHTML = `
          <div class="title">${run.name}</div>
          <div class="meta">Latest steps: ${run.latest_steps ?? 'n/a'}</div>
          <div class="meta">Checkpoint: ${run.latest_checkpoint ? run.latest_checkpoint : 'n/a'}</div>
          <div class="meta">Streaming: ${run.metadata.stream_enabled ? 'on' : 'off'}</div>
          <div class="meta">Num envs: ${run.metadata.train_config ? run.metadata.train_config.num_envs : 'n/a'}</div>
          <div class="meta">Batch size: ${run.metadata.train_config ? run.metadata.train_config.batch_size : 'n/a'}</div>
        `;
        runsDiv.appendChild(card);
        const optA = document.createElement('option');
        optA.value = run.latest_checkpoint || '';
        optA.text = run.name;
        const optB = optA.cloneNode(true);
        selA.appendChild(optA);
        selB.appendChild(optB);
        if (idx === 1) selB.selectedIndex = 1;
      });
    }
    function compare() {
      const a = document.getElementById('runA').value;
      const b = document.getElementById('runB').value;
      if (!a || !b) {
        document.getElementById('compareCmd').innerText = 'Select two runs with checkpoints.';
        return;
      }
      const cmd = `python tools/compare_runs.py --checkpoint-a "${a}" --checkpoint-b "${b}" --rom PokemonRed.gb --state init.state`;
      document.getElementById('compareCmd').innerText = cmd;
    }
    loadRuns();
  </script>
</body>
</html>
"""


class DashboardHandler(SimpleHTTPRequestHandler):
    def do_GET(self):
        if self.path == "/" or self.path.startswith("/index.html"):
            self.send_response(200)
            self.send_header("Content-Type", "text/html")
            self.end_headers()
            self.wfile.write(DASHBOARD_HTML.encode("utf-8"))
            return
        if self.path.startswith("/api/runs"):
            runs = collect_runs(RUNS_DIR)
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(json.dumps(runs).encode("utf-8"))
            return
        return super().do_GET()


def main(port: int):
    server = HTTPServer(("0.0.0.0", port), DashboardHandler)
    print(f"Serving dashboard on http://localhost:{port}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Serve a simple dashboard for runs/")
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()
    main(args.port)
