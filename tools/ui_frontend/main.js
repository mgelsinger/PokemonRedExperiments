const form = document.getElementById("run-form");
const stopBtn = document.getElementById("stop-btn");
const formStatus = document.getElementById("form-status");

function numberFmt(num) {
  if (num === null || num === undefined) return "-";
  if (Math.abs(num) >= 1000000) return (num / 1_000_000).toFixed(2) + "M";
  if (Math.abs(num) >= 1000) return (num / 1000).toFixed(1) + "k";
  return typeof num === "number" ? num.toLocaleString() : String(num);
}

function percentFmt(progress) {
  if (progress === null || progress === undefined) return "-";
  return (progress * 100).toFixed(2) + "%";
}

function setText(id, value) {
  const el = document.getElementById(id);
  if (el) el.textContent = value;
}

async function startRun(event) {
  event.preventDefault();
  const payload = {
    rom: document.getElementById("rom").value,
    state: document.getElementById("state").value,
    run_name: document.getElementById("run_name").value,
    preset: document.getElementById("preset").value || null,
    num_envs: parseInt(document.getElementById("num_envs").value, 10),
    batch_size: parseInt(document.getElementById("batch_size").value, 10),
    total_multiplier: parseInt(document.getElementById("total_multiplier").value, 10),
    checkpoint_freq: parseInt(document.getElementById("checkpoint_freq").value, 10) || null,
    status_interval: parseFloat(document.getElementById("status_interval").value),
    seed: document.getElementById("seed").value ? parseInt(document.getElementById("seed").value, 10) : null,
    eval_every_steps: parseInt(document.getElementById("eval_every_steps").value, 10) || null,
    eval_episodes: parseInt(document.getElementById("eval_episodes").value, 10),
    eval_max_steps: parseInt(document.getElementById("eval_max_steps").value, 10) || null,
    stream: document.getElementById("stream").checked,
    wandb: document.getElementById("wandb").checked,
    eval_stream: document.getElementById("eval_stream").checked,
  };

  formStatus.textContent = "Starting...";
  try {
    const res = await fetch("/api/run", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
    });
    if (!res.ok) {
      const msg = await res.text();
      throw new Error(msg || "Failed to start");
    }
    formStatus.textContent = "Run started.";
    setTimeout(() => (formStatus.textContent = ""), 2000);
    refreshStatus();
  } catch (err) {
    console.error(err);
    formStatus.textContent = err.message || "Failed to start.";
  }
}

async function stopRun() {
  try {
    const res = await fetch("/api/stop", { method: "POST" });
    if (!res.ok) {
      const msg = await res.text();
      throw new Error(msg || "Failed to stop");
    }
    formStatus.textContent = "Stop signal sent.";
    refreshStatus();
  } catch (err) {
    console.error(err);
    formStatus.textContent = err.message || "Failed to stop.";
  }
}

function renderMetrics(metrics) {
  const list = document.getElementById("metric-list");
  list.innerHTML = "";
  if (!metrics || Object.keys(metrics).length === 0) {
    list.innerHTML = "<li class='muted'>No metrics yet.</li>";
    return;
  }
  Object.entries(metrics).forEach(([k, v]) => {
    const li = document.createElement("li");
    li.textContent = `${k}: ${typeof v === "number" ? v.toFixed(5) : v}`;
    list.appendChild(li);
  });
}

function renderEvals(evals) {
  const el = document.getElementById("eval-list");
  el.innerHTML = "";
  if (!evals || evals.length === 0) {
    el.innerHTML = "<div class='muted'>No evals yet.</div>";
    return;
  }
  evals.forEach((row) => {
    const div = document.createElement("div");
    const ts = new Date(row.timestamp * 1000).toLocaleTimeString();
    div.innerHTML = `<strong>${ts}</strong> — mean reward ${row.mean_reward.toFixed(
      3
    )}, len ${row.mean_length.toFixed(1)} (episodes ${row.episodes})`;
    el.appendChild(div);
  });
}

function renderCheckpoints(list) {
  const el = document.getElementById("ckpt-list");
  el.innerHTML = "";
  if (!list || list.length === 0) {
    el.innerHTML = "<div class='muted'>No checkpoints yet.</div>";
    return;
  }
  list.forEach((ckpt) => {
    const div = document.createElement("div");
    const ts = new Date(ckpt.mtime * 1000).toLocaleString();
    div.innerHTML = `<strong>${ckpt.name}</strong> — ${ts} — ${Math.round(ckpt.size / 1024)} KB`;
    el.appendChild(div);
  });
}

function renderLogs(stdoutLines, stderrLines) {
  document.getElementById("stdout-log").textContent = stdoutLines.join("\n");
  document.getElementById("stderr-log").textContent = stderrLines.join("\n");
}

async function refreshStatus() {
  try {
    const res = await fetch("/api/status");
    if (!res.ok) return;
    const data = await res.json();
    const proc = data.process || {};
    const status = data.status || {};

    setText("status-run", status.run_name || (proc.run && proc.run.run_name) || "-");
    const progress = status.progress;
    setText("status-progress", progress !== undefined && progress !== null ? percentFmt(progress) : "-");
    setText("status-steps", `${numberFmt(status.timesteps_done)} / ${numberFmt(status.timesteps_total)}`);
    const throughput = status.throughput_steps_per_sec;
    setText(
      "status-throughput",
      throughput === undefined || throughput === null ? "-" : `${throughput.toFixed(1)} steps/s`
    );
    const lastEval = status.last_eval ? `r ${status.last_eval.mean_reward.toFixed(3)} len ${status.last_eval.mean_length.toFixed(1)}` : "-";
    setText("status-last-eval", lastEval);
    setText("status-pid", proc.pid || "-");

    renderMetrics(status.latest_metrics);
    renderEvals(data.evals);
    renderCheckpoints(data.checkpoints);
    renderLogs(proc.stdout_tail || [], proc.stderr_tail || []);
  } catch (err) {
    console.error("status refresh failed", err);
  }
}

form.addEventListener("submit", startRun);
stopBtn.addEventListener("click", stopRun);

refreshStatus();
setInterval(refreshStatus, 4000);
