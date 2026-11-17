// Session: connect form, hand chips, step guide, buttons, shortcuts, log.
import { getJSON, post, recall, remember } from "./api.js";
import { refreshEpisodes } from "./episodes.js";
import { showEpisode } from "./result.js";

const $ = (id) => document.getElementById(id);
const SHORTCUTS = { h: "H", c: "C", e: "E", x: "X" };
let state = { running: false };
let lastSaved = undefined;     // folder of the newest saved episode seen this page load
let formOpen = true;
let logOpen = false;

function setForm(open) {
  formOpen = open;
  $("connect").hidden = !open;
  $("settings-link").hidden = open || !state.running;
}

function render() {
  const running = state.running;
  $("live").hidden = !running;
  $("start").disabled = running;
  if (running && formOpen && !$("connect").contains(document.activeElement)) setForm(false);
  if (!running) setForm(true);
  const status = state.status || {};
  for (const side of ["left", "right"]) {
    const chip = $(`chip-${side}`);
    chip.className = `chip ${state.chips ? state.chips[side] : "grey"}`;
    const s = (status.sides || {})[side] || {};
    chip.title = !s.live ? "no tracking" : s.follow ? "LIVE · FOLLOW" : "LIVE · press Calibrate";
  }
  for (const step of document.querySelectorAll("#steps li")) {
    const name = step.dataset.step;
    step.classList.toggle("next", state.next_step === name);
    step.classList.toggle("recording", name === "record" && state.next_step === "recording");
  }
  const key = { reset: "H", calibrate: "C", record: "E" }[state.next_step];
  for (const button of document.querySelectorAll(".buttons button[data-key]")) {
    button.classList.toggle("next", button.dataset.key === key);
  }
  const record = $("record");
  if (status.recording) {
    const seconds = status.rec_seconds || 0;
    const mm = String(Math.floor(seconds / 60)).padStart(2, "0");
    const ss = (seconds % 60).toFixed(1).padStart(4, "0");
    record.innerHTML = `● Stop ${mm}:${ss} <kbd>E</kbd>`;
    record.classList.add("on");
  } else {
    record.innerHTML = "Record <kbd>E</kbd>";
    record.classList.remove("on");
  }
  $("notice").textContent = status.error ? `Tracking: ${status.error}`
    : status.paused ? "Paused" : status.notice || "";
  $("exit-code").textContent = state.crashed ? `Simulation stopped unexpectedly (exit code ${state.exit_code})` : "";
}

async function poll() {
  try {
    const previous = state;
    state = await getJSON("/api/session");
    if (state.crashed && !previous.crashed) showLog(true);
    const saved = state.status && state.status.last_saved;
    if (saved && saved.folder !== lastSaved) {
      lastSaved = saved.folder;
      await refreshEpisodes();
      await showEpisode(saved.folder, { autoplay: true });
    }
    render();
    if (logOpen) await loadLog();
  } catch { /* server restarting: keep polling */ }
  setTimeout(poll, state.running ? 250 : 1000);
}

async function loadLog() {
  const { lines } = await getJSON("/api/session/log");
  const pre = $("log");
  const atBottom = pre.scrollTop + pre.clientHeight >= pre.scrollHeight - 4;
  pre.textContent = lines.join("\n");
  if (atBottom) pre.scrollTop = pre.scrollHeight;
}

function showLog(open) {
  logOpen = open;
  $("log").hidden = !open;
  $("log-toggle").textContent = open ? "Hide log" : "Show log";
  if (open) loadLog().catch(() => {});
}

async function key(name) {
  if (!state.running) return;
  try { await post("/api/session/key", { key: name }); } catch { /* not connected yet */ }
}

export function initSession() {
  $("address").value = recall("address", "");
  $("operator").value = recall("operator", "");
  for (const id of ["auto-recenter", "headset-view", "grasp-assist"]) {
    $(id).checked = recall(id, $(id).checked);
  }
  $("connect").addEventListener("submit", async (event) => {
    event.preventDefault();
    $("form-error").textContent = "";
    const body = {
      address: $("address").value.trim(), operator: $("operator").value.trim(),
      auto_recenter: $("auto-recenter").checked, headset_view: $("headset-view").checked,
      grasp_assist: $("grasp-assist").checked,
    };
    remember("address", body.address);
    remember("operator", body.operator);
    for (const id of ["auto-recenter", "headset-view", "grasp-assist"]) remember(id, $(id).checked);
    try {
      state = await post("/api/session/start", body);
      lastSaved = undefined;
      setForm(false);
      render();
    } catch (error) {
      $("form-error").textContent = error.message;
    }
  });
  $("settings-link").addEventListener("click", () => setForm(true));
  $("stop").addEventListener("click", async () => {
    $("stop").disabled = true;
    $("stop").textContent = "Stopping…";
    try { state = await post("/api/session/stop"); } finally {
      $("stop").disabled = false;
      $("stop").textContent = "Stop session";
      render();
    }
  });
  for (const button of document.querySelectorAll(".buttons button[data-key]")) {
    button.addEventListener("click", () => { key(button.dataset.key); button.blur(); });
  }
  document.addEventListener("keydown", (event) => {
    if (event.metaKey || event.ctrlKey || event.altKey || event.repeat) return;
    if (event.target.closest("input, textarea, select")) return;
    const name = SHORTCUTS[event.key.toLowerCase()];
    if (name) { event.preventDefault(); key(name); }
  });
  $("log-toggle").addEventListener("click", () => showLog(!logOpen));
  poll();
}
