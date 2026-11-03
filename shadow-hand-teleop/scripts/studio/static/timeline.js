// Phase timeline under the video. Video time <-> sim time goes through the
// recorded frame_time (frame k shows sim time frame_time[k]), not currentTime
// alone, because the recorder may drop frames.
export const PHASE_COLORS = {
  idle: "#3a414d", reach: "#6aa9ff", grasp: "#f2c14e", lift: "#f28f3b", transport: "#b48cf2",
  place: "#4fd1c5", release: "#7bd88f", retreat: "#8e99ab", recover: "#e5484d",
};

export function simTimeAt(frameTime, fps, currentTime) {
  if (!frameTime || !frameTime.length || !fps) return currentTime;
  const k = Math.min(frameTime.length - 1, Math.max(0, Math.floor(currentTime * fps)));
  return frameTime[k];
}

export function videoTimeFor(frameTime, fps, simTime) {
  if (!frameTime || !frameTime.length || !fps) return simTime;
  let lo = 0, hi = frameTime.length - 1;
  while (lo < hi) {                      // first frame at or after simTime
    const mid = (lo + hi) >> 1;
    if (frameTime[mid] < simTime) lo = mid + 1; else hi = mid;
  }
  return (lo + 0.5) / fps;               // middle of that frame's display interval
}

export function renderTimeline(container, detail, video, frameTime) {
  container.replaceChildren();
  const duration = detail.duration_s || 1;
  const pct = (t) => `${Math.max(0, Math.min(100, (100 * t) / duration))}%`;
  for (const segment of detail.segments) {
    const div = document.createElement("div");
    div.className = "seg";
    div.style.left = pct(segment.start_t);
    div.style.width = `calc(${pct(segment.end_t - segment.start_t)} - 1px)`;
    div.style.background = PHASE_COLORS[segment.phase] || "#888";
    div.title = `${segment.phase} ${segment.start_t.toFixed(1)}-${segment.end_t.toFixed(1)} s`;
    if (segment.phase !== "idle") {
      const label = document.createElement("span");
      label.textContent = segment.phase;
      div.append(label);
    }
    container.append(div);
  }
  for (const event of detail.events) {
    const tick = document.createElement("div");
    tick.className = "tick";
    tick.style.left = pct(event.t);
    tick.title = `${event.kind.replaceAll("_", " ")} ${event.t.toFixed(2)} s`;
    container.append(tick);
  }
  const playhead = document.createElement("div");
  playhead.className = "playhead";
  container.append(playhead);
  const fps = detail.video_fps;
  container.onclick = (event) => {
    const box = container.getBoundingClientRect();
    const simTime = ((event.clientX - box.left) / box.width) * duration;
    if (video.src) video.currentTime = videoTimeFor(frameTime, fps, simTime);
  };
  let frame = null;
  const draw = () => {
    playhead.style.left = pct(simTimeAt(frameTime, fps, video.currentTime || 0));
    frame = requestAnimationFrame(draw);
  };
  draw();
  return () => cancelAnimationFrame(frame);
}
