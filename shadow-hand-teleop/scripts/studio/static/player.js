// Custom player bar for the result video: time readout, scrubber, back/play/forward,
// loop and speed. The phase timeline (timeline.js) sits between scrubber and buttons.
const $ = (id) => document.getElementById(id);
const SKIP_S = 5;
const SPEEDS = [0.25, 0.5, 1, 2];
const PLAY = '<path d="M7 4.5 L15.5 10 L7 15.5 Z"/>';
const PAUSE = '<rect x="5.5" y="4.5" width="3" height="11" rx=".5"/><rect x="11.5" y="4.5" width="3" height="11" rx=".5"/>';

function clock(seconds) {
  if (!Number.isFinite(seconds)) seconds = 0;
  const m = Math.floor(seconds / 60);
  const s = Math.floor(seconds % 60);
  return `${m}:${String(s).padStart(2, "0")}`;
}

export function initPlayer() {
  const video = $("video");
  let speed = 1;

  const toggle = () => {
    if (!video.src) return;
    if (video.paused || video.ended) video.play().catch(() => {}); else video.pause();
  };
  const skip = (delta) => {
    if (!video.src || !Number.isFinite(video.duration)) return;
    video.currentTime = Math.max(0, Math.min(video.duration, video.currentTime + delta));
  };

  const drawState = () => {
    const playing = !video.paused && !video.ended;
    $("play-icon").innerHTML = playing ? PAUSE : PLAY;
    $("play").setAttribute("aria-label", playing ? "Pause" : "Play");
  };
  const drawTime = () => {
    const duration = video.duration;
    const fraction = Number.isFinite(duration) && duration > 0 ? video.currentTime / duration : 0;
    const percent = `${100 * Math.min(1, fraction)}%`;
    $("scrub-fill").style.width = percent;
    $("scrub-knob").style.left = percent;
    $("time-now").textContent = clock(video.currentTime);
    $("time-total").textContent = clock(duration);
    requestAnimationFrame(drawTime);
  };
  drawTime();

  for (const name of ["play", "pause", "ended", "emptied"]) video.addEventListener(name, drawState);
  // A new source resets playbackRate; keep the chosen speed across episodes and cameras.
  video.addEventListener("loadedmetadata", () => { video.playbackRate = speed; });

  video.addEventListener("click", toggle);
  $("play").addEventListener("click", toggle);
  $("back").addEventListener("click", () => skip(-SKIP_S));
  $("forward").addEventListener("click", () => skip(SKIP_S));
  $("loop").addEventListener("click", () => {
    video.loop = !video.loop;
    $("loop").setAttribute("aria-pressed", String(video.loop));
  });
  $("speed").addEventListener("click", () => {
    speed = SPEEDS[(SPEEDS.indexOf(speed) + 1) % SPEEDS.length];
    video.playbackRate = speed;
    $("speed").textContent = `${speed}×`;
  });

  // Scrubber: click or drag to seek.
  const scrub = $("scrub");
  const seekTo = (event) => {
    if (!video.src || !Number.isFinite(video.duration)) return;
    const box = scrub.getBoundingClientRect();
    const fraction = Math.max(0, Math.min(1, (event.clientX - box.left) / box.width));
    video.currentTime = fraction * video.duration;
  };
  scrub.addEventListener("pointerdown", (event) => {
    scrub.setPointerCapture(event.pointerId);
    seekTo(event);
    const move = (e) => seekTo(e);
    const up = () => {
      scrub.removeEventListener("pointermove", move);
      scrub.removeEventListener("pointerup", up);
      scrub.removeEventListener("pointercancel", up);
    };
    scrub.addEventListener("pointermove", move);
    scrub.addEventListener("pointerup", up);
    scrub.addEventListener("pointercancel", up);
  });

  // Space plays/pauses and arrows skip, unless typing in the form or a button has focus.
  document.addEventListener("keydown", (event) => {
    if (event.metaKey || event.ctrlKey || event.altKey) return;
    if (event.target.closest("input, textarea, select, button")) return;
    if (event.key === " ") { event.preventDefault(); toggle(); }
    else if (event.key === "ArrowLeft") { event.preventDefault(); skip(-SKIP_S); }
    else if (event.key === "ArrowRight") { event.preventDefault(); skip(SKIP_S); }
  });
}
