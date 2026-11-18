// Result: video with camera tabs and timeline in the stage; outcome, confidence,
// placement and phases in the properties panel.
import { getJSON } from "./api.js";
import { renderTimeline, simTimeAt, videoTimeFor } from "./timeline.js";

const $ = (id) => document.getElementById(id);
let current = null;      // { folder, detail, camera, stopTimeline }

export function currentFolder() {
  return current && current.folder;
}

function badgeText(detail) {
  if (detail.outcome === "success") return "SUCCESS";
  if (detail.outcome === "fail") return `FAIL · ${detail.friendly_reason || "unknown"}`;
  return "NOT LABELED";
}

function setCamera(camera, { keepTime = true, autoplay = false } = {}) {
  const { detail } = current;
  const video = $("video");
  const cameras = detail.cameras;
  if (!cameras[camera]) camera = Object.keys(cameras)[0];
  for (const tab of document.querySelectorAll(".tabs button")) {
    tab.classList.toggle("active", tab.dataset.camera === camera);
    tab.disabled = !cameras[tab.dataset.camera];
  }
  if (current.stopTimeline) current.stopTimeline();
  if (!camera) {
    video.removeAttribute("src");
    video.load();
    current.stopTimeline = renderTimeline($("timeline"), detail, video, null);
    return;
  }
  const previous = current.camera && cameras[current.camera];
  const simTime = keepTime && previous ? simTimeAt(previous.frame_time, detail.video_fps, video.currentTime) : 0;
  current.camera = camera;
  const frameTime = cameras[camera].frame_time;
  video.src = `/api/episodes/${encodeURIComponent(current.folder)}/video/${camera}`;
  video.onloadedmetadata = () => {
    if (simTime > 0) video.currentTime = videoTimeFor(frameTime, detail.video_fps, simTime);
    if (autoplay) video.play().catch(() => {});
  };
  current.stopTimeline = renderTimeline($("timeline"), detail, video, frameTime);
}

export async function showEpisode(folder, { autoplay = false } = {}) {
  const detail = await getJSON(`/api/episodes/${encodeURIComponent(folder)}`);
  if (current && current.stopTimeline) current.stopTimeline();
  current = { folder, detail, camera: null, stopTimeline: null };
  const name = `Episode ${String(detail.number).padStart(4, "0")}`;
  $("result-empty").hidden = true;
  $("result-card").hidden = false;
  $("props-empty").hidden = true;
  $("props-list").hidden = false;
  $("crumb-episode").textContent = name;
  const badge = $("badge");
  badge.textContent = badgeText(detail);
  badge.className = `badge ${detail.outcome}`;
  $("episode-name").textContent = name;
  $("placement").textContent = detail.placement || "—";
  $("assist").textContent = detail.grasp_assist ? "On" : "Off";
  const confidence = detail.confidence == null ? 0 : detail.confidence;
  const percent = `${Math.round(100 * confidence)}%`;
  $("confidence-fill").style.width = percent;
  $("confidence-text").textContent = detail.confidence == null ? "—" : percent;
  $("phase-strip").textContent = detail.phase_strip || "—";
  setCamera("overview", { keepTime: false, autoplay });
  document.dispatchEvent(new CustomEvent("episode-shown", { detail: { folder } }));
}

export function initResult() {
  for (const tab of document.querySelectorAll(".tabs button")) {
    tab.addEventListener("click", () => current && setCamera(tab.dataset.camera, { autoplay: !$("video").paused }));
  }
}
