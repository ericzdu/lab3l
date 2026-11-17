// Episodes strip: recent kept episodes; clicking one loads it into the result card.
import { getJSON } from "./api.js";
import { currentFolder, showEpisode } from "./result.js";

const $ = (id) => document.getElementById(id);
const MINI = { success: "SUCCESS", fail: "FAIL", unlabeled: "—" };

export async function refreshEpisodes() {
  const data = await getJSON("/api/episodes");
  $("counts").textContent = `${data.count} episode${data.count === 1 ? "" : "s"} · `
    + `${data.successes} success${data.successes === 1 ? "" : "es"}`;
  const list = $("episode-list");
  list.replaceChildren(...data.episodes.map((episode) => {
    const card = document.createElement("button");
    card.type = "button";
    card.className = "card";
    card.dataset.folder = episode.folder;
    card.title = episode.friendly_reason ? `FAIL · ${episode.friendly_reason}` : episode.outcome;
    const number = document.createElement("span");
    number.className = "num";
    number.textContent = String(episode.number).padStart(4, "0");
    const mini = document.createElement("span");
    mini.className = `mini ${episode.outcome}`;
    mini.textContent = MINI[episode.outcome];
    card.append(number, mini);
    card.addEventListener("click", () => showEpisode(episode.folder, { autoplay: true }));
    return card;
  }));
  markCurrent();
  return data;
}

export function markCurrent() {
  for (const card of document.querySelectorAll(".card")) {
    card.classList.toggle("current", card.dataset.folder === currentFolder());
  }
}
