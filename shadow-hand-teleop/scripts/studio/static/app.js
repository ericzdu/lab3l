// lab3l Studio (demo cut): one page in three panes. Episodes on the left; session and the
// result video in the middle; the shown episode's properties on the right.
import { recall, remember } from "./api.js";
import { markCurrent, refreshEpisodes } from "./episodes.js";
import { initPlayer } from "./player.js";
import { initResult, showEpisode } from "./result.js";
import { initSession } from "./session.js";

function initPanes() {
  const app = document.getElementById("app");
  for (const [button, cls] of [["sidebar-toggle", "no-sidebar"], ["props-toggle", "no-props"]]) {
    app.classList.toggle(cls, recall(cls, false));
    document.getElementById(button).addEventListener("click", () => {
      remember(cls, app.classList.toggle(cls));
    });
  }
}

initPanes();
initResult();
initPlayer();
document.addEventListener("episode-shown", markCurrent);
refreshEpisodes().then((data) => {
  // Start with the newest kept episode on screen (not playing) until a new one arrives.
  if (data.episodes.length) showEpisode(data.episodes[0].folder);
}).catch(() => {});
initSession();
