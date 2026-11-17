// lab3l Studio (demo cut): one page. Session on top, result in the middle, episodes beside it.
import { markCurrent, refreshEpisodes } from "./episodes.js";
import { initResult, showEpisode } from "./result.js";
import { initSession } from "./session.js";

initResult();
document.addEventListener("episode-shown", markCurrent);
refreshEpisodes().then((data) => {
  // Start with the newest kept episode on screen (not playing) until a new one arrives.
  if (data.episodes.length) showEpisode(data.episodes[0].folder);
}).catch(() => {});
initSession();
