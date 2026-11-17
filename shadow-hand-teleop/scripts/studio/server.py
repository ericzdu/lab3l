"""lab3l Studio (demo cut): a localhost web page for recording sessions and results.

    ./run_studio.command                              # from the project folder
    .venv/bin/python -m studio.server --port 8765     # from scripts/

Binds 127.0.0.1 only. Requests must name 127.0.0.1:<port> or localhost:<port>
as Host (DNS-rebinding guard), and every POST must carry the X-Studio-Token
generated at start and injected into the page. Episode folders are resolved
inside data/episodes only (studio.episodes.resolve).
"""
from __future__ import annotations

import argparse
import hmac
from pathlib import Path
import secrets
import sys

from flask import Flask, abort, jsonify, request, send_file

SCRIPTS = Path(__file__).resolve().parents[1]
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

from studio import episodes  # noqa: E402

STATIC = Path(__file__).with_name("static")
DEFAULT_PORT = 8765
HOST = "127.0.0.1"


def create_app(episodes_dir=None, port=DEFAULT_PORT, token=None, session=None):
    """Flask app; `session` is a studio.session.Session (or None: no live controls)."""
    from episode_recorder import DEFAULT_RECORD_DIR
    app = Flask(__name__, static_folder=None)
    app.config.update(EPISODES=Path(episodes_dir or DEFAULT_RECORD_DIR), TOKEN=token or secrets.token_urlsafe(24),
                      PORT=int(port), SESSION=session)
    allowed_hosts = {f"{HOST}:{port}", f"localhost:{port}"}

    @app.before_request
    def guard():
        if request.host not in allowed_hosts:
            abort(403, "bad host")
        if request.method not in ("GET", "HEAD", "OPTIONS"):
            sent = request.headers.get("X-Studio-Token", "")
            if not hmac.compare_digest(sent.encode(), app.config["TOKEN"].encode()):
                abort(403, "bad token")

    @app.after_request
    def headers(response):
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["Referrer-Policy"] = "no-referrer"
        if request.path.startswith("/api/") and response.mimetype == "application/json":
            response.headers["Cache-Control"] = "no-store"
        return response

    @app.get("/")
    def index():
        page = STATIC / "index.html"
        if not page.is_file():
            return "<!doctype html><title>lab3l Studio</title><p>Studio API is running.</p>"
        html = page.read_text().replace("{{STUDIO_TOKEN}}", app.config["TOKEN"])
        return html, 200, {"Content-Type": "text/html; charset=utf-8", "Cache-Control": "no-store"}

    @app.get("/static/<path:name>")
    def static_file(name):
        path = (STATIC / name).resolve()
        if STATIC.resolve() not in path.parents or not path.is_file():
            abort(404)
        return send_file(path, conditional=True, max_age=0)

    @app.get("/api/episodes")
    def episode_list():
        return jsonify(episodes.list_episodes(app.config["EPISODES"]))

    @app.get("/api/episodes/<folder>")
    def episode_detail(folder):
        path = episodes.resolve(app.config["EPISODES"], folder)
        if path is None:
            abort(404)
        try:
            return jsonify(episodes.detail(path))
        except (OSError, ValueError, KeyError) as exc:
            abort(500, f"unreadable episode: {type(exc).__name__}")

    @app.get("/api/episodes/<folder>/video/<camera>")
    def episode_video(folder, camera):
        path = episodes.video_path(app.config["EPISODES"], folder, camera)
        if path is None:
            abort(404)
        return send_file(path, mimetype="video/mp4", conditional=True, max_age=0)

    def live():
        if app.config["SESSION"] is None:
            abort(503, "no live session support")
        return app.config["SESSION"]

    @app.get("/api/session")
    def session_state():
        return jsonify(live().state())

    @app.get("/api/session/log")
    def session_log():
        return jsonify({"lines": live().log_lines()})

    @app.post("/api/session/start")
    def session_start():
        body = request.get_json(silent=True) or {}
        try:
            live().start(body.get("address"), operator=body.get("operator", ""),
                         auto_recenter=bool(body.get("auto_recenter", True)),
                         headset_view=bool(body.get("headset_view", True)),
                         grasp_assist=bool(body.get("grasp_assist", False)))
        except ValueError as exc:
            return jsonify({"error": str(exc)}), 400
        except RuntimeError as exc:
            return jsonify({"error": str(exc)}), 409
        return jsonify(live().state())

    @app.post("/api/session/stop")
    def session_stop():
        live().stop()
        return jsonify(live().state())

    @app.post("/api/session/key")
    def session_key():
        key = (request.get_json(silent=True) or {}).get("key")
        try:
            sent = live().send_key(key)
        except ValueError as exc:
            return jsonify({"error": str(exc)}), 400
        return jsonify({"sent": sent}), 200 if sent else 409

    return app


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--port", type=int, default=DEFAULT_PORT)
    parser.add_argument("--episodes", help="Episode folder (default: data/episodes)")
    parser.add_argument("--open", action="store_true", help="Open the page in the default browser")
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    from studio.session import Session
    session = Session()
    app = create_app(args.episodes, port=args.port, session=session)
    url = f"http://{HOST}:{args.port}/"
    print(f"lab3l Studio: {url}  (Ctrl-C stops it)", flush=True)
    if args.open:
        import threading
        import webbrowser
        threading.Timer(.8, webbrowser.open, args=(url,)).start()
    import signal

    def closing(signum, frame):
        raise SystemExit(0)

    # Closing the terminal window (SIGHUP) or a plain kill still stops the sim cleanly.
    signal.signal(signal.SIGHUP, closing)
    signal.signal(signal.SIGTERM, closing)
    try:
        app.run(host=HOST, port=args.port, threaded=True, use_reloader=False)
    finally:
        # Ctrl-C or the terminal closing: Esc to the sim (saves an open episode), then kill.
        if session.running():
            print("Stopping the simulation…", flush=True)
            session.stop()


if __name__ == "__main__":
    main()
