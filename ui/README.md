# LAB3L UI (optional)

Black-and-white React / TypeScript / shadcn/ui workspace for the existing Python project. All UI files and Node dependencies are isolated in `ui/`. The existing Python launchers, requirements, data files and tests are unchanged.

## Start

Use Node.js 22.16 or newer. Set up the existing `shadow-hand-teleop` Python environment according to its README, then:

```sh
cd ui
npm ci
npm run dev
```

Open http://localhost:5173. Close an existing server on that port before starting this one. The native MuJoCo viewer remains a separate desktop window.

The UI discovers the repository as the parent of `ui/`, and runs the reader with `shadow-hand-teleop/.venv/bin/python` (or `.venv/Scripts/python.exe` on Windows). To use another interpreter or checkout:

```sh
LAB3L_PYTHON=/absolute/path/to/python npm run dev
LAB3L_REPO=/absolute/path/to/lab3l npm run dev
```

The interpreter needs the existing project's dependencies. No additional Python dependency or change to its requirements is introduced. Missing environments produce a readable unavailable state. Windows users can set variables with their shell's normal environment syntax.

## Behavior and compatibility

- Teleoperation: native launcher commands and actual keyboard references. The browser does not send robot commands, launch programs, connect to a headset or simulate a successful connection.
- Datasets: read actual `.npz`, `.h5` and `.hdf5` files under `shadow-hand-teleop/data/` using the checkout's `scripts/replay_data.py`. Display actual array dimensions, timestamp counts and numeric rows, plus available JSON-sidecar metadata. Time span comes from recorded timestamps. Unknown values stay unknown.
- Inspection: download original data/sidecar bytes. Copy a command for native MuJoCo replay. Holos files use the existing loader's default `demo_0`; use native `--demo` for another group. The UI does not substitute fabricated robot states for tracking.
- Workflows / Setup: implemented entry points, source links and the exact checkout commit. Source links refer to GitHub main; the local commit is separate.

A fresh clone correctly starts with an empty library until you add or collect real data. Existing mock output stays labeled `mock_input` and `not_evaluated`. There are no seeded records, automatic success labels or browser-timer recordings.

The local reader is read-only. Downloads are limited to allowed extensions inside the real data directory; traversal and escaping symlinks are rejected. Python bytecode writing is disabled for reader subprocesses. No script or dataset is modified. Keep the server local: it is a development tool, not an authenticated hosted service.

## Build and checks

```sh
npm run build
npm run preview
```

The bundle alone has no Python backend. `vite preview` supplies the local reader; static hosting cannot read these local datasets. No Sites registration, deployment credentials, machine-specific paths or generated dataset copies are included.

For browser checks, install Playwright Chromium if needed (`npx playwright install chromium`), start the dev server and run `npm run test:ui`. Set `CHROME_PATH` to use an installed browser, `LAB3L_UI_URL` for another local URL, and `LAB3L_PYTHON` if needed. The check adapts to an empty checkout and compares available records to the actual reader. Only ignored screenshots and temporary downloads are created; source recordings are unchanged.

The accessibility audit describes the earlier audited prototype and recommendations. This integration does not claim those findings have been fixed.

Run `npm run test:api` against the running local server to compare the listing to the existing Python loader, verify byte-identical downloads, reject POST writes, and check traversal / escaping-symlink protection. It uses a temporary test-only symlink when a data directory exists and removes only that link afterward.
