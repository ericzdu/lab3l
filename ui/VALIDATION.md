# Integration validation

Validated on 2026-11-17 against upstream main commit `38ffed82b0e458e925e2b205db054a9aa0a2ad91`.

- `npm ci` and `npm run build`: passed TypeScript and production build.
- Existing `shadow-hand-teleop` pytest suite: 18 passed, using its existing Python dependencies and cached model assets.
- Fresh checkout: actual reader returned zero files; browser workflow correctly displayed the empty library.
- Real-data compatibility: a byte-identical local copy of an existing HDF5 recording was loaded through the current main `replay_data.py`. No generated dataset was used in the UI. This ignored recording is not committed.
- Browser workflow: real file listing, obsolete prototype-cache removal, unsupported controls disabled, actual metadata and numeric values, original-file download, search, four-page mobile layout, and unavailable-reader state passed without browser errors.
- Five API checks passed: repository-loader consistency, rejection of write methods, byte-identical recording download, path traversal rejection, and external JSON symlink rejection.
- The UI uses a configurable Python executable; the existing Python requirements and launchers are untouched. Testing used macOS Chrome; other operating systems and screen readers have not been validated here.

All additions are confined to `ui/`. Existing repository files, CI, recordings, model assets and other branches are outside this change. The accessibility report remains an audit of the earlier prototype, not a claim of completed accessibility remediation.
