# Integration boundary

This directory is additive. It does not replace Python launchers, change actuator mapping, install Python packages, alter existing CI, generate datasets, delete files or merge into main.

The reader relies on `replay_data.load_demo(path)` and its flat timestamp/qpos/qvel/action arrays. Loader errors are displayed per file. Holos inputs are delegated to the repository loader; absent robot-state arrays remain unknown. Root discovery uses this directory's location rather than a developer's Desktop path.

Validate the frontend build, original Python suite, reader against current main, empty-data behavior, original-byte downloads and rejected path traversal. Commit only ui/ and use a normal non-forced branch update.

No robot-control or write API is implemented. Adding one later requires a separate integration with actual device lifecycle, tracking timeout and recording semantics.
