"""Read and write the project's flat, sample-major demonstration format."""
from pathlib import Path
import json

import numpy as np


def _h5py():
    try:
        import h5py
    except ImportError as exc:
        raise RuntimeError("HDF5 support requires h5py: install requirements.txt") from exc
    return h5py


def save_demo(path, arrays, metadata):
    path = Path(path)
    if path.suffix.lower() not in (".npz", ".h5", ".hdf5"):
        raise ValueError("Demo path must end in .npz, .h5, or .hdf5")
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.suffix.lower() == ".npz":
        np.savez_compressed(path, **arrays)
    else:
        with _h5py().File(path, "w") as file:
            file.attrs["schema_version"] = 1
            file.attrs["metadata"] = json.dumps(metadata)
            for name, values in arrays.items():
                file.create_dataset(name, data=values, compression="gzip")
    path.with_suffix(".json").write_text(json.dumps(metadata, indent=2) + "\n")


def load_demo(path, demo="demo_0"):
    path = Path(path)
    if path.suffix.lower() == ".npz":
        with np.load(path, allow_pickle=False) as file:
            arrays = {name: file[name] for name in file.files}
    elif path.suffix.lower() in (".h5", ".hdf5"):
        with _h5py().File(path, "r") as file:
            if "holos_convention_version" in file.attrs and "qpos" not in file:
                from holos_replay import read_holos
                return read_holos(file, demo)
            arrays = {name: file[name][...] for name in ("timestamp", "qpos", "qvel", "action") if name in file}
    else:
        raise ValueError("Replay path must end in .npz, .h5, or .hdf5")
    missing = set(("timestamp", "qpos", "qvel", "action")) - arrays.keys()
    if missing:
        raise ValueError(f"Missing replay datasets: {', '.join(sorted(missing))}")
    times = arrays["timestamp"]
    if times.ndim != 1 or not len(times):
        raise ValueError("timestamp must be a nonempty 1D array")
    for name in ("timestamp", "qpos", "qvel", "action"):
        values = arrays[name]
        if values.dtype.kind not in "fiu" or not np.isfinite(values).all():
            raise ValueError(f"{name} must contain finite numeric values")
        if name != "timestamp" and (values.ndim != 2 or len(values) != len(times)):
            raise ValueError(f"{name} must be a 2D array with one row per timestamp")
    if np.any(np.diff(times) <= 0) or times[0] < 0:
        raise ValueError("timestamps must be nonnegative and strictly increasing")
    return arrays


def validate_model(arrays, model):
    for name, width in (("qpos", model.nq), ("qvel", model.nv), ("action", model.nu)):
        if arrays[name].shape[1] != width:
            raise ValueError(f"{name} has {arrays[name].shape[1]} columns; model requires {width}")
