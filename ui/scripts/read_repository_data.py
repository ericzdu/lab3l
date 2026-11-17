"""Read real repository datasets without generating or modifying records."""
import json
import os
from pathlib import Path
import subprocess
import sys

root = Path(os.environ.get('LAB3L_REPO', str(Path(__file__).resolve().parents[2]))).resolve()
project = root / 'shadow-hand-teleop'
sys.path.insert(0, str(project / 'scripts'))

def main():
    from replay_data import load_demo
    records = []
    directory = project / 'data'
    if directory.exists():
        for path in sorted(directory.rglob('*')):
            if path.suffix.lower() not in ('.h5', '.hdf5', '.npz') or not path.is_file():
                continue
            if not path.resolve().is_relative_to(directory.resolve()):
                continue
            record = {'path': str(path.relative_to(directory)), 'name': path.name, 'bytes': path.stat().st_size, 'metadata': None, 'error': None, 'samples': None, 'duration': None, 'fields': [], 'preview': []}
            sidecar = path.with_suffix('.json')
            if sidecar.is_file() and sidecar.resolve().is_relative_to(directory.resolve()):
                try:
                    record['metadata'] = json.loads(sidecar.read_text())
                except (ValueError, OSError) as exc:
                    record['metadata_error'] = str(exc)
            try:
                arrays = load_demo(path)
                record['fields'] = [{'name': key, 'shape': list(value.shape)} for key, value in arrays.items() if hasattr(value, 'shape')]
                times = arrays.get('timestamp')
                if times is not None:
                    record['samples'] = len(times)
                    record['duration'] = float(times[-1] - times[0])
                    # Actual recorded values, never interpolated poses or inferred phases.
                    indices = sorted(set([0, len(times)//4, len(times)//2, len(times)*3//4, len(times)-1]))
                    record['preview'] = [{'index': i, 'timestamp': float(times[i]), 'qpos': arrays['qpos'][i].tolist(), 'action': arrays['action'][i].tolist()} for i in indices]
            except Exception as exc:
                record['error'] = str(exc)
            records.append(record)
    commit = subprocess.run(['git', '-C', str(root), 'rev-parse', 'HEAD'], capture_output=True, text=True, check=True).stdout.strip()
    print(json.dumps({'repository': 'ericzdu/lab3l', 'checkout': str(root), 'commit': commit, 'data_directory': str(directory), 'records': records}, allow_nan=False))

if __name__ == '__main__':
    main()
