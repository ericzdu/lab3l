"""Adapt HolosCapture world-space skeletons to fixed Shadow Hand flexion."""
import numpy as np
import mujoco

from hand_retargeting import finger_targets, actuator_indices, apply_targets

THUMB = ['thumbKnuckle', 'thumbIntermediateBase', 'thumbIntermediateTip', 'thumbTip']
FINGERS = [prefix + part for prefix in ('indexFinger', 'middleFinger', 'ringFinger', 'littleFinger')
           for part in ('Metacarpal', 'Knuckle', 'IntermediateBase', 'IntermediateTip', 'Tip')]


def read_holos(file, demo):
    path = f'data/{demo}'
    if path not in file:
        raise ValueError(f'Missing Holos demonstration: {path}')
    group = file[path]
    times = np.asarray(group['time'][...], dtype=float)
    if times.ndim != 1 or not len(times) or not np.isfinite(times).all() or np.any(np.diff(times) <= 0) or times[0] < 0:
        raise ValueError('Holos time must be finite, nonnegative and strictly increasing')
    result = {'timestamp': times}
    for side in ('left', 'right'):
        key = f'obs/hand_{side}_joints_world'
        if key not in group:
            continue
        dataset = group[key]
        names = dataset.attrs.get('joint_names')
        if names is None and 'joint' in file:
            names = file['joint'].asstr()[...]
        if names is None:
            raise ValueError('Holos skeleton is missing joint names')
        names = [name.decode() if isinstance(name, bytes) else str(name) for name in names]
        try:
            order = [names.index(name) for name in ['forearmWrist'] + THUMB + FINGERS]
        except ValueError as exc:
            raise ValueError('Holos skeleton is missing required hand joints') from exc
        poses = dataset[...]
        if poses.shape != (len(times), len(names), 7):
            raise ValueError(f'{key} must have shape (samples, joints, 7)')
        # World positions share one frame; parent-local positions cannot be used
        # directly to measure bends. Quaternions are unnecessary for flexion.
        result[f'hand_{side}'] = poses[:, order, :3]
        result[f'wrist_{side}'] = poses[:, order[0], 3:7]
    if len(result) == 1:
        raise ValueError('Holos demo has no world-space hand skeletons')
    return result


def retarget_holos(arrays, model):
    """Simulate targets at the recorded times; invalid hands hold their target."""
    data = mujoco.MjData(model)
    indices = {side: actuator_indices(model, side) for side in ('left', 'right')}
    target = data.ctrl.copy()
    output = {key: [] for key in ('qpos', 'qvel', 'action')}
    valid = {side: 0 for side in indices}
    times = arrays['timestamp']
    previous_target = target.copy()
    integrated_steps = 0
    for frame, stamp in enumerate(times):
        for side in indices:
            positions = arrays.get(f'hand_{side}')
            if positions is None:
                continue
            transforms = np.repeat(np.eye(4)[None], 25, axis=0)
            transforms[:, :3, 3] = positions[frame]
            try:
                values = finger_targets(transforms)
            except ValueError:
                continue
            apply_targets(target, values, indices[side], model.actuator_ctrlrange)
            valid[side] += 1
        # Integrate the preceding interval using the previous frame's command.
        # The new target then governs the next interval (zero-order hold).
        desired_steps = int(round((stamp - times[0]) / model.opt.timestep))
        for _ in range(desired_steps - integrated_steps):
            data.ctrl[:] += np.clip(previous_target - data.ctrl, -0.015, 0.015)
            mujoco.mj_step(model, data)
        integrated_steps = desired_steps
        previous_target = target.copy()
        if not np.isfinite(data.qpos).all():
            raise ValueError('Non-finite simulation state while retargeting Holos capture')
        output['qpos'].append(data.qpos.copy())
        output['qvel'].append(data.qvel.copy())
        output['action'].append(data.ctrl.copy())
    if not any(valid.values()):
        raise ValueError('Holos capture contains no usable hand tracking')
    print(f'Holos frames with usable tracking: {valid}; missing tracking holds targets.')
    return {'timestamp': times, **{key: np.asarray(value) for key, value in output.items()}}
