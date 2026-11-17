"""Optional grasp assist (--grasp-assist): weld a closed, touching hand to the held object.

Works for the task's objects (the cup, or props such as the apple or banana;
workspace_scene.task_held_objects); "the cup" below stands for any of them.

The weld equality constraints exist in every workspace model (added by
workspace_scene.add_workspace, inactive), so nq/nv/nu never change and old
episodes still replay. This only switches one on and off:

- on, at the palm-cup pose of that instant (the cup is never moved), when the
  thumb AND at least one finger group touch the cup and the commanded grip
  closure exceeds ON_CLOSURE and has settled (stopped closing);
- off when closure drops below OFF_CLOSURE (hysteresis), when the hand has
  not touched the cup for LOST_SECONDS, or on H/reset (release_all).

Only one hand holds the cup at a time. Episodes record it honestly: per-sample
assist/active, metadata grasp_assist, and assist_on/assist_off events. The
labeler sees only simulated state, so assisted grasps are labeled the same way.
"""
from __future__ import annotations

import mujoco
import numpy as np

from arm_control import SIDES

ON_CLOSURE = .5      # mean finger closure, 0 open .. 1 fully flexed (keyboard fist ~0.8)
OFF_CLOSURE = .35
LOST_SECONDS = .3    # sim time without any hand-cup contact before release
# Engage only once the commanded grip has stopped closing (changed by less
# than SETTLE_CHANGE over SETTLE_SECONDS): welding mid-close freezes the cup
# low in the hand, where the closing fingers would have drawn it in.
SETTLE_SECONDS = .15
SETTLE_CHANGE = .02


def weld_name(side, obj="cup"):
    return f"{side}_grasp_assist" if obj == "cup" else f"{side}_grasp_assist_{obj}"


def palm_name(side):
    return f"{side}_{side[0]}h_palm"


def grip_closure(ctrl, indices):
    """Mean commanded closure of FF/MF/RF/LF as a fraction of J3 (pi/2) and J0 (pi)."""
    return float(np.mean([(ctrl[indices[f"{p}J3"]] / (np.pi / 2) + ctrl[indices[f"{p}J0"]] / np.pi) / 2
                          for p in ("FF", "MF", "RF", "LF")]))


class GraspAssist:
    def __init__(self, model, hands, objects=("cup",)):
        from episode_recorder import ContactTables
        self.model, self.hands = model, hands
        self.objects = list(objects)
        self.contacts = ContactTables(model, [o for o in self.objects if o != "cup"])
        self.welds = {(side, obj): model.equality(weld_name(side, obj)).id
                      for side in SIDES for obj in self.objects}
        self.palms = {side: model.body(palm_name(side)).id for side in SIDES}
        self.bodies = {obj: model.body(obj).id for obj in self.objects}
        self.active = {side: False for side in SIDES}
        self.holding = {side: None for side in SIDES}   # object welded to that hand
        self.last_touch = {side: -np.inf for side in SIDES}
        self.history = {side: [] for side in SIDES}   # (sim time, closure)

    def touching(self, data):
        """side -> {object: (thumb, other finger group, any hand part) touching it}."""
        measured = self.contacts.measure(self.model, data)
        result = {side: {} for side in SIDES}
        for obj in self.objects:
            hand = measured[f"hand_{obj}"]
            for s, side in enumerate(SIDES):
                result[side][obj] = (bool(hand[s, 0]), bool(hand[s, 1:5].any()), bool(hand[s, :6].any()))
        return result

    def update(self, data):
        """Call once per tick after physics; returns [(event kind, payload)]."""
        events = []
        touch = self.touching(data)
        for side in SIDES:
            held = self.holding[side]
            closure = grip_closure(data.ctrl, self.hands[side])
            if (touch[side][held][2] if held else any(t[2] for t in touch[side].values())):
                self.last_touch[side] = data.time
            history = self.history[side]
            history.append((data.time, closure))
            while len(history) > 1 and history[1][0] <= data.time - SETTLE_SECONDS:
                history.pop(0)
            settled = (history[0][0] <= data.time - SETTLE_SECONDS + 1e-9
                       and max(c for _, c in history) - min(c for _, c in history) < SETTLE_CHANGE)
            if self.active[side]:
                reason = ("opened" if closure < OFF_CLOSURE else
                          "contact_lost" if data.time - self.last_touch[side] > LOST_SECONDS else None)
                if reason:
                    events.append(self._release(data, side, reason))
            elif closure > ON_CLOSURE and settled and not any(self.active.values()):
                gripped = [obj for obj, (thumb, finger, _) in touch[side].items() if thumb and finger]
                if gripped:
                    events.append(self._engage(data, side, closure, gripped[0]))
        return events

    def _engage(self, data, side, closure, obj="cup"):
        palm, body = self.palms[side], self.bodies[obj]
        rotation = data.xmat[palm].reshape(3, 3)
        relative = rotation.T @ (data.xpos[body] - data.xpos[palm])
        quat = np.zeros(4)
        inverse = np.zeros(4)
        mujoco.mju_negQuat(inverse, data.xquat[palm])
        mujoco.mju_mulQuat(quat, inverse, data.xquat[body])
        weld = self.welds[side, obj]
        # Weld data: anchor (3), relative position (3), relative quaternion (4), torque scale.
        self.model.eq_data[weld, :3] = 0
        self.model.eq_data[weld, 3:6] = relative
        self.model.eq_data[weld, 6:10] = quat
        data.eq_active[weld] = 1
        self.active[side], self.holding[side] = True, obj
        payload = {"side": side, "closure": round(closure, 3)}
        if obj != "cup":
            payload["object"] = obj
        return "assist_on", payload

    def _release(self, data, side, reason):
        if self.holding[side] is not None:
            data.eq_active[self.welds[side, self.holding[side]]] = 0
        self.active[side], self.holding[side] = False, None
        return "assist_off", {"side": side, "reason": reason}

    def release_all(self, data, reason="reset"):
        """H or reset: drop any hold; returns the assist_off events."""
        return [self._release(data, side, reason) for side in SIDES if self.active[side]]

    def forget(self):
        """After a reset that already cleared data.eq_active (MuJoCo window Reset)."""
        for side in SIDES:
            self.active[side], self.holding[side] = False, None
            self.last_touch[side] = -np.inf
            self.history[side] = []
