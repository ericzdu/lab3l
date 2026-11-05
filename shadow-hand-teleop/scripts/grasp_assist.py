"""Optional grasp assist (--grasp-assist): weld a closed, touching hand to the cup.

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


def weld_name(side):
    return f"{side}_grasp_assist"


def palm_name(side):
    return f"{side}_{side[0]}h_palm"


def grip_closure(ctrl, indices):
    """Mean commanded closure of FF/MF/RF/LF as a fraction of J3 (pi/2) and J0 (pi)."""
    return float(np.mean([(ctrl[indices[f"{p}J3"]] / (np.pi / 2) + ctrl[indices[f"{p}J0"]] / np.pi) / 2
                          for p in ("FF", "MF", "RF", "LF")]))


class GraspAssist:
    def __init__(self, model, hands):
        from episode_recorder import ContactTables
        self.model, self.hands = model, hands
        self.contacts = ContactTables(model)
        self.welds = {side: model.equality(weld_name(side)).id for side in SIDES}
        self.palms = {side: model.body(palm_name(side)).id for side in SIDES}
        self.cup = model.body("cup").id
        self.active = {side: False for side in SIDES}
        self.last_touch = {side: -np.inf for side in SIDES}
        self.history = {side: [] for side in SIDES}   # (sim time, closure)

    def touching(self, data):
        """side -> (thumb, other finger group, any hand part) touching the cup."""
        hand_cup = self.contacts.measure(self.model, data)["hand_cup"]
        return {side: (bool(hand_cup[s, 0]), bool(hand_cup[s, 1:5].any()), bool(hand_cup[s, :6].any()))
                for s, side in enumerate(SIDES)}

    def update(self, data):
        """Call once per tick after physics; returns [(event kind, payload)]."""
        events = []
        touch = self.touching(data)
        for side in SIDES:
            thumb, finger, anything = touch[side]
            closure = grip_closure(data.ctrl, self.hands[side])
            if anything:
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
            elif (thumb and finger and closure > ON_CLOSURE and settled
                    and not any(self.active.values())):
                events.append(self._engage(data, side, closure))
        return events

    def _engage(self, data, side, closure):
        palm = self.palms[side]
        rotation = data.xmat[palm].reshape(3, 3)
        relative = rotation.T @ (data.xpos[self.cup] - data.xpos[palm])
        quat = np.zeros(4)
        inverse = np.zeros(4)
        mujoco.mju_negQuat(inverse, data.xquat[palm])
        mujoco.mju_mulQuat(quat, inverse, data.xquat[self.cup])
        weld = self.welds[side]
        # Weld data: anchor (3), relative position (3), relative quaternion (4), torque scale.
        self.model.eq_data[weld, :3] = 0
        self.model.eq_data[weld, 3:6] = relative
        self.model.eq_data[weld, 6:10] = quat
        data.eq_active[weld] = 1
        self.active[side] = True
        return "assist_on", {"side": side, "closure": round(closure, 3)}

    def _release(self, data, side, reason):
        data.eq_active[self.welds[side]] = 0
        self.active[side] = False
        return "assist_off", {"side": side, "reason": reason}

    def release_all(self, data, reason="reset"):
        """H or reset: drop any hold; returns the assist_off events."""
        return [self._release(data, side, reason) for side in SIDES if self.active[side]]

    def forget(self):
        """After a reset that already cleared data.eq_active (MuJoCo window Reset)."""
        for side in SIDES:
            self.active[side] = False
            self.last_touch[side] = -np.inf
            self.history[side] = []
