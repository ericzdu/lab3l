import unittest
import mujoco
import numpy as np
from smoke_plume import SmokePlume


class SmokeTests(unittest.TestCase):
    def test_smoke_deflects_at_solid_surface(self):
        model = mujoco.MjModel.from_xml_string('''<mujoco><worldbody>
          <geom type="box" size=".2 .2 .1" pos="0 0 .6"/>
          <geom name="smoke_probe_geom" type="sphere" size=".001" pos="0 0 -20" contype="0" conaffinity="0" rgba="0 0 0 0"/>
        </worldbody></mujoco>''')
        data = mujoco.MjData(model)
        mujoco.mj_forward(model, data)
        plume = SmokePlume(model, [[0, 0, 0]], count=1)
        plume.positions[0] = [0, 0, .485]
        plume.velocities[0] = [0, 0, .4]
        plume.ages[0] = 0
        # Keep this single-particle test independent of emission.
        plume.carry = -100
        plume.step(data, .05)
        self.assertGreaterEqual(plume.collisions, 1)
        self.assertLess(plume.positions[0, 2], .5)
        self.assertLess(plume.velocities[0, 2], 0)
        self.assertGreater(np.linalg.norm(plume.velocities[0, :2]), 0)

    def test_existing_penetration_and_puff_radius(self):
        model = mujoco.MjModel.from_xml_string('''<mujoco><worldbody>
          <geom type="box" size=".2 .2 .1" pos="0 0 .6"/>
          <geom name="smoke_probe_geom" type="sphere" size=".001" pos="0 0 -20" contype="0" conaffinity="0"/>
        </worldbody></mujoco>''')
        data = mujoco.MjData(model)
        mujoco.mj_forward(model, data)
        for height in (.55, .495):
            plume = SmokePlume(model, [[0, 0, 0]], count=1)
            plume.positions[0] = [0, 0, height]
            plume.velocities[0] = [0, 0, .3]
            plume.ages[0] = 2
            plume.carry = -100
            plume.step(data, .01)
            self.assertLessEqual(plume.positions[0, 2] + plume.radius(plume.ages[0]), .498)

    def test_actual_shadow_palm_excludes_smoke_volume(self):
        from view_dual_shadow_hands import build_model
        model = build_model(moving=True)
        data = mujoco.MjData(model)
        mujoco.mj_forward(model, data)
        # Start inside an actual hand collision geometry, as a moving hand
        # sweeping into smoke would do. All surviving puffs need clearance.
        geom = next(i for i in range(model.ngeom)
                    if model.geom_group[i] == 3 and model.geom_type[i] == mujoco.mjtGeom.mjGEOM_BOX)
        plume = SmokePlume(model, [[0, 0, 0]], count=1)
        plume.positions[0] = data.geom_xpos[geom]
        plume.ages[0] = 1
        plume.carry = -100
        plume.step(data, .01)
        if plume.ages[0] < 4:
            plume.query_data.geom_xpos[plume.probe] = plume.positions[0]
            for obstacle in plume.obstacles:
                distance = mujoco.mj_geomDistance(model, plume.query_data, int(obstacle), plume.probe, .1, None)
                self.assertGreaterEqual(distance + .001, plume.radius(plume.ages[0]))
        self.assertGreater(plume.collisions, 0)

    def test_particle_budget_and_render(self):
        model = mujoco.MjModel.from_xml_string('<mujoco><worldbody><geom name="smoke_probe_geom" type="sphere" size=".001" contype="0" conaffinity="0"/></worldbody></mujoco>')
        data = mujoco.MjData(model)
        plume = SmokePlume(model, [[0, 0, .1]], count=20)
        for stamp in np.arange(0, 1, .02):
            plume.advance(data, stamp)
        self.assertTrue(np.isfinite(plume.positions).all())
        self.assertGreater(np.median(plume.positions[:, 2]), .1)
        scene = mujoco.MjvScene(model, maxgeom=10)
        plume.draw(scene)
        self.assertEqual(scene.ngeom, 10)
        plume.advance(data, 0)
        self.assertTrue((plume.ages == 99).all())


if __name__ == '__main__':
    unittest.main()
