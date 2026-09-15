"""Obstacle geometry, collision support, ray scans and safe approach positions."""
from copy import deepcopy
import unittest
import uuid
import xml.etree.ElementTree as ET

import mujoco
import numpy as np

from isal2 import PROJECT_ROOT
from isal2.deployment.parkour import EXTRA_TILES
from isal2.deployment.terrain import build_scene


class ParkourTerrainTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        root = ET.parse(PROJECT_ROOT / "assets/data/rpo/mjcf/rpo.xml").getroot()
        names = [motor.get("joint") for motor in root.find("actuator")]
        cls.meta = {"joint_names": names, "physics_dt": .001,
                    "joints": {"effort_limit": [120.] * 23, "armature": [.01] * 23}}
        cls.output = PROJECT_ROOT / "outputs/parkour_validation" / ("unit_" + uuid.uuid4().hex)
        cls.model, cls.details = build_scene(cls.meta, output=cls.output)
        cls.data = mujoco.MjData(cls.model)
        mujoco.mj_forward(cls.model, cls.data)

    def ray(self, x, y):
        distance = mujoco.mj_ray(self.model, self.data, np.array([x, y, 10.]), np.array([0., 0., -1.]),
            np.array([1, 0, 0, 0, 0, 0], dtype=np.uint8), True, -1, np.empty(1, dtype=np.int32))
        self.assertGreaterEqual(distance, 0)
        return 10. - distance

    def test_sixteen_distinct_bays_and_safe_approaches(self):
        tiles = self.details["tiles"]
        self.assertEqual(len(tiles), 16)
        self.assertEqual(len({tile["kind"] for tile in tiles}), 16)
        self.assertEqual([t["kind"] for t in tiles[8:]], EXTRA_TILES)
        self.assertEqual(self.details["layout"], {"rows": 4, "columns": 4})
        for tile in tiles:
            x, y, z = tile["spawn_position"]
            self.assertAlmostEqual(self.ray(x, y), z, places=6)
            if tile["kind"] in EXTRA_TILES:
                # Both feet and a small reset perturbation stay on the approach.
                for dx in (-.2, 0, .2):
                    for dy in (-.2, 0, .2):
                        self.assertAlmostEqual(self.ray(x + dx, y + dy), 0., places=6)

    def test_pillar_tops_bridges_and_real_pit_floors(self):
        for tile in self.details["tiles"][8:11]:
            x, y = tile["center"][:2]
            self.assertAlmostEqual(self.ray(x, y + 1.6), -1., places=6)
            for geom in tile["geoms"]:
                suffix = geom["name"][len(tile["name"]) + 1:]
                if suffix == "beam" or suffix.startswith(("pillar_", "bridge_")):
                    self.assertAlmostEqual(self.ray(*geom["position"][:2]), geom["top"], places=6)
        gaps = self.details["tiles"][12]
        self.assertAlmostEqual(self.ray(*gaps["center"][:2]), -1., places=6)
        bridge = self.details["tiles"][9]
        x, y = bridge["center"][:2]
        self.assertAlmostEqual(self.ray(x, y), 0., places=6)
        # Check the actual lateral edge rather than the metadata alone.
        self.assertAlmostEqual(self.ray(x, y + .19), 0., places=6)
        self.assertAlmostEqual(self.ray(x, y + .21), -1., places=6)

    def test_sphere_lands_on_bridge_but_falls_through_gap(self):
        tile = self.details["tiles"][9]
        source = ET.parse(self.output / "scene.xml").getroot()
        root = ET.Element("mujoco")
        ET.SubElement(root, "option", timestep=".001")
        world = ET.SubElement(root, "worldbody")
        for geom in source.find("worldbody").findall("geom"):
            if geom.get("name", "").startswith(tile["name"] + "_"):
                world.append(deepcopy(geom))
        x, y = tile["center"][:2]
        for name, dy in (("on_bridge", 0.), ("in_gap", 1.5)):
            body = ET.SubElement(world, "body", name=name, pos=f"{x} {y + dy} .3")
            ET.SubElement(body, "freejoint")
            ET.SubElement(body, "geom", type="sphere", size=".04", mass=".1")
        model = mujoco.MjModel.from_xml_string(ET.tostring(root, encoding="unicode"))
        data = mujoco.MjData(model)
        for _ in range(900):
            mujoco.mj_step(model, data)
        mujoco.mj_forward(model, data)
        self.assertAlmostEqual(data.body("on_bridge").xpos[2], .04, delta=.005)
        self.assertAlmostEqual(data.body("in_gap").xpos[2], -.96, delta=.005)

    def test_difficulty_changes_physical_width_and_height(self):
        dimensions = []
        for difficulty in (0., 1.):
            _, details = build_scene(self.meta, difficulty=difficulty, output=self.output / str(difficulty))
            beam = next(g for g in details["tiles"][9]["geoms"] if g["name"].endswith("_beam"))
            pillar = next(g for g in details["tiles"][8]["geoms"] if "_pillar_" in g["name"])
            dimensions.append([beam["size"][1], pillar["size"][0], details["tiles"][11]["max_height"]])
        self.assertGreater(dimensions[0][0], dimensions[1][0])
        self.assertGreater(dimensions[0][1], dimensions[1][1])
        self.assertLess(dimensions[0][2], dimensions[1][2])


if __name__ == "__main__":
    unittest.main()
