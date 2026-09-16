"""Real mesh/metadata checks without importing the simulator."""
import math
import unittest
import numpy as np
import torch
from isal2.tasks.sparse.geometry import build_tile
from isal2.tasks.sparse.terrain_cfg import TYPES, WIDTHS, columns
from isal2.tasks.sparse.commands import route_velocity
from isal2.tasks.sparse.outcomes import inside_rectangles


class SparseGeometryTests(unittest.TestCase):
    def test_mixture_and_exact_widths(self):
        names = columns()
        self.assertEqual(len(names), 20)
        self.assertEqual(names.count('single_beam'), 6)
        self.assertEqual(names.count('radial_beams'), 4)
        self.assertEqual(len(columns(True, 40)), 40)
        with self.assertRaises(ValueError):
            columns(count=10)
        for level, width in enumerate(WIDTHS):
            tile = build_tile('single_beam', level)
            self.assertEqual(tile.supports[2][3]*2, width)
            self.assertEqual(tile.supports[2][2]*2, 3.)

    def test_holes_not_covered_at_support_height(self):
        for kind, point in [('single_beam', (0, 1, 0)), ('single_gap', (0, 0, 0)),
                            ('grid_stones', (-1.45, 0, 0))]:
            tile = build_tile(kind, 0)
            support = torch.tensor([tile.supports], dtype=torch.float)
            self.assertFalse(inside_rectangles(torch.tensor([[point]], dtype=torch.float), support).any())
            # The only mesh covering a side-of-beam XY point is the deep pit floor.
            if kind == 'single_beam':
                tops = [m.bounds[1, 2] for m in tile.meshes if
                        m.bounds[0, 0] <= 0 <= m.bounds[1, 0] and m.bounds[0, 1] <= 1 <= m.bounds[1, 1]]
                self.assertEqual(tops, [-1.])

    def test_every_tile_has_safe_spawn_and_exit(self):
        for kind in TYPES:
            for level in (0, 4, 9):
                tile = build_tile(kind, level)
                self.assertTrue(all(np.isfinite(m.vertices).all() for m in tile.meshes))
                for route in tile.routes:
                    p = torch.tensor([[route.spawn]], dtype=torch.float)
                    self.assertTrue(inside_rectangles(p, torch.tensor([tile.supports], dtype=torch.float)).any(), (kind, level))
                    self.assertLess(route.entry, route.exit)

    def test_legacy_dimensions_and_rescue(self):
        for i, length in enumerate((1, 2, 3)):
            tile = build_tile('single_beam', i, rescue=True)
            self.assertEqual(tile.params['width'], .8)
            self.assertEqual(tile.params['length'], length)
        for d in (0., 1/3, 2/3, 1.):
            tile = build_tile('legacy_star', round(d*9), legacy_difficulty=d)
            self.assertAlmostEqual(tile.params['width'], .4-.15*d)
            self.assertEqual(tile.params['pit_depth'], 10.)

    def test_world_route_command_rotates_with_robot(self):
        speed = torch.tensor([.5, .5, .5])
        result = route_velocity(speed, torch.zeros(3), torch.tensor([0., math.pi/2, math.pi]))
        self.assertTrue(torch.allclose(result, torch.tensor([[.5, 0], [0, -.5], [-.5, 0]]), atol=1e-6))

    def test_mesh_rays_across_edge_sampling_phases(self):
        import trimesh
        for level in (0, 4, 9):
            tile = build_tile('single_beam', level)
            mesh = trimesh.util.concatenate(tile.meshes)
            half = tile.params['width']/2
            for phase in (0., .025, .049):
                starts = np.array([[phase, half-.01, 20.], [phase, half+.01, 20.]])
                hits, rays, _ = mesh.ray.intersects_location(starts, np.tile((0., 0., -1.), (2, 1)))
                tops = [hits[rays == i, 2].max() for i in range(2)]
                np.testing.assert_allclose(tops, [0., -1.], atol=1e-7)


if __name__ == '__main__':
    unittest.main()
