"""One source of truth for collision meshes and support/route metadata.

Coordinates are tile-centred. Rectangles are (x, y, half_x, half_y, yaw, top_z).
The pit floor is a collision surface, but is never a valid support rectangle.
"""
from dataclasses import dataclass, asdict
import math
import numpy as np
import trimesh
from .terrain_cfg import parameters, REVIEW


@dataclass
class Route:
    spawn: tuple
    yaw: float
    entry: float
    exit: float
    half_width: float
    exit_region: tuple


@dataclass
class Tile:
    kind: str
    level: int
    seed: int
    params: dict
    supports: list
    routes: list
    meshes: list

    def metadata(self):
        return dict(kind=self.kind, level=self.level, seed=self.seed, params=self.params,
                    supports=self.supports, routes=[asdict(r) for r in self.routes])


def box(rect, depth=1.):
    x, y, hx, hy, yaw, z = rect
    transform = trimesh.transformations.rotation_matrix(yaw, (0, 0, 1))
    transform[:3, 3] = (x, y, z-depth/2)
    return trimesh.creation.box((2*hx, 2*hy, depth), transform)


def verify_legacy_star(reference_cfg):
    """Compare actual ray heights against the unchanged Isaac MeshStar generator."""
    probes = np.array([(x, y, 20.) for x in np.linspace(-3.93, 3.93, 35)
                       for y in np.linspace(-3.93, 3.93, 35)])
    directions = np.tile((0., 0., -1.), (len(probes), 1))
    def heights(mesh):
        hits, ids, _ = mesh.ray.intersects_location(probes, directions)
        result = np.full(len(probes), -np.inf)
        np.maximum.at(result, ids, hits[:, 2])
        return result
    for difficulty in (0., 1/3, 2/3, 1.):
        cfg = reference_cfg.copy()
        cfg.size = (8., 8.)
        original, _ = cfg.function(difficulty, cfg)
        original = trimesh.util.concatenate(original)
        original.apply_translation((-4., -4., 0.))
        actual = trimesh.util.concatenate(build_tile('legacy_star', round(difficulty*9), legacy_difficulty=difficulty).meshes)
        np.testing.assert_allclose(heights(original), heights(actual), atol=1.e-6)
    return 4*len(probes)


def build_tile(kind, level, seed=42, rescue=False, legacy_difficulty=None):
    p = parameters(kind, level, rescue)
    rng = np.random.default_rng(seed)
    supports, routes = [], []
    w, length = p['width'], p['length']
    if kind in ('radial_beams', 'legacy_star'):
        # Same 24-sided central platform and 12 diametral bars as MeshStarTerrain.
        d = level / 9 if legacy_difficulty is None else legacy_difficulty
        if kind == 'legacy_star':
            w = .4 - .15*d
            p.update(width=w, difficulty=d)
        depth = p['pit_depth']
        meshes = [trimesh.creation.cylinder(1., depth, sections=24,
                   transform=trimesh.transformations.translation_matrix((0, 0, -depth/2)))]
        p['central_platform_radius'] = 1.
        # The central polygon is queried analytically; this inscribed rectangle
        # also makes conservative spawn checks available to plain rectangle users.
        supports.append((0., 0., .70, .70, 0., 0.))
        rim = w if kind == 'legacy_star' else .8
        for i in range(12):
            yaw = i*math.pi/12
            radius = 4/max(abs(math.cos(yaw)), abs(math.sin(yaw)))
            supports.append((0., 0., radius-w/2, w/2, yaw, 0.))
        supports.extend([(0, -4+rim/2, 4, rim/2, 0, 0), (0, 4-rim/2, 4, rim/2, 0, 0),
                         (-4+rim/2, 0, rim/2, 4-rim, 0, 0), (4-rim/2, 0, rim/2, 4-rim, 0, 0)])
        # Full bars and exterior rim match the legacy generator dimensions.
        meshes.extend(box(r, depth) for r in supports[1:])
        for i in range(24):
            yaw = i*math.pi/12
            t = np.array((math.cos(yaw), math.sin(yaw)))
            radius = 4/max(abs(t))
            end = radius - rim*.5/max(abs(t))
            routes.append(Route(tuple(-.25*t)+(0.,), yaw, 1., end-.05, w/2,
                                tuple(end*t)+(max(.05, rim*.4), w*.4, yaw, 0.)))
    else:
        # Spawn and exit platforms cannot connect around the sides of a pit.
        supports.extend([(-2.75, 0, 1.25, 4, 0, 0), (2.75, 0, 1.25, 4, 0, 0)])
        if kind == 'single_beam':
            supports = [(-2-length/4, 0, (4-length/2)/2, 4, 0, 0),
                        (2+length/4, 0, (4-length/2)/2, 4, 0, 0),
                        (0, 0, length/2, w/2, 0, 0)]
        elif kind in ('grid_stones', 'single_stones', 'pallets'):
            sw = p['pallet_width'] if kind == 'pallets' else p['stone_width']
            gap = p['pallet_gap'] if kind == 'pallets' else p['stone_gap']
            n = max(2, math.ceil(3/(sw+gap)))
            # Exact gaps; the final stone may join the exit platform.
            actual_gap = gap
            p['actual_gap'] = actual_gap
            ys = (0.,) if kind != 'grid_stones' else (-sw-gap, 0., sw+gap)
            for i in range(n):
                x = -1.5+actual_gap+sw/2+i*(sw+actual_gap)
                for y in ys:
                    supports.append((x, y, sw/2, sw/2, 0, 0))
        elif kind in ('single_gap', 'repeated_gaps'):
            gaps = [0.] if kind == 'single_gap' else [-.9, 0., .9]
            left = -1.5
            for x in gaps + [1.5+p['gap']/2]:
                right = x-p['gap']/2
                if right > left:
                    supports.append(((left+right)/2, 0, (right-left)/2, .7, 0, 0))
                left = x+p['gap']/2
        elif kind in REVIEW:
            supports = []
            for i in range(40):
                x = -3.9+i*.2
                z = 0. if kind == 'flat' else (.025*(min(i, 39-i)//3) if kind == 'stairs' else float(rng.uniform(-.015, .015)))
                supports.append((x, 0, .1, 4, 0, z))
        meshes = [box(r) for r in supports]
        routes = [Route((-2.5, 0, 0), 0., -length/2, 1.9,
                        .7 if kind != 'single_beam' else w/2, (2.4, 0, .5, .7, 0, 0))]
        if kind in REVIEW:
            top = next(r[5] for r in supports if abs(-2.5-r[0]) <= r[2]+1.e-6)
            routes[0].spawn = (-2.5, 0, top)
    # Floor only at pit depth, never closing the holes at walking height.
    meshes.append(box((0, 0, 4, 4, 0, -p['pit_depth']), .1))
    return Tile(kind, level, seed, p, supports, routes, meshes)
