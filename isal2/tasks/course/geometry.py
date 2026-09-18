"""Simulator-independent convex support geometry; metres in tile-local coordinates.

Meshes and support queries are both derived from the same polygons. The floor is
collision geometry only, never a valid support (except the intentional shallow pit).
"""
from dataclasses import dataclass, field, asdict
import math
import numpy as np
import trimesh

STAGES = {
    1: ("stairs", "pits", "rough", "pallets", "gaps", "grid_stones", "beams"),
    2: ("pentagon_stones", "single_column_stones", "narrow_pallets", "consecutive_gaps", "narrow_stairs"),
}
GEOMETRY_VERSION = 1


def parameters(kind, level):
    if kind not in STAGES[1] + STAGES[2] or not 0 <= level <= 9:
        raise ValueError(f"Unknown terrain or level: {kind}/{level}")
    t = level / 9
    mix = lambda lo, hi: lo + (hi - lo) * t
    if kind in ("stairs", "narrow_stairs"):
        return dict(height=mix(.05, .15), tread=mix(.35, .25) if kind == "narrow_stairs" else .35, width=1.2)
    if kind == "pits":
        return dict(depth=mix(.05, .18), length=.8, width=1.2)
    if kind == "rough":
        return dict(amplitude=mix(.01, .04), cell=.2, width=1.2)
    if kind in ("pallets", "narrow_pallets"):
        narrow = kind == "narrow_pallets"
        return dict(width=mix(.5, .3) if narrow else mix(.8, .5), length=.5,
                    gap=mix(.15, .25) if narrow else mix(.1, .2))
    if kind in ("grid_stones", "single_column_stones", "pentagon_stones"):
        return dict(width=mix(.5 if kind == "grid_stones" else .45, .3),
                    gap=mix(.1, .2 if kind == "grid_stones" else .25))
    if kind in ("gaps", "consecutive_gaps"):
        return dict(gap=mix(.1 if kind == "gaps" else .15, .3), landing=mix(.5, .35), width=1.2)
    return dict(width=mix(.6, .3), length=4.)


@dataclass
class Support:
    polygon: list
    top: float = 0.

    def contains(self, xy, margin=0.):
        points = np.asarray(xy)
        vertices = np.asarray(self.polygon)
        edge = np.roll(vertices, -1, axis=0) - vertices
        delta = points[..., None, :] - vertices
        cross = edge[:, 0]*delta[..., 1] - edge[:, 1]*delta[..., 0]
        return (cross >= margin*np.linalg.norm(edge, axis=-1)-1.e-9).all(axis=-1)


@dataclass
class Tile:
    kind: str
    level: int
    seed: int
    params: dict
    supports: list = field(default_factory=list)
    spawn: tuple = (-2.7, 0., 0.)
    goal: tuple = (2.7, 0., 0.)
    route_half_width: float = .6

    def metadata(self):
        return asdict(self)

    def heights(self, xy):
        result = np.full(np.asarray(xy).shape[:-1], -1.)
        for support in self.supports:
            result = np.where(support.contains(xy), np.maximum(result, support.top), result)
        return result

    def meshes(self):
        return [prism(s.polygon, s.top, -1.) for s in self.supports] + [
            prism(rect(-4, 4, -4, 4), -1., -1.1)]


def rect(left, right, bottom, top):
    return [(left, bottom), (right, bottom), (right, top), (left, top)]


def prism(polygon, top, bottom):
    """Triangulate a convex CCW polygon, with correctly wound closed faces."""
    p = np.asarray(polygon)
    n = len(p)
    vertices = np.concatenate((np.c_[p, np.full(n, bottom)], np.c_[p, np.full(n, top)]))
    faces = []
    for i in range(1, n-1):
        faces.extend(((0, i+1, i), (n, n+i, n+i+1)))
    for i in range(n):
        j = (i+1) % n
        faces.extend(((i, j, n+j), (i, n+j, n+i)))
    return trimesh.Trimesh(vertices=vertices, faces=faces, process=False)


def build_tile(kind, level, seed=42):
    p = parameters(kind, level)
    tile = Tile(kind, level, seed, p, route_half_width=1.2 if kind == "grid_stones" else .6)
    def add(left, right, width, z=0.):
        tile.supports.append(Support(rect(left, right, -width/2, width/2), z))
    add(-4., -2., 2.4)
    add(2., 4., 2.4)
    if kind in ("stairs", "narrow_stairs"):
        # Flat shoulders absorb the remainder without shortening a tread.
        n = int(2 / p['tread'])
        left = -n*p['tread']
        add(-2., left, 1.2)
        for i in range(2*n):
            height = min(i+1, 2*n-i)*p['height']
            add(left+i*p['tread'], left+(i+1)*p['tread'], 1.2, height)
        add(-left, 2., 1.2)
    elif kind == "pits":
        add(-2., -.4, 1.2)
        add(-.4, .4, 1.2, -p['depth'])
        add(.4, 2., 1.2)
    elif kind == "rough":
        rng = np.random.default_rng(seed)
        for i in range(20):
            for j in range(6):
                tile.supports.append(Support(rect(-2+i*.2, -2+(i+1)*.2, -.6+j*.2, -.6+(j+1)*.2),
                                             float(rng.uniform(-p['amplitude'], p['amplitude']))))
    elif kind == "beams":
        add(-2., 2., p['width'])
    elif kind in ("gaps", "consecutive_gaps"):
        gap = p['gap']
        centers = (0.,) if kind == "gaps" else (-(gap+p['landing']), 0., gap+p['landing'])
        left = -2.
        for center in centers:
            add(left, center-gap/2, 1.2)
            left = center+gap/2
        add(left, 2., 1.2)
    else:
        # Fit whole stones with exact edge-to-edge gaps, and connected shoulders
        # at both ends. No final sliver or overlapped landing is generated.
        w, gap = p['width'], p['gap']
        if kind == "pentagon_stones":
            radius = w/(2*math.cos(math.pi/5))
            angles = math.pi/2 + np.arange(5)*2*math.pi/5
            polygon = np.c_[radius*np.cos(angles), radius*np.sin(angles)]
            length = float(np.ptp(polygon[:, 0]))
        else:
            length = p.get('length', w)
            polygon = np.array(rect(-length/2, length/2, -w/2, w/2))
        count = int((4-gap)/(length+gap))
        occupied = count*length+(count+1)*gap
        left = -occupied/2
        add(-2., left, 2.4)
        add(-left, 2., 2.4)
        ys = (-w-gap, 0., w+gap) if kind == "grid_stones" else (0.,)
        for i in range(count):
            x = left+gap+length/2+i*(length+gap)
            for y in ys:
                tile.supports.append(Support((polygon+np.array((x, y))).tolist()))
        p['count'] = count
        p['actual_gap'] = gap
    # Filter zero-width stair shoulders at exact divisibility.
    tile.supports = [s for s in tile.supports if np.ptp(np.asarray(s.polygon)[:, 0]) > 1.e-8]
    return tile


def validate_perception(tile, resolution=.1, size=(1.6, 1.0)):
    """Allow .2 m approach margin and .2 m visible landing past every gap."""
    if resolution <= 0 or len(size) != 2 or min(size) <= 0:
        raise ValueError('Scan resolution and extents must be positive')
    if tile.params.get('gap', 0.) + .2 + .2 > size[0]/2 + 1.e-8:
        raise ValueError(f"Gap and landing exceed forward scan extent: {tile.kind}")
    if 'stones' in tile.kind and (tile.params['width'] < .3-1.e-8 or tile.params['width'] <= 2*resolution):
        raise ValueError('Stone support must exceed two scan cells and be at least 0.30 m wide')
