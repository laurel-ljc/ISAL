"""Static box/cylinder obstacle bays with real gaps and level approach platforms."""
import numpy as np
import xml.etree.ElementTree as ET


SOLID_TILES = {"pillars", "balance_beam", "zigzag_bridge", "hurdles", "gaps", "staggered_blocks"}
EXTRA_TILES = ["pillars", "balance_beam", "zigzag_bridge", "hurdles", "gaps", "staggered_blocks",
               "wave", "ramp_platform"]
TILE_LABELS = {
    "flat": "Flat / warm-up", "rough": "Rough ground", "slope": "Pyramid slope",
    "inv_slope": "Inverted slope", "stairs": "Pyramid stairs", "inv_stairs": "Inverted stairs",
    "boxes": "Random boxes", "stones_gaps": "Stepping stones / trench",
    "pillars": "Stepping pillars", "balance_beam": "Balance beam", "zigzag_bridge": "Zigzag bridge",
    "hurdles": "Low hurdles", "gaps": "Repeated gaps", "staggered_blocks": "Staggered blocks",
    "wave": "Rolling waves", "ramp_platform": "Ramp / platform / ramp",
}
COLORS = [".43 .49 .54 1", ".42 .52 .36 1", ".65 .56 .37 1", ".44 .56 .54 1",
          ".68 .48 .31 1", ".52 .42 .60 1", ".53 .58 .35 1", ".39 .56 .61 1",
          ".80 .48 .25 1", ".72 .55 .32 1", ".28 .64 .67 1", ".76 .62 .25 1",
          ".59 .41 .35 1", ".45 .56 .73 1", ".40 .64 .46 1", ".66 .47 .60 1"]


def solid_tile(world, name, kind, center, cfg, difficulty, color):
    """Return geometry descriptions in world coordinates for saving and verification."""
    half = cfg["tile_size"] / 2
    border, depth = cfg["border_width"], cfg["pit_depth"]
    inner = half - border
    bank = cfg["platform_width"] / 2
    reach = inner - bank
    lerp = lambda key: float(np.interp(difficulty, [0, 1], cfg[key]))
    geoms = []

    def add(suffix, shape, xy, size, top, bottom, rgba=color, yaw=0.):
        position = [float(center[0] + xy[0]), float(center[1] + xy[1]), (top + bottom) / 2]
        dimensions = [*size, (top - bottom) / 2]
        geom_name = f"{name}_{suffix}"
        ET.SubElement(world, "geom", name=geom_name, type=shape, pos=" ".join(map(str, position)),
                      size=" ".join(map(str, dimensions)), rgba=rgba, group="0", contype="1",
                      conaffinity="15", condim="3", friction=".9 .2 .2",
                      quat=f"{np.cos(yaw / 2)} 0 0 {np.sin(yaw / 2)}")
        geoms.append({"name": geom_name, "type": shape, "position": position,
                      "size": dimensions, "top": top, "yaw": yaw})

    pits = kind in ("pillars", "balance_beam", "zigzag_bridge", "gaps")
    floor = -depth if pits else 0.
    add("floor", "box", [0, 0], [half, half], floor, floor - .15, ".19 .23 .28 1" if pits else color)
    if pits:
        # These are perimeter walkways and two banks, never a hidden floor below a gap.
        for sign in (-1, 1):
            add(f"walkway_y{sign}", "box", [0, sign * (half - border / 2)], [half, border / 2], 0., -depth)
            add(f"bank_x{sign}", "box", [sign * (half + reach) / 2, 0], [(half - reach) / 2, inner], 0., -depth)
    if kind == "pillars":
        radius, gap = lerp("pillar_radius_range"), lerp("pillar_gap_range")
        count = max(3, int(np.ceil(2 * reach / (2 * radius + gap))))
        xs = np.linspace(-reach + radius * .65, reach - radius * .65, count)
        for row, y in enumerate((-.75, 0., .75)):
            for col, x in enumerate(xs):
                # Stagger adjacent rows to offer multiple foot-placement routes.
                x = float(np.clip(x + (.12 if row != 1 and col % 2 else 0), -reach + radius * .65, reach - radius * .65))
                top = lerp("pillar_height_range") * ((col + row) % 3) / 2
                add(f"pillar_{row}_{col}", "cylinder", [x, y], [radius], top, -depth)
    elif kind == "balance_beam":
        add("beam", "box", [0, 0], [reach + .05, lerp("beam_width_range") / 2], 0., -.12)
    elif kind == "zigzag_bridge":
        width = lerp("zigzag_width_range")
        xs = np.linspace(-reach - .05, reach + .05, 5)
        ys = [0., .55, -.55, .55, 0.]
        for i in range(4):
            start, end = np.array([xs[i], ys[i]]), np.array([xs[i + 1], ys[i + 1]])
            delta = end - start
            add(f"bridge_{i}", "box", (start + end) / 2,
                [np.linalg.norm(delta) / 2 + width / 2, width / 2], 0., -.12,
                yaw=np.arctan2(delta[1], delta[0]))
    elif kind == "hurdles":
        for i, x in enumerate(np.linspace(-reach + .2, reach - .2, 5)):
            height = lerp("hurdle_height_range") * (.7 + .3 * (i % 2))
            add(f"hurdle_{i}", "box", [x, 0], [.07, 1.1], height, 0., ".92 .69 .20 1")
    elif kind == "gaps":
        width = lerp("gap_width_range")
        cuts = np.linspace(-reach * .7, reach * .7, 3)
        left = -reach
        for i, cut in enumerate([*cuts, reach + width / 2]):
            right = cut - width / 2
            add(f"landing_{i}", "box", [(left + right) / 2, 0], [(right - left) / 2, inner], 0., -depth)
            left = cut + width / 2
    elif kind == "staggered_blocks":
        for i, x in enumerate(np.linspace(-reach + .3, reach - .3, 6)):
            add(f"block_{i}", "box", [x, .28 if i % 2 else -.28], [.26, .32],
                lerp("block_height_range") * (.5 + .25 * (i % 3)), 0.)
    else:
        raise ValueError(f"Unknown solid obstacle tile: {kind}")
    return geoms
