"""Exact discrete geometry and column allocation, independent of Isaac Sim."""
WIDTHS = (.80, .70, .60, .55, .50, .45, .40, .35, .30, .25)
ACQUIRE = dict(single_beam=6, radial_beams=4, grid_stones=3, single_gap=2,
               pallets=1, flat=2, stairs=1, rough=1)
ROBUST = dict(single_beam=4, radial_beams=3, grid_stones=2, single_gap=2,
              pallets=1, flat=2, stairs=1, rough=1, single_stones=2, repeated_gaps=2)
REVIEW = ("flat", "stairs", "rough")
TYPES = tuple(ROBUST) + ("legacy_star",)


def columns(topology=False, count=20):
    if count < 20 or count % 20:
        raise ValueError("Sparse terrain columns must be a positive multiple of 20 (exact proportions)")
    return [name for name, n in (ROBUST if topology else ACQUIRE).items()
            for _ in range(n * (count // 20))]


def parameters(kind, level, rescue=False):
    if kind not in TYPES or not 0 <= level < 10:
        raise ValueError(f"Invalid sparse terrain {kind}/{level}")
    f = level / 9
    return dict(width=.8 if rescue else WIDTHS[level], length=(1., 2., 3.)[min(level, 2)] if rescue else 3.,
                stone_width=.6 - .3*f, stone_gap=.1 + .1*f,
                pallet_width=.8 - .45*f, pallet_gap=.1 + .15*f, gap=.1 + .2*f,
                pit_depth=10. if kind == "legacy_star" else 1.)
