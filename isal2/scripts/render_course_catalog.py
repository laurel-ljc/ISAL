"""Render all 120 stage tiles and export their exact geometric metadata (no simulator)."""
import argparse
import json
from pathlib import Path
try:
    from ._bootstrap import bootstrap
except ImportError:
    from _bootstrap import bootstrap
ROOT = bootstrap()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=ROOT / 'outputs/course_catalog')
    args = parser.parse_args()
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import numpy as np
    from isal2.tasks.course.geometry import STAGES, build_tile, validate_perception
    args.output.mkdir(parents=True, exist_ok=True)
    atlas = []
    for stage, kinds in STAGES.items():
        fig, axes = plt.subplots(len(kinds), 10, figsize=(24, len(kinds)*2.1), squeeze=False)
        x, y = np.meshgrid(np.linspace(-4, 4, 401), np.linspace(-1.3, 1.3, 131))
        xy = np.stack((x, y), -1)
        for row, kind in enumerate(kinds):
            for level in range(10):
                tile = build_tile(kind, level)
                validate_perception(tile)
                atlas.append(dict(stage=stage, **tile.metadata()))
                ax = axes[row, level]
                ax.imshow(tile.heights(xy), origin='lower', extent=(-4, 4, -1.3, 1.3),
                          vmin=-1., vmax=1.2, cmap='terrain', interpolation='nearest')
                ax.scatter([-2.7, 2.7], [0, 0], c=['lime', 'red'], s=10)
                ax.set_xticks([])
                ax.set_yticks([])
                if row == 0:
                    ax.set_title(f'Level {level+1}')
                if level == 0:
                    ax.set_ylabel(kind.replace('_', ' '), fontsize=10)
        fig.suptitle(f'AME Stage {stage} | green: start / red: goal | actual support heights', fontsize=16)
        fig.tight_layout()
        fig.savefig(args.output / f'stage{stage}_all_levels.png', dpi=140)
        plt.close(fig)
        # Side profiles expose stair/pit/rough height changes hidden in plan views.
        fig, axes = plt.subplots(len(kinds), 1, figsize=(12, 2*len(kinds)), squeeze=False)
        for row, kind in enumerate(kinds):
            for level in (0, 4, 9):
                tile = build_tile(kind, level)
                xs = np.linspace(-4, 4, 2001)
                axes[row, 0].plot(xs, tile.heights(np.c_[xs, np.zeros_like(xs)]), label=f'Level {level+1}')
            axes[row, 0].set_title(kind.replace('_', ' '))
            axes[row, 0].set_ylabel('Height (m)')
            axes[row, 0].legend(loc='upper right', ncol=3)
        axes[-1, 0].set_xlabel('Course x (m)')
        fig.tight_layout()
        fig.savefig(args.output / f'stage{stage}_profiles.png', dpi=120)
        plt.close(fig)
    (args.output / 'terrain_parameters.json').write_text(json.dumps(atlas, indent=2), encoding='utf-8')
    print(f'Wrote {len(atlas)} tiles to {args.output.resolve()}')


if __name__ == '__main__':
    main()
