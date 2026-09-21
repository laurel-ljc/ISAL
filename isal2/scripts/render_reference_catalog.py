"""Render collision-ray atlases and audit the pinned reference terrain recipe."""
import argparse
import json
from pathlib import Path
from _bootstrap import bootstrap
ROOT = bootstrap()


def main():
    from isaaclab.app import AppLauncher
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,default=ROOT/'outputs/reference_catalog')
    parser.add_argument('--seed',type=int,default=42)
    AppLauncher.add_app_launcher_args(parser)
    args = parser.parse_args()
    args.kit_args += f' --portable-root={ROOT.as_posix()}/.cache/catalog'
    app = AppLauncher(args).app
    import copy
    import numpy as np
    import trimesh
    from scipy.ndimage import label
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from unittest.mock import patch
    import isaaclab.terrains as terrain_gen
    from isaaclab.terrains.height_field import HfTerrainBaseCfg
    from isal2.tasks.common.reference import SOURCE_SHA
    from isal2.tasks.common.reference.terrain import terrain_config,ReferenceTerrainGenerator,plain
    from isal2.tasks.common.reference.seeding import terrain_seed,tile_seed
    from isal2.tasks.common.reference import loco_hf_terrains_cfg as custom
    from isal2.tests.reference_sim_checks import check_config_regression
    from isal2.tasks.ame_stage1.env_cfg import RPOAMEStage1EnvCfg
    from isal2.tasks.ame_stage2.env_cfg import RPOAMEStage2EnvCfg
    assert check_config_regression(RPOAMEStage1EnvCfg())
    assert check_config_regression(RPOAMEStage2EnvCfg())
    source_dir = ROOT/'tasks/common/reference/upstream'
    source_functions = {}
    exec((source_dir/'loco_hf_terrains.py.txt').read_text(encoding='utf-8'),source_functions)
    args.output.mkdir(parents=True,exist_ok=True)
    progress = args.output/'measurements_progress.jsonl'
    progress.write_text('',encoding='utf-8')

    def ray_heights(mesh,xy):
        hit,indices,_ = mesh.ray.intersects_location(np.c_[xy,np.full(len(xy),10.)],
            np.tile((0,0,-1),(len(xy),1)),multiple_hits=True)
        z = np.full(len(xy),-np.inf)
        np.maximum.at(z,indices,hit[:,2])
        # Box seams can lie exactly on ray/triangle edges after float32 mesh
        # construction. Retry only misses at a 1/2 micrometre XY offset; this
        # changes the probe, never the collision geometry.
        missing = np.flatnonzero(~np.isfinite(z))
        if len(missing):
            hit,index,_ = mesh.ray.intersects_location(np.c_[xy[missing]+[1e-6,2e-6],np.full(len(missing),10.)],
                np.tile((0,0,-1),(len(missing),1)),multiple_hits=True)
            np.maximum.at(z,missing[index],hit[:,2])
        return z

    reports = []
    for stage in (1,2):
        cfg = terrain_config(stage,args.seed)
        # Compare every parameter to the actual pinned upstream config file.
        text = (source_dir/('terrain_cfg.py.txt' if stage==1 else 'finetune_terrain_cfg.py.txt')).read_text(encoding='utf-8')
        text = '\n'.join(line for line in text.splitlines() if not line.startswith('import ') and not line.startswith('from '))
        for name in ('HfConcentricGapTerrainCfg','HfDoubleColumnStakesTerrainCfg','HfAlternateColumnStakesTerrainCfg','HfStonesBridgeTerrainCfg'):
            text = text.replace('terrain_gen.'+name,name)
        namespace = dict(vars(custom),terrain_gen=terrain_gen,TerrainGeneratorCfg=terrain_gen.TerrainGeneratorCfg)
        exec(text,namespace)
        upstream = namespace['ROUGH_TERRAINS_CFG' if stage==1 else 'FINETUNE_ROUGH_TERRAINS_CFG']
        for name,sub in cfg.sub_terrains.items():
            assert plain(sub.to_dict())==plain(upstream.sub_terrains[name].to_dict()),name
        for name in ('size','border_width','num_rows','num_cols','horizontal_scale','vertical_scale','slope_threshold'):
            assert getattr(cfg,name)==getattr(upstream,name),name
        for sub in cfg.sub_terrains.values():
            sub.size = cfg.size
            if isinstance(sub,HfTerrainBaseCfg):
                sub.horizontal_scale,sub.vertical_scale,sub.slope_threshold = cfg.horizontal_scale,cfg.vertical_scale,cfg.slope_threshold
        # Generate each tile independently with the same column/row RNG sequence
        # as Isaac Lab, avoiding a second full 200-tile map in memory.
        generator = object.__new__(ReferenceTerrainGenerator)
        generator.cfg = cfg
        kinds = list(cfg.sub_terrains)
        proportions = np.array([x.proportion for x in cfg.sub_terrains.values()])
        rng = np.random.default_rng(args.seed)
        atlas = []
        first_columns = {}
        fig,axes = plt.subplots(8,10,figsize=(25,19),squeeze=False)
        for col in range(cfg.num_cols):
            kind = kinds[np.where(col/cfg.num_cols+.001 < np.cumsum(proportions/proportions.sum()))[0][0]]
            first_columns.setdefault(kind,col)
            sub = cfg.sub_terrains[kind]
            for row in range(10):
                difficulty = (row+rng.uniform())/10
                item = dict(stage=stage,kind=kind,column=col,row=row,difficulty=difficulty,parameters=plain(sub.to_dict()))
                atlas.append(item)
                if col != first_columns[kind]:
                    continue
                mesh,origin = generator._get_terrain_mesh(difficulty,sub)
                mesh2,origin2 = generator._get_terrain_mesh(difficulty,sub)
                np.testing.assert_array_equal(mesh.vertices,mesh2.vertices)
                np.testing.assert_array_equal(mesh.faces,mesh2.faces)
                np.testing.assert_array_equal(origin,origin2)
                if sub.function.__module__.endswith('loco_hf_terrains'):
                    source = source_functions[sub.function.__name__]
                    seeded = sub.copy(); seeded.seed = args.seed
                    factory = np.random.default_rng
                    with terrain_seed(args.seed,difficulty,sub.function.__name__):
                        with patch('numpy.random.default_rng',side_effect=lambda seed=None: factory(tile_seed(seeded,difficulty) if seed is None else seed)):
                            meshes,expected_origin = source(difficulty,seeded)
                    expected = trimesh.util.concatenate(meshes)
                    expected.apply_translation((-4,-4,0))
                    np.testing.assert_array_equal(mesh.vertices,expected.vertices)
                    np.testing.assert_array_equal(mesh.faces,expected.faces)
                    np.testing.assert_array_equal(origin,expected_origin-np.array([4,4,0]))
                # Real triangle intersections (including vertical-slope correction).
                axis = np.arange(-3.975,4,.05)
                xx,yy = np.meshgrid(axis,axis,indexing='ij')
                z = ray_heights(mesh,np.c_[xx.ravel(),yy.ravel()]).reshape(xx.shape)
                assert np.isfinite(z).all(),(stage,kind,row,'collision holes')
                spawn_axis = np.linspace(-.5,.5,11) if stage==1 else np.array([0.])
                sx,sy = np.meshgrid(spawn_axis,spawn_axis)
                spawn_z = ray_heights(mesh,np.c_[sx.ravel(),sy.ravel()])
                assert (spawn_z>origin[2]-.3).all(),(stage,kind,row,'unsupported spawn')
                measure = dict(stage=stage,kind=kind,row=row,column=col,difficulty=difficulty,
                    origin=origin.tolist(),minimum_collision_z=float(z.min()),
                    spawn_min_z=float(spawn_z.min()),spawn_max_z=float(spawn_z.max()),
                    reproducible=True,upstream_geometry_match=True)
                if kind in ('hf_steppingstones','stakes1','stakes2','stakes3','stonebridge'):
                    # Global 0.1m lattice, 16 phases; disconnected islands are
                    # counted separately. Clip edge/border components, central platform.
                    components,n = label(z>-.5)
                    sizes = np.bincount(components.ravel())
                    interior = [i for i in range(1,n+1) if sizes[i]>=4 and not
                        ((components[0]==i).any() or (components[-1]==i).any() or
                         (components[:,0]==i).any() or (components[:,-1]==i).any() or
                         components[len(axis)//2,len(axis)//2]==i)]
                    minimum = None
                    for dx in (0.,.025,.05,.075):
                        for dy in (0.,.025,.05,.075):
                            gx,gy = np.meshgrid(np.arange(-3.9,3.901,.1)+dx,np.arange(-3.9,3.901,.1)+dy,indexing='ij')
                            xy = np.c_[gx.ravel(),gy.ravel()]
                            heights = ray_heights(mesh,xy)
                            ij = np.clip(np.rint((xy-axis[0])/.05).astype(int),0,len(axis)-1)
                            island = components[ij[:,0],ij[:,1]]
                            for index in interior:
                                count = int(((island==index)&(heights>-.5)).sum())
                                minimum = count if minimum is None else min(minimum,count)
                    measure.update(isolated_stones_measured=len(interior),minimum_scan_hits_over_16_phases=minimum)
                if kind=='hf_gaps':
                    x = np.arange(-3.99,4,.01)
                    zz = ray_heights(mesh,np.c_[x,np.zeros_like(x)])
                    pit = zz < -1
                    changes = np.diff(np.r_[False,pit,False].astype(int))
                    widths = [(b-a)*.01 for a,b in zip(np.flatnonzero(changes==1),np.flatnonzero(changes==-1))]
                    max_gap = max(widths,default=0.)
                    measure.update(max_measured_gap_m=max_gap,forward_scan_m=.8,edge_setback_m=.2,landing_margin_m=.15,
                        gap_and_landing_visible=max_gap+.2+.15<=.8+1e-6)
                reports.append(measure)
                with progress.open('a',encoding='utf-8') as stream:
                    stream.write(json.dumps(plain(measure))+'\n')
                ax = axes[kinds.index(kind),row]
                ax.imshow(z.T,origin='lower',extent=(-4,4,-4,4),vmin=-2,vmax=1,cmap='terrain')
                ax.plot(0,0,'r.',markersize=2)
                ax.set_title(f'{kind}\nrow {row} / d={difficulty:.3f}',fontsize=7)
                ax.set_xticks([]); ax.set_yticks([])
                print(f'CATALOG stage={stage} kind={kind} row={row}',flush=True)
        fig.tight_layout()
        fig.savefig(args.output/f'stage{stage}.png',dpi=110)
        plt.close(fig)
        (args.output/f'stage{stage}_parameters.json').write_text(json.dumps(atlas,indent=2),encoding='utf-8')
    result = dict(source_sha=SOURCE_SHA,seed=args.seed,config_regression=True,
        seam_ray_retry_xy_m=[1e-6,2e-6],tiles_measured=reports)
    (args.output/'measurements.json').write_text(json.dumps(plain(result),indent=2),encoding='utf-8')
    print('REFERENCE_CATALOG_COMPLETE',flush=True)
    app.close(skip_cleanup=True)


if __name__ == '__main__':
    main()
