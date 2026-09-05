# Stage 2 acceptance record

Date: 2026-09-04

## Scope delivered

- Added `ISAL-Humanoid-Rough-HeightScan-v0` without changing the registered Stage 1 task.
- Added root-relative Actor terrain perception with dynamically inferred RayCaster grid geometry.
- Added a separate current-frame `height_scan` observation group for ordinary PPO.
- Extended the existing symmetry transform to mirror the height scan across the lateral axis.
- Added a zero-action RayCaster visualization utility.

No interaction tracker, affordance model, auxiliary loss, depth camera, policy export, or training was added or run.

## Public classes and configuration

```text
Environment:        ISALHumanoidHeightScanEnv
Environment config: ISALHumanoidRoughHeightScanEnvCfg
Agent config:       ISALHumanoidRoughHeightScanAgentCfg
Experiment:         isal_humanoid_rough_height_scan
```

`TerrainPerceptionCfg` contains all Stage 2 perception parameters:

```text
size=(1.6, 1.0), resolution=0.1, offset_x=0.4
min_height=-0.8, max_height=0.4, height_scale=0.5
noise_std=0.0, dropout_prob=0.0
```

The actual initialized RayCaster pattern resolves to `(H, W) = (17, 11)` and 187 rays. The environment computes `ray_hit_world_z - root_world_z`, maps non-finite hits to `min_height`, clips, and scales. It preserves a canonical `(N, 1, H, W)` tensor and exposes its current-frame flattened view to standard RSL-RL `ActorCritic`.

## Observation and learning contract

```text
policy       (N, 780)   10-frame proprioception history
height_scan  (N, 187)   current canonical height grid, flattened
critic       (N, 3260)  10-frame privileged history with clean scan
Actor input  (N, 967)   policy + height_scan
action       (N, 23)
```

The new agent uses explicit observation groups and fully qualified names for the same standard classes:

```text
policy:    rsl_rl.modules:ActorCritic
algorithm: rsl_rl.algorithms:PPO
```

All reward, terrain, command, event/randomization, time-scale, action, PPO, and base symmetry parameters match Stage 1. The Stage 1 task continues to produce only `(N,780)` policy and `(N,3260)` critic observations.

## Files

Primary additions:

```text
isal/isal/tasks/direct/humanoid_rough/height_scan.py
isal/isal/tasks/direct/humanoid_rough/height_scan_env.py
isal/isal/tasks/direct/humanoid_rough/terrain_perception_cfg.py
isal/scripts/debug_height_scan.py
isal/tests/stage2/
isal/STAGE2_ACCEPTANCE.md
```

Existing files changed only for task registration, the new config/agent subclass, dynamic validation, shared test fixture, asset-import namespace compatibility, and documentation.

## Verification results

Complete Stage 1 and Stage 2 suite:

```powershell
conda run --no-capture-output -n env_isaaclab python -u -m pytest `
  isal\tests -q --tb=short -p no:cacheprovider
```

```text
15 passed, 1 warning in 18.07s
```

The warning is RSL-RL's existing deprecation notice for the inherited `empirical_normalization` field; Actor and Critic normalization remain explicitly configured as in Stage 1.

Stage 2 public `--validate-only` result:

```text
observation_keys=['critic', 'height_scan', 'policy']
policy_shape=(1, 780)
height_scan_shape=(1, 187)
critic_shape=(1, 3260)
actor_input_shape=(1, 967)
height_scan_grid_shape=(1, 1, 17, 11)
action_shape=(1, 23)
reward_finite=True
extras_keys=['log', 'time_outs']
stepped_with_zero_actions=True
runner.learn_called=False
```

Stage 1 public `--validate-only` was rerun and retained policy `(1,780)`, critic `(1,3260)`, Actor input `(1,780)`, and action `(1,23)`.

Both training and visualization scripts pass `--help`. The interactive RayCaster debug command is documented in `README.md`; only its non-mutating CLI path was checked automatically because the acceptance suite is headless.

## Repository boundary

Before and after implementation:

```text
git -C robolab status --short  # empty
git -C rsl_rl status --short   # empty
```

The only RoboLab imports in the ISAL package are the standard and monorepo-fallback paths to the same read-only `RPO_CFG` asset. There are no `robolab.tasks` imports.

## Next gate

Phase 3 has not started. It will add the GPU-vectorized foot interaction tracker and label logging only after Stage 2 acceptance.
