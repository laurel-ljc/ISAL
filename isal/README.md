# ISAL-Humanoid

Independent Isaac Lab workspace for interaction-supervised affordance learning on humanoid rough-terrain locomotion.

The Stage 1 task is a behavior-preserving copy of RoboLab's `RPO-Rough` baseline. Stage 2 adds a separate ordinary-PPO task whose Actor also receives a root-relative RayCaster height scan. Environment, MDP, terrain, and agent implementation live in this package. Robot configuration and URDF/data assets are loaded read-only from `robolab.assets` (with a monorepo namespace fallback).

Registered tasks:

- `ISAL-Humanoid-Rough-v0`: frozen Stage 1 proprioceptive Actor baseline.
- `ISAL-Humanoid-Rough-HeightScan-v0`: Stage 2 Actor height-scan baseline.

## Install

```powershell
conda run -n env_isaaclab python -m pip install -e .\isal --no-deps
```

## Validate without training

```powershell
conda run --no-capture-output -n env_isaaclab python -u `
  isal\scripts\rsl_rl\train.py `
  --task ISAL-Humanoid-Rough-HeightScan-v0 `
  --headless `
  --num_envs 1 `
  --validate-only
```

`--validate-only` constructs the environment, RSL-RL wrapper, policy, algorithm, and runner, resets the environment, performs one fixed zero-action step, and exits before `runner.learn()`.

The Stage 2 observation contract is:

```text
policy       (num_envs, 780)   # 10-frame proprioception history
height_scan  (num_envs, 187)   # current 17 x 11 canonical grid, flattened for standard ActorCritic
critic       (num_envs, 3260)  # 10-frame privileged history with clean scan
actor input  (num_envs, 967)   # policy + height_scan
```

## Inspect the RayCaster grid without training

Run without `--headless` to view the height-scan markers. The robot receives zero actions and no runner is created:

```powershell
conda run --no-capture-output -n env_isaaclab python -u `
  isal\scripts\debug_height_scan.py `
  --task ISAL-Humanoid-Rough-HeightScan-v0 `
  --num_envs 1 `
  --steps 1000
```

## Training (external training machine only)

Do not run training on the local development machine.

```powershell
conda run --no-capture-output -n env_isaaclab python -u `
  isal\scripts\rsl_rl\train.py `
  --task ISAL-Humanoid-Rough-HeightScan-v0 `
  --headless `
  --num_envs 4096
```
