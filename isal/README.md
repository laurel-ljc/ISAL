# ISAL-Humanoid

Independent Isaac Lab workspace for interaction-supervised affordance learning on humanoid rough-terrain locomotion.

The Stage 1 task is a behavior-preserving copy of RoboLab's `RPO-Rough` baseline. Environment, MDP, terrain, and agent implementation live in this package. Robot configuration and URDF/data assets are loaded read-only from the installed `robolab` package.

## Install

```powershell
conda run -n env_isaaclab python -m pip install -e .\isal --no-deps
```

## Validate without training

```powershell
conda run --no-capture-output -n env_isaaclab python -u `
  isal\scripts\rsl_rl\train.py `
  --task ISAL-Humanoid-Rough-v0 `
  --headless `
  --num_envs 1 `
  --validate-only
```

`--validate-only` constructs the environment, RSL-RL wrapper, policy, algorithm, and runner, resets the environment, performs one fixed zero-action step, and exits before `runner.learn()`.

## Training (external training machine only)

Do not run training on the local development machine.

```powershell
conda run --no-capture-output -n env_isaaclab python -u `
  isal\scripts\rsl_rl\train.py `
  --task ISAL-Humanoid-Rough-v0 `
  --headless `
  --num_envs 4096
```
