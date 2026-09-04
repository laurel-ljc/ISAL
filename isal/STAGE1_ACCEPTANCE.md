# Stage 1 acceptance record

Date: 2026-09-03

## Scope delivered

- Independent distribution `isal-humanoid` and Python package `isal`.
- One registered task: `ISAL-Humanoid-Rough-v0`.
- Independent DirectRLEnv, scene, MDP/rewards, rough-terrain generators, PPO configuration, and symmetry mapping.
- Independent RSL-RL entry point with `--validate-only`.
- Read-only robot dependency: `from robolab.assets.robots import RPO_CFG`.
- Baseline source commit: `6b1c3d9988497c8961dcba77892de32edc1770e1`.

No training was executed.

## Files

```text
isal/
├── BASELINE_PROVENANCE.md
├── README.md
├── STAGE1_ACCEPTANCE.md
├── setup.py
├── isal/
│   ├── __init__.py
│   └── tasks/
│       ├── __init__.py
│       └── direct/
│           ├── __init__.py
│           └── humanoid_rough/
│               ├── __init__.py
│               ├── base_config.py
│               ├── base_env.py
│               ├── isal_env_cfg.py
│               ├── scene_cfg.py
│               ├── terrain_generator_cfg.py
│               ├── agents/
│               │   ├── __init__.py
│               │   └── isal_agent_cfg.py
│               └── mdp/
│                   ├── __init__.py
│                   └── rewards.py
├── scripts/rsl_rl/
│   ├── cli_args.py
│   └── train.py
└── tests/stage1/
    ├── conftest.py
    ├── test_config_parity.py
    └── test_source_boundaries.py
```

## Intentional configuration differences

The parity tests compare the complete serialized `RPORoughEnvCfg` and `RPORoughAgentCfg` trees. The only normalized differences are:

- Python module namespaces (`robolab.tasks...` to `isal.tasks...`);
- environment/config/agent class names and Gym task ID;
- experiment and logger project names (`isal_humanoid_rough`).

The action dimension (23), physics rate (200 Hz), control rate (50 Hz), episode length (20 s), history length (10), reward, commands, randomization, terrain, PPO, and symmetry values are unchanged. Stage 1 keeps height scan enabled for the critic and disabled for the actor.

## Verification results

Editable installation:

```text
Successfully installed isal-humanoid-0.1.0
package path: C:\Users\Admin\Documents\ISAL\isal\isal\__init__.py
```

Complete test suite:

```powershell
conda run --no-capture-output -n env_isaaclab python -u -m pytest `
  isal\tests\stage1 -q --tb=short -p no:cacheprovider
```

```text
........ [100%]
8 passed in 9.23s
```

Runtime validation:

```text
policy_shape=(1, 780)
critic_shape=(1, 3260)
action_shape=(1, 23)
reward_finite=True
extras_keys=['log', 'time_outs']
stepped_with_zero_actions=True
runner.learn_called=False
```

`--validate-only` reduces the generated terrain grid to one tile solely for the local smoke test. The registered task and normal training path retain the baseline 10-by-20 terrain configuration.

Both before and after implementation:

```text
git -C robolab status --short  # empty
git -C rsl_rl status --short   # empty
```

## Known risks and boundaries

- Runtime still depends on the pinned RoboLab robot configuration and its URDF/data paths remaining available.
- Training has not been run on this machine by design; only configuration, construction, reset, inference-policy construction, and one fixed zero-action step were tested.
- Isaac Sim emitted non-fatal plugin shutdown and graphics cache warnings during tests; all acceptance commands exited successfully.
- Phase 2 has not been started.
