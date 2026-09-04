# Baseline provenance

The Stage 1 implementation was copied from RoboLab's `RPO-Rough` task at commit:

```text
6b1c3d9988497c8961dcba77892de32edc1770e1
```

Copied responsibilities:

- Direct RL environment and configuration;
- scene, contact sensors, and ray casters;
- reward/MDP implementation;
- terrain generators;
- PPO agent configuration and symmetry mapping.

Namespace and public-name changes are limited to the `isal` package, `ISALHumanoid*` classes, task ID `ISAL-Humanoid-Rough-v0`, and experiment/logger project name `isal_humanoid_rough`.

The robot articulation and URDF remain read-only references to `robolab.assets.robots.RPO_CFG`; no robot assets are duplicated into this project.

The existing BSD-3-Clause copyright and license notices are retained in copied source files.

Robot configuration and URDF/data assets are not copied. They are referenced read-only through `robolab.assets.robots.RPO_CFG`.
