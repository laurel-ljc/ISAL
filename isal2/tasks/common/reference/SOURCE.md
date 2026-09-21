# AME_Locomotion 来源与适配

固定来源：[SII-FUSC/AME_Locomotion](https://github.com/SII-FUSC/AME_Locomotion/tree/1d3519ee946c846c1c89dd660539a224f99972b5)，commit `1d3519ee946c846c1c89dd660539a224f99972b5`。

原始目录：`source/ame_locomotion/ame_locomotion/tasks/manager_based/ame_locomotion/`。

| 原始路径 | 本地用途 |
|---|---|
| `terrains/terrain_cfg.py` | Stage1 参数与比例 |
| `terrains/finetune_terrain_cfg.py` | Stage2 参数与比例 |
| `terrains/loco_hf_terrains.py` | 石桥、双列踏桩、交替踏桩、同心沟槽高度场 |
| `terrains/loco_hf_terrains_cfg.py` | 上述高度场配置 |
| `29dof/velocity_env_cfg_29dof.py` | 出生、速度/heading、距离课程的配置来源 |

`upstream/*.txt` 保存固定 commit 的原文，运行时无需联网或导入 AME_Locomotion。`terrain.py` 明确列出实际使用的参数；标准地形及高度场转网格使用本机 Isaac Lab。图册工具将每个自定义地形在相同随机数下与原始函数的网格逐点比较。

许可证：两个地形配置文件原文带 Isaac Lab 的版权声明及 `SPDX-License-Identifier: BSD-3-Clause`；保留于 upstream 副本。固定 commit 的根目录未提供 LICENSE，两个自定义高度场文件未带许可证头，因此不能把整个参考仓库宣称为 BSD-3-Clause。Isaac Lab 标准函数使用其自身 BSD-3-Clause 许可；本项目既有机器人、MDP 的版权头保持不变。

适配仅补齐随机性：双列踏桩原来无 seed 的 `default_rng()` 使用 `(seed*1000003 + round(difficulty*1000000)) % 2**32`；交替踏桩沿用已有同式种子。石桥、沟槽和标准地形使用的 NumPy/Python 全局 RNG，以及随机方块使用的 PyTorch CPU/CUDA RNG，在每块生成期间确定性播种，退出后恢复训练 RNG。随机种子由地图 seed、实际难度与函数名产生。保留原几何算法、像素取整、坑底、边框、中央平台，不添加端点路线限制。

Isaac Lab 每行实际难度为 `(row + U[0,1))/10`，不是固定 `row/9`；配置范围端点并不保证正好出现在生成地图中。标准地形实现版本来自安装的 Isaac Lab，配置迁移不等同于复现其完整训练软件版本。

ISAL2 保留 RPO 奖励、网络、PPO 和 0.1 m 策略扫描。本次仅迁移课程相关设置，不宣称复现参考仓库的训练效果。
