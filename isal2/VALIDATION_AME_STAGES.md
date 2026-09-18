# AME 两阶段任务验收记录

日期：2026-09-18。环境：本机 `env_isaaclab`，Windows、Isaac Sim、CUDA。

## 实现范围

- 完整迁移旧 tasks 到 deprecated_tasks，旧任务 ID 保留。
- 实现 AME Stage1/Stage2，共十二种地形、每种十级，干净 actor/critic 高程、终点命令、每环境每类型独立课程。
- 新 base/AME 公共实现与迁移前源码一致；机器人、MDP 和网络复用原配置。地形、课程、出生、命令、成功终止语义和阶段推扰由独立 course 模块提供。
- Affordance Stage1/Stage2 尚未实现、未注册；既有 deprecated Affordance 保持原行为。

## 自动化与仿真结果

| 检查 | 结果 | 本地证据（相对 isal2） |
|---|---|---|
| 全量单元/回归测试 | 85 项通过 | `outputs/stage_acceptance_final.log` |
| 新 course 专项测试 | 11 项通过，包含全部120个地形的网格射线与支撑区域一致性、网格相位覆盖、间隙可见性、课程、命令、推扰、检查点 | `tests/test_course.py`、`outputs/course_tests_final.log` |
| Stage1 生命周期 smoke | 8环境×80步；成功终止、超时、局部重置、干净高程通过 | `outputs/rpo_ame_stage1/ame_stage1_lifecycle/result.json` |
| Stage2 持续 smoke | 8环境×450步，45次重置；观测、奖励和动作有限 | `outputs/rpo_ame_stage2/ame_stage2_smoke/result.json` |
| Stage2 生命周期和实际推扰 | 8环境×80步，最终代码再验30步；成功/超时分类及水平增量推扰通过 | `outputs/rpo_ame_stage2/ame_stage2_final/result.json` |
| Stage1 PPO | 0→3轮，CNN/位置编码/query/attention/actor/critic 均更新 | `outputs/rpo_ame_stage1/ame_stage1_train/result.json` |
| Stage1 resume | 3→5轮，模型继续更新 | `outputs/rpo_ame_stage1/ame_stage1_resume/result.json` |
| Stage1→Stage2 warm-start | 严格加载 Stage1 模型，Stage2从第0轮训练至第3轮，各模块更新 | `outputs/rpo_ame_stage2/ame_stage2_train/result.json` |
| Stage2 resume | 3→5轮，模型继续更新 | `outputs/rpo_ame_stage2/ame_stage2_resume/result.json` |
| 旧 AME ID 兼容入口 | 4环境×30步，rough地形、局部重置通过 | `outputs/rpo_ame/deprecated_ame_migration/result.json` |
| 旧任务源码迁移核对 | 45个文件，Python源码除包引用外与迁移前一致 | Git HEAD 与 deprecated_tasks 源码逐文件比对 |

新检查点包含 `course_state` 和 Python/NumPy/Torch/CUDA 随机数状态；Stage1 保存的等级矩阵为 `[8,7]`，Stage2为 `[8,5]`。测试用非零等级验证保存恢复；不兼容阶段的 resume 在修改模型前拒绝，warm-start 保留模型与归一化统计，重置优化器与课程。

成功终止在真实终点平台上验证，至少一脚接触并持续0.2秒；不设置 timeout bootstrap、不触发失败惩罚，并只升级本回合地形类型。超时在真实环境中强制触发，降级且保留终止前价值 bootstrap。

## 地形预览

- `outputs/course_catalog/stage1_all_levels.png`：一阶段全部70个地形。
- `outputs/course_catalog/stage2_all_levels.png`：二阶段全部50个地形。
- `outputs/course_catalog/stage1_profiles.png`、`stage2_profiles.png`：第1/5/10级中心线高度剖面。
- `outputs/course_catalog/terrain_parameters.json`：全部120个地形的几何参数和支撑多边形。

已检查俯视图及侧面高度图。五边形使用真实正五边形；浅坑有合法坑底，悬空间隙没有隐藏平路。扫描校验读取实际配置的尺寸和分辨率，不硬编码为默认扫描范围。

## 验证边界

目录整理后，公共模块统一位于 `tasks/common/{base,ame,course}`，任务顶层仅保留两个阶段入口和 `common`。已重新通过85项测试（`outputs/common_layout_tests.log`），并通过4环境×30步的Stage2仿真、成功终止、超时、局部重置和实际推扰检查（`outputs/rpo_ame_stage2/common_layout_smoke/result.json`）。

短程 PPO 和 smoke 验证工程闭环，不证明机器人已学会所有地形，尚未进行4096环境长程收敛实验。

本机 Isaac Sim 日志出现 Vulkan/ShaderCache、缓存权限和 SharedMutex 相关信息；若干进程在写出成功 `result.json` 后停留于 `Closing simulator`，已在确认结果保存后终止这些验证进程。部分运行正常退出。此关闭阶段现象没有在本次任务中修改或宣称修复。
