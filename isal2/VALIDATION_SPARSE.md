# 稀疏地形任务实现与验收

日期：2026-09-16。正式两组各 500 次 PPO 更新初筛未执行。

## 交付

- `ISAL2-RPO-AME-Sparse-v0` 与 `ISAL2-RPO-Affordance-Sparse-v0` 已注册；两个薄任务共用 `tasks/sparse/`。
- acquire/robust、C0/C1/C2、准确离散宽度、补救长度、路线出生/命令、接触支撑判定、分类型课程与回放已接入。
- warm-start/resume/advance、独立固定评估、阶段门槛与状态持久化已接入。网络与导出维度不变。
- Affordance 保留原接触监督、双优化器和门控；漂移地图的落脚 query 使用实际采样原点。
- 目录与逐文件职责见 `AME_SPARSE_CURRICULUM_PLAN.md` 的实现修订；运行命令见 README。

## 已完成验证

运行环境：Windows、RTX 3060 12 GB，`env_isaaclab` Python 3.11。修复本地外部 RSL-RL 3.3.0 editable 安装；
构建时临时使用 packaging 25.0，安装后恢复 Isaac Sim 所需 packaging 23.0。没有复制外部库源码。

| 验证 | 实际结果 | 证据目录/文件 |
|---|---|---|
| CPU 回归 | 74 项通过，包含原测试与新增稀疏测试 | `outputs/sparse_unit_tests.log` |
| AME acquire 仿真 | 32 环境、200 步；有限观测/奖励、局部 reset、干净 critic | `outputs/rpo_ame_sparse/sparse_smoke_ame_final/result.json` |
| Affordance acquire 仿真 | 32 环境、200 步；采集与 reset 检查通过 | `outputs/rpo_affordance_sparse/sparse_smoke_affordance/result.json` |
| Affordance robust 仿真 | drift_02 档，32 环境、200 步；actor ±2 cm 回合固定漂移、critic 保持干净 | `outputs/rpo_affordance_sparse/sparse_smoke_robust/result.json` |
| AME 热启动训练 | rough/model_9001.pt，2 次新增 PPO 更新；所有 AME 模块更新 | `outputs/rpo_ame_sparse/sparse_train_ame/result.json` |
| Affordance 热启动训练 | 10 次新增 PPO 更新；379 条有效样本、64 次辅助更新，U-Net 参数更新 | `outputs/rpo_affordance_sparse/sparse_train_affordance/result.json` |
| Affordance 续训 | 更新 10→11；样本累计 379→384，辅助更新累计 64→72，gate/replay 连续 | `outputs/rpo_affordance_sparse/sparse_resume_affordance/result.json` |
| AME 续训与自动评估 | 更新 2→3；训练入口调用独立进程评估并保存验证历史/平地基线 | `outputs/rpo_ame_sparse/sparse_resume_ame/` |
| AME 固定评估 | 640 次尝试、20 个类型/难度组；包含核心 384 场景和补充集 | `outputs/sparse_evaluation_ame/stage/` |
| Affordance 固定评估 | 640 次尝试、20 个类型/难度组；归一化冻结，成功/失败 bootstrap 断言通过 | `outputs/sparse_evaluation_affordance/stage/` |
| 原星形梁几何 | 对原 MeshStar 配置四个难度共 4,900 条射线，表面高度误差 ≤1e-6 m | AME 最终 smoke 的 `legacy_star_reference_rays=4900` |
| AME ONNX | batch 1/4/32，最大绝对误差约 1.43e-6 | `outputs/rpo_ame_sparse/sparse_train_ame/export/deployment.json` |
| Affordance ONNX | batch 1/4/32，最大绝对误差约 2.86e-6 | `outputs/rpo_affordance_sparse/sparse_train_affordance/export/deployment.json` |
| 地形图册 | 11 类 × 最低/中间/最高档，共 33 张碰撞网格投影和射线高度剖面 | `outputs/sparse_catalog_software/manifest.json` |

Affordance 短程训练验收显式使用 `--affordance_warmup 0 --affordance_ramp 2`，以验证门控和辅助更新完整闭环。
正式训练默认仍为 500 轮预热＋1000 轮渐增，没有将验收参数写入任务默认值。

CPU 测试覆盖两个指定 AME 历史 checkpoint 的严格加载、归一化逐张量保留、加载后 std=0.30、
Affordance 权重保留与门控/replay 清零、跨类型拒绝、RNG 往返、阶段门槛、辅助状态迁移、
回放不误升降级、空 reset 批次、推理后 reset、悬空脚不误判、失败优先及停滞不终止。
原 PPO 终态 bootstrap 测试一并通过；实际评估同时断言成功使用截断、真实失败不 bootstrap。

## 结果边界

- 以上为工程闭环验收，不是稀疏地形技能收敛证明。AME 仅微调 2 次更新的诊断评估仍有大量停滞；
  单梁各档未通过，80 cm 新放射梁通过 6/32。没有解锁高难度或进入 robust 训练。
- Affordance 第 11 次新增更新的诊断评估：80 cm 单梁 32/32，新放射梁 27/32；40 cm 目标单梁/放射梁和原星形梁仍未通过。
  两类模型的微调预算与验收门控不同，这些数值不能用作公平算法对照或两组 AME 初始化优劣结论。
- 进入 robust 的能力门槛由合成通过/失败报告做正反向单元测试；当前真实模型不满足门槛，未伪造通过结果开展实际阶段迁移。
- RPO 足部 STL 包围盒约 15.8 × 7.685 cm；25 cm 梁已做几何与射线检查，但真实足部凸包/姿态可达性和策略通行能力仍需专项运动验证。
  默认梁类训练上限锁在等级 6（40 cm），较窄档保留评估；验证后再提高 `SparseCfg.target_level`。
- RTX 图册在本机产生空白帧，显式渲染重试也未完成，因此交付的是同一碰撞三角网格的软件投影与高度剖面，**不是 RTX 仿真截图**。图册脚本保留 RTX 路径并提供 `--software`。
- 标准化评估在独立 Isaac Sim 进程运行，会额外占用显存与内存；本机已验证 32 个训练环境与 640 个评估环境同时存在的入口链路。1024 环境正式训练预算未执行。
- 长训练、多种子对照、40 cm 稳定通行、robust 各档长期收敛和 25 cm 极限可达性均未在本轮宣称完成。

## 复现要点

从 `isal2` 目录使用 `env_isaaclab` 运行 README 的命令。短程调试使用 `--skip-evaluation`，不会解锁课程。
正式初筛不传该选项，初始/250/500 更新均生成固定评估报告；通过率统计包含全部尝试。
resume 使用相同环境数、地形行列、seed 和辅助训练配置；advance 需要匹配源模型指纹的报告，且一次只变更一个阶段因素。
成功与时间限制截断均保留 terminal observation；resume 重新开始物理回合，不恢复瞬时接触/pending 或声称逐轨迹重现。
