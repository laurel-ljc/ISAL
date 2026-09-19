# Affordance 两阶段任务验收

验证日期：2026-09-18 至 2026-09-19。环境：Windows、`env_isaaclab`、Python 3.11、Isaac Sim 5.1、PyTorch 2.7.0+cu128、RSL-RL 3.3.0。

## 实现范围

- 注册 `ISAL2-RPO-Affordance-Stage1-v0`、`ISAL2-RPO-Affordance-Stage2-v0`，默认训练任务仍为 AME Stage1。
- 新环境、采集器和配置位于 `tasks/common/affordance`，运行不导入 `deprecated_tasks`。四个阶段任务共用对应的地形、课程、终点命令和干净高程图。
- Stage1 保留500轮等待、1000轮渐进及累计256条有效样本门槛；未接入预测时使用常数质量图0.5。Stage2 从首次推理开始 alpha=1，拒绝非零 warm-up/ramp 覆盖。
- 两阶段均在回放有效样本至少64条时，每轮 PPO 后执行8次监督更新，batch=256。warm-up 不阻止 U-Net 学习。
- 成功和超时保留已完成评价窗的样本，清理未完成窗；真实失败沿用零标签规则。手动和局部 reset 清理对应环境的接触状态。
- warm-start 加载完整 Affordance 模型及归一化，重置优化器、回放、课程、迭代数和动作标准差；目标阶段决定初始 alpha。同阶段 resume 恢复模型、两组优化器、回放、统计、课程和随机数状态，重新开始回合。

## 自动回归

从仓库根目录执行：

```powershell
python -m unittest discover -s isal2/tests -p "test_*.py"
```

最终结果：**104项通过，耗时9.271秒**，退出码0。日志：[aff_stages_all_tests_final.log](outputs/aff_stages_all_tests_final.log)。包含已有 AME、deprecated Affordance、课程几何、部署测试，以及新增19项阶段测试。

新增测试覆盖调度边界0/499/500/1000/1499/1500、255/256条样本门槛、warm-up 中监督更新、首次推理使用预测、优化器与梯度隔离、成功/失败/超时及局部重置标签、两阶段保存恢复、跨阶段 warm-start、拒绝 AME 检查点及导出包装器 alpha=0/0.5/1 的推理一致性。

任务列表包含四个 active 阶段任务和原有 deprecated ID；新任务目录不存在 `deprecated_tasks` 导入。

## Isaac Sim 验证

两阶段使用8个环境完成 smoke test，并直接检查成功终止、超时 bootstrap、独立类型等级、actor/critic 干净扫描、局部 reset。Stage2 还检查实际叠加的水平速度扰动。两阶段均在创建场景前比较与对应 AME 的机器人、奖励、仿真、噪声、归一化、命令、场景、事件及课程配置，结果一致。

| 检查 | 步数 | 随机运行 reset 次数 | 结果文件 |
|---|---:|---:|---|
| Stage1 首次 smoke | 150 | 15 | [result.json](outputs/rpo_affordance_stage1/aff_stage1_smoke/result.json) |
| Stage1 最终配置与生命周期检查 | 40 | 1 | [result.json](outputs/rpo_affordance_stage1/aff_stage1_smoke_final/result.json) |
| Stage2 最终检查，包含扰动 | 150 | 16 | [result.json](outputs/rpo_affordance_stage2/aff_stage2_smoke_final/result.json) |

最初的 Stage2 配置比较在场景创建后执行，因 Isaac 会修改已解析的场景配置而失败；已将比较移至场景创建前，并重新通过两个阶段的检查。

短程 PPO 使用64个环境，Stage1为7列地形，Stage2为5列；监督配置使用任务默认值。样本来自真实仿真接触。

| 运行 | 迭代范围 | 累计有效样本 | 累计监督更新 | alpha | 结果 |
|---|---|---:|---:|---:|---|
| Stage1 从零训练 | 0→10 | 915 | 72 | 0 | [result.json](outputs/rpo_affordance_stage1/aff_stage1_train/result.json) |
| Stage1 resume | 10→12 | 1114 | 88 | 0 | [result.json](outputs/rpo_affordance_stage1/aff_stage1_resume/result.json) |
| Stage2 从 Stage1 warm-start | 0→10 | 194 | 56 | 1 | [result.json](outputs/rpo_affordance_stage2/aff_stage2_train/result.json) |
| Stage2 resume | 10→12 | 215 | 72 | 1 | [result.json](outputs/rpo_affordance_stage2/aff_stage2_resume/result.json) |

四次运行均确认 Actor、Critic、注意力相关模块和 U-Net 权重实际更新，损失有限。Stage1 在 alpha=0 时持续学习 U-Net；Stage2 样本不足256条时也保持 alpha=1。

## ONNX 导出

从两个阶段的 `model_10.pt` 导出，分别比较原 Actor、标准算子包装器与 ONNX Runtime，batch为1/4/32，容差为 atol=1e-5、rtol=1e-4，均通过。

| 检查点 | 导出 alpha | 最大绝对误差 | 元数据 |
|---|---:|---:|---|
| Stage1 | 0 | 1.78814e-7 | [deployment.json](outputs/rpo_affordance_stage1/aff_stage1_train/export/deployment.json) |
| Stage2 | 1 | 2.38419e-7 | [deployment.json](outputs/rpo_affordance_stage2/aff_stage2_train/export/deployment.json) |

## 复现命令

以下在 `isal2` 目录执行。更换 `--run_name` 可保留本次验收输出；resume 的 `--max_iterations` 是本次追加的迭代数。

```powershell
python scripts/train.py --task ISAL2-RPO-Affordance-Stage1-v0 --headless --num_envs 8 --terrain_cols 7 --smoke_steps 40 --check_reset --run_name aff_stage1_smoke_final
python scripts/train.py --task ISAL2-RPO-Affordance-Stage2-v0 --headless --num_envs 8 --terrain_cols 5 --smoke_steps 150 --check_reset --run_name aff_stage2_smoke_final
python scripts/train.py --task ISAL2-RPO-Affordance-Stage1-v0 --headless --num_envs 64 --terrain_cols 7 --max_iterations 10 --run_name aff_stage1_train
python scripts/train.py --task ISAL2-RPO-Affordance-Stage2-v0 --headless --num_envs 64 --terrain_cols 5 --max_iterations 10 --warm-start outputs/rpo_affordance_stage1/aff_stage1_train/model_10.pt --run_name aff_stage2_train
python scripts/train.py --task ISAL2-RPO-Affordance-Stage1-v0 --headless --num_envs 64 --terrain_cols 7 --max_iterations 2 --resume outputs/rpo_affordance_stage1/aff_stage1_train/model_10.pt --run_name aff_stage1_resume
python scripts/train.py --task ISAL2-RPO-Affordance-Stage2-v0 --headless --num_envs 64 --terrain_cols 5 --max_iterations 2 --resume outputs/rpo_affordance_stage2/aff_stage2_train/model_10.pt --run_name aff_stage2_resume
python scripts/export_onnx.py --checkpoint outputs/rpo_affordance_stage1/aff_stage1_train/model_10.pt
python scripts/export_onnx.py --checkpoint outputs/rpo_affordance_stage2/aff_stage2_train/model_10.pt
```

## 验证边界

本机 Isaac Sim 存在 Vulkan/ShaderCache/SharedMutex 相关告警；部分运行在写出结果和检查点、进入 `Closing simulator` 后未自行退出，已中断本次启动的进程。训练与检查结果已保存，不能据此声称仿真器退出流程正常。

本次验证证明短程采集、PPO、监督更新、恢复和导出链路可运行；没有执行1500轮真实训练或证明复杂地形长期收敛。Stage1 长调度边界由自动测试覆盖，Stage2 独立初始化与空回放由自动测试覆盖，实际 Stage2 PPO 使用 Stage1 warm-start。
