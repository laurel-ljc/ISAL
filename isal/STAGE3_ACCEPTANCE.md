# 阶段三验收包：Foot Interaction Tracker 与交互标签

日期：2026-09-06  
状态：实现及本机工程测试完成，待用户验收。阶段四未开始。

## 1. 交付范围与隔离边界

新增任务 `ISAL-Humanoid-Rough-Interaction-v0`，继承阶段二 HeightScan 环境和普通 PPO agent。
`self_supervised.enabled=False` 时不创建 tracker、不输出辅助数据。

本阶段没有新增网络、affordance head、auxiliary loss、optimizer 或 rollout learning buffer；
没有执行训练入口、`runner.learn()` 或任何策略训练。本机只执行了配置/注册检查、
CPU/CUDA 合成 tensor 测试、固定动作仿真和文件导出检查。

阶段一、二环境实现文件保持不变。新旧任务的完整配置对照测试确认 observation、action、
reward、termination、terrain、commands、randomization、symmetry、Actor/Critic 和 PPO 设置一致；
仅新增采样配置与新的实验名称。`robolab`、`rsl_rl` 工作区均无修改。

### 新增文件

| 文件（相对仓库根目录） | 用途 |
|---|---|
| `isal/isal/interaction/__init__.py` | 无 Isaac Sim 依赖的公共采样接口 |
| `isal/isal/interaction/config.py` | `SelfSupervisedCfg` 与配置检查 |
| `isal/isal/interaction/tracker.py` | GPU 向量化事件、快照、pending 槽及标签 |
| `isal/isal/interaction/recording.py` | 有界调试导出及全样本统计 |
| `isal/isal/tasks/direct/humanoid_rough/interaction_env.py` | 自动 reset 前采样与 reset 后发布 |
| `isal/scripts/debug_interactions.py` | 1～4 环境、固定零动作调试入口 |
| `isal/tests/stage3/test_tracker.py` | CPU/CUDA 合成生命周期、坐标与标签测试 |
| `isal/tests/stage3/test_recording.py` | 导出上限、全量统计及空数据测试 |
| `isal/tests/stage3/test_interaction_runtime.py` | 配置公平性、注册、开关和仿真接入测试 |
| `isal/STAGE3_ACCEPTANCE.md` | 本验收记录 |

### 修改文件

- `isal/isal/tasks/direct/humanoid_rough/__init__.py`：新增独立任务注册。
- `isal/isal/tasks/direct/humanoid_rough/isal_env_cfg.py`：新增继承阶段二的采样配置类。
- `isal/isal/tasks/direct/humanoid_rough/agents/isal_agent_cfg.py`：新增普通 PPO 实验名称子类。
- `isal/README.md`：增加采样命令、完整 tensor 契约和标签解释。

## 2. 关键设计决策

### 数据和时序

- 每脚保存一条当前 swing snapshot，以及独立的多个 pending outcome。
  因此上一条记录等待 survival 时仍可接收新的起落事件。
- liftoff snapshot 使用阶段二同定义的 root-relative scan，detach/copy 后保存。
  本阶段禁止对采样任务启用 Actor scan noise/dropout，保证尺度和输入一致。
- query 使用 liftoff 的 root xy/yaw 转换；scan 范围来自实际 ray origins，包含且只包含一次前向偏移。
- 实际传感器和机器人索引分别解析：本机左/右 robot link ID 为 `[20,21]`，
  ContactSensor body ID 为 `[6,12]`。没有将两类索引混用。
- 接触力取当前正 world-z 分量，严格大于 20 N；不使用 reward 的历史力范数判据。
  脚位置和速度均使用 ankle-roll link frame，不混用 link 位置和 COM 速度。
- `_get_dones()` 在 control step 内、原环境自动 reset 前调用 tracker；同一步只推进一次。
  `step()` 在父类 reset 完成后发布结果。单独读取 observation/dones 不推进 tracker。
- explicit reset 清理选中环境的状态与旧输出；automatic reset 保留当前步已结算的样本。
  已返回 packet 拥有独立数据，不被后续 step/reset 改写。

### 窗口与早期失败

使用 `round(window_s / step_dt)`；本机 `step_dt=0.02` 时，outcome 为 12 帧，
survival 为 25 步。Touchdown 是 outcome 第 1 帧、survival age 0；到 age 25 才判定存活成功。
每脚默认 `ceil(25/2)+1=14` 个 pending 槽。显式配置更小容量时溢出计数，不覆盖旧数据。

标签为：

```text
0.40 * exp(-contact_frame_mean_horizontal_speed / 0.15)
+ 0.25 * exp(-max_wrapped_roll_pitch_change / 0.20)
+ 0.20 * contacted_frames / outcome_steps
+ 0.15 * survival
```

输出限制在 `[0,1]`。按用户确认的选择，早期摔倒也结算有效 touchdown：
slip/tilt 使用已观察数据，缺失时段对 contact persistence 贡献为零，survival=0，
并记录 `partial_window`、`observed_frames`。timeout 不标成摔倒；窗口不足则丢弃。
没有有效 touchdown/query 的 swing 失败不生成伪造样本。

### 公共接口

`extras["auxiliary"]` 前缀为 `[N,2,P]`，包括左右脚与 pending 槽。
scan 尾部为 `[1,H,W]`，query/one-hot 为 `[2]`，context 为 `[3]`，target 为 `[1]`。
额外附带观察帧数、部分窗口标志、评分分量和 peak force；无效槽全部为零。
完整字段/dtype 见 README。

`extras["interaction_stats"]` 为累计 GPU scalar 计数和槽位占用。
阶段五可以通过 `packet[key][packet["valid"]]` 获取展平样本，无须修改 wrapper。
未完成的物理事件可以跨 control steps 保留，但 tracker 不承担 replay 或网络更新。

所有标签输入均为机器人交互反馈；没有 terrain type、difficulty、friction 等 metadata 输入。
Peak force 仅记录，不参与 target。CPU 列表和磁盘写入仅存在于独立调试 recorder，
不进入 env → extras 的 GPU 数据链路。

## 3. 测试命令与真实结果

工作目录：`C:\Users\Admin\Documents\ISAL`。

### 完整阶段一至三回归

```powershell
conda run --no-capture-output -n env_isaaclab python -u -m pytest `
  isal/tests -q --tb=short -p no:cacheprovider `
  > outputs/stage3_full_acceptance_tests.log 2>&1
```

最终结果：**58 passed, 1 warning in 59.91s**。
其中既有阶段一/二 15 项，新增阶段三 43 项；无跳过项，CPU 和 CUDA 分支均执行。
唯一 pytest warning 为既有 `empirical_normalization` 弃用提示。

覆盖：静止/初始悬空、左右脚独立及同时事件、重叠记录、同一步批量摔倒结算、
最快交替接触、显式槽位溢出、不变快照、非零平移/yaw、边界和越界、
接触帧 slip、tilt 周期边界、窗口时序、高 slip 低分、早期失败、timeout、
部分 reset、无效输入隔离、输出所有权、GPU/dtype、开关和完整配置一致性。

仿真用例分别使用 1 环境 aux-off、4 环境 aux-on，各先执行 5 步真实零动作。
4 环境用例再用 4 个固定动作 control steps 配合明确标记的合成反馈，验证
终止前快照、reset 后 packet、左右脚同时结算及未 reset 环境隔离。
这些注入标签仅用于测试，未混入下面的真实导出。

独立集成测试命令及结果：

```powershell
conda run --no-capture-output -n env_isaaclab python -u -m pytest `
  isal/tests/stage3/test_interaction_runtime.py -q --tb=short -p no:cacheprovider `
  > outputs/stage3_runtime_isolated.log 2>&1
```

**4 passed in 47.05s**。

### 真实零动作采样

```powershell
conda run --no-capture-output -n env_isaaclab python -u `
  isal/scripts/debug_interactions.py --headless --num_envs 1 --steps 200 `
  --max_samples 1000 --output outputs/stage3_acceptance_zero_action `
  > outputs/stage3_debug_zero_action.log 2>&1
```

seed=42，200 control/environment steps，4 秒仿真时间，scan 为 `[1,1,17,11]`。
`training_started=false`、`synthetic=false`。

| 指标 | 结果 |
|---|---:|
| liftoff / touchdown | 8 / 10 |
| 有效样本 / 已保存 | 2 / 2 |
| 早期部分窗口样本 | 1 |
| query 越界丢弃 | 3 |
| 不完整记录清理 | 3 |
| 没有有效 snapshot 的 touchdown | 5 |
| 非有限输入丢弃 / 槽位溢出 | 0 / 0 |
| 最大每脚 pending 占用 | 1 |
| 结束时 pending | 0 |

| 脚 | target | 观察帧数 | survival | partial |
|---|---:|---:|---:|---|
| 左 | 0.33024466 | 12 | 1 | false |
| 右 | 0.12847789 | 2 | 0 | true |

target 均值 0.22936127，标准差 0.10088339；两条均落入 bad 区间。
这只是零动作采样链路检查，不能作为步态质量、标签总体分布或学习有效性的证据。

产物：

- `outputs/stage3_acceptance_zero_action/summary.json`：全量计数、均值/标准差、query 范围和 histogram。
- `outputs/stage3_acceptance_zero_action/samples.csv`：两条样本的 query 与分量。
- `outputs/stage3_acceptance_zero_action/samples.pt`：原始 snapshot/context/诊断 tensor。

### CLI 与产物交叉核对

```powershell
conda run --no-capture-output -n env_isaaclab python -u `
  isal/scripts/debug_interactions.py --help
```

CLI 检查通过。另在 `env_isaaclab` 中以 `torch.load(..., weights_only=True)` 读取实际 PT，
重新计算 target，并核对 CSV、JSON 与 PT：

```text
artifact_check=PASS
samples=2
height_scan_shape=(2,1,17,11)
finite=True
targets_recomputed=True
csv_json_pt_consistent=True
```

```powershell
git diff --check
git -C robolab status --short
git -C rsl_rl status --short
```

无 whitespace error；两个上游 submodule 均为空输出。

## 4. 已处理问题、限制与阶段门禁

- 初轮合成测试发现非有限接触力与终止同帧时无效记录计数遗漏，已修复并在 CPU/CUDA 回归通过。
- 默认受限沙箱不能写入 Isaac Lab USD 缓存及 pytest 临时目录；最终仿真验收使用了获准的缓存访问。
- 本地 Isaac Lab 会在重复建环境时沿 RayCaster 静态 mesh 缓存遍历 Warp 循环引用；
  清理缓存后，同进程重复建立 SimulationContext 又出现停滞。
  最终将每个仿真测试放入全新进程，统一由 pytest 调度，子进程仍显式使用
  `conda run --no-capture-output -n env_isaaclab python`。未修改上游源码或全局运行时类。
  子进程同时检查 pytest 文本结果，避免本机 conda 偶尔以零退出码转发失败的问题。
- Isaac Sim 退出时仍打印既有 USD detach / recursive unload 警告；完整测试与导出均已成功完成。
- 工程验收没有未通过项。真实步态标签统计、每 rollout 样本量、训练收敛及方法收益尚未验证，
  按计划留给外部训练阶段。早期部分窗口标签存在观察时长差异，导出已显式标记，后续应单独分析。

下一阶段准备：在阶段三验收通过后，才新增 terrain encoder、独立 Actor/Critic 和 query affordance head，
首先固定 `lambda_aux=0`，进行前向、梯度隔离、checkpoint 与导出一致性测试。
本次没有开始阶段四。
