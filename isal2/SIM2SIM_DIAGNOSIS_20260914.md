# sim2sim 行走异常排查与修复

2026-09-14。使用本机 `C:/Users/Admin/miniconda3/envs/env_isaaclab/python.exe`，MuJoCo 3.3.3、PyTorch 2.7.0、ONNX Runtime 1.27.0 CPU。

## 结论

确认并修复两个部署代码错误。主要问题是角速度坐标系不匹配，另一个是生成地形写入错误的 heightfield。两组 model_9001.pt 的 ONNX 数值验证通过；本次修复可直接使用现有 ONNX，无需重新训练或导出。

实际下载目录是 `outputs_download/`，不是 `outputs/_download/`。检查的两组模型为：

- `outputs_download/rpo_ame/ISAL2-RPO-AME-v0-Rough/model_9001.pt`
- `outputs_download/rpo_affordance/ISAL2-RPO-Affordance-v0-Rough/model_9001.pt`

## 1. 角速度读取到了惯性主轴坐标系

`deployment/runtime.py` 的 `Simulator.observation()` 原来使用 `mj_objectVelocity(..., mjOBJ_BODY, ..., 1)`。该接口的 BODY 局部坐标是惯性坐标系；训练的 `root_ang_vel_b` 使用机器人 link 坐标系。MuJoCo 提供 XBODY 来访问 link 坐标系，见 [3.3.3 官方对象类型说明](https://mujoco.readthedocs.io/en/3.3.3/APIreference/APItypes.html#mjtobj)。

RPO 的 base_link 惯性矩阵不是对角矩阵，因此这两个坐标系实际差异很大。本机对机器人设置机身角速度 `[1, 2, 3]` 时：

- 旧 BODY 读取值为 `[-0.99718622, 3.00100479, -1.99989748]`。
- 修正后的 XBODY 读取值为 `[1, 2, 3]`。

重力观测一直使用 `xmat` 对应的 link 坐标系，所以旧实现向网络提供了坐标系不一致的重力与角速度。机器人静止时角速度为零，无法暴露此问题；运动后错误反馈会导致动作放大、踢腿和失稳。这也解释了为什么训练指标正常不能排除部署故障。

已将该调用改成 `mjOBJ_XBODY`。新增测试对单位姿态和任意旋转姿态逐轴检查角速度，并与 link 上的 IMU gyro 交叉验证。

## 2. 高度场索引整体错位

`deployment/terrain.py` 从原始 MJCF 继承了一个名为 `hf0` 的 heightfield，随后才追加生成地形。但旧代码从 `model.hfield_adr[0]` 开始依次写入新地形数据，导致写入对象错位。

在默认 mixed 场景中，tile 0 本应完全平坦却出现约 1 cm 高度变化；最后的踏石区没有获得数据，整块变成 -1 m 坑底。其他 tile 的形状也被错配，`terrain.json` 的出生高度因此可能不对应真实地面。

修复为按每个 tile 的名称解析 `mjOBJ_HFIELD` ID，检查尺寸后写入。新增测试逐格核对全部八块地形，并核对保存后重新加载的 MJB。

## 前后对照

同一原 ONNX、平地、命令 `(0.3, 0, 0)`、6 秒。标注“仅修正角速度”的两行只修改了角速度读取，地形代码尚未修改，因此能独立验证角速度修复效果。

| 模型与实现 | 跌倒次数 | 力矩饱和累计次数 | 动作绝对值最大值 |
|---|---:|---:|---:|
| AME 原实现 | 1 | 3387 | 22.625 |
| AME 仅修正角速度 | 0 | 0 | 2.215 |
| Affordance 原实现 | 0 | 984 | 3.944 |
| Affordance 仅修正角速度 | 0 | 0 | 1.911 |

饱和次数是所有物理步、所有关节超出力矩上限的累计数量。原 AME 测试发生跌倒后按原 headless 行为自动 reset，不能将整段位移理解成单次连续行走。

## 修复后真实权重验证

全部使用原来的 `export/model.onnx`，命令 `(0.3, 0, 0)`，seed=42，difficulty=0.5；下表测试没有发生 reset。

| 模型 | 场景 | 时长 | 跌倒 | 饱和次数 | 前向位移 | 根部 z 范围 |
|---|---|---:|---:|---:|---:|---|
| AME | 扩大平地 | 20 s | 0 | 0 | 4.710 m | 0.727–0.751 m |
| Affordance | 扩大平地 | 20 s | 0 | 0 | 5.294 m | 0.733–0.752 m |
| AME | mixed 的平坦 tile 0 | 10 s | 0 | 0 | 2.256 m | 0.727–0.751 m |
| Affordance | mixed 的平坦 tile 0 | 10 s | 0 | 0 | 2.591 m | 0.733–0.752 m |
| AME | rough 的粗糙 tile 1 | 10 s | 0 | 88 | 2.253 m | 0.743–0.788 m |
| Affordance | rough 的粗糙 tile 1 | 10 s | 0 | 283 | 1.312 m | 0.752–0.786 m |

20 秒测试使用 32 m 平地、0.1 m 高度场分辨率，避免走出默认 8 m 地块。mixed tile 0 测试没有跨越全部复杂地形。Affordance 在粗糙地面的速度跟踪仍不理想；这些结果不代表完整楼梯、踏石通过率或全部手柄命令已验收。

## ONNX 与回归检查

- 两个 model_9001.pt 均重新导出到独立诊断目录，batch 1/4/32 的 ONNX checker 和原 Actor 数值对照通过，最大绝对误差分别为 2.623e-6、2.384e-6。
- 六段真实 rollout 每 10 步核对一次原 ONNX 与 checkpoint 的 `act_inference`，合计 400 个观测；最大动作误差 1.907e-6。
- 核对了名称驱动的关节/电机映射、默认姿态、五帧历史顺序、观测缩放、动作缩放、20 ms 策略周期、PD 增益和力矩限幅；本次未发现这些项存在对应的映射错误。
- 新增两项回归测试在备份的旧实现上均失败，在修复后通过；完整测试集 38 项通过。
- 仍存在一般 sim2sim 动力学差异：训练使用 5 ms 物理步长与随机 PD 延迟，部署使用 1 ms、零延迟；训练开启自碰撞，原 MJCF 机器人碰撞掩码关闭自碰撞。它们没有在本次修复中调整，复杂地形的剩余表现需单独评估。

## 复现与文件

从 `isal2` 目录执行，`python` 指向上述解释器：

```powershell
python scripts/sim2sim.py --model outputs_download/rpo_ame/ISAL2-RPO-AME-v0-Rough/export/model.onnx --terrain flat --headless --duration 10 --command 0.3 0 0
python scripts/sim2sim.py --model outputs_download/rpo_affordance/ISAL2-RPO-Affordance-v0-Rough/export/model.onnx --terrain flat --headless --duration 10 --command 0.3 0 0
python -m unittest discover -s tests -p 'test_*.py'
python outputs/deployment_debug/verify_trained.py
```

手柄启动可继续使用原命令，建议先加 `--terrain flat` 检查基本行走；需要关闭旧仿真进程并重新启动以加载修复代码。

诊断数据在 `outputs/deployment_debug/`：`verification_summary.json`、各组 `result.json` / `telemetry.json`、场景、`unit_tests.log`、`regression_before.log`、原部署源码备份 `baseline_source/`。原权重和原 ONNX 未改写。
