# 第五阶段验证：ONNX 与 MuJoCo

日期：2026-09-11。项目：`C:/Users/Admin/Documents/ISAL/isal2`。

解释器：`C:/Users/Admin/miniconda3/envs/env_isaaclab/python.exe`；Python 3.11、PyTorch 2.7、RSL-RL 3.3.0、onnx 1.20.1、onnxruntime-gpu 1.27.0、MuJoCo 3.3.3。运行使用 CPUExecutionProvider，没有更换已有依赖。

## 结果

- Base、AME、Affordance 均导出成功，ONNX checker 通过；batch 1/4/32 与原 Actor 一致，阈值 `atol=1e-5, rtol=1e-4`。
- 现有 27 项测试加新增 9 项部署测试，共 36 项通过。
- 九组 MuJoCo 验收各运行 200 个控制步（4 秒仿真时间、4,000 物理步），有限值检查通过，进程正常退出，无 Isaac Sim 退出停留。
- 额外在隔离 Python 进程、非项目根工作目录运行 mixed 方块区 200 步，通过；实际运行未加载 torch、rsl_rl、isaaclab、isal 或 robolab。
- 官方 viewer 打开、暂停、重置、继续和关闭通过真实窗口测试，输入来自模拟 XInput 状态。运行 22 步、1 次手动 reset、0 次跌倒，正常退出。
- 物理手柄：探测 XInput 0–3 均未连接，**实物按键、摇杆及实际拔插未验证**。模拟状态测试不能代替此项。
- wheel 构建及内容检查通过，含 31 个原始资源文件、部署模块和两个脚本，不包含输出、build 或旧工程包。

## 导出误差

| 模型与 checkpoint | α | B=1 最大绝对误差 | B=4 | B=32 |
|---|---:|---:|---:|---:|
| Base `external_rsl_train/model_5.pt` | — | 6.71e-8 | 1.19e-7 | 1.34e-7 |
| AME `ame_train_acceptance/model_5.pt` | — | 8.20e-8 | 1.49e-7 | 1.79e-7 |
| Affordance `aff_train_acceptance/model_5.pt` | 0 | 7.45e-8 | 1.49e-7 | 1.34e-7 |
| Affordance `aff_gate_acceptance/model_10.pt` | 1 | 1.34e-7 | 1.19e-7 | 2.09e-7 |

导出保留模型中的 α；没有修改训练 checkpoint。数值测试也检查归一化状态保留、图中无 Critic、高程改变动作、α=0 时修改 U-Net 不改变动作，以及 α=1 时蓝色网络参与 Actor 推理。

导出器会打印 PyTorch 关于 Slice 无法常量折叠的 warning；这不影响 ONNX checker 和 ONNX Runtime 的实际运行，未屏蔽该 warning。

## MuJoCo 运行结果

所有行命令均为 `(vx,vy,wz)=(0.3,0,0)`，seed=42、difficulty=0.5。flat 出生在 tile 0，rough 出生在 tile 1 的真实粗糙地面，rough_hard 出生在 tile 7 的踏石中心平台。

| 模型 | 地形 | 控制步 | 跌倒/reset | 饱和计数 | ONNX 平均耗时 ms |
|---|---|---:|---:|---:|---:|
| Base | flat | 200 | 2/2 | 0 | 0.162 |
| Base | rough | 200 | 2/2 | 0 | 0.216 |
| Base | rough_hard | 200 | 2/2 | 34 | 0.156 |
| AME | flat | 200 | 2/2 | 0 | 0.534 |
| AME | rough | 200 | 1/1 | 0 | 0.459 |
| AME | rough_hard | 200 | 2/2 | 45 | 0.487 |
| Affordance α=1 | flat | 200 | 2/2 | 0 | 0.858 |
| Affordance α=1 | rough | 200 | 1/1 | 0 | 0.732 |
| Affordance α=1 | rough_hard | 200 | 2/2 | 45 | 0.746 |
| Affordance α=1，隔离进程 | mixed tile 6 | 200 | 4/4 | 103 | 0.755 |

饱和计数为各物理步、各关节 PD 原始力矩超过对应限幅的累计数量；下发力矩已经裁剪。耗时测量有并发进程影响，仅统计 ONNX 前向，不代表完整仿真周期或长期性能保证。

这些短程 checkpoint 会跌倒。本阶段通过代表观测、动作、PD、地形与 reset 链路可运行，不代表机器人已学会稳定行走或通过复杂地形。

几何测试覆盖 xy 排序、平移和 90° yaw、已知斜面高度、机器人几何排除、射线未命中、八种混合地形的有限值与边界，以及踏石顶部和降低的坑底。地形为离散 heightfield，边缘存在网格插值；不是 PhysX 地形的逐三角形复刻。

## 实际命令

以下从项目根 `isal2` 执行，`python` 指向上述解释器。

```powershell
python scripts/export_onnx.py --checkpoint outputs/rpo_base/external_rsl_train/model_5.pt
python scripts/export_onnx.py --checkpoint outputs/rpo_ame/ame_train_acceptance/model_5.pt
python scripts/export_onnx.py --checkpoint outputs/rpo_affordance/aff_train_acceptance/model_5.pt
python scripts/export_onnx.py --checkpoint outputs/rpo_affordance/aff_gate_acceptance/model_10.pt

# 执行过的 9 组仿真命令，以等价循环列出。
$runs = @{
  base='rpo_base/external_rsl_train'
  ame='rpo_ame/ame_train_acceptance'
  affordance='rpo_affordance/aff_gate_acceptance'
}
foreach ($kind in $runs.Keys) {
  foreach ($terrain in @('flat','rough','rough_hard')) {
    $tile=0
    if ($terrain -eq 'rough') { $tile=1 }
    if ($terrain -eq 'rough_hard') { $tile=7 }
    python scripts/sim2sim.py --model "outputs/$($runs[$kind])/export/model.onnx" --terrain $terrain --spawn-tile $tile --headless --duration 4 --command 0.3 0 0 --output "outputs/deployment_validation/${kind}_$terrain"
  }
}

python -m unittest discover -s tests -p 'test_*.py'
python tests/deployment_viewer_check.py
python -m isal2.scripts.export_onnx --help
python -m isal2.scripts.sim2sim --help
python setup.py bdist_wheel --dist-dir outputs/deployment_validation/wheels
git diff --check
```

隔离测试在 `isal2/outputs/deployment_validation` 工作目录执行：

```powershell
python -I C:/Users/Admin/Documents/ISAL/isal2/scripts/sim2sim.py --model C:/Users/Admin/Documents/ISAL/isal2/outputs/rpo_affordance/aff_gate_acceptance/export/model.onnx --terrain mixed --spawn-tile 6 --headless --duration 4 --command 0.3 0 0 --output C:/Users/Admin/Documents/ISAL/isal2/outputs/deployment_validation/isolated_mixed
```

原始结果在 `outputs/deployment_validation/`：各仿真目录含 `result.json`、逐步 `telemetry.json`、场景 MJB 与地形参数；`unit_tests.log` 保存测试输出；`viewer/viewer_check.json` 标明模拟输入与实际设备探测结果。每个导出目录的 `deployment.json` 保存模型哈希与数值误差。

## 验证过程中处理的问题

本机受限 Windows 进程不能访问 Python `TemporaryDirectory` 创建的私有 ACL 子目录。首次新增 checkpoint 测试因此失败，改用继承工作区 ACL 的普通 UUID 输出目录后，36 项测试通过。测试产物保留在 ignored outputs 中。

`pip wheel . --no-deps --no-build-isolation --no-index -w outputs/deployment_validation/wheels` 同样因临时 build tracker 权限失败。使用工作区内 `setup.py bdist_wheel` 完成离线构建并检查 wheel，未修改环境权限或安装依赖。该命令仅用于本次构建验证，日常安装仍使用 pip。

没有执行长程训练或最终通过率评估；实物手柄验收需要接入 Xbox 兼容 XInput 设备。
