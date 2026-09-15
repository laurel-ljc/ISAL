# ISAL2：D435i 高程感知 sim2real 实施计划

日期：2026-09-14。性质：代码与官方仓库调研后的实施建议，尚未实现或在实机验证。

已知硬件：Jetson Orin、RealSense D435i、ROS、机器人 IMU。尚待实机盘点：Orin 型号/内存、JetPack/CUDA、Ubuntu/ROS 发行版、IMU 安装位置与时间戳、是否已有可靠里程计、相机真实安装外参。

## 1. 决策与边界

保留 ISAL2 的 PPO/AME/Affordance 结构，增加可部署的局部高程输入。第一版不增加 AMP、MoE、图像循环网络或地图补全网络。部署依托 RoboParty 的驱动、机器人接口和控制框架；新增高程建图、地图观测适配和 ISAL2 导出适配。

采用重力对齐、机器人附近滚动维护的短时地图。不要求全局建图或回环，但历史点云融合必须有平移和旋转的相对运动估计。IMU 不能单独提供可靠的平移里程计，D435i 本身也不是现成的六自由度定位系统。

## 2. 已确认的代码事实

- ISAL2 当前地图为 11×17、间隔 0.1 m、中心在 base_link、x 向前/y 向左。Actor 的本体观测为 78×5=390，地图为 187 个数，输出 23 关节动作。
- `_height_scan()` 返回 `(base_z-ground_z-0.75).clamp(-1,1)` 再乘观测缩放，而非直接返回世界地面 z。具体偏置/缩放以导出元数据为准。
- 当前 TerrainAttention 和 AffordanceUNet 都为单通道高程；XY 位置编码以米为单位。接触采集器保存候选抬脚帧地图，再把首次候选落地脚掌中心投回旧地图，之后评价接触质量。
- 本地 RoboLab parkour 使用 GroupedRayCaster 深度相机，64×36，经裁剪变成 18×32，取 8 个历史帧，再编码为 128 维。策略是 EncoderMoEActorCritic + AMP，不是 ISAL2 架构。
- parkour 相机配置写有 `ray_alignment="yaw"`，但本地 `GroupedRayCasterCamera._update_ray_infos()` 明确使用完整姿态。相机几何应检查实际实现，不能仅由这个配置字段判断。该函数对 camera quaternion 字段的约定也应通过已知三维点验证，不能仅依赖字段名。
- 相机安装在 torso_link；URDF 中 torso_link 与 base_link 之间有活动 torso_joint，因此相机到 base 的变换可能随腰关节变化。
- RoboParty 部署主仓库所引用的 camera 子模块：D435i 原始深度配置为 480×270×60，depth_node 默认 50 Hz，输出 `/depth_obs` 的 Float32MultiArray；parkour 输出为 128 维。
- 官方 parkour 策略输入为 78×8+128=752，采用 obs_major。当前 ISAL2 为 frame_major 的 5 帧历史。
- 官方推理 setup_model 目前只支持一个 ONNX 输入；ISAL2 当前 AME/Affordance ONNX 为 policy、height_scan 两个输入。
- 官方 IMU ROS 发布函数使用发布时刻 `now()`，本次检查的函数仅填姿态和角速度，未填线加速度；Python SDK 文档说明 getter 没有采样时间戳。部署侧需补充时间同步与数据有效性，不能把消息 stamp 当成经过验证的硬件采样时刻。
- 现有 sim2sim 诊断确认修正过角速度坐标系错误；已验证范围不能替代楼梯/踏石全课程验收。

本地入口：`tasks/base/scene_cfg.py`、`tasks/base/base_env.py`、`tasks/ame/ame_env.py`、`tasks/affordance/collection.py`、`modified_rsl/modules/terrain_attention.py`、`modified_rsl/modules/affordance.py`、`deployment/export.py`；对照 `../robolab/robolab/tasks/manager_based/parkour/`。

## 3. 工具与数据链路

建议链路：

```text
D435i / realsense2_camera
  ├─ 米制深度 + CameraInfo + 原始采集时间 ──→ 点云与机器人自身过滤
  └─ RGB + 对齐深度 ──→ RGB-D 里程计（若无现成机器人里程计）
机器人 IMU + 关节反馈 ──→ 姿态/运动估计与关节 TF
点云 + 对应时刻 TF ──→ elevation_mapping_cupy 局部地图
局部地图 + 当前 base 位姿 ──→ ISAL2 固定网格采样 H/M/A
本体历史 + H/M/A ──→ 完整 ISAL2 Actor ONNX ──→ 官方关节控制接口
```

### 工具选择

1. 深度采集：复用官方相机包中的 realsense-ros 驱动；高程分支直接订阅原始 rectified depth 和对应 CameraInfo。不要对同一台相机再开第二个驱动实例。
2. 投影：librealsense 的去投影约定或等价 GPU 针孔去投影。使用米制深度和实际内参；改变图像尺寸时同步改内参。无效点、超量程点、机器人自身点单独标记，不能把 2.5 m 的占位值投成地面。
3. 建图：优先评估 `leggedrobotics/elevation_mapping_cupy` 的 ROS 2 版本，利用 Orin GPU。若系统为 Humble，先固定 `v2.0.0` 等经检查的兼容版本；当前 `ros2` 分支 README 已面向 Ubuntu 24.04/Jazzy/CUDA 12，不能不看版本就直接拉最新分支。Orin 上的构建、CUDA/CuPy 兼容性和延迟均需实测。
4. 地图传输与显示：ROS 2 GridMap、tf2、robot_state_publisher、RViz2、rosbag2。格点排列由坐标采样确定，不能直接 flatten GridMap 内部循环缓冲矩阵。
5. 平移里程计：先盘点现有机器人状态估计。若没有，先用 RTAB-Map `rgbd_odometry` 做低速数据采集和地图验证；它有官方 D435i 示例。此时须开启颜色和用于里程计的深度对齐；建图仍可走原始深度。先不启用全局 SLAM/回环。
6. 动态行走时若视觉里程计质量不足，再评估关节运动学、接触约束与 IMU 的融合估计。接触可靠性和传感器标定是前置条件，不能把 IMU 积分或指令速度充当可靠里程计。RGB-D 里程计也不应未经行走验证就被视为实机最终定位方案。

第一版关闭不必要的语义、平面分解和学习式通行性插件，先验证 elevation、validity、variance/age。工具提供的 traversability 分数不能替代 ISAL2 的接触结果标签。

### 初始几何与时间参数（建议起点，不是实测性能）

- 保留网络 11×17、0.1 m 网格，便于与旧策略比较；覆盖 x≈[-0.8,0.8]、y≈[-0.5,0.5]。
- 底层滚动地图可先约 2×2 m、0.05 m 分辨率，再按网络网格采样。网格尺寸与库的 cell-center 定义必须测试。
- 深度先保留 480×270 有效视野，按性能测量再降采样；不要照搬 parkour 的 18×32 裁剪，因为它会丢掉建图需要的点。
- 建图更新目标先定 20–30 Hz，策略保持 50 Hz；用 p95 延迟、帧龄、掉帧率测是否满足要求，不能将平均 FPS 当作时延验收。
- 历史保留长度按“地面退出视野到落脚”的实测时长选择，可从约 0.5–1 s 做实验，随位置不确定性和数据年龄淘汰。
- 10 cm 网络网格先用于接口验证和宽台阶。精细落足若需要 5 cm，同一覆盖范围将变成 21×33；要同时修改 XY 编码、采集、回放与导出，并重新训练。不要把当前网格宣称为厘米级落足表示。

## 4. 网络改动

第一版输入改为三通道 `[B,3,11,17]`：

| 通道 | 定义 |
|---|---|
| H | 有效格子的相对高程，保持原偏置、符号、截断和缩放 |
| M | 观测有效性；未知、过期、融合冲突等位置为 0 |
| A | 距最后一次有效观测的时间，截断并归一化；未知为最大龄 |

未知 H 可以填 0，但必须连同 M=0 进入网络，不能单独解释成真实平地。建图保留方差，第一版用于有效性判定；后续做消融时再考虑增加连续置信度通道。

- `TerrainAttention.policy_encoder`：第一层 Conv2d 输入从 1 改为 3，后续 CNN、32 维注意力和本体 query 保留。
- `AffordanceUNet`：第一层从 1 改为 3，仍输出一张接触质量图。左右脚监督关系、PPO/监督优化器隔离和 no_grad 路径保留。
- XY 位置编码保持真实米制坐标；不需要改成相机 UV 编码。
- 对过期/未知格子的质量融合设置中性值，例如 `Q_used=0.5+alpha*valid*(Q-0.5)`，而非标成失败。若增加 attention mask，要实现全部无效时的有限值回退或 null token，避免 softmax 全被遮蔽而出 NaN。
- 第一版不加 GRU，地图提供显式短时记忆；若测试表明仍有时间信息不足，再单独评估循环网络。
- 本体 390 维、Actor MLP、Critic、奖励先保留。Critic 继续使用干净高程/特权状态。
- 保留并正确导出 ISAL2 自己的 actor normalizer；RoboLab parkour 关于关闭 normalizer 的注释针对它的分离深度编码实现，不能直接套到 ISAL2。

旧权重只允许显式迁移：首层旧高程核复制到 H 通道，新通道可零初始化以保留初始响应，其余同形参数按名称校验；重新创建优化器和回放缓存。几何一致仍需再训练。不要通过放宽 strict 加载伪装成普通 resume。

## 5. 新增训练任务

保留现有 Base/AME/Affordance 作为基线。以下任务名为建议新增，当前不可运行。

| 任务 | Actor 地形输入 | 目的 |
|---|---|---|
| `ISAL2-RPO-AME-MapRobust-v0` | 干净高程经视野/块状缺测/延迟/位姿误差扰动，附 M/A | 快速检验网络改动、失效回退和训练稳定性 |
| `ISAL2-RPO-AME-DepthMap-v0` | 实际仿真相机深度，经去投影、局部融合和采样得到 H/M/A | 必需的完整感知训练与 AME 部署基线 |
| `ISAL2-RPO-Affordance-DepthMap-v0` | 与上项相同，再开启接触标签与质量图 | 验证 Affordance 在真实感知条件下的增益 |

MapRobust 是快速过渡实验，不能替代 DepthMap。若资源有限，最少新增后两个任务即可；第一个可作为共用环境的配置模式。

训练顺序：旧基线复杂地形验收 → MapRobust 调试/预训练 → AME-DepthMap → Affordance-DepthMap。若旧高程教师足够好，可选择动作/特征蒸馏加速学生训练，但不是第一版的必要新训练系统，也不能强迫学生在不可观测地形上完美模仿教师。

### 传感器与扰动

- 从 RoboLab 复用 grouped ray-caster、机器人遮挡模型与部分深度噪声组件；高程分支在归一化/裁剪前取米制深度，并显式传递有效 mask。
- 相机安装实际位置与标定误差分开建模：真实相机姿态决定图像，估计姿态用于重建；两边施加相同变化并不能模拟错误标定。
- 加入边缘缺测、成片空洞、深度偏差、相机刷新节奏、传输/处理延迟、丢帧和冻结帧；范围最终由 D435i 记录数据拟合，不能只叠独立高斯噪声。
- 位姿误差作用于建图变换，包含相关漂移、延迟和失效；reset/里程计跳变清空历史，不将旧地图带入新 episode。
- 场景先平地、缓坡、低台阶，再到上下楼梯、粗糙地面和较窄落脚区域。加入原地转向、俯仰/腰部变化、停走、侧移导致的视野变化。
- 训练保留部分失效输入，使策略经历未知区域；在难地形上长期失明没有性能保证，部署必须能识别并处理失效。

### 仿真与实机建图的一致性

不在数千个 Isaac 环境中启动数千个 ROS 节点。训练端使用批量 Torch/Warp 建图，复用可抽取的建图规则，并与部署所用 CuPy 版本做对照；这不是自动成立的等价。

必须统一：去投影、坐标约定、格心/边界、点融合、有效性、遮挡清理、历史衰减、地图采样、H 归一化。用同一份带时间戳的深度/轨迹序列比较两端输出与下游动作；再以少量环境实际接入生产建图节点做闭环评估。若近似训练 mapper 无法匹配部署误差，先修一致性或修改训练扰动，不能带着差异直接上实机。

## 6. 标签采集改动

- 抬脚候选时保存 Actor 实际可用的 H/M/A、地图 frame/pose、地图时刻及观测版本；不得重新从干净扫描器生成监督输入。
- 异步深度与当前网格区分：若地图被重采样到当前 base 网格，记录该网格的当前位姿；若直接保存旧网格，则记录其旧位姿。单元格年龄仍按来源深度计。
- 使用相同网格定义把落足世界点转换为旧网格 query；保留原接触时序与支撑/滑动等仿真真值评分。估计误差和几何偏差应通过点/表面一致性及置信度检查，低可信样本丢弃或降权。
- 对落脚 query 检查有效性、年龄以及双线性插值邻域，避免插值跨越未知格子或台阶边缘而产生错误监督。保留有效样本率与被拒原因统计。
- 未观测区域没有接触标签，不补 0；不将“建图未知”当“落脚失败”。
- 第一版实机只做推理与记录，不做在线 SL/PPO。仿真足部扫描支撑率并非 D435i 能直接替代的实机标签，后续实机自监督采集需要单独设计。

## 7. 官方部署适配

建议新增 `isal2_elevation` 建图/适配包，以及 inference 的 `isal2_ame_map.yaml`、`isal2_affordance_map.yaml`，原 parkour 配置保留作对照。

### 导出和观测

- 将完整 Actor（本体归一化、CNN、U-Net、注意力、动作 MLP）导出为一个 ONNX，避免套用 parkour 的 128 维 depth_encoder 接口。ISAL2 的注意力 query 依赖本体状态，不能直接当作纯图像编码器替换。
- 为官方单输入运行时增加导出 wrapper：输入 `[policy390, H187, M187, A187]`，总长 951，内部切分；输出 23 动作。这是导出适配，不必改变训练时的多组 TensorDict。
- 可用官方 sparse history 配置表达：`frame_stacks:[5]`、`obs_stack_orders:["frame_major"]`，每帧本体六项共 78，`perception:561@0` 只使用当前地图。用单元测试和固定观测确认最终确为 390+561，而非每帧都堆叠地图。
- 官方 parkour 把角速度缩放设为 0.25，当前 ISAL2 默认为 1；官方命令角速度缩放还复用了 ang_vel 参数。缩放、历史、关节顺序、零偏、PD、动作缩放必须来自 ISAL2 元数据，不能整份复制 parkour.yaml。
- 增加地图 schema 版本，写明形状、米制 XY、通道顺序、H 偏置、年龄尺度和缺测语义。部署 adapter、ONNX 与 checkpoint 必须一致。

### 时间戳和执行节奏

- 新增带 Header/采集时间、地图重采样时刻、frame_id、序号/epoch、数据层的地图消息；不要仅发送无时间戳 Float32MultiArray。推理收到后校验，再复制为数值向量供模型使用。
- 同一深度帧只能作为一次新观测融合，不能因 50 Hz 定时器反复处理缓存图像而把旧地图变“新”。
- 点云、关节 TF 与 IMU/里程计按采集时刻对齐；不使用消息到达时的最新姿态代替历史姿态。
- 建图异步执行，策略线程仅取完整快照，不能等待 GPU 建图锁而拖慢关节控制。
- 短时缺测走训练过的 mask/age 路径；长期数据过期或位姿失效触发经过验证的停走/稳定控制流程。不能把直接置零动作等同于稳定停步。
- Orin 首先验证 ONNX Runtime 单图数值与延迟；是否采用 GPU/TensorRT 由共享 GPU 建图下的 p95 实测决定。CPU 推理也可作为对照，不能把桌面 x86 的速度当成 Orin 速度。

## 8. 分阶段产出与验收

| 阶段 | 产出 | 进入下一阶段的条件 |
|---|---|---|
| P0 基线和接口 | 现有策略按地形成功率、传感器/TF/时钟清单、Orin 版本清单 | 先明确控制基线，确认相机与机器人状态能记录和对齐 |
| P1 静态几何 | D435i bag、点云、局部地图、落足/标尺叠加 | 平地和台阶符号正确；改变相机俯仰、腰部角度后地形保持一致；未知区域不伪造 |
| P2 运动与建图 | 带里程计的短时滚动地图、原始数据与时延报告 | 平移/转向下边缘无严重拖影，失效可识别，地图年龄正确 |
| P3 网络与训练 | MapRobust、AME-DepthMap、完整输入/导出适配 | 缺测/延迟回归通过，训练与部署 mapper 对照通过，无干净地图泄漏 |
| P4 Affordance | 深度地图标签采集与 Affordance-DepthMap | 旧地图落脚对齐可信；有足够有效样本；相同感知条件下与 AME 对照 |
| P5 sim2sim 与回放 | MuJoCo 相机→生产建图→同一 ONNX；实机 bag shadow inference | 不再依赖 mj_ray 完整地面扫描；观测和动作数值对齐，长时间延迟测试通过 |
| P6 实机分级 | 平地低速、停走/转向、缓坡、宽低台阶，再扩大地形 | 各级重复验证成功率、饱和、接触、丢帧/过期处理，失败可归因 |

建议 P1 初始几何目标：可见静态区域误差在约 1–2 cm、台阶边缘位置误差低于一个 5 cm 底层格；按实测距离/深度质量调整。单独统计稳定平面误差和不连续边缘误差，避免用全图平均掩盖边缘问题。目标是验收建议，不是 D435i 精度承诺。

正式比较最少包含：干净高程 AME、DepthMap AME、DepthMap Affordance。报告地形成功率、速度误差、滑动/跌倒、有效落脚样本比例、地图覆盖/年龄、p50/p95 端到端延迟；只比较总 reward 不足以定位问题。

## 9. 建议文件布局（尚未创建）

```text
isal2/perception/{map_spec,depth_projection,local_mapping,noise}.py
isal2/tasks/depth_map/{depth_map_env,depth_map_env_cfg}.py
isal2/tasks/depth_map/agents/ppo_cfg.py
isal2/tests/test_depth_mapping.py
isal2/tests/test_map_collection.py
isal2/tests/test_map_export.py
部署工作区/src/isal2_elevation/  # ROS 2 点云、建图、采样和带时间戳观测
```

环境建议共享感知模块，以 AME/Affordance 配置切换监督分支，避免复制整套 BaseEnv。

## 10. 调研来源

官方部署读取的主仓库提交：`a8a0f1557cc5d085234b8bac248a8f543342f531`。camera 子模块：`7434a8dbd36e97466a96e85fa8d088f528a2e8dc`；inference 子模块：`ac448b12c382e3fab42c0e8467a6943987223c4b`。以下链接用于核对已发现的实现，非安装完成证明。

- [RoboParty 部署](https://github.com/Roboparty/roboparty_deploy/tree/a8a0f1557cc5d085234b8bac248a8f543342f531)
- [D435i 配置](https://github.com/Roboparty/roboparty_camera/blob/7434a8dbd36e97466a96e85fa8d088f528a2e8dc/configs/realsense_d435i.yaml)
- [深度处理配置](https://github.com/Roboparty/roboparty_camera/blob/7434a8dbd36e97466a96e85fa8d088f528a2e8dc/configs/parkour.yaml)
- [深度处理与缓存](https://github.com/Roboparty/roboparty_camera/blob/7434a8dbd36e97466a96e85fa8d088f528a2e8dc/src/depth_provider.cpp)
- [官方 parkour 策略配置](https://github.com/Roboparty/roboparty_inference/blob/ac448b12c382e3fab42c0e8467a6943987223c4b/robots/rpo/configs/parkour.yaml)
- [官方单输入 ONNX 运行时](https://github.com/Roboparty/roboparty_inference/blob/ac448b12c382e3fab42c0e8467a6943987223c4b/src/inference_node.cpp)
- [官方观测、IMU 发布](https://github.com/Roboparty/roboparty_inference/blob/ac448b12c382e3fab42c0e8467a6943987223c4b/src/ros_interface.cpp)
- [Elevation Mapping CuPy ROS 2](https://github.com/leggedrobotics/elevation_mapping_cupy/tree/ros2)
- [Humble v2.0.0 发布说明](https://github.com/leggedrobotics/elevation_mapping_cupy/releases/tag/v2.0.0)
- [RTAB-Map D435i 官方示例](https://github.com/introlab/rtabmap_ros/blob/ros2/rtabmap_examples/launch/realsense_d435i_color.launch.py)

本计划只新增本文档；未修改训练/部署源码、安装 Orin 依赖、运行相机或发送机器人控制指令。
