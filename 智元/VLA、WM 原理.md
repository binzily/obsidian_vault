# 1.现在的世界模型是怎么做 WAM 的？

## 核心关系可以写成：

```text
当前观测 + 语言任务
        ↓
预测未来世界状态
        ↓
从状态变化推断动作
        ↓
输出机器人控制指令
```

## 基本结构

典型 WAM 包含三部分：
`观测编码器 → 世界动态模型 → 动作解码器`
输入通常是：

```text
context = {
    "images": image_t,
    "robot_state": state_t,
    "instruction": language,
}
```

模型内部预测：
`future_latent = world_model(context, action_or_latent_action)`
最后输出：

```text
action = action_decoder(
    current_latent,
    future_latent,
    robot_state
)
```

## 两种主流做法

### 1. Action-conditioned World Model

训练时动作是已知条件：
`(z_t, a_t) → z_(t+1)`
其中：
- z_t 是当前图像或状态的 latent
- a_t 是机器人动作
- z_(t+1) 是预测的未来状态
推理时不能直接给真实动作，因此需要搜索候选动作：

```text
candidate_actions = sample_actions()
future_states = world_model(current_state, candidate_actions)
score = goal_model(future_states, instruction)
action = candidate_actions[argmax(score)]
```

这类似模型预测控制 MPC：

```text
采样动作序列
→ 模拟每条序列的未来
→ 根据任务目标打分
→ 执行第一步
→ 获取新观测后重新规划
```

这种方法可解释性较强，但推理成本高。

### 2. Latent Action World Model

这是当前很重要的方向。模型先从视频变化中学习“潜在动作”，不一定需要真实机器人动作标签。
训练数据：
`图像_t → 图像_(t+1)`
通过逆动力学编码器得到 latent action：
`latent_action = inverse_model(image_t, image_t1)`
再训练世界模型：
`predicted_image_t1 = world_model(image_t, latent_action)`
完整过程：

```text
image_t ───────────────┐
                          ├→ inverse model → latent action
image_(t+1) ───────────┘

image_t + latent action
          ↓
     world model
          ↓
预测 image_(t+1)
```

用数学表示：
`z_t = inverse_model(o_t, o_t1)`
`o_t1_pred = world_model(o_t, z_t)`
训练目标只是让预测结果接近真实下一帧：
`loss = distance(o_t1_pred, o_t1)`

这样可以先用大量没有机器人动作标签的视频学习：
- 物体怎样移动
- 手与物体怎样交互
- 什么变化是可控的
- 一段任务通常包含哪些动作阶段
之后再用少量有真实动作的数据，把 latent action 映射成机器人控制量：
`latent action → joint action / end-effector action`

## WAM 如何输出真实动作

世界模型产生的未来变化不能直接控制机器人，还需要动作落地模块。常见有三种方式。

### 逆动力学模型

```text
action_t = inverse_dynamics(
    observation_t,
    predicted_observation_t1,
)
```

例如看到“当前夹爪在杯子左侧”和“目标下一帧夹爪更靠近杯子”，逆动力学模型输出：
`[dx, dy, dz, dRx, dRy, dRz, gripper]`

### Action Head

在世界模型的 latent 上直接增加动作预测头：

```text
future_latent, action = model(
    image,
    robot_state,
    instruction,
)
```

训练损失一般是：

```text
loss = (
    world_prediction_loss
    + λ1 * action_loss
    + λ2 * temporal_consistency_loss
)
```

动作可以通过回归、离散 token 或 diffusion 生成。

### Diffusion Action Decoder

世界模型提供场景理解和未来目标，Diffusion Policy 生成动作块：

```text
action_chunk = diffusion_decoder(
    current_latent,
    predicted_future_latent,
    robot_state,
)
```

输出不是单步，而是：
`[a_t, a_(t+1), ..., a_(t+H)]`
执行前几步后再根据新观测重新规划。

## 训练数据

WAM 往往混合两类数据：

```text
互联网/人类操作视频
    → 学习通用世界变化和潜在动作

机器人轨迹
    → 学习 latent action 到真实控制指令的映射
机器人轨迹需要：
image_t
state_t
action_t
image_(t+1)
language_instruction
```

世界模型损失可能包括：

```text
L_world   = distance(predicted_future, real_future)
L_action  = distance(predicted_action, real_action)
L_inverse = distance(inverse(z_t, z_t1), action_t)
L_goal    = task_consistency(predicted_future, instruction)
```

如果生成像素，常用视频 diffusion 或 autoregressive video token；如果只建模 latent，则使用 Transformer、JEPA 类目标或 latent diffusion。

## 与 VLA 的区别

```text
普通 VLA 更接近：
图像 + 语言 + 状态 → 动作
WAM 更接近：
图像 + 语言 + 状态
        ↓
“如果这样行动，世界会怎样变化？”
        ↓
选择能实现目标的未来
        ↓
动作
所以 WAM 显式或隐式包含因果转移：
当前世界 + 动作 → 未来世界
```

它的潜在优势是能够规划、检查动作后果，并利用无动作标签的视频；主要困难则是视频预测误差、动作语义对齐、实时推理成本，以及从视觉变化映射到精确控制。

# 2.术语

| 术语 | 属于什么 | 一句话解释 |
|---|---|---|
| LIBERO | 仿真基准 | 用一组机械臂操作任务测试模型是否能持续学习、迁移和避免遗忘 |
| LeRobot | 开源工具链 | Hugging Face 推出的机器人数据采集、训练、评估和部署框架 |
| Open X-Embodiment / OXE | 大型数据集合 | 汇集许多实验室、机器人和任务的真实机器人演示数据 |
| RLDS | 数据格式 | 用统一的 episode -> step 结构保存机器人或强化学习轨迹 |
| ACT | 控制策略算法 | Transformer 一次预测未来一段动作，而不是每次只预测一个动作 |

## LIBERO

全称来自 Lifelong Robot Learning，主要研究“终身机器人学习”。它建立在 MuJoCo、robosuite 等仿真工具上，包含桌面机械臂完成抓取、放置、开关物体等任务。
常见子集包括：
- LIBERO-Spatial：考察空间关系，例如把杯子放到盘子左边。
- LIBERO-Object：考察不同物体。
- LIBERO-Goal：初始场景相似，但目标不同。
- LIBERO-100：规模更大的 100 个任务集合。
它主要是考试场，不是机器人模型。

## LeRobot

Hugging Face 的机器人学习开源生态，提供：
- 真实机器人数据采集与回放。
- LeRobotDataset 数据格式。
- ACT、Diffusion Policy、VLA 等策略的训练实现。
- 摄像头、机械臂、遥操作设备的接口。
- 数据集和模型在 Hugging Face Hub 上的发布能力。
它更像机器人学习领域的“训练工具箱和生态”，既不是单独的数据集，也不是单独的算法。

## Open X-Embodiment，简称 OXE

Google DeepMind 联合许多机构整理的跨机器人数据集合。“X-Embodiment”表示跨多种机器人形态，包括不同机械臂、夹爪、摄像头位置和控制方式。
它试图解决的问题是：能否像大语言模型使用互联网文本一样，让机器人模型从许多来源的数据中获得通用能力。
注意：OXE 是许多数据集的集合，不是一台机器人，也不是一种算法。

## RLDS

全称 Reinforcement Learning Datasets。它规定如何组织序列数据：

```text
Dataset
└── Episode，一次完整尝试
    ├── Step 1：图像、关节状态、动作、奖励
    ├── Step 2：图像、关节状态、动作、奖励
    └── Step N：终止状态
```

RLDS 解决的是“数据怎么存、怎么读”的问题。OXE 的很多数据使用 RLDS 表示，但 RLDS 本身不包含特定任务数据。

## ACT

全称 Action Chunking with Transformers，即“基于 Transformer 的动作分块”。

```text
普通策略可能每次只预测下一个动作：
当前观测 → 下一个动作
ACT 一次预测未来一段动作：
当前观测 → 接下来 50 个关节动作
```

这样可以减少动作抖动，更容易学习擦桌子、穿线、折叠等连续操作。ACT 最初因 ALOHA 双臂机器人系统而流行，通常使用模仿学习训练。

## 常见数据集

| 术语 | 解释 |
|---|---|
| DROID | 大规模真实机械臂操作数据集，覆盖许多场景、任务和采集地点 |
| BridgeData V2 | 以 WidowX 等机械臂采集的真实操作数据，常用于通用策略训练 |
| RoboNet | 多机器人、多实验室的视频与动作数据集 |
| RH20T | 包含多种真实机器人、传感器和操作任务的数据集 |
| ALOHA Dataset | 使用低成本双臂 ALOHA 系统采集的遥操作演示 |
| RT-X Data | 从 OXE 数据中整理、转换出来供 RT-X 系列模型训练的数据 |
| Ego4D / EPIC-KITCHENS | 人类第一视角视频，不是标准机器人动作数据，但可用于学习人类操作知识 |

## 常见模型和算法

| 术语 | 解释 |
|---|---|
| BC | Behavior Cloning，行为克隆；直接学习“看到这个观测时，人类做了什么动作” |
| Imitation Learning / IL | 模仿学习；BC、ACT、Diffusion Policy 都可以属于这一大类 |
| Diffusion Policy | 用扩散模型生成一段连续动作，通常对多峰、精细操作表现较好 |
| RT-1 | Robotics Transformer 1；用 Transformer 根据图像和语言预测机器人动作 |
| RT-2 | 将视觉语言模型的知识迁移到机器人动作生成 |
| RT-X | 使用多机器人 OXE 数据训练的跨机器人模型系列 |
| Octo | 开源通用机器人策略，可在不同任务和机器人数据上预训练、微调 |
| OpenVLA | 开源视觉-语言-动作模型，根据图像和文字指令输出动作 |
| π0 / pi0 | Physical Intelligence 提出的通用机器人基础模型，使用流匹配生成动作 |
| GR00T | NVIDIA 的通用人形机器人基础模型与相关训练生态 |

## 常见仿真和评测环境

| 术语 | 解释 |
|---|---|
| CALVIN | 测试模型能否根据语言连续完成多个桌面操作任务 |
| RLBench | 基于 CoppeliaSim 的大量机械臂任务基准 |
| Meta-World | 常用于强化学习和多任务学习的机械臂任务集合 |
| robomimic | 模仿学习数据集、算法库和评测工具 |
| ManiSkill | 面向机器人操作的高性能仿真与基准平台 |
| BEHAVIOR | 更偏向家庭环境中的长流程任务，如清理、整理和烹饪 |
| MuJoCo | 机器人动力学仿真引擎 |
| Isaac Sim / Isaac Lab | NVIDIA 的 GPU 机器人仿真与训练平台 |
| robosuite | 基于 MuJoCo 的机器人操作环境，LIBERO 建立在其生态上 |
| SimPLer / SIMPLER | 在仿真中评估真实世界机器人策略的方法和环境 |

## 硬件与软件术语

| 术语 | 解释 |
|---|---|
| ALOHA | 低成本双臂遥操作硬件系统，也是 ACT 常用的数据采集平台 |
| Mobile ALOHA | 给 ALOHA 增加移动底盘，可执行整理、烹饪等移动操作 |
| ROS / ROS 2 | 机器人软件通信框架，让摄像头、机械臂、控制器等模块互相通信 |
| UR5 / Franka / WidowX | 常见机械臂型号或系列 |
| Teleoperation | 遥操作；人通过主臂、手柄或 VR 设备控制机器人采集演示 |

## 论文中高频出现的基础词

| 术语 | 解释 |
|---|---|
| DAgger | 模型运行时让专家纠错，再把纠错数据加入训练 |

最实用的辨别方法是每看到一个新词，先问它属于哪一层：硬件、仿真器、数据格式、数据集、算法、模型还是评测基准。这一步分清后，机器人学习论文会容易读很多。
