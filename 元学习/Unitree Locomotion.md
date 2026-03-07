[unitreerobotics/unitree_rl_lab: This is a repository for reinforcement learning implementation for Unitree robots, based on IsaacLab.](https://github.com/unitreerobotics/unitree_rl_lab?tab=readme-ov-file) Unitree RL Lab 是一个用于 Unitree 机器人强化学习（locomotion policy）训练、验证和部署的完整工作流框架，不是新算法，而是RL infra。
**面向 Unitree Go2 的鲁棒速度跟踪强化学习：基于 Unitree RL Lab 的 Go2 locomotion，研究“面向 MuJoCo 目标域的 actuator 参数辨识”是否能比常规 domain randomization 更有效地缩小 Isaac→MuJoCo 的 transfer gap。**

## 0.前置知识

> 机器人强化学习基本流程是：
> 训练：仿真环境（Isaac / MuJoCo / Bullet）
>           ↓
> 验证：另一个仿真器（Sim2Sim）
>           ↓
> 部署：真实机器人（Sim2Real）
> 
> 机器人公司一般的流程：
> 仿真训练
> ↓
> 策略导出
> ↓
> 控制器集成
> ↓
> 机器人通信
> ↓
> 运行在真实系统
> 
> 机器人控制的本质：
> 传感 → 计算 → 执行，也就是：
> 观测
    ↓
   决策
    ↓
   执行 的循环，通常是**200Hz~1000Hz**。
> 
> 例如，每5ms，系统要完成：
> 读取传感器
>  ↓
> 运行控制算法
>  ↓
> 发送电机命令

sim2sim、sim2real gap的原因：ETH 的 `legged_gym` README 直接把 **actuator network** 列为 sim-to-real transfer 的核心组件之一，和 friction/mass randomization、噪声观测、随机推搡并列。

