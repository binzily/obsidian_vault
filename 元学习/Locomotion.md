[unitreerobotics/unitree_rl_lab: This is a repository for reinforcement learning implementation for Unitree robots, based on IsaacLab.](https://github.com/unitreerobotics/unitree_rl_lab?tab=readme-ov-file)
**面向 Unitree Go2 的鲁棒速度跟踪强化学习：在 Isaac Lab 中训练策略，并通过 MuJoCo Sim2Sim 评估跨物理引擎泛化；针对 actuator/延迟/地形课程带来的 sim2sim gap 进行建模与改进。**

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

sim2sim、sim2real gap的原因有这么几类：
- actuator ：真实有延迟/摩擦。
- 控制延迟：真实机器人感知 → 控制 → 执行，存在10ms ~ 30ms latency，而仿真几乎没有延迟。
- 噪声。
- 动力学参数误差：质量、摩擦、惯量。
