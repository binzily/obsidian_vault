[unitreerobotics/unitree_rl_lab: This is a repository for reinforcement learning implementation for Unitree robots, based on IsaacLab.](https://github.com/unitreerobotics/unitree_rl_lab?tab=readme-ov-file)
**面向 Unitree Go2 的鲁棒速度跟踪强化学习：在 Isaac Lab 中训练策略，并通过 MuJoCo Sim2Sim 评估跨物理引擎泛化；针对 actuator/延迟/地形课程带来的 sim2sim gap 进行建模与改进。**

机器人强化学习基本流程是：
训练：仿真环境（Isaac / MuJoCo / Bullet）
          ↓
验证：另一个仿真器（Sim2Sim）
          ↓
部署：真实机器人（Sim2Real）

