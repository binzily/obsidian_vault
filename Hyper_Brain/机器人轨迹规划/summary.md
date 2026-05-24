## 1. MπNets和RobotDiffuse

​	MπNets 是端到端的模仿学习，用 PointNet++ 处理点云，结合关节状态，输出关节位移，损失函数包括行为克隆和碰撞损失。而 RobotDiffuse 是基于扩散模型的运动规划，用 Transformer 捕捉时序依赖，结合物理约束，生成轨迹。

首先，模型架构方面，MπNets 是 encoder-decoder 结构，PointNet++ 处理点云，MLP 处理关节状态，实时单步输出。RobotDiffuse 是扩散模型，前向加噪和反向去噪，用 Transformer 捕捉长时序依赖，生成完整轨迹。

然后是数据和训练。MπNets 用专家轨迹做行为克隆，注重单步模仿和碰撞避免；RobotDiffuse 用扩散模型学习轨迹的噪声分布，训练时加入物理约束，如碰撞和关节限制。

核心机制方面，MπNets 是反应式策略，实时根据当前状态调整；RobotDiffuse 是生成式，一次性生成完整轨迹，适合复杂高维空间。

还要考虑应用场景，MπNets 适合实时动态环境，RobotDiffuse 适合高自由度机械臂的复杂避障。



## 2. 