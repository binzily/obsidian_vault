[【强化学习的数学原理】课程：从零开始到透彻理解（完结）_哔哩哔哩_bilibili](https://www.bilibili.com/video/BV1sd4y167NS/?spm_id_from=333.788.recommend_more_video.1&vd_source=40871d74fa8db37f3b1352390d563aa5)

[零基础学习强化学习算法：ppo_哔哩哔哩_bilibili](https://www.bilibili.com/video/BV1iz421h7gb/?spm_id_from=333.788.recommend_more_video.0&vd_source=40871d74fa8db37f3b1352390d563aa5)

## 0.基本概念

==model free==：迷宫例子中的 Model-Free 学习

假设有以下环境动态：

- **转移概率**：每次移动到相邻的格子的概率为 1（确定性环境）
- **奖励函数**：只有到达出口 *S*9 时获得 +100 的奖励，其他移动没有奖励

在 model-free 的强化学习中：

1. **智能体不知道转移概率**。它不知道从 *S*1 向右移动一定会到达 *S*2，只能通过尝试来学习。
2. **智能体不知道奖励函数**。它不知道到达 *S*9 会获得 +100 的奖励，只能通过实际到达 *S*9 后获得奖励来学习。

与 Model-Based 的对比

- **Model-Based**：智能体知道（或学习）环境的转移概率和奖励函数。例如，在迷宫中，智能体可能通过观察或训练来构建一个模型，知道从 *S*1 向右移动会到达 *S*2，以及到达 *S*9 会获得奖励。
- **Model-Free**：智能体不依赖于环境的动态模型。它通过与环境的交互（试错）来学习状态价值或状态-动作价值。

Model-Free 的优点

- **无需环境模型**：在许多实际问题中，环境的动态模型可能难以获取或过于复杂。例如，在机器人控制中，精确建模环境的物理特性可能是不现实的。
- **适应性强**：model-free 方法能够适应环境的变化。如果环境的动态发生变化（例如迷宫的某些路径被堵住），智能体可以通过继续与环境交互来更新其策略。

Model-Free 的缺点

- **学习效率较低**：由于不利用环境模型，智能体需要通过大量的试错来学习，这可能在某些情况下效率较低。
- **依赖于环境交互**：model-free 方法需要智能体不断地与环境交互，这对于某些实际应用（如机器人在危险环境中操作）可能是不可行的。

总结

在迷宫例子中，model-free 方法意味着智能体不知道以下内容：

- **转移概率**：从一个状态采取某个动作后转移到下一个状态的概率。
- **奖励函数**：在某个状态下采取某个动作后获得的即时奖励。

==Action Space==: 可选择的动作，比如 {left, up, right}

==Policy==: 策略函数，输入 State，输出 Action 的概率分布。一般用 $\pi$ 表示。

$$\begin{align*}\pi(\text{left} | s_t) &= 0.1 \\\pi(\text{up} | s_t) &= 0.2 \\ \pi(\text{right} | s_t) &= 0.7 \\ \end{align*}$$

==Trajectory==: 轨迹，用$ \tau$表示，一连串状态和动作的序列。==Episode==（停掉的trajectory）, trial。$\{s_0, a_0, s_1, a_1, \dots\}$

$s_{t+1} = f(s_t, a_t)$（确定）

$s_{t+1} = P(\cdot | s_t, a_t)$（随机）

==Return==: 回报，从当前时间点到游戏结束的 Reward 的累积和。

==目标==：训练一个$Policy$ 神经网络$\pi$ ，在所有状态$S$下，给出相应的Action，得到$Return$的期望最大。

==目标==：训练一个$Policy$ 神经网络$\pi$ ，在所有的$Trajectory$ 中，得到$Return$的期望最大。



用公式表达这个期望：$E(R(\tau))_{\tau\sim P_\theta(\tau)}=\sum_{\tau}R(\tau)P_\theta(\tau)$



目标是最大化这个期望，用梯度上升法。直观理解就是，我们要知道不同轨迹的概率及其回报，以及参数变化对轨迹概率的影响，从而找到提升期望回报的参数更新方向。求梯度：

$$\begin{aligned}
\nabla E(R(\tau))_{\tau\sim P_{\theta}(\tau)} & =\nabla\sum_{\tau}R(\tau)P_{\theta}(\tau) \\
 & =\sum_\tau R(\tau)\nabla P_\theta(\tau) \\
 & =\sum_{\tau}R(\tau)\nabla P_{\theta}(\tau)\frac{P_{\theta}(\tau)}{P_{\theta}(\tau)} \\
 & =\sum_\tau P_\theta(\tau)R(\tau)\frac{\nabla P_\theta(\tau)}{P_\theta(\tau)} \\
 & =\sum_{\tau}P_\theta(\tau)R(\tau)\frac{\nabla P_\theta(\tau)}{P_\theta(\tau)} \\
 & \approx\frac{1}{N}\sum_{n=1}^NR(\tau^n)\frac{\nabla P_\theta(\tau^n)}{P_\theta(\tau^n)} \\
 & =\frac{1}{N}\sum_{n=1}^{N}R(\tau^n)\nabla\mathrm{log}P_{\theta}(\tau^n)
\end{aligned}$$

​												$$\nabla\log f(x)=\frac{\nabla f(x)}{f(x)}
\\\tau{\sim}P_\theta(\tau)$$

​				$$\begin{aligned}
 & =\frac{1}{N}\sum_{n=1}^NR(\tau^n)\nabla\log P_\theta(\tau^n) \\
 & =\frac{1}{N}\sum_{n=1}^{N}R(\tau^n)\nabla\mathrm{log}\prod_{t=1}^{T_n}P_\theta(a_n^t|s_n^t) \\
 & =\frac{1}{N}\sum_{n=1}^NR(\tau^n)\sum_{t=1}^{T_n}\nabla\mathrm{log}P_\theta(a_n^t|s_n^t) \\
 & =\frac{1}{N}\sum_{n=1}^N\sum_{t=1}^{T_n}R(\tau^n)\nabla\mathrm{log}P_\theta(a_n^t|s_n^t) \\
 & 
\begin{aligned}
\frac{1}{N}\sum_{n=1}^{N}\sum_{t=1}^{T_{n}}R(\tau^{n})\mathrm{log}P_{\theta}(a_{n}^{t}|s_{n}^{t})
\end{aligned}
\end{aligned}$$

由两部分构成：**第一部分是一个trajetory得到的Return（有N个这样的trajectory），第二部分是每一步t根据当前的state做出action的概率。**直观意义是，如果一个trajetory得到的Return大于零，那么就增大**这个trajectory里面所有状态下采取当前action**的概率。这就叫Policy Gradient 策略梯度算法。

==损失函数==：那么我们怎么训练这样一个$P_\theta$ 策略神经网络呢？定义的loss函数如下：

$$\mathrm{Loss}=-\frac{1}{N}\sum_{n=1}^{N}\sum_{t=1}^{T_n}R(\tau^n)\log P_\theta(a_n^t|s_n^t)$$

![image-20250513110526442](G:\software\Typora\Typora_files\Hyper_Brain\强化学习\RL.assets\image-20250513110526442.png)

==on policy==：采集和 训练用的policy都是同一个。导致大部分时间在采集数据，训练非常慢，这就是**PPO**算法要解决的问题。

思考目前的Loss（Policy Gradient）存在的问题：

- 是否增大或减少在状态$s$下做动作$a$的概率，应该看做了这个动作之后到游戏介绍累计的$reward$，而不是整个$trajectory$累计的$reward$，因为一个动作只能影响后面的$reward$，不能影响它之前的。

- 上一点的当前的$reward$应该对后面的影响越来越弱。

改进如下，其中$R(\tau^n)\to\sum_{t^{\prime}=t}^{T_n}\gamma^{t^{\prime}-t}r_t^n=R_t^n$，**$R_t^n$代表在第n次采样，从第t个时间步往后的累计奖励：**

$$\begin{gathered}
=\frac{1}{N}\sum_{n=1}^N\sum_{t=1}^{T_n}R(\tau^n)\nabla\mathrm{log}P_\theta(a_n^t|s_n^t) \\
=\frac{1}{N}\sum_{n=1}^N\sum_{t=1}^{T_n}R_t^n\nabla\mathrm{log}P_\theta(a_n^t|s_n^t)
\end{gathered}$$

- 在好的情况或者坏的情况时，所有的reward都会增加（减少），好的动作增加的多，坏的动作减少的多，这样会让训练很慢。最好是相对好的增加，相对差的减少，这样训练会加快。办法是给所有动作的reward都减去一个baseline，这个baseline也需要神经网络估算，用的是**actor-critic**算法。

$$=\frac{1}{N}\sum_{n=1}^N\sum_{t=1}^{T_n}\left(R_t^n-B(s_n^t)\right)\nabla\mathrm{log}P_\theta(a_n^t|s_n^t)$$

==Action-Value Function==:

$R_t^n$ 每次都是一次随机采样，如果只采样一个trajectory，方差很大，训练不稳定，采样次数N要无限多次才能反应当前Action的好坏。那么能不能有个函数能帮我们**估计一下这个Action能够得到的Return的期望**呢？

$Q_\theta(s,a)$ 在state s下，做出Action a，期望的回报。**动作价值函数**。

==State-Value Function==：

$\color{}{V_\theta(s)}$ 在state s下，后面所有tractory期望的回报。**状态价值函数**。**$v_\pi(s)=\mathbb{E}[G_t|S_t=s]$**，其中discount return为：$\begin{aligned}
\mathrm{Gt} & 
\begin{aligned}
=R_{t+1}+\gamma R_{t+2}+\gamma^2R_{t+3}+\ldots,
\end{aligned} \\
 & =R_{t+1}+\gamma(R_{t+2}+\gamma R_{t+3}+\ldots), \\
 & =R_{t+1}+\gamma G_{t+1},
\end{aligned}$ 其实就是Return的期望。

==State-Value 和Return 的区别==：

前者是从一个状态出发所有的trajectory的Return的期望，后者是一条trajectory的Return。

==贝尔曼公式==：

**计算state value 可以评估一个策略$\pi$ 的好坏。**由于state value的重要性，那么引入贝尔曼公式来计算state value，**贝尔曼公式就是在一个给定策略$\pi$下，描述不同state value关系的公式，最终目的是得到类似套娃的方程组解出state value。**

- 贝尔曼公式推导：

$$\begin{aligned}
v_{\pi}(s) & =\mathbb{E}{|G_t|}S_t=s{]}{} \\
 & =\mathbb{E}[R_{t+1}{}{+}\gamma G_{t+1}|S_t=s] \\
 & 
\begin{aligned}
=\mathbb{E}[R_{t+1}|S_t=s]+\gamma\mathbb{E}[G_{t+1}|S_t=s]
\end{aligned}
\end{aligned}$$

$$\begin{aligned}
\mathbb{E}[R_{t+1}|S_{t}=s] & 
\begin{aligned}
=\sum_a\pi(a|s)\mathbb{E}[R_{t+1}|S_t=s,A_t=a]
\end{aligned} \\
 & =\sum_{s}\pi(a|s)\sum p(r|s,a)r
\end{aligned}$$

$$\begin{aligned}
\mathbb{E}[G_{t+1}|S_{t}=s] & 
\begin{aligned}
=\sum_{s^{\prime}}\mathbb{E}[G_{t+1}|S_{t}=s,S_{t+1}=s^{\prime}]p(s^{\prime}|s)
\end{aligned} \\
 & =\sum_{s^{\prime}}\mathbb{E}[G_{t+1}|S_{t+1}=s^{\prime}]p(s^{\prime}|s) \\
 & =\sum_{s^{\prime}}v_{\pi}(s^{\prime})p(s^{\prime}|s) \\
 & 
\begin{aligned}
=\sum_{s^{\prime}}v_{\pi}(s^{\prime})\sum_{a}p(s^{\prime}|s,a)\pi(a|s)
\end{aligned}
\end{aligned}$$

- 简单应用：

  ![image-20250514155541789](G:\software\Typora\Typora_files\Hyper_Brain\强化学习\RL.assets\image-20250514155541789.png)

可以直接手写出$V\pi(S_{1})= \begin{cases} 0.5(-1+\gamma V_{\pi}(S_{2}) \\ 0.5(0+\gamma V_{\pi}(S_{3}) & \end{cases}$

- 简化贝尔曼公式为矩阵形式：[第2课-贝尔曼公式（公式向量形式与求解）_哔哩哔哩_bilibili](https://www.bilibili.com/video/BV1sd4y167NS?spm_id_from=333.788.videopod.episodes&vd_source=40871d74fa8db37f3b1352390d563aa5&p=7)

$v_\pi=r_\pi+\gamma P_\pi v_\pi$ ，**$r_\pi$为immediate reward exception**。 

- 求解贝尔曼公式：
  - closed-form solutuon：$\begin{aligned} v_\pi=(I-\gamma P_\pi)^{-1}r_\pi \end{aligned}$ 求逆计算量大。
  - iterative soluton:$v_{k+1}=r_{\pi}+\gamma P_{\pi}v_{k}$



==Advantage Function==：

$A_\theta(s,a)=Q_\theta(s,\alpha)-\color{r}{V_\theta(s)}$ 在state s下，做出Action a，比其他动作能带来多少优势。**优势函数**。



$\frac{1}{N}\sum_{n=1}^N\sum_{t=1}^{T_n}A_\theta(s_n^t,a_n^t)\nabla\mathrm{log}P_\theta(a_n^t|s_n^t)$ 

由于$Q_\theta(s_t,a)=r_t+\gamma*V_\theta(s_{t+1})$ ,带入$A_\theta(s,a)=Q_\theta(s,\alpha)-\color{r}{V_\theta(s)}$ :



$A_{\theta}(s_{t},a)=r_{t}+\gamma*V_{\theta}(s_{t+1})-V_{\theta}(s_{t})$，这里$r_{t}$为第t步做出Action a的reward。

这样一来，原来需要训练两个神经网络现在只需要**训练一个神经网络**就可以了。



现在用$V_\theta(s_{t+1})\approx r_{t+1}+\gamma*V_\theta(s_{t+2})$对优势函数简化，这里$r_{t+1}$ 为第t+1步做出所有动作的期望：





## 1. 贝尔曼最优公式BOE

怎么比较两个策略？

$v_{\pi_1}(s) \geq v_{\pi_2}(s)$ for all $s$ in $S$ 

那么$\pi_1$ 就是比 $\pi_2$ 好。

那么就可以定义最优策略：A policy $\pi^*$ is optimal if $v_{\pi^*}(s) \geq v_{\pi}(s)$ for all s and for any other policy $\pi$.

==BOE：==

element-wise form：

$$\begin{aligned}
v(s) & 
\begin{aligned}
=\max_{\pi}\sum_{a}\pi(a|s)\left(\sum_{r}p(r|s,a)r+\gamma\sum_{s^{\prime}}p(s^{\prime}|s,a)v(s^{\prime})\right),\quad\forall s\in\mathcal{S}
\end{aligned} \\
 & 
\begin{aligned}
=\max_{\pi}\sum_{a}\pi(a|s)q(s,a)\quad s\in\mathcal{S}
\end{aligned}
\end{aligned}$$

$p(r|s,a),p(s^{\prime}|s,a)$是已知的，和系统模型有关；$v(s),v(s^\prime)$未知，是要求的；$\pi\left(s\right)$策略在贝尔曼公式是已知的，在最优公式中是未知的。

matrix-vector form：

$$v=\max_\pi(r_\pi+\gamma P_\pi v)$$

求解贝尔曼最优公式：

$$\max_{\pi}\sum_a\pi(a|s)q(s,a)=\max_{a\in\mathcal{A}(s)}q(s,a),$$ 其实就是让策略$\pi$ 在action value最大的概率为1，即只有一个最优策略。

$\pi(a|s)= \begin{cases} 1 & a=a^* \\ 0 & a\neq a^* & \end{cases}$且 $\begin{aligned} a^*=\arg\max_aq(s,a) \end{aligned}$



​	可以证明$\begin{aligned} v=f(v)=\max_\pi(r_\pi+\gamma P_\pi v) \end{aligned}$ 的$f(v)$ 是一个contraction mapping，过程略。

那么可以迭代求出$f(v)$ 的不动点$v^*$ ,且可以证明这个不动点是唯一且是最大，并且证明$v^*$对应的$\pi*$ 是greedy且deterministic的。  

$$\begin{aligned}
v^* & =\max_{\pi}(r_\pi+\gamma P_\pi v^*) \\
\pi^* & =\arg\max_{\pi}(r_\pi+\gamma P_\pi v^*) \\
v^* & =r_{\pi^*}+\gamma P_{\pi^*}v^*
\end{aligned}$$

==关于贝尔曼公式和贝尔曼最优公式的思考==：

$v_{\pi}(s)=\sum_{a}\pi(a|s)\left[\sum_{r}p(r|s,a)r+\gamma\sum_{s^{\prime}}p(s^{\prime}|s,a)v_{\pi}(s^{\prime})\right]$



$v(s)=\max_\pi\sum_{a}\pi(a|s)\left(\sum_{r}p(r|s,a)r+\gamma\sum_{s^{\prime}}p(s^{\prime}|s,a)v(s^{\prime})\right)$

$\begin{aligned} v=\max_\pi(r_\pi+\gamma P_\pi v) \end{aligned}$

看两个方程就可以知道，贝尔曼公式是在策略和奖励定下来之后，求解state value的工具。但是由于我们要寻找最优策略，最优策略和奖励定下来之后，不就是一个普通的贝尔曼公式吗？所以贝尔曼最优公式解决的是：先求出$v^*$ ，再求出$\pi^*$ ，这个$\pi^*$就是greedy且deterministic。

==贝尔曼最优公式例子==：

给定奖励和gamma，下图中$r_{\mathrm{boundary}}=r_{\mathrm{forbidden}}=-1,r_{\mathrm{target}}=1,\gamma=0.9$。就能利用贝尔曼公式迭代（迭代算法在后面）算出所有的state value。

![image-20250515144046710](G:\software\Typora\Typora_files\Hyper_Brain\强化学习\RL.assets\image-20250515144046710.png)

==贝尔曼公式可以给定策略求state value ，也可以给定state value 求 策略（greedy）。==

## 2.迭代算法

### 2.1 value iteration

怎么求解贝尔曼最优方程$v=f(v)=\max_\pi(r_\pi+\gamma P_\pi v)$？

上节课利用 contraction mapping theorem 可以用一个迭代算法求解，这个算法叫值迭代：$v_{k+1}=f(v_k)=\max_{\pi}(r_\pi+\gamma P_\pi v_k),\quad k=1,2,3\ldots$

- <u>step 1：policy update</u>

$\large{\pi_{k+1}}=\arg\max_{\pi}(r_{\pi}+\gamma P_{\pi}v_{k})$

给定任意一个$v_k$ ,求得$q_k$ ,然后得到一个$\pi_{k+1}$，**这里的$\pi_{k+1}$就是greedy和deterministi**c的 。**注意这里的$v_k$ 不是state value，是一个随机值，迭代后才逼近state value。**

- <u>setp 2：value update</u>

$v_{k+1}=r_{\pi_{k+1}}+\gamma P_{\pi_{k+1}}v_k$

**这里的 $v_{k+1}$ 就是 $q_k$ 里面最大的那个数**。

- <u>summary</u>

$$v_k(s)\to q_k(s,a)\to\text{greedy policy }\pi_{k+1}(a|s)\to\text{new value }v_{k+1}=\max_aq_k(s,a)$$

伪代码：![image-20250515154359429](G:\software\Typora\Typora_files\Hyper_Brain\强化学习\RL.assets\image-20250515154359429.png)

- <u>考虑下面这个例子：</u>

$\begin{aligned} & \text{The reward setting is }r_{\mathrm{boundary}}=r_{\mathrm{forbidden}}=-1,r_{\mathrm{target}}=1.\mathrm{The}\ \text{discount rate is }\gamma=0.9. \end{aligned}$

![image-20250515160605652](G:\software\Typora\Typora_files\Hyper_Brain\强化学习\RL.assets\image-20250515160605652.png)

q-table，给定 $v$ 可以计算 $q$：

![image-20250515160638791](G:\software\Typora\Typora_files\Hyper_Brain\强化学习\RL.assets\image-20250515160638791.png)

![image-20250515161503233](G:\software\Typora\Typora_files\Hyper_Brain\强化学习\RL.assets\image-20250515161503233.png)

依次迭代计算。**由 $\pi$ 得到 $v$ 是需要求解贝尔曼方程迭代无限次的，但是这里没有迭代，所以这里的 $v$ 并不是实际的state value。**



### 2.2 policy iteration

- <u>step 1：policy evaluation （PE）</u>

给定初始policy计算state value , 这里也是通过一个迭代算法（因为求矩阵的逆不好求）。

$v_{\pi_k}=r_{\pi_k}+\gamma P_{\pi_k}v_{\pi_k}$ 

- <u>step 2: policy improvement (PI)</u>

由state value 更新policy （greedy and deterministic）

$\pi_{k+1}=\arg\max_{\pi}(r_{\pi}+\gamma P_{\pi}(\nu_{\pi_{k}}))$

- <u>summary</u>

$\pi_0\xrightarrow{PE}v_{\pi_0}\xrightarrow{PI}\pi_1\xrightarrow{PE}v_{\pi_1}\xrightarrow{PI}\pi_2\xrightarrow{PE}v_{\pi_2}\xrightarrow{PI}\ldots$

![image-20250515164808920](G:\software\Typora\Typora_files\Hyper_Brain\强化学习\RL.assets\image-20250515164808920.png)

### 2.3 truncated policy iteration 

![image-20250516101335221](G:\software\Typora\Typora_files\Hyper_Brain\强化学习\RL.assets\image-20250516101335221.png)

## 3.Monte Carlo

### 3.1 MC Basic

MC Basic算法和Policy iteration的区别在于：在第k步，后者是通过模型求得 $v_{\pi_k}(s)$ ，前者是通过数据估计得到 $q_{\pi_{k}}(s,a)$ ，（注意这里是直接估计q而不是v，因为v得到q又要依赖模型）这就是model free，用数据来代替模型。

![image-20250516104713329](G:\software\Typora\Typora_files\Hyper_Brain\强化学习\RL.assets\image-20250516104713329.png)

### 3.2 MC Exploring Starts

![image-20250516135831956](G:\software\Typora\Typora_files\Hyper_Brain\强化学习\RL.assets\image-20250516135831956.png)

### 3.3 MC $\epsilon$ - Greedy

soft policy和deteministc policy是对立的，为什么需要soft policy呢？实际上，由于soft policy是不确定的，所以只要episode足够长，从某一些 $(s,a)$ 出发可以保证 visit 到所有的 $state-action$ ，不需要exploring starts这个条件了。

![image-20250516145900008](G:\software\Typora\Typora_files\Hyper_Brain\强化学习\RL.assets\image-20250516145900008.png)



## 3. 随机近似与随机梯度下降

### 3.1 iterative mean expectation

平均数迭代求法 $w_{k+1}=w_k-\frac{1}{k}(w_k-x_k)$ 

当 $\alpha_k$ 满足某些条件时，这个 $\frac{1}{k}$ 可以用一个$\alpha_k$ 代替。

### 3.2 Robbins Monro Algorithm

用来解方程零点的算法。

$w_{k+1}=w_k-a_k\widetilde{g}(w_k,\eta_k),\quad k=1,2,3,\ldots$

![image-20250516161312804](G:\software\Typora\Typora_files\Hyper_Brain\强化学习\RL.assets\image-20250516161312804.png)

这里迭代解出来的 **$w$ 是方程 $g(w)=0$ 的解**。体现的思想还是：没有模型（这里是函数表达式），那就要有数据。

### 3.3 SGD

==梯度下降（GD）==：$w_{k+1}=w_k-\alpha_k\nabla_w\mathbb{E}[f(w_k,X)]=w_k-\alpha_k\mathbb{E}[\nabla_wf(w_k,X)]$ 

这个公式中，$w_k$ 相当于神经网络的参数的一个估计值；$f$ 是损失函数；$X$ 是神经网络的输入，是一个随机变量，服从某种概率分布。

==随机梯度下降（SGD）==：$w_{k+1}=w_{k}-\alpha_{k}\nabla_{w}f(w_{k},x_{k})$

这个公式中，$x_k$  是 $ X$ 的一个估计值。



## 4.时序差分

### 4.1 TD算法

**TD算法的目的是在给定一个策略 $\pi$ 的情况下 , 求解贝尔曼公式。本质上是求解贝尔曼公式的RM算法。**

​	==首先考虑例子==：$w=\mathbb{E}[R+\gamma v(X)]$ 其中 $R$ 和 $X$ 是随机变量，$\gamma$ 是系数，式子是要求一个期望，用RM算法。现在有一些 {r} 和 {x} 这些估计值，定义 $g(w)=w-\mathbb{E}[R+\gamma v(X)]$ ，所以可以写出 $g(w)$ 的观测值 $\begin{aligned} \tilde{g}(w,\eta) & =w-[r+\gamma v(x)] \\ & =(w-\mathbb{E}[R+\gamma v(X)])+(\mathbb{E}[R+\gamma v(X)]-[r+\gamma v(x)]) \\ & \dot{=}g(w)+\eta \end{aligned}$ ，所以RM公式可以写成 $w_{k+1}=w_k-\alpha_k\tilde{g}(w_k,\eta_k)=w_k-\alpha_k[w_k-(r_k+\gamma v(x_k))]$ 。



​	**TD算法是用来做policy evaluation的，**==算法如下==：

$$\underbrace{v_{t+1}(s_t)}_{\text{new estimate}}=\underbrace{v_t(s_t)}_{\text{current estimate}}-\alpha_t(s_t)[\overbrace{v_t(s_t)-[\underbrace{r_{t+1}+\gamma v_t(s_{t+1})}_{\text{TD target }\bar{v}_t}^{\text{} }]}^{\text{TD error } \delta_t}$$

==$v_t(s_t)$  是 $v_\pi(s_t)$ 的估计。==

新的estimate是由旧的estimate加上一个修正项所得，算法是让estimate像TD target改进。

![image-20250519154355597](G:\software\Typora\Typora_files\Hyper_Brain\强化学习\RL.assets\image-20250519154355597.png)![image-20250519155913426](G:\software\Typora\Typora_files\Hyper_Brain\强化学习\RL.assets\image-20250519155913426.png)

### 4.2 Sarsa

Sarsa的含义：${(s_t,a_t,r_{t+1},s_{t+1},a_{t+1})}$ ，Sarsa实际上和TD learning是一模一样的，只不过是把对state value的估计换成了对action value 的估计。Sarsa实际上也是求解一个贝尔曼公式，不过是action value的贝尔曼公式：$q_{\pi}(s,a)=\mathbb{E}\left[R+\gamma q_{\pi}(S^{\prime},A^{\prime})|s,a\right],\quad\forall s,a$

$\left.\left.q_{t+1}(s_t,a_t)=q_t(s_t,a_t)-\alpha_t(s_t,a_t)\right|q_t(s_t,a_t)-[r_{t+1}+\gamma q_t(s_{t+1},a_{t+1})]\right|$

其中 ==$q_t(s_t,a_t)$ 是  $q_\pi(s_t,a_t)$ 的估计==。



![image-20250520154507470](G:\software\Typora\Typora_files\Hyper_Brain\强化学习\RL.assets\image-20250520154507470.png)

### 4.3 Expected Sarsa

​	Expected Sarsa 使用所有可能动作的期望值来更新状态-动作价值，而不是仅仅使用下一个状态下的一个实际选取的动作，避免了 SARSA 中因随机选择动作导致的高方差问题。

### 4.4 n-step Sarsa

首先action value的定义：$q_{\pi}(s,a)=\mathbb{E}[G_{t}|S_{t}=s,A_{t}=a]$

discount return $G_t$ 有不同的写法：

$$\begin{aligned}
\mathsf{Sarsa}\longleftrightarrow G_t^{(1)} & =R_{t+1}+\gamma q_{\pi}(S_{t+1},A_{t+1}), \\
G_{t}^{(2)} & =R_{t+1}+\gamma R_{t+2}+\gamma^2q_\pi(S_{t+2},A_{t+2}), \\
\mathrm{:} \\
n\text{-step Sarsa}\leftarrow G_{t}^{(n)} & =R_{t+1}+\gamma R_{t+2}+\cdots+\gamma^nq_{\pi}(S_{t+n},A_{t+n}), \\
\mathrm{:} \\
MC\leftarrow G_{t}^{\left(\infty\right)} & =R_{t+1}+\gamma R_{t+2}+\gamma^{2}R_{t+3}+\ldots
\end{aligned}$$

tips：这里所有含有不同上标的 $G_t$ 都是等价的，只不过是分解的方式不同。

- **Sarsa要解决的数学问题**：$$q_{\pi}(s,a)=\mathbb{E}[G_{t}^{(1)}|s,a]=\mathbb{E}[R_{t+1}+\gamma q_{\pi}(S_{t+1},A_{t+1})|s,a]$$
- **MC要解决的数学问题**：$$q_\pi(s,a)=\mathbb{E}[G_t^{(\infty)}|s,a]=\mathbb{E}[R_{t+1}+\gamma R_{t+2}+\gamma^2R_{t+3}+\ldots|s,a]$$
- **n-step Sarsa要解决的数学问题**：$q_\pi(s,a)=\mathbb{E}[G_t^{(n)}|s,a]=\mathbb{E}[R_{t+1}+\gamma R_{t+2}+\cdots+\gamma^nq_\pi(S_{t+n},A_{t+n})|s,a]$

都用RM算法解出他们三者的不同就在于**TD target不同**，所以他们**需要的数据也不同**。

### 4.5 Q-learning

**公式** $q_{t+1}(s_t,a_t)=q_t(s_t,a_t)-\alpha_t(s_t,a_t)\left[q_t(s_t,a_t)-[r_{t+1}+\gamma\max_{a\in\mathcal{A}}q_t(s_{t+1},a)]\right]$

- Q-learning的**TD target**是 $\begin{aligned} r_{t+1}+\gamma\max_{a\in\mathcal{A}}q_t(s_{t+1},a) \end{aligned}$ ，对 $a$ 选择，最大化 $q_t(s_{t+1},a)$ 。

- Sarsa的**TD target** 是 $r_{t+1}+\gamma q_{t}(s_{t+1},a_{t+1})$ 。

- Q-learning解决的数学问题：不是在求解一个贝尔曼方程，不是说你给定我一个策略，他对应的贝尔曼方程是什么，那么我求出对应的action value。他在求解一个贝尔曼最优方程，但是这个和之前的贝尔曼最优方程不太一样，因为他有expectation并且基于action value，**贝尔曼最优方程是和策略无关的**：$$\left.q(s,a)=\mathbb{E}\left[R_{t+1}+\gamma\max_aq(S_{t+1},a)\right|S_t=s,A_t=a\right],\quad\forall s,a$$

Q-Learning 的特点

1. **Off-Policy**：Q-Learning 学习的是最优策略 *π*∗，而不管当前策略是什么。这使得 Q-Learning 能够独立于当前策略进行学习。
2. **Greedy 在更新中**：在更新 *q*(*s*,*a*) 时，使用下一个状态下的最大价值 max*a*∈A*q*(*s**t*+1,*a*)，这使得 Q-Learning 能够逐步逼近最优策略。

![image-20250521142135722](G:\software\Typora\Typora_files\Hyper_Brain\强化学习\RL.assets\image-20250521142135722.png)

![image-20250521143435773](G:\software\Typora\Typora_files\Hyper_Brain\强化学习\RL.assets\image-20250521143435773-1747809276855-1.png)

### 4.6 summary

所有的这节的算法都可以用一个式子表示：$$q_{t+1}(s_t,a_t)=q_t(s_t,a_t)-\alpha_t(s_t,a_t)[q_t(s_t,a_t)-\bar{q_t}]$$

$\bar{q}_{t}$ 就是TD target。

![image-20250521143953909](G:\software\Typora\Typora_files\Hyper_Brain\强化学习\RL.assets\image-20250521143953909.png)

在数学上他们要解决的问题是：

![image-20250521144330731](G:\software\Typora\Typora_files\Hyper_Brain\强化学习\RL.assets\image-20250521144330731.png)



## 5.值函数近似

​	目标是找到一个 $\hat{v}(s,w)$ 去接近真实的 $v_{\pi}(s)$ ，是一个policy evaluation的问题，这里是给定初始策略，后面可以结合policy improvement更新这个初始策略。

目标函数 $J(w)=\mathbb{E}[(v_\pi(S)-\hat{v}(S,w))^2]$ 找到参数 $w$ 去最小化 $J(w)$ 。

- 对每一个状态 $S$ 都要估计他的state value，在把所有的state value求expectation。这里就有问题，状态 $S$ 是符合一个什么样的properbility distribution？这里引入stationary distribution $d_\pi$ 。

- 用随机梯度下降优化 $J(w)$ ：$w_{t+1}=w_{t}+\alpha_{t}(\underbrace{v_{\pi}(s_{t})}-\hat{v}(s_{t},w_{t}))\nabla_{w}\hat{v}(s_{t},w_{t})$
  - MC方法替代 $v_\pi(s_t)$ ：$w_{t+1}=w_t+\alpha_t(g_t-\hat{v}(s_t,w_t))\nabla_w\hat{v}(s_t,w_t)$
  - TD learning替代 $ v_\pi(s_t)$ :$w_{t+1}=w_t+\alpha_t\left[r_{t+1}+\gamma\hat{v}(s_{t+1},w_t)-\hat{v}(s_t,w_t)\right]\nabla_w\hat{v}(s_t,w_t)$
  - Sarsa替代 $ v_\pi(s_t)$ ：$\begin{aligned} w_{t+1}=w_t+\alpha_t\left[r_{t+1}+\gamma\hat{q}(s_{t+1},a_{t+1},w_t)-\hat{q}(s_t,a_t,w_t)\right]\nabla_w\hat{q}(s_t,a_t,w_t) \end{aligned}$
  - Q-learning替代 $v_\pi(s_t)$ ：$\begin{aligned} w_{t+1}=w_t+\alpha_t\left[r_{t+1}+\gamma\max_{a\in\mathcal{A}(s_{t+1})}\hat{q}(s_{t+1},a,w_t)-\hat{q}(s_t,a_t,w_t)\right]\nabla_w\hat{q}(s_t,a_t,w_t) \end{aligned}$
  - 神经网络替代 $v_\pi(s_t)$ ：$$\theta_{t+1}=\theta_t+\alpha_t\left[r_{t+1}+\gamma\max_aQ(s_{t+1},a,\theta_t)-Q(s_t,a_t,\theta_t)\right]\nabla_\theta Q(s_t,a_t,\theta_t)$$
  
  但是这样子比较底层，我们引入DQN的思想。、
  
  

我要训练神经网络，我就需要一个**损失函数**：$J(w)=\mathbb{E}\left[\left(R+\gamma\max_{a\in\mathcal{A}(S^{\prime})}\hat{q}(S^{\prime},a,w)-\hat{q}(S,A,w)\right)^2\right]$

怎么最小化这个损失函数呢？用Gradient descent。

要计算 $J(w)$ 相对于 $w$ 的梯度，这个损失函数有两个地方含有 $w$ 。DQN引入了**两个神经网络**，main network和target network：$J(w)=\mathbb{E}\left[\left(R+\gamma\max_{a\in\mathcal{A}(S^{\prime})}\hat{q}(S^{\prime},a,w_T)-\hat{q}(S,A,w)\right)^2\right]$ ， $R$ 和 $S'$ 是和模型有关的随机变量。

这里涉及到一个**experience replay**：$\begin{aligned} \mathcal{B} & \doteq\{(s,a,r,s^{\prime})\} \end{aligned}$ ，从缓冲区拿 ${(s,a)}$ 是服从**均匀分布**的，如果是别的分布的话需要有先验知识，知道谁重要谁不重要。![image-20250522101156891](G:\software\Typora\Typora_files\Hyper_Brain\强化学习\RL.assets\image-20250522101156891.png)

（因为是**off-policy所以假设现有了一个策略 $\pi_b$ ，是behavier policy，它产生了很多的sample**，把这些sample放到集合 replay buffer中。） 



## 6.策略梯度方法

值函数近似是用函数代替表格来表达action value/state value，现在我们用函数替代表格表达policy。

下面有两个等价的metric：

1.用**average value**： $\bar{v}_\pi=\mathbb{E}[v_\pi(S)=\mathbb{E}\left[\sum_{t=0}^\infty\gamma^tR_{t+1}\right]^{\color{red}{}}$ 的大小来评估策略的好坏。2.用**average reward对immediate reward求平均**：

$$\bar{r}_{\pi}\doteq\sum_{s\in\mathcal{S}}d_{\pi}(s){r_{\pi}(s)}=\mathbb{E}[r_{\pi}(S)]=\lim_{n\to\infty}\frac{1}{n}\mathbb{E}\left[\sum_{k=1}^{n}R_{t+k}\right]$$

$\bar{r}_{\pi}$ 和 $\bar{v}_{\pi}$ 是等价的，$\bar{r}_\pi=(1-\gamma)\bar{v}_\pi$

对metric求梯度有：$\nabla_\theta J(\theta)=\sum_{s\in\mathcal{S}}\eta(s)\sum_{a\in\mathcal{A}}\nabla_\theta\pi(a|s,\theta)q_\pi(s,a)$ ，${J(\theta)}\mathrm{~can~be~}\bar{v}_\pi,\bar{r}_\pi,\mathrm{or~}\bar{v}_\pi^0$



$$\begin{aligned}
\nabla_{\theta}J & 
\begin{aligned}
 & =\sum_sd(s)\sum_a\nabla_\theta\pi(a|s,\theta)q_\pi(s,a)
\end{aligned} \\
 & 
\begin{aligned}
=\sum_sd(s)\sum_a\pi(a|s,\theta)\nabla_\theta\ln\pi(a|s,\theta)q_\pi(s,a)
\end{aligned} \\
 & =\mathbb{E}_{S\sim d}\left[\sum_{a}\pi(a|S,\theta)\nabla_{\theta}\ln\pi(a|S,\theta)q_{\pi}(S,a)\right] \\
 & =\mathbb{E}_{S\sim d,A\sim\pi}
\begin{bmatrix}
\nabla_\theta\ln\pi(A|S,\theta)q_\pi(S,A)
\end{bmatrix} \\
 & \doteq\mathbb{E}\left[\nabla_\theta\ln\pi(A|S,\theta)q_\pi(S,A)\right]
\end{aligned}$$



**梯度上升算法**：

$\begin{aligned} \theta_{t+1} & =\theta_{t}+\alpha\nabla_{\theta}J\left(\theta\right) \\ & =\theta_t+\alpha\mathbb{E}\left[\nabla_\theta\ln\pi(A|S,\theta_t)q_\pi(S,A)\right] \end{aligned}$

有expectation所以上面的公式不能直接用，**要用随机梯度上升算法**：

$\theta_{t+1}=\theta_t+\alpha\nabla_\theta\ln\pi(a_t|s_t,\theta_t)q_\pi(s_t,a_t)$

但是上面的式子也是不能用的，因为 $q_\pi(s_t,a_t)$ 是策略 $\pi$ 真实对应的action value，那么把 $q_\pi(s_t,a_t)$ 换成 $q_t(s_t,a_t)$ 近似或者采样。

- 用MC 近似 $q_t(s_t,a_t)$ 的算法叫 REINFORCE
- 用TD 近似 $q_t(s_t,a_t)$

![image-20250522161914072](G:\software\Typora\Typora_files\Hyper_Brain\强化学习\RL.assets\image-20250522161914072.png)



## 7. actor-critic

 ### 7.1 AC

**介绍actor-critic基本思想**：

![image-20250526102138627](G:\software\Typora\Typora_files\Hyper_Brain\强化学习\RL.assets\image-20250526102138627.png)

### 7.2 Advantage actor-critic

**引入baseline 来减少方差，而 $\left(q_\pi(S,A)-b(S)\right)$ = $\left(q_\pi(S,A)-v_\pi(S)\right)$ = $\delta$ ，这里是令baseline= $v_\pi(s)$ 。**

![image-20250526144930468](G:\software\Typora\Typora_files\Hyper_Brain\强化学习\RL.assets\image-20250526144930468.png)

policy network的参数为 $\theta$ ,value network的参数为 $w$ 。TD误差 $\delta_t=r_{t+1}+\gamma v(s_{t+1},w_t)-v(s_t,w_t)$ 就是优势函数。

### 7.3 off-policy actor-critic

- **importance sampling**

$$\mathbb{E}_{X\sim p_0}[X]=\sum_{x}p_0(x)x=\sum_{x}p_1(x)\underbrace{\frac{p_0(x)}{p_1(x)}x}_{f(x)}=\mathbb{E}_{X\sim p_1}[f(X)]$$

$\quad\frac{p_0(x_i)}{p_1(x_i)}$ 叫做importance weight；用 $p_1$ 这个概率分布去估计 $p_0$ 。

![image-20250527090729831](G:\software\Typora\Typora_files\Hyper_Brain\强化学习\RL.assets\image-20250527090729831-1748308050907-1.png)

### 7.4 deterministic actor-critic

$\beta$ 是behavior policy，$\mu$ 是target policy。

![image-20250527092542562](G:\software\Typora\Typora_files\Hyper_Brain\强化学习\RL.assets\image-20250527092542562.png)

