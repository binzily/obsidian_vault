## 1. 正向过程

​	正向过程的特点在于，可以根据系数$\beta$ 和x~0~直接求出任意时刻的转移分布${p_\theta(\mathbf{x}_{t}|\mathbf{x}_0)}\mathbf{}$。

## 2.反向过程

- 由于前向过程是我们手动添加的噪声，所以有：

  $${
  \begin{aligned}
  q(x_t\mid x_{t-1}) & \sim\mathcal{N}\left(x_t;\sqrt{\alpha_t}x_{t-1},1-\alpha_t\right) \\
   \\
  q(x_{t-1}\mid x_0) & \sim\mathcal{N}\left(x_{t-1};\sqrt{\overline{\alpha}_{t-1}}x_0,1-\overline{\alpha}_{t-1}\right) \\
   \\
  q(x_t\mid x_0) & \sim\mathcal{N}\left(x_t;\sqrt{\overline{\alpha}_t}x_0,1-\overline{\alpha}_t\right)
  \end{aligned}}$$

- **所以正向过程的后验分布${q(\mathbf{x}_{t-1}|\mathbf{x}_t)}\mathbf{}$ 的均值**：

  $\mu_q(x_t,x_0)=\frac{1}{\sqrt{\alpha_t}}\left(x_t-\frac{1-\alpha_t}{\sqrt{1-\bar{\alpha}_t}}\epsilon\right)$
  
- **直接规定，模型输出预测噪声，在通过输出的确定的噪声求得的均值为，这个用在后面的推理部分**：

​	$\mu_\theta(x_t,t)=\frac{1}{\sqrt{\alpha_t}}\left(x_t-\frac{1-\alpha_t}{\sqrt{1-\bar{\alpha}_t}}\epsilon_\theta(x_t,t)\right)$

（正向过程的后验分布可视化看b站https://www.bilibili.com/video/BV1hZ421y7id）

假设正向扩散过程是马尔科夫链（即每个时间步的状态之和前一个时间步的状态有关而与之前的无关），那么可以加高斯分布的噪声得到X~t~，时间步很多的话那么X~t~会接近X~T~，这是一个符合N（0，1）的高斯分布。后面训练的过程就可以直接拿X~T~作为输入，由于上面的假设，前向过程作为已知。

![image-20250512105745233](G:\software\Typora\Typora_files\Hyper_Brain\DDPm.assets\image-20250512105745233.png)

​	X~T~减去ϵ就能生成图片了，加上文字token就能生成准确的图片了。所以是先有X~T~，假设每一个时间步的去噪过程都是高维高斯分布（**高维高斯分布后加多个高维高斯分布可以组合出复杂的概率分布**），这样单个时间步的去噪过程都是高斯分布的话就可以往前推前面时间步的状态了。

## 3.损失函数

**反向传播算法**：在扩散模型中不能直接使用反向传播算法。

- **扩散过程引入随机噪声**：扩散模型通过在数据上逐步添加噪声来模拟扩散过程，这个过程中涉及到随机变量。例如，在每个时间步，会根据一定的概率分布向数据中添加高斯噪声。由于随机变量的取值是不确定的，在某一点处不存在确定的导数。
- **导数定义的不适用**：从导数的定义来看，导数是函数在某一点处的变化率。对于随机变量，由于其取值的随机性，无法明确地定义在某一点处的变化率。以高斯随机变量为例，它的取值是根据概率分布随机出现的，在不同的样本点上取值不同，不存在一个固定的、可计算的变化率，也就无法像确定性函数那样求导。

**重参数化**：$ z=\mu+\sigma\epsilon $，这样就可以**求导**了。

- **确定部分**：$\mu$和$\sigma$是模型给定的参数 ，基于这两个参数以及四则运算规则形成了$\mu + \sigma\epsilon$ 这样一个确定的数学运算形式。只要$\epsilon$确定下来（尽管它是随机采样的），按照这个公式就能确定地计算出结果 ，所以从整体运算形式角度把它归为确定部分。
- **不确定部分**：$\epsilon $本身是从标准正态分布中随机采样而来 ，其取值具有随机性和不可预测性，这是整个过程中不确定性的来源，所以单独把它看作不确定部分。 这种划分是从采样过程的结构和性质角度出发的 。

**KL散度**：
$$
D_{KL}(P||Q)=\int_{-\infty}^{\infty}p(x)\log\left(\frac{p(x)}{q(x)}\right)dx=\mathbb{E}_{x\sim P}\left[\log\left(\frac{p(x)}{q(x)}\right)\right]
$$
​	这里p(x)和q(x)分别是概率分布P和Q的概率密度函数，KL 散度衡量了用概率分布Q来近似概率分布P时所损失的信息。

在公式的期望形式E*x*∼*P*[log(*q*(*x*)*p*(*x*))]中，

- E*x*∼*P*表示对服从概率分布P的随机变量x求期望 ，是一种运算符号 。
- 其作用对象是后面方括号内的函数log(*q*(*x*)*p*(*x*))，意思是在概率分布P下，求函数log(*q*(*x*)*p*(*x*)) 的平均取值 。

**损失函数**：

​	先求x~0~的负对数似然函数，再加上一个KL散度作为上界。这样一来**，目标是最大化对数似然（让模型输出正确的图形的概率最大）**，等价于最小化负对数似然，等价于最小化它的上界。  

$$\begin{aligned}
-\log p_\theta(\mathbf{x}_0) & \leq-\log p_\theta(\mathbf{x}_0)+D_{\mathrm{KL}}(q(\mathbf{x}_{1:T}|\mathbf{x}_0)\|p_\theta(\mathbf{x}_{1:T}|\mathbf{x}_0)) \\
 & =-\log p_\theta(\mathbf{x}_0)+\mathbb{E}_{\mathbf{x}1:T\sim q(\mathbf{x}_1:T|\mathbf{x}_0)}\left[\log\frac{q(\mathbf{x}_{1:T}|\mathbf{x}_0)}{p_\theta(\mathbf{x}_{0:T})/p_\theta(\mathbf{x}_0)}\right] \\
 & =-\log p_\theta(\mathbf{x}_0)+\mathbb{E}_q\left[\log\frac{q(\mathbf{x}_{1:T}|\mathbf{x}_0)}{p_\theta(\mathbf{x}_{0:T})}+\log p_\theta(\mathbf{x}_0)\right] \\
 & =\mathbb{E}_q\left[\log\frac{q(\mathbf{x}_{1:T}|\mathbf{x}_0)}{\boldsymbol{p_\theta}(\mathbf{x}_{0:T})}\right]
\end{aligned}$$

在经过一系列推导后的损失函数：

$$\begin{aligned}
L_t^{\mathrm{simple}} & =\mathbb{E}_{\boldsymbol{t}\sim[1,T],\mathbf{x}_0\boldsymbol{\epsilon}_t}\left[\|\boldsymbol{\epsilon}_t-\boldsymbol{\epsilon}_\theta(\mathbf{x}_t,t)\|^2\right] \\
 & =\mathbb{E}_{t\sim[1,T],\mathbf{x}_0,\boldsymbol{\epsilon}_t}\left[\|\boldsymbol{\epsilon}_t-\boldsymbol{\epsilon}_\theta(\sqrt{\bar{\alpha}_t}\mathbf{x}_0+\sqrt{1-\bar{\alpha}_t}\boldsymbol{\epsilon}_t,t)\|^2\right]
\end{aligned}$$

这个式子是什么意思呢？就是有一个神经网络，它的输入是 x~0~, $\epsilon_t$，还有时间戳 t，输出值是预测的 $\epsilon_\theta$，用来逼近扩散过程的噪声 $\epsilon_t$。这样我们就实现了对负对数似然的优化。

## 4.训练和采样算法

![image-20250512160025022](G:\software\Typora\Typora_files\Hyper_Brain\DDPm.assets\image-20250512160025022.png)

​	先看左边是个循环，里面x~0~符合分布q(x~0~)，也就是从训练集中采样数据。然后随机生成一个时刻t。第四步再生成一个标准正态分布的噪声。然后，把这些值带入下面目标函数式子里。其中$\epsilon_\theta$是个神经网络，且$\mathbf{x}_t=\sqrt{\bar{\alpha}_t}\mathbf{x}_0+\sqrt{1-\bar{\alpha}_t}\epsilon$，**模型训练时，输入x~t~和 t（训练时已知x~0~），目的是让神经网络  $\epsilon_\theta$学会预测出这个与相x~0~关联的真实噪声 $\epsilon$**，损失函数的本质是**让模型学习噪声与数据、时间步之间的潜在关系**。

 训练完毕推理过程看右边f。x~T~从标准正态分布开始，然后逆向循环迭代T步。从正态分布中采样z。这个z乘以当前分布的标准差，然后再加上均值，就能得到x~T-1~，经过T次后就能得到x~0~。









==李宏毅==：

**文生图模型的frame**：

![image-20250605112424063](G:\software\Typora\Typora_files\Hyper_Brain\扩散模型\DDPM.assets\image-20250605112424063.png)

**所有文生图模型共同的的本质目标（像VAE，Diffusion）**：

$P_{data}(x)$ 是真实世界的数据分布，是个很复杂的分布。

![image-20250605133412003](G:\software\Typora\Typora_files\Hyper_Brain\扩散模型\DDPM.assets\image-20250605133412003.png)

**用公式描述这个目标，这个目标等价于解决ELBO**：

![image-20250605133942777](G:\software\Typora\Typora_files\Hyper_Brain\扩散模型\DDPM.assets\image-20250605133942777.png)

