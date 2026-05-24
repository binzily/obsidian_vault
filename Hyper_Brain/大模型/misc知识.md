[深入理解进程，如何进行上下文切换？计算机科学中最深刻和最成功的想法之一。_哔哩哔哩_bilibili](https://www.bilibili.com/video/BV1Cfx7e2Eqr?spm_id_from=333.788.videopod.sections&vd_source=40871d74fa8db37f3b1352390d563aa5)

# 1. AMP混合精度训练

- 训练时，梯度相对参数是很小的数，那么更新参数时，要把梯度和参数相加（梯度下降算法），当精度不够时（比如说梯度和参数都用FP16表示），会出现大数吃小数的问题，导致参数不更新：<img src="G:\software\Typora\Typora_files\Hyper_Brain\大模型\misc知识.assets\image-20250527101703440.png" alt="image-20250527101703440" style="zoom: 50%;" />

解决方法：保存一份高精度的参数weight，在训练时用低精度，在更新时用高精度：

<img src="G:\software\Typora\Typora_files\Hyper_Brain\大模型\misc知识.assets\image-20250527101849421.png" alt="image-20250527101849421" style="zoom:50%;" />

- 第二个问题是梯度大部分绝对值都很小，在FP16精度之外，但是FP16还可以表示大数，这就导致还有很大一部分没有没用到。那么我们可以对梯度进行缩放，把他移动到FP16可以表示的范围内：![image-20250527102331051](G:\software\Typora\Typora_files\Hyper_Brain\大模型\misc知识.assets\image-20250527102331051.png)



# 2.分布式训练技术

## 2.1 data parallel

![image-20250527151057230](G:\software\Typora\Typora_files\Hyper_Brain\大模型\misc知识.assets\image-20250527151057230.png)

- 问题1：单进程，多线程，Python GIL只能利用一个cpu核
- 问题2：GPU0负责收集梯度，更新参数，同步参数。通信，计算压力大。

## 2.2 distributed data parallel

![image-20250527172639865](G:\software\Typora\Typora_files\Hyper_Brain\大模型\misc知识.assets\image-20250527172639865.png)

<img src="G:\software\Typora\Typora_files\Hyper_Brain\大模型\misc知识.assets\image-20250527172858019.png" alt="image-20250527172858019" style="zoom: 33%;" />

通过Ring-AllReduce使得DDP每个GPU里都要存储完整的神经网络优化器状态，显存开销大，通信开销小。