# 一.yolov1



![image-20241112201514598](C:\Users\26344\AppData\Roaming\Typora\typora-user-images\image-20241112201514598.png)

![image-20241112201436707](C:\Users\26344\AppData\Roaming\Typora\typora-user-images\image-20241112201436707.png)

![image-20241112201407941](C:\Users\26344\AppData\Roaming\Typora\typora-user-images\image-20241112201407941.png)

![image-20241112203044501](C:\Users\26344\AppData\Roaming\Typora\typora-user-images\image-20241112203044501.png)

![image-20241112203152733](C:\Users\26344\AppData\Roaming\Typora\typora-user-images\image-20241112203152733.png)

yolov1的问题：一个cell只有B1和B2两个框，导致小物体检测不到；重叠的东西不好做；每个cell只预测一个类别导致多标签不好做；



# 二.yolov2

### 1.Batch Normalization

V2版本舍弃Dropout，没有全连接层了，卷积后全部加入Batch Normalization（均值为0，方差为σ），防止网络越学越不收敛（走偏）。

从现在的角度来看，Batch Normalization已经成网络必备处理。（Conv-BN当成一个整体的组合）

### 2.更大的分辨率

V1训练时用的是224\*224,测试时使用448\*448。可能导致模型水土不服,V2训练时额外又进行了10次448*448的微调，使用高分辨率分类器后,YOLOv2的mAP提升了约4%。

### 3.网络结构

由于全连接层容易过拟合且训练比较慢，v2的网络中没有全连接层，拿卷积来代替（就是提取特征）。

![image-20241112205826758](C:\Users\26344\AppData\Roaming\Typora\typora-user-images\image-20241112205826758.png)

反正7\*7\*1024最后要变成7\*7\*30，所以不如直接把7\*7\*1024直接\卷积成7\*7\*30。

做了五次降采样，其实就是五次池化。

 ![image-20241112210336971](C:\Users\26344\AppData\Roaming\Typora\typora-user-images\image-20241112210336971.png)

416/2^5 = 13(五次池化)，变成13\*13的网络（v1是7\*7）。

 ### 4.（类似B1和B2的线性框）先验框

把实际的真实的框，用k-means聚类，v2是k = 5类（v1是b = 2类）。

recall相当于查全率。

### 5.位置误差

由于刚开始训练的参数都是随机的，框可能会发生”飘“出去的情况，v2中使用相对偏移。

### 6.感受野

特征图相当于原始特征图的多大区域就叫感受野。

<img src="C:\Users\26344\AppData\Roaming\Typora\typora-user-images\image-20241112224332178.png" alt="image-20241112224332178" style="zoom: 33%;" />

此图最后一层的感受野就是5\*5。

![image-20241112225012736](C:\Users\26344\AppData\Roaming\Typora\typora-user-images\image-20241112225012736.png)

![image-20241112225155612](C:\Users\26344\AppData\Roaming\Typora\typora-user-images\image-20241112225155612.png)

卷积核的维度一定是和输入图像维度一样。上图的3个3\*3卷积每个都会加入BN。

![image-20241112230055344](C:\Users\26344\AppData\Roaming\Typora\typora-user-images\image-20241112230055344.png)

### 7.多样尺寸

![image-20241112230252793](C:\Users\26344\AppData\Roaming\Typora\typora-user-images\image-20241112230252793.png)



全连接层定了，那么图像尺寸也定了。但是卷积是什么大小的图像都能卷。



# 三.yolov3

### 1. 九种先验框，三个检测头

![image-20241113094827170](C:\Users\26344\AppData\Roaming\Typora\typora-user-images\image-20241113094827170.png)

![image-20241113095251039](C:\Users\26344\AppData\Roaming\Typora\typora-user-images\image-20241113095251039.png)

![image-20241113100258130](C:\Users\26344\AppData\Roaming\Typora\typora-user-images\image-20241113100258130.png)

### 2.残差链接

<img src="C:\Users\26344\AppData\Roaming\Typora\typora-user-images\image-20241113101612851.png" alt="image-20241113101612851" style="zoom:33%;" />

残差链接保证网络至少不比原来差。

### 3.网络结构

![image-20241113110629928](C:\Users\26344\AppData\Roaming\Typora\typora-user-images\image-20241113110629928.png)

### 4.先验框设计

![image-20241113111013067](C:\Users\26344\AppData\Roaming\Typora\typora-user-images\image-20241113111013067.png)

![image-20241113112048819](C:\Users\26344\AppData\Roaming\Typora\typora-user-images\image-20241113112048819.png)

![image-20241113112123458](C:\Users\26344\AppData\Roaming\Typora\typora-user-images\image-20241113112123458.png)

### 5. softmax层改进

可以预测一个目标同时属于多个类别。

![image-20241113114407995](C:\Users\26344\AppData\Roaming\Typora\typora-user-images\image-20241113114407995.png)

# 四.yolov4

### 1.数据层面的改进

![image-20241113195602634](C:\Users\26344\AppData\Roaming\Typora\typora-user-images\image-20241113195602634.png)

方法一：

![image-20241113200136465](C:\Users\26344\AppData\Roaming\Typora\typora-user-images\image-20241113200136465.png)



方法二：标签平滑

![image-20241113200844196](C:\Users\26344\AppData\Roaming\Typora\typora-user-images\image-20241113200844196.png)

方式三：CIOU损失

![image-20241113202049313](C:\Users\26344\AppData\Roaming\Typora\typora-user-images\image-20241113202049313.png)

### 2.网络结构的改进

 ![image-20241113205006743](C:\Users\26344\AppData\Roaming\Typora\typora-user-images\image-20241113205006743.png)

channel变为一半，计算量会少。速度提升很多，精度且意外有一点点提升。

![image-20241113205933784](C:\Users\26344\AppData\Roaming\Typora\typora-user-images\image-20241113205933784.png)

在chanell维度和位置维度都加入权重值，因为有的重要有的不重要。

![image-20241113211338167](C:\Users\26344\AppData\Roaming\Typora\typora-user-images\image-20241113211338167.png)

![image-20241113211452437](C:\Users\26344\AppData\Roaming\Typora\typora-user-images\image-20241113211452437.png)

# 五.yolov7



