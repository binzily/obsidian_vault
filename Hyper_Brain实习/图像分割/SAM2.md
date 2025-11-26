首先回顾一下经典的ViT的结构

![img](https://picx.zhimg.com/v2-b657ff78abf08822bea2df9e9bf43b3b_1440w.jpg)

论文开篇就是这张图：

![image-20250305145313945](C:\Users\26344\AppData\Roaming\Typora\typora-user-images\image-20250305145313945.png)

简单解释一下：

（a）提出了一种新的视觉任务，通过prompt引导模型分割图像，prompt的形式有4种：point、box、mask和text。但开源的代码中缺少text部分。

（b）模型整体由3部分组成，prompt encoder对prompt进行编码，image encoder对图片进行编码，lightweight mask decoder融合prompt和image的编码，生成分割masks。

（c）为了快速标注数据训练模型而提出的辅助图片标注的流程。

------

下面这张图对模型进行更具体的展开，

![image-20250305145348069](C:\Users\26344\AppData\Roaming\Typora\typora-user-images\image-20250305145348069.png)

从中可以看出几点：

- image encoder对image进行编码，得到image embedding。
- Conv对mask进行编码，再和image embedding相加。
- points、box和text经过prompt encoder编码，再与image embedding融合，最后输出多个masks。

但是还不够详细，本人对流程中Tensor的shape的变化过程有墙裂的求知欲。

image encoder和conv还好理解一些，图像特征提取那一套，都是（B, C, H, W）进(B, C', H', W')出。

prompt encoder和mask decoder就没那么容易想象，看论文也没看懂，于是看源代码。

根据对代码的理解，画出下面几个图。

------

image经过ViT编码，mask经过conv编码，得到shape相同的embedding，然后相加。

为了方便理解（偷懒），下面画的图都把batch size默认为1，省略掉1维。

![img](https://pica.zhimg.com/v2-516a6d861c5ebb826a12b6e52b874b7e_1440w.jpg)

​								image和mask编码

------

points和box两种prompts都统一成points的形式，因为box（矩形）可以用左上角和右下角两个点表示。

points分正样本（前景）和负样本（背景）两类，所以一个point prompt不仅有坐标coordinate，还有标签label。

同样，box的两个点也有labels，区分哪个是左上角哪个是右下角，同时也能跟points区分开来。

如此一来，points和box都可以统一由坐标（coordinates）和标签（labels）表示，于是就可以用同一个网络对它们进行编码（尽管我画成了两个分支，但其实模型参数是共享的）。

points可以有多个，所以coordinates.shape=(P, 2)，对应的labels.shape=(P, 1)。

box按理说只需要一个，所以coordinates.shape=(2, 2)，对应的labels.shape=(2, 1)。

coordinates经过一顿处理，变成了shape=(P, C)的coord embedding，可以简单理解为经过一个Linear层，提升了维度。

labels不直接输入模型计算，而是用对应的tokens，一个token是一个长度为C的向量，所以tokens.shape=(P, C)。

然后，coord embedding和label tokens相加，得到的结果称之为point embedding吧。

然后，points和box得到的point embedding拼接成新的point embedding，shape=(P+2, C)。

到此，points和box的编码阶段就结束啦。

![img](https://picx.zhimg.com/v2-0dc62fb6a7444aa155f8a7c9c7202297_1440w.jpg)

​						prompt encoder细节（points和box编码）

------

前面两张流程图描述了image embedding和point embedding的生成流程，接下来介绍这两者的融合过程，也就是mask decoder。

这里突然冒出了iou tokens和mask tokens。

先说mask tokens，类似目标检测DETR里面的anchors，输入一个query得到一个output，这里输入一个mask token就得到一个分割mask。

这里用了3+1个mask tokens，3对应着论文中所说的整体、部分和子部分（whole, part, and subpart），论文中Figure4的剪刀就说明了这一点，即一个prompt可以让模型分割出三个masks。

如果不想要那么多个模棱两可的输出，而是要更加坚定的唯一结果，那就不用这3个mask tokens，用剩下的那个mask token。

再就是iou tokens，如果说mask tokens是用来查询masks的，那么iou tokens就是用来查询masks的置信度的，或者说是分割的质量。类似目标检测里bounding box和score的关系。

mask tokens和iou tokens会和前面介绍的point embedding拼接在一起，统称为output tokens。

output tokens和image embedding被送进transformer做融合，输出的结果的shape和输入时是一样的，只是经过交融后达到了你中有我和我中有你的状态，至于交融的过程这里就不展开了。

到这里，masks已经呼之欲出了。

还记得output tokens是由iou tokens、mask tokens和point embedding三部分组成的吗？现在要把它们也拆分成三部分。

其中，mask tokens和image embedding进行矩阵相乘，直接得到最终的masks。

iou tokens经过一个MLP，也直接得到4个masks的置信度。

![img](https://pic4.zhimg.com/v2-878ca0722c930b6760ecaca8841601dd_1440w.jpg)

​									mask decoder

![image-20250305145422585](C:\Users\26344\AppData\Roaming\Typora\typora-user-images\image-20250305145422585.png)

​									mask decoder

decoder的结构之所以看起来复杂，主要原因是prompt embedding和image embedding在这个结构中反复融合并且反复更新，从这里同样可以看出prompt在这个任务中的重要地位。

我们从左至右逐步分析decoder的流程，

- 在prompt embedding进入decoder之前，先在它上面concat了一组可学习的output tokens，output tokens由两个部分构成：
  - 一个是iou token，它会在后面被分离出来用于预测iou的可靠性（对应结构图右侧的IoU output token），它受到模型计算出的iou与模型计算出的mask与GT实际的iou之间的MSE loss监督；
  - 另一个是mask token，它也会在后面被分离出来参与预测最终的mask（对应结构图右侧的output token per mask），mask受到[focal loss](https://zhida.zhihu.com/search?content_id=226202437&content_type=Article&match_order=1&q=focal+loss&zhida_source=entity)和[dice loss](https://zhida.zhihu.com/search?content_id=226202437&content_type=Article&match_order=1&q=dice+loss&zhida_source=entity) 20:1的加权组合监督。
  - 这两个token的意义我感觉比较抽象，因为理论来说进入decoder的变量应该是由模型的输入，也就是prompt和image的映射构成，但这两个token的定义与prompt和image完全没有关系，而是凭空出现的。从结果反推原因，只能把它们理解成对模型的额外约束，因为它们两个参与构成了模型的两个输出并且有loss对他们进行监督。
  - 最终prompt embedding（这一步改名叫prompt token）和刚才提到这两个token concat到一起统称为tokens进入decoder。
- image embedding在进入decoder之前也要进行一步操作：dense prompt由于包含密集的空间信息，与image embedding所在的特征空间一致性更高，所以直接与image embedding相加融合。因为后面要与prompt做cross attention融合，这里还要先算一下image embedding的位置编码。
- 接下来{image embedding，image embedding的位置编码，tokens}进入一个两层transformer结构的decoder做融合。值得注意的是，在transformer结构中，为了保持位置信息始终不丢失，每做一次attention运算，不管是self-attention还是cross-attention，tokens都叠加一次初始的tokens，image embedding都叠加一次它自己的位置编码，并且每个attention后边都接一个layer_norm。
  - tokens先过一个self-attention。
  - tokens作为q，对image embedding做cross attention，更新tokens。
  - tokens再过两层的mlp做特征变换。
  - image embedding作为q，对tokens做cross attention，更新image embedding。
- 更新后的tokens作为q，再对更新后的image embedding做cross attention，产生最终的tokens。
- 更新后的image embedding过两层kernel_size=2, stride=2的转置卷积，升采样到4x大小（依然是4x降采样原图的大小），产生最终的image embedding。
- 接下来兵分两路：
  - mask token被从tokens中分离出来（因为他一开始就是concat上去的，可以直接按维度摘出来），过一个三层的mlp调整channel数与最终的image embedding一致，并且他们两个做矩阵乘法生成mask的预测。
  - iou token被从tokens中分离出来，也过一个三层的mlp生成最终的iou预测。
- 最后，如前文所述，分别对mask的预测和iou预测进行监督，反向传播，更新参数。

参考文献：

[SAM（Segment Anything Model）模型结构 - 知乎](https://zhuanlan.zhihu.com/p/661793344)

[Segment Anything Model（SAM）模型结构介绍 - 知乎](https://zhuanlan.zhihu.com/p/6717260596)

[SAM模型详解 - 知乎](https://zhuanlan.zhihu.com/p/621320070)

