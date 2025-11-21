# 一.前置知识学习

## 1.传统RNN网络

![image-20241115095950842](C:\Users\26344\AppData\Roaming\Typora\typora-user-images\image-20241115095950842.png)

## 2.长短期记忆网络LSTM

![image-20241115102958132](C:\Users\26344\AppData\Roaming\Typora\typora-user-images\image-20241115102958132.png)

# 二.Transformer

### 2.1多头机制

![image-20241116113301460](C:\Users\26344\AppData\Roaming\Typora\typora-user-images\image-20241116113301460.png)

多头机制即平行做多次。

### 2.2Layer Norm

![image-20241116120106995](C:\Users\26344\AppData\Roaming\Typora\typora-user-images\image-20241116120106995.png)

### 2.3.整体梳理

![image-20241116125929780](C:\Users\26344\AppData\Roaming\Typora\typora-user-images\image-20241116125929780.png)

解码器输出是个向量，如何变成单词？这就由一个简单的全连接层和softmax层来处理了。

全连接层输出一个很长的一维向量，softmax输出概率最高的哪个向量。

### 2.4.Encoder-Decoder

![image-20241116143412534](C:\Users\26344\AppData\Roaming\Typora\typora-user-images\image-20241116143412534.png)

这样无论输入多长，都压缩成统一长度编码c，导致精度下降。所以有如下改进：

![image-20241116143641247](C:\Users\26344\AppData\Roaming\Typora\typora-user-images\image-20241116143641247.png)

# 三.BERT

![image-20241116130246787](C:\Users\26344\AppData\Roaming\Typora\typora-user-images\image-20241116130246787.png)

![](C:\Users\26344\AppData\Roaming\Typora\typora-user-images\image-20241116130716770.png)

![image-20241116130725351](C:\Users\26344\AppData\Roaming\Typora\typora-user-images\image-20241116130725351.png)

  BERT预训练：海量无监督语料。用上下文预测屏蔽的单词就是bidirectional。模型输出被mask的词后，会和mask之前的正确词做对比。

# 四.VIT

![image-20241116155524274](C:\Users\26344\AppData\Roaming\Typora\typora-user-images\image-20241116155524274.png)

class embeding是整个图片的全局特征，比如说代表了这张图片是一个苹果，是模型理解图像后给出的综合表述。

 

# 五.Prompt-Tuning

### 5.1 经典的Pre-Trained任务

#### 5.1.1 Masked Language Modeling （MLM）

​	传统的语言模型是以word2vec、GloVe为代表的词向量模型,他们主要是以==词袋==(N-Gram)为基础。例如在word2vec的CBOW方法中,==随机选取一个固定长度的词袋区间,然后挖掉==中心部分的词后,让模型(一个简单的深度神经网络四)预测该位置的词,如下图所示:![image-20241122170247289](C:\Users\26344\AppData\Roaming\Typora\typora-user-images\image-20241122170247289.png)

MLM则采用了N-Gram的方法，不同的是，N-Gram喂入的是被截断的短文本，而==MLM则是完整的文本==，因此MLM更能够保留原始的语义：![image-20241122170556815](C:\Users\26344\AppData\Roaming\Typora\typora-user-images\image-20241122170556815.png)

​	MLM是一种==自监督的训练方法==，其先从大规模的无监督语料上通过固定的替换策略获得自监督语料，设计预训练的目标来训练模型，具体的可以描述为：

- 替换策略：在所有语料中，随机抽取15%的文本。被选中的文本中，则有80%的文本中，随机挑选一个token并替换为 `[mask]`，10%的文本中则随机挑选一个token替换为其他token，10%的文本中保持不变。

- 训练目标：当模型遇见 `[mask]` token时，则根据学习得到的上下文语义去预测该位置可能的词，因此，训练的目标是对整个词表上的分类任务，可以使用交叉信息熵作为目标函数。

因此以BERT为例，首先喂入一个文本`It is very cold today, we need to wear more clothes.` ，然后随机mask掉一个token，并结合一些特殊标记得到：`[cls] It is very cold today, we need to [mask] more clothes. [sep]` ，喂入到多层的Transformer结构中，则可以得到最后一层每个token的隐状态向量。MLM则通过在`[mask]`头部添加一个MLP映射到词表上，得到所有词预测的概率分布。

#### 5.1.2 Next Sentence Prediction（NSP）

​	其主要目标是==给定两个句子，来判断他们之间的关系==，属于一种自然语言推理（NLI）任务。

在BERT中，NSP任务则视为sentence-pair任务，例如输入两个句子`S1：It is very cold today.` 和 `S2：We need to wear more clothes.`，通过拼接特殊字符后，得到：`[cls] It is very cold today. [sep] We need to wear more clothes. [sep]`，然后喂入到多层Transformer中，可以得到`[cls]`token的隐状态向量，同样通过MLP映射到一个3分类上获得各个类的概率分布：![image-20241122234140925](C:\Users\26344\AppData\Roaming\Typora\typora-user-images\image-20241122234140925.png)

### 5.2 Task-specific Fine-tuning

​	==获得了预训练的语言模型后==，在面对具体的下游任务时，则需要进行微调。通常微调的任务目标取决于下游任务的性质。我们简单列举了几种NLP有关的下游任务：

- **Single-text Classification（单句分类）**：常见的单句分类任务有短文本分类、长文本分类、意图识别、情感分析、关系抽取等。给定一个文本，喂入多层==Transforme==r模型中，获得最后一层的隐状态向量后，再输入到新添加的分类器==MLP==中进行分类。在Fine-tuning阶段，则通过交叉信息熵损失函数训练分类器；
- **Sentence-pair Classification（句子匹配/成对分类）**：给定两个文本，用于判断其是否存在匹配关系。此时将两个文本拼接后喂入模型中，训练策略则与Single-text Classification一样；
- **Span Text Prediction（区间预测）**：常见的任务类型有抽取式阅读理解、实体抽取、抽取式摘要等。给定一个passage和query，根据query寻找passage中可靠的字序列作为预测答案。通常该类任务需要模型预测区间的起始位置，因此在Transformer头部添加两个分类器以预测两个位置。

​	这几类任务在Fine-tuning阶段几乎都涉及**在模型头部引入新参数**的情况，且都存在**小样本场景过拟合**的问题，因此Prompt-Tuning的引入非常关键。

### 5.3 Prompt-Tuning的定义

​	**Prompt的目的是将Fine-tuning的下游任务目标转换为Pre-training的任务**。那么具体如何工作呢？

​	以二分类的情感分析作为例子，描述Prompt-tuning的工作原理。给定一个句子`[CLS] I like the Disney films very much. [SEP]` 传统的Fine-tuning方法是将其通过BERT的Transformer获得 `[CLS]`表征之后再喂入新增加的MLP分类器进行二分类，预测该句子是积极的（positive）还是消极的（negative），因此需要一定量的训练数据来训练。

​	而Prompt-Tuning则执行如下步骤：

- **构建模板（Template Construction）**：通过人工定义、自动搜索、文本生成等方法，生成与给定句子相关的一个含有`[MASK]`标记的模板。例如`It was [MASK].`，并拼接到原始的文本中，获得Prompt-Tuning的输入：`[CLS] I like the Disney films very much. [SEP] It was [MASK]. [SEP]`。将其喂入BERT模型中，并复用预训练好的MLM分类器（在huggingface中为BertForMaskedLM），即可直接得到`[MASK]`预测的各个token的概率分布；
- **标签词映射（Label Word Verbalizer）**：因为`[MASK]`部分我们只对部分词感兴趣，因此需要建立一个映射关系。例如如果`[MASK]`预测的词是“great”，则认为是positive类，如果是“terrible”，则认为是negative类。

*每个句子可能期望预测出来的label word都不同，因此如何最大化的寻找当前任务更加合适的template和label word是Prompt-tuning非常重要的挑战。*

- **训练**：根据Verbalizer，则可以获得指定label word的预测概率分布，并采用交叉信息熵进行训练。此时因为只对预训练好的MLM head进行微调，所以避免了过拟合问题

*引入的模板和标签词本质上也属于一种数据增强，通过添加提示的方式引入==先验知识==*

### 5.4 Prompt-tuning的研究进展

#### 5.4.1 Prompt-Tuning的鼻祖——GPT3与PET

​	Prompt-Tuning其起源于GPT3提出的：

- **In-context Learning**：通过上下文的示例来启发模型，上下文是用户一并输入的。

- **Demonstration Learning**：和上下文有相似性，但是不同的是，它更注重展示得到答案的过程；上下文学习更简洁一点，给出的是结果示例。

这类方法有一个明显的缺陷是——**其建立在超大规模的预训练语言模型上**，此时的模型参数数量通常超过100亿，**在真实场景中很难应用**，所以PET问世。PET详细地设计了Prompt-Tuning的重要组件——Pattern-Verbalizer-Pair（PVP）

​	PET设计了两个组件：

- **Pattern（Template）**：即上文提到的Template。
- **Verbalizer**：即标签词的映射，对于具体的分类任务，需要选择指定的标签词（label word）。

当一个句子有多个PVP时，即为Prompt-Tuning的集成（PVPs-Ensembling）。

​	PET还提供了半监督的学习方法——Iterative PET（iPET）![img](https://i-blog.csdnimg.cn/blog_migrate/564361ddf3d986a2cd557722868d76a7.png)

iPET原理是通过多次迭代的方式来优化提示（Prompt）。一开始，会设计一个初始的提示，然后根据模型对这个提示的输出结果来分析和改进提示。

#### 5.4.2 如何挑选合适的Pattern

​	现阶段有一些成熟的==自动构建Pattern==的方法，分为：

- ==Hard Prompt==:也称离散Prompt,是一个实际的文本字符串(自然语言,人工可读),通常由中文或英文词汇组成;不需引入任何参数。无法在训练中被优化。
- ==Soft Prompt==:也称连续Prompt,通常是在向量空间优化出来的提示,通过梯度搜索之类的方式进行优化;需要引入少量的参数。可以在训练中优化。

Hard Prompt包括：

- 启发式：基于经验、直觉和一些一般性的原则来构建模板，相对较为灵活但可能不够精确。
- 生成式：通常利用算法和模型根据大量的数据自动生成模板，具有一定的创新性但可能缺乏可解释性。
- 人工构建：完全由人根据自己的知识和判断来设计模板，更具针对性和明确的意图，但可能受个人局限。

Soft Prompt包括：

- 词向量微调（Word Embedding）：显式地定义离散字符的模板，但在训练时这些模板字符的词向量参与梯度下降，初始定义的离散字符用于作为向量的初始化；

- 伪标记（Pseudo Token）：不显式地定义离散的模板，而是将模板作为可训练的参数；

​	基于**连续提示**（Soft-Tuning）的Prompt-Tuning方法：**Prompt Tuning**，**P-tuning**，**PPT（Pre-trained Prompt Tuning）**。下面是PPT的例子：![在这里插入图片描述](https://i-blog.csdnimg.cn/blog_migrate/d4297a3a810075b6a38c8e563f68f885.png)

​	基于**离散提示**（hard-Tuning）的Prompt-Tuning方法**RLprompt**。

Pre-trained Prompt Tuning 可以看作是在 P-tuning 的思路基础上，更注重对已经存在的、在预训练阶段形成的输入模板或提示进行优化和改进，以使其更适合特定的任务和数据。

在 RLprompt 中，模型根据当前的提示生成回答，然后根据某种奖励机制（比如回答的准确性、合理性等）来评估这个回答的好坏。基于这个评估结果，模型调整提示的生成策略，以便在后续的交互中能够获得更高的奖励，也就是生成更优质的回答。



#### 5.4.3 如何挑选合适的Verbalizer

​	传统的方法是人工设计，除此之外还有两个有代表性的方法：**KPT（Knowledgeable Prompt Tuning）**和**ProtoVerb**。

- KPT

![img](https://i-blog.csdnimg.cn/blog_migrate/53d61c3dc18f8f60fc47697718a8ccf4.png)

- ProtoVerb![img](https://i-blog.csdnimg.cn/blog_migrate/6a45cd14ce6c4f79712d0d670c07c505.png)

 Verbalizer 就像是一个 “翻译官”，把模型给出的一些不太好理解的结果 “翻译” 成我们能明白的自然语言。比如说我们要判断一段文字是表达 “高兴” 还是 “悲伤” 的情绪，可能有几个不同的 “翻译官”（也就是 Verbalizer）方案，有的可能直接翻译成 “高兴”“悲伤”，有的可能会翻译成 “心情愉悦”“心情低落” 等等。而 ProtoVerb 呢，就像是一个专门负责挑选 “翻译官” 的 “管理员”。

### 5.5 Prompt-Tuning的本质

- Prompt的本质是一种对任务的指令；
- Prompt的本质是一种对预训练任务的复用；
- Prompt的本质是一种参数有效性学习；

#### 5.5.1 Prompt是一种针对任务的指令

​	看似设计指令是一件容易的事情，但是在真实使用过程中，预训练模型很难“理解”这些指令，根据最近研究工作发现，主要总结如下几个原因：

- **预训练模型不够大**：对比一下传统的Fine-tuning，每个样本的输入几乎都是不同的，然而基于Prompt的方法中，所有的样本输入都会包含相同的指令，这就导致小模型很容易受到这些指令带来的干扰。
- **缺乏指令相关的训练**

​	也许读者想到了前面所讲到的Pre-trained Prompt Tuning（PPT），即再次对预训练语言模型进行一次Continual Pre-training。然而我们忽略了一点，即**我们期望预训练模型不止是在我们已经设计好的指令上进行学习，还应该在未知的指令上具备一定的泛化性能**。为了达到这个目的，最常用的方法是**元学习（Meta Learning）**，下面介绍几个代表性的工作：

- **TransPrompt**：

