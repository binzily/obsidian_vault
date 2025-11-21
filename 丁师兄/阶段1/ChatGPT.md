 # 一.预训练阶段

## 1.1 Tokenizer Training

​	通常，tokenizer 有 3 种常用形式：WordPiece 、 BPE和BBPE。

- **WordPiece**（Subwords）：

  假设我们有相同的语料库：["hug", "hugs", "hugging", "hugged"]。

  1）**初始化**：同样将所有单词拆分为单个字符：

  - hug -> h, u, g
  - hugs -> h, u, g, s
  - hugging -> h, u, g, g, i, n, g
  - hugged -> h, u, g, g, e, d

  2）**合并过程**：

  - WordPiece会考虑合并后对语言模型似然值的提升程度。假设合并h和u为hu能最大化似然值，进行合并.
  - 更新词汇表：[hu, g, s, i, n, e, d]
  - 继续合并，假设合并hu和g为hug能最大化似然值，进行合并.
  - 更新词汇表：[hug, s, i, n, e, d]
  - 继续合并，直到达到预定的词汇表大小.

- **BPE**（Subwords）：

  假设一个简单的语料库：["hug", "hugs", "hugging", "hugged"]。

  1）**初始化**：首先，将所有单词拆分为单个字符：

  - hug -> h, u, g
  - hugs -> h, u, g, s
  - hugging -> h, u, g, g, i, n, g
  - hugged -> h, u, g, g, e, d

  2）**合并过程**：

  - 统计字符对的频率，选择频率最高的字符对进行合并。假设h和u出现频率最高，合并为hu。
  - 更新词汇表：[hu, g, s, i, n, e, d]
  - 继续合并，假设hu和g出现频率最高，合并为hug。
  - 更新词汇表：[hug, s, i, n, e, d]
  - 继续合并，直到达到预定的词汇表大小。

- **Byte-level BPE**（基于字节）：

  1）**初始化**：基于字节初始化。

  2）**统计相邻字节对的频率**。

  3）**合并频率最高的字节对**。

## 1.2 LM Pretraining

​	预训练有3个点：数据源采样，数据预处理，模型结构。



## 1.3 数据集清理



## 1.4 模型效果评测

​	LM的量化指标，较为普遍的有PPL，BPC等，可以简单理解为在生成结果和目标文本之间的Cross Entropy Loss 上做了一些处理，用来评估拟合程度（生成的句子通不通顺）。

​	还有一个知识蕴含能力。一个很好的中文知识能力测试数据集是 [[C-Eval](https://link.zhihu.com/?target=https%3A//github.com/SJTU-LIT/ceval)]，涵盖1.4w 道选择题，共 52 个学科。由于是选择题的形式，我们可以通过将题目写进 prompt 中，并让模型续写 1 个 token，判断这个续写 token 的答案是不是正确答案即可。但大部分没有精调过的预训练模型可能无法续写出「A B C D」这样的选项答案，因此，官方推荐使用 5-shot 的方式来让模型知道如何输出答案。我们获得模型续写后的第一个 token 的概率分布（logits），并取出「A B C D」这 4 个字母的概率，通过 softmax 进行归一化：

```python
probs = (
    torch.nn.functional.softmax(
        torch.tensor(
            [
                logits[self.tokenizer.encode(
                    "A", bos=False, eos=False)[0]],
                logits[self.tokenizer.encode(
                    "B", bos=False, eos=False)[0]],
                logits[self.tokenizer.encode(
                    "C", bos=False, eos=False)[0]],
                logits[self.tokenizer.encode(
                    "D", bos=False, eos=False)[0]],
            ]
        ),
        dim=0,
    ).detach().cpu().numpy()
)
pred = {0: "A", 1: "B", 2: "C", 3: "D"}[np.argmax(probs)] 
```

由于是 4 选项问题，所以基线（随机选择）的正确率是 25%。

# 二.指令微调阶段

​	==预训练任务的本质在于「续写」==，而「续写」的方式并不一定能够很好的回答用户的问题，因为训练大多来自互联网中的数据，我们无法保证数据中只存在存在规范的「一问一答」格式，这就会造成预训练模型通常无法直接给出人们想要的答案，但这不能说明模型无知。

​	既然模型知道知识，只是不符合我们人类的对话习惯，那么我们只要再去教会模型「如何对话」就好了。

**这就是 Instruction Tuning 要做的事情，即指令对齐**。

OpenAI 展示了 GPT-3 和经过指令微调前后模型的区别：

![image-20250107153326280](C:\Users\26344\AppData\Roaming\Typora\typora-user-images\image-20250107153326280.png)

## 2.1 Self Instruction

​	既然我们需要去教会模型说人话，那么我们就需要去精心编写各式各样人们在对话中可能询问的问题，以及问题的答案。

在Instruct Paper中，使用了 1.3w 的数据来对 GPT-3.5 进行监督学习，成本高。

但是现在我们已经有了ChatGPT，让ChatGPT训练我们的模型就是 **Self Instruction** 的思路。

通俗来讲，就是人为的先给一些「训练数据样例」让 ChatGPT 看，紧接着利用 ChatGPT 的续写功能，让其不断地举一反三出新的训练数据集。

## 2.2 开源数据集构建

​	stanford_alpaca采用上述的 self instruction 的方式采集了 5200 条指令训练数据集，样例如下：

```python
{
    "instruction": "Arrange the words in the given sentence to form a grammatically correct sentence.",
    "input": "quickly the brown fox jumped",
    "output": "The quick brown fox jumped quickly."
}
```

其中，instruction 代表要求模型做的任务，input 代表用户输入， output 代表喂给模型的 label。

​	BELLE（Self Instruction中文项目）数据集样例如下：

```python
{
    "instruction": "判断给定的文章是否符合语法规则。如果不符合，请提供修改建议。 下面是一篇文章的开头: ‘为了探讨这个主题，本文将提供一系列数据和实例，以证明这一观点。’",
    "input": "",
    "output": "这个开头符合语法规则。"
}
```

## 2.3 模型的评测方法

​	一种比较流行的方式是用GPT-4为模型生成的结果打分，但是事实上这种方式没有想象中的靠谱。

# 三.奖励模型（RM）

## 3.1 奖励模型（RM）的必要性

​	前面SFT（Supervised Fine-Tuning）**监督式微调**（Self Instruction）是在**告诉模型什么是好的数据，没有给出不好的数据**。 SFT 的目的只是将 Pretrained Model 中的知识给引导出来的一种手段，而在SFT 数据有限的情况下，我们对模型的「引导能力」就是有限的。这将导致预训练模型中原先「错误」或「有害」的知识没能在 SFT 数据中被纠正，从而出现「有害性」或「幻觉」的问题。

​	为此，一些让模型脱离昂贵标注数据，自我进行迭代的方法被提出，比如：RLHF（基于人类反馈的强化学习），DPO，RL 是直接告诉模型当前样本的（好坏）得分，DPO 是同时给模型一条好的样本和一条坏的样本。但无论是 RL 还是 DPO，我们都需要让告知模型什么是「好的数据」，什么是「不好的数据」。

​	而判断样本数据的「好坏」除了昂贵的人工标注之外，那就是 Reward Model 大显身手的时候了。

## 3.2利用偏序对训练奖励模型

 

























# 四.强化学习（RLHF）





















