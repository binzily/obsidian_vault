# 一.Llama

​	Llama是和GPT2一样的基于Transformer的Decoder架构。但是做了如下改变：

## 1.1 RoPE

​	Llama采用RoPE（旋转位置）编码，基于旋转矩阵，通过对向量进行旋转操作（相乘）来编码位置信息。原始 Transformer 位置编码通常是通过正弦和余弦函数直接生成固定的位置向量，然后与词嵌入相加。

由于RoPE本质上是旋转矩阵作用于Wq和Wk上，所以不改变向量范数；原始 Transformer 改变了向量范数，可能会出现性能下降或信息丢失的问题。（向量范数可以理解为词在高纬度空间中的语义特征）

![image-20250116105401070](C:\Users\26344\AppData\Roaming\Typora\typora-user-images\image-20250116105401070.png)

## 2.2 GQA



# 二.T5

​	**所谓的 T5 模型其实就是个 Transformer 的 Encoder-Decoder 模型。**在预训练阶段，它使用类似于BERT的Masked Language Model（MLM）方法来训练Encoder，同时Decoder的目标是根据Encoder的输出来重建原始文本。这种架构使得T5能够将各种自然语言处理任务转化为文本到文本的任务，从而实现统一的框架来处理不同的NLP任务。**在预训练阶段，不包含下游任务**，只有无监督数据训练。



## 2.1 代码注意点

### 2.1.1 

何时需要设置 **past_key_values** 为 None：

在生成任务中，past_key_values 用于存储前一个 batch 的隐藏状态，以便在生成下一个 token 时继续使用这些状态，从而加速生成过程并保持生成的连贯性。

1.独立 batch 处理：在大多数监督学习任务中，每个 batch 通常是独立的样本，不需要利用之前的隐藏状态。此时，设置 past_key_values 为 None 可以确保每个 batch 从头开始处理。
2.避免缓存问题：如果不正确初始化或传递 past_key_values，可能会导致模型使用错误的缓存状态，从而影响训练效果。设置为 None 可以确保每次 batch 处理时都是独立的，避免缓存带来的问题。

### 2.1.2 

在使用自动混合精度（Automatic Mixed Precision, AMP）训练模型时，缩放损失（scaling the loss）是一个重要的步骤，主要用于防止梯度下溢（gradient underflow）。以下是详细的解释：

### 自动混合精度（AMP）
自动混合精度是一种训练技术，它结合了单精度（float32）和半精度（float16）浮点数格式来加速训练过程并减少内存使用。然而，使用半精度浮点数时，梯度可能会变得非常小，导致梯度下溢问题，从而影响模型的训练稳定性。

### 梯度下溢
梯度下溢是指在反向传播过程中，梯度值变得非常小，以至于在使用半精度浮点数表示时，梯度值可能变为零。这会导致模型参数无法更新，从而影响训练效果。

### 损失缩放
为了防止梯度下溢，AMP 使用损失缩放（loss scaling）。具体步骤如下：

1. **损失缩放**：在反向传播之前，将损失值乘以一个缩放因子（scale factor）。这个缩放因子通常是一个大于1的值，例如2的幂（如128或256）。
2. **反向传播**：使用缩放后的损失值进行反向传播，计算梯度。
3. **梯度缩放**：在更新模型参数之前，将梯度除以相同的缩放因子，以恢复原始的梯度值。

### 代码解释
在你的代码中，`self.scaler.scale(loss).backward()` 这一行代码实现了损失缩放：

```python
with autocast('cuda'):  # 启用自动混合精度
    outputs = self.model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
    loss = outputs.loss
self.scaler.scale(loss).backward()  # 缩放损失并反向传播
```


- `autocast('cuda')`：启用自动混合精度，将部分操作使用半精度浮点数进行计算。
- `self.scaler.scale(loss)`：将损失值乘以缩放因子。
- `.backward()`：使用缩放后的损失值进行反向传播，计算梯度。

### 更新参数
在反向传播之后，使用 `self.scaler.step(optimizer)` 和 `self.scaler.update()` 来更新模型参数并调整缩放因子：

```python
self.scaler.step(self.optimizer)  # 更新参数
self.scaler.update()  # 更新缩放器
```


### 总结
缩放损失在反向传播中的作用是防止梯度下溢，确保梯度值在半精度浮点数范围内保持稳定，从而提高模型训练的稳定性和效率。
