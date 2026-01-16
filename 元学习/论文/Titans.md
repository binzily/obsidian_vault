```python
# TITANS — 清晰伪代码
# 输入:
#   tokens[1..T] : 一个序列（语言/时间序列都可）
#   mode         : "MAC" | "MAG" | "MAL"   (三种 Titans 变体)
#   W            : 短期窗口大小（core 只看最近 W 个 token）
# 模块/参数:
#   Embed(.)     : 把 token 变成向量 x_t
#   Core(.)      : 你的主干（通常是有限窗口注意力 + FFN；注意力只看窗口内）
#   M_theta(.)   : 长期记忆网络（一个深一点的 memory 网络；参数 theta）
#   P[1..p]      : persistent memory（p 个“固定前缀 token”，训练好后推理时不更新）
#   WQ, WK, WV   : 三个线性投影（把 x_t 投影成 query/key/value）
#   u            : 动量缓存（和 theta 同形状，初始为 0）
# 超参:
#   mu           : 动量系数（如 0.9）
#   eta          : 基础学习率（写入强度）
#   wd           : weight decay / forgetting 强度（遗忘强度）
#
# 关键约定:
#   - “读记忆” = 对 M_theta 做一次 forward，不更新参数（retrieve 时不训练）
#   - “写记忆” = 只更新 theta（长期记忆网络参数），不更新 Core 的参数
#   - persistent memory P 在测试/推理时固定不变

init window = empty queue   # 保存最近 W 个向量（短期上下文）
init u = 0                  # 动量缓存

for t in 1..T:

  # ===== (0) token -> 向量 =====
  x = Embed(tokens[t])      # x = x_t

  # ===== (1) 读长期记忆 retrieve（只 forward，不更新）=====
  q   = x * WQ              # query
  mem = M_theta(q)          # mem 是“从长期记忆读出的摘要/提示”
                            # 注意：这里不更新 theta

  # ===== (2) 用短期 core 处理当前（有限窗口注意力）=====
  # persistent memory 的作用：把 P 当成“永远可见的前缀 token”
  # core 的作用：只在“最近 W 个 token（+前缀/摘要）”上做注意力，省计算

  if mode == "MAC":         # Memory as Context：把记忆当“上下文 token”拼进去
      # core_input 是一个 token 序列： [P 前缀] + [mem 摘要] + [window 最近W] + [x 当前]
      core_input = concat(P, mem, window, x)
      y = Core(core_input)  # 产出当前步的输出表示/logits（实现里通常是因果注意力取最后位置）

  else if mode == "MAG":    # Memory as Gated branch：core 和 memory 并行，再用门融合
      core_input = concat(P, window, x)
      core_out   = Core(core_input)          # 短期精确推理
      y          = Gate(core_out, mem)       # 用门把“短期输出”和“长期摘要”融合

  else if mode == "MAL":    # Memory as Layer：先过“记忆层”压缩，再进 core
      z = MemoryLayer(mem, x)                # 把 mem 作为一层，对当前表示做变换/压缩
      core_input = concat(P, window, z)
      y = Core(core_input)

  # ===== (3) 计算 surprise：新信息是否“值得记”=====
  # 论文思想：越“违背预期/越新奇”的 token，梯度越大 => 更值得写入长期记忆
  k    = x * WK
  v    = x * WV
  pred = M_theta(k)                           # 记忆网络对 key 的“联想输出”
  loss = MSE(pred, v)                         # 联想记忆的训练目标（让 key -> value）
  g    = grad(loss, theta)                    # 对记忆参数 theta 求梯度
  s    = norm(g)                              # surprise 分数：梯度越大越“惊讶/重要”

  # ===== (4) 写入长期记忆：带动量 + 带遗忘（只更新 theta）=====
  u     = mu * u + g                          # 动量：让“惊讶”在时间上有延续
  theta = (1 - wd * f(s)) * theta - eta * h(s) * u
          # 遗忘/衰减：控制容量，避免“什么都记导致溢出”
          # f(s), h(s) 是把 surprise 映射到“忘多少/写多狠”的函数：
          #   - s 大：h(s) 更大（写得更狠），f(s) 更小（少忘点）
          #   - s 小：h(s) 很小（几乎不写），f(s) 更大（更快淡忘不重要的）

  # ===== (5) 更新短期窗口 =====
  push(window, x)           # 或 push(window, y) 取决于实现（用输入态 or 隐状态）
  if size(window) > W:
      pop_left(window)

return outputs/logits
```

Q1：Titan的大模型在训练时，要训练MAC、MAL、MAG吗？是混在一起训练还是单独训出三个模型比如：Titans-MAC、Titans-MAL、Titans-MAG？
答：论文先说“提出三种 Titans 变体”，并强调每种都有不同 trade-off（权衡）。论文里确实是“**Titans-MAC**”、“**Titans-MAG**”、“**Titans-MAL**”（再加 Titans(LMM)）这种“分别训练、分别评测”的做法，而不是一个模型里混着三种模式随时切换。

Q2：为什么一般不会“混在一起训练成一个可切换模型”？
因为三者的计算图/接口不一样：
- **MAC** 是把 `mem` 当成“额外上下文 token”拼进注意力输入（还会分段 chunk）。
- **MAG** 是“滑窗注意力一条支路 + 记忆网络一条支路”，最后 gate 融合。
- **MAL** 是把记忆当成一层（layer）插在注意力前/中间。



