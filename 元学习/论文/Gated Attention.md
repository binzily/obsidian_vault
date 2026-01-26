```python
# Shapes (typical):
# X:      [B, T, D]          # 当前层输入 hidden states
# WQ,WK,WV: [D, H*Dh]
# WO:     [H*Dh, D]
# WG_head: [D, H]            # headwise gate (每个 head 一个标量)
# (elementwise 版本可用 WG_elem: [D, H*Dh])

def gated_attention_block(X, attn_mask=None):
    Xn = RMSNorm(X)                           # paper uses pre-norm hidden states as gate input X :contentReference[oaicite:1]{index=1}

    # QKV projections
    Q = (Xn @ WQ).reshape(B, T, H, Dh).transpose(1, 2)  # [B, H, T, Dh]
    K = (Xn @ WK).reshape(B, T, H, Dh).transpose(1, 2)  # [B, H, T, Dh]
    V = (Xn @ WV).reshape(B, T, H, Dh).transpose(1, 2)  # [B, H, T, Dh] :contentReference[oaicite:2]{index=2}

    # SDPA
    scores = (Q @ K.transpose(-1, -2)) / sqrt(Dh)       # [B, H, T, T]
    if attn_mask is not None:
        scores += attn_mask
    A = softmax(scores, dim=-1)                          # [B, H, T, T]
    Y = A @ V                                            # [B, H, T, Dh]  (SDPA output) :contentReference[oaicite:3]{index=3}

    # G1 gate: head-specific sigmoid gate AFTER SDPA output
    gate = sigmoid(Xn @ WG_head)                         # [B, T, H]      (headwise gate) :contentReference[oaicite:4]{index=4}
    gate = gate.transpose(1, 2).unsqueeze(-1)            # [B, H, T, 1]
    Y = Y * gate                                         # [B, H, T, Dh]  (modulate each head) :contentReference[oaicite:5]{index=5}

    # concat heads + output projection
    Ycat = Y.transpose(1, 2).reshape(B, T, H*Dh)          # [B, T, H*Dh]
    O = Ycat @ WO                                         # [B, T, D]
    return O

# elementwise gate 版本（论文也比较过）：
# gate = sigmoid(Xn @ WG_elem).reshape(B,T,H,Dh).transpose(1,2)  # [B,H,T,Dh]
# Y = Y * gate
```
gated attention 做的事非常简单：**在标准 softmax attention 的输出后面，加一个“门”把输出再筛一遍。** PS:headwise（每头一个标量） 和 elementwise（每维一个门）。

* 没有gate时： (把注意力权重 $A = \text{softmax}(QK^\top)$ 当成已算好的系数)：$$\text{out} = (A (X W_V)) W_O$$这条从$X \to W_V \to W_O$的通路，在$A$固定时就是“线性的”，而$W_V$把$D$压到$H \cdot D_h$再由$W_O$拉回$D$，所以是“低秩瓶颈”的线性变换。
* 加了gate后：$$\text{out} = \left(A (X W_V) \odot \sigma(X W_G)\right) W_O$$ 这里多了sigmoid和逐元素乘法(⊙)，于是这条通路不能再合并成一个线性矩阵了——这就是“把非线性插进那条低秩线性链路里”。
* 论文发现 gate 分数往往很“稀疏”（很多接近 0），并且能缓解 **attention sink**（前几个 token 异常吸走大量注意力）从而更利于长上下文外推。
