DSA 的思路可以简单理解成两层结构：有一个 Lightning Indexer，跑在 MLA 的压缩表示上，给历史 token 打一个相关性分数，这一步本身还是 O(L²)，但是非常轻，支持 FP8，算力基本可以忽略。后面有一个 Fine‑grained Token Selector，根据打分做 top‑k 选择，只保留大约 2048 个最相关 token 进入真正的注意力计算。
这样主注意力的复杂度就从 O(L²) 变成了 O(L·k)，k 是固定常数，和 L 无关。
 
这里面有两个细节：一个是 DSA 分两阶段训练——**先 dense warm‑up 只训 Indexer，让它的分布对齐原始注意力，然后再做 sparse training，让主模型适应稀疏模式。**