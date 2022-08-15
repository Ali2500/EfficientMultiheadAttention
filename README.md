# Efficient Multihead Attention

## PyTorch implementation of "Self-attention Does Not Need O(n2) Memory"

This one-file repo provides a PyTorch implementation of the work by Rabe et al which provides code in JAX: https://arxiv.org/abs/2112.05682

The attention operation coded here is identical to the [standard multi-head attention proposed by Vaswani et al.](https://arxiv.org/abs/1706.03762?context=cs), but uses some mathematical tricks and gradient checkpointing to process the input features in chunks, thereby significantly reducing memory overhead.
