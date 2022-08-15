from torch import Tensor
from typing import Optional
from torch.utils.checkpoint import checkpoint as gradient_checkpoint

import math
import torch
import torch.nn as nn


__all__ = [
    "MultiheadAttention"
]


class MultiheadAttention(nn.Module):
    def __init__(self, embed_dim: int, num_heads: int, bias: bool = True):
        super().__init__()

        self.num_heads = num_heads
        self.embed_dim = embed_dim
        self.proj_q = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.proj_k = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.proj_v = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.proj_out = nn.Linear(embed_dim, embed_dim, bias=bias)

    def forward(self, query: Tensor, key: Tensor, value: Tensor, query_chunk_size: Optional[int] = 1024):
        """
        Forward method
        :param query: tensor of shape [B, T, C]
        :param key: tensor of shape [B, S, C]
        :param value: tensor of shape [B, S, C]
        :param query_chunk_size: size of query chunk used for splitting the attention op
        :return: output tensor
        """
        assert key.shape == value.shape, f"Key and value must have the same shape, but got {key.shape} and {value.shape}"
        assert key.shape[0] == query.shape[0], f"Key/value and query must have the same size at dims 0"
        assert key.shape[2] == query.shape[2], f"Key/value and query must have the same size at dims 2"

        def fn(x: Tensor, proj_layer: nn.Linear):
            batch_sz, seq_len = x.shape[:2]
            x = proj_layer(x)  # [B, N, C]
            x = x.reshape(batch_sz, seq_len, self.num_heads, -1)  # [B, N, H, C']
            return x.permute(0, 2, 1, 3).contiguous()

        query, key, value = [
            fn(x, layer)
            for x, layer in zip([query, key, value], [self.proj_q, self.proj_k, self.proj_v])
        ]

        output = _efficient_attention(query, key, value, query_chunk_size)  # [B, H, N, C']
        output = output.permute(0, 2, 1, 3).flatten(2).contiguous()  # [B, N, C]
        return self.proj_out(output)


def _summarize_chunk(q, k, v):
    attn_wts = torch.einsum("bhqd,bhkd->bhqk", q, k)  # [batch_sz, n_heads, tgt_len, src_len]
    max_score = attn_wts.max(-1, keepdim=True)[0].detach()  # [batch_sz, n_heads, tgt_len, 1]
    exp_wts = (attn_wts - max_score).exp()  # [batch_sz, n_heads, tgt_len, src_len]
    exp_values = torch.einsum("bhvd,bhqv->bhqd", v, exp_wts)  # [batch_sz, n_heads, tgt_len, n_dims]

    return exp_values, exp_wts.sum(-1), max_score.squeeze(-1)


def _query_chunk_attention(query: Tensor, key: Tensor, value: Tensor, key_chunk_size: Optional[int] = 4096):
    batch_sz, n_heads, src_len, n_dims = key.shape
    query = query / math.sqrt(n_dims)

    chunk_values, chunk_wts, chunk_max = [], [], []

    key = key.split(key_chunk_size, 2)
    value = value.split(key_chunk_size, 2)

    for key_chunk, value_chunk in zip(key, value):
        # chunk_output = summarize_chunk(query, key_chunk, value_chunk)
        chunk_output = gradient_checkpoint(_summarize_chunk, query, key_chunk, value_chunk)

        chunk_values.append(chunk_output[0])
        chunk_wts.append(chunk_output[1])
        chunk_max.append(chunk_output[2])

    chunk_max = torch.stack(chunk_max, 0)  # [#chunks, batch_sz, n_heads, tgt_len]
    global_max = chunk_max.max(0, keepdim=True)[0]  # [1, batch_sz, n_heads]
    max_diffs = (chunk_max - global_max).exp()  # [#chunks, batch_sz, n_heads, tgt_len]

    chunk_values = torch.stack(chunk_values, 0)  # [#chunks, batch_sz, n_heads, tgt_len, n_dims]
    chunk_values = chunk_values * max_diffs.unsqueeze(-1)

    chunk_wts = torch.stack(chunk_wts, 0)  # [#chunks, batch_sz, n_heads, tgt_len]
    chunk_wts = chunk_wts * max_diffs

    all_values = chunk_values.sum(0)  # [batch_sz, n_heads, tgt_len, n_dims]
    all_wts = chunk_wts.sum(0).unsqueeze(-1)  # [batch_sz, n_heads, tgt_len, 1]

    return all_values / all_wts  # [batch_sz, n_heads, tgt_len, n_dims]


def _efficient_attention(query: Tensor, key: Tensor, value: Tensor, query_chunk_size: int):
    """
    Efficient implementation of multi-head attention op
    :param query: tensor of shape [batch_size, n_heads, tgt_len, n_dims]
    :param key: tensor of shape [batch_size, n_heads, src_len, n_dims]
    :param value: tensor of shape [batch_size, n_heads, src_len, n_dims]
    :param query_chunk_size: int
    :return:
    """
    query = query.split(query_chunk_size, 2)

    attn_output = [_query_chunk_attention(query_chunk, key, value) for query_chunk in query]
    attn_output = torch.cat(attn_output, 2)
    return attn_output


def _test():
    def vanilla_attn(query, key, value):
        bs, nh = query.shape[:2]
        query = query.flatten(0, 1)
        key = key.flatten(0, 1)
        value = value.flatten(0, 1)

        dotprod = torch.bmm(query, key.transpose(1, 2))  # [batch_sz * n_heads, tgt_len, src_len]
        # x = dotprod.softmax(2) / math.sqrt(query.shape[-1])  # [batch_sz * n_heads, tgt_len, src_len]
        x = (dotprod / math.sqrt(query.shape[-1])).softmax(2)  # [batch_sz * n_heads, tgt_len, src_len]
        output = torch.bmm(x, value)

        return output.reshape(bs, nh, *output.shape[1:])

    BATCH_SIZE = 2
    NUM_HEADS = 4
    SRC_SEQ_LEN = 128
    TGT_SEQ_LEN = 128
    EMBED_DIM = 256

    TOTAL_RUNS = 100
    ABSOLUTE_TOLERANCE = 1e-3
    RELATIVE_TOLERANCE = 1e-4

    num_failed_runs = 0

    attn = MultiheadAttention(EMBED_DIM, NUM_HEADS).cuda()

    for _ in range(TOTAL_RUNS):
        query = torch.normal(0., 1., (BATCH_SIZE, NUM_HEADS, TGT_SEQ_LEN, EMBED_DIM)).cuda()
        key = torch.normal(0., 1., (BATCH_SIZE, NUM_HEADS, SRC_SEQ_LEN, EMBED_DIM)).cuda()
        value = torch.normal(0., 1., (BATCH_SIZE, NUM_HEADS, SRC_SEQ_LEN, EMBED_DIM)).cuda()

        attn(query=torch.rand(2, 10, 256).cuda(), key=torch.rand(2, 20, 256).cuda(), value=torch.rand(2, 20, 256).cuda())

        efficient_output = _efficient_attention(query, key, value, query_chunk_size=1024)
        vanilla_output = vanilla_attn(query, key, value)

        num_failed_runs += not torch.allclose(
            efficient_output, vanilla_output,
            rtol=RELATIVE_TOLERANCE,
            atol=ABSOLUTE_TOLERANCE
        )

    print(f"Number of failed runs: {num_failed_runs}/{TOTAL_RUNS}")


if __name__ == '__main__':
    _test()
