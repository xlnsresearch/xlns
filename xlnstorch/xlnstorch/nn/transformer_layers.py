from typing import Optional
import torch
from xlnstorch import LNSTensor, zeros, empty, LNS_NEG_INF
import xlnstorch.nn as nn
import xlnstorch.nn.init as init
from . import LNSModule

class LNSMultiheadAttention(LNSModule):
    """
    An LNS multi-head attention layer.

    See also: :py:class:`torch.nn.MultiheadAttention`

    Parameters
    ----------
    embed_dim : int
        Total dimension of the model.
    num_heads : int
        Number of parallel attention heads.
    dropout : float, optional
        Dropout probability on attention weights. Default: 0.0.
    bias : bool, optional
        If specified, add bias to the projection layers. Default: True.
    add_bias_kv : bool, optional
        If specified, add bias to the key and value sequences at dim=0. Default: False.
    add_zero_attn : bool, optional
        If specified, add a new batch of zeros to the key and value sequences at dim=1. Default: False.
    kdim : int, optional
        Total number of features in key. Default: None (uses `embed_dim`).
    vdim : int, optional
        Total number of features in value. Default: None (uses `embed_dim`).
    batch_first : bool, optional
        If True, then the input and output tensors are provided as (batch, seq, features). Default: False (seq, batch, features).

    Attributes
    ----------
    q_proj : LNSLinear
        Linear layer to project the queries.
    k_proj : LNSLinear
        Linear layer to project the keys.
    v_proj : LNSLinear
        Linear layer to project the values.
    out_proj : LNSLinear
        Linear layer to project the output.
    bias_k : LNSTensor
        Bias for the key sequence to be added at dim=0.
    bias_v : LNSTensor
        Bias for the value sequence to be added at dim=0.
    attn_dropout : torch.nn.Dropout
        Dropout layer on attention weights.
    """

    def __init__(
            self,
            embed_dim: int,
            num_heads: int,
            dropout: float = 0.0,
            bias: bool = True,
            add_bias_kv: bool = False,
            add_zero_attn: bool = False,
            kdim: Optional[int] = None,
            vdim: Optional[int] = None,
            batch_first: bool = False,
    ):
        super().__init__()

        if embed_dim % num_heads != 0:
            raise ValueError("embed_dim must be divisible by num_heads.")

        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.scaling = self.head_dim ** -0.5
        self.dropout = dropout
        self.add_zero_attn = add_zero_attn
        self.kdim = kdim if kdim is not None else embed_dim
        self.vdim = vdim if vdim is not None else embed_dim
        self.batch_first = batch_first

        self.q_proj = nn.LNSLinear(embed_dim, embed_dim, bias=bias)
        self.k_proj = nn.LNSLinear(self.kdim, embed_dim, bias=bias)
        self.v_proj = nn.LNSLinear(self.vdim, embed_dim, bias=bias)
        self.out_proj = nn.LNSLinear(embed_dim, embed_dim, bias=bias)

        for proj in (self.q_proj, self.k_proj, self.v_proj, self.out_proj):
            init.xavier_uniform_(proj.weight)
            if proj.bias is not None:
                init.zeros_(proj.bias)

        if add_bias_kv:

            bias_k = empty(1, 1, embed_dim)
            bias_v = empty(1, 1, embed_dim)
            init.xavier_normal_(bias_k)
            init.xavier_normal_(bias_v)

            self.register_parameter("bias_k", bias_k)
            self.register_parameter("bias_v", bias_v)

        else:
            self.bias_k = None
            self.bias_v = None

        self.attn_dropout = torch.nn.Dropout(dropout)

    def _shape(self, x: LNSTensor, B: int):
        """
        (B, L, D) -> (B, num_heads, L, head_dim)
        """
        return x.view(B, -1, self.num_heads, self.head_dim).transpose(1, 2).contiguous()

    def forward(
            self,
            query: LNSTensor,
            key: LNSTensor,
            value: LNSTensor,
            key_padding_mask: Optional[LNSTensor] = None,
            need_weights: bool = True,
            attn_mask: Optional[LNSTensor] = None,
            average_attn_weights: bool = True,
    ):
        assert attn_mask is None, "Attn mask not supported yet."
        assert key_padding_mask is None, "Key padding mask not supported yet."

        if self.batch_first:
            query = query.transpose(0, 1)
            key = key.transpose(0, 1)
            value = value.transpose(0, 1)

        Lq, Nt, _ = query.shape
        Sk = key.shape[0]

        q = self.q_proj(query) * self.scaling
        k = self.k_proj(key)
        v = self.v_proj(value)

        if self.bias_k is not None and self.bias_v is not None:
            k = torch.cat([k, self.bias_k.repeat(1, Nt, 1)], dim=0)
            v = torch.cat([v, self.bias_v.repeat(1, Nt, 1)], dim=0)

            if attn_mask is not None:
                attn_mask = torch.nn.functional.pad(attn_mask, (0, 1))

            if key_padding_mask is not None:
                key_padding_mask = torch.nn.functional.pad(key_padding_mask, (0, 1))

        if self.add_zero_attn:
            zero = zeros(1, Nt, self.embed_dim)
            k = torch.cat([k, zero], dim=0)
            v = torch.cat([v, zero], dim=0)

            if attn_mask is not None:
                attn_mask = torch.nn.functional.pad(attn_mask, (0, 1))

            if key_padding_mask is not None:
                key_padding_mask = torch.nn.functional.pad(key_padding_mask, (0, 1))

        q = self._shape(q.transpose(0, 1), Nt) # (N, h, Lq, d)
        k = self._shape(k.transpose(0, 1), Nt) # (N, h, Sk, d)
        v = self._shape(v.transpose(0, 1), Nt) # (N, h, Sk, d)

        attn_scores = torch.matmul(q, k.transpose(-2, -1))

        if attn_mask is not None:
            if attn_mask.dim() == 2:
                attn_mask = attn_mask.unsqueeze(0).unsqueeze(0) # (1, 1, Lq, Sk)
            elif attn_mask.dim() == 3: # (N or 1, Lq, Sk)
                attn_mask = attn_mask.unsqueeze(1) # (N, 1, Lq, Sk)

            attn_scores += attn_mask

        if key_padding_mask is not None:
            attn_scores = torch.where(
                key_padding_mask.unsqueeze(1).unsqueeze(2), # (N, Sk) -> (N, 1, 1, Sk)
                LNS_NEG_INF,
                attn_scores,
            )

        attn_weights = torch.nn.functional.softmax(attn_scores, dim=-1)
        attn_weights = self.attn_dropout(attn_weights)

        attn_output = torch.matmul(attn_weights, v)
        attn_output = attn_output.transpose(1, 2).contiguous().view(Nt, Lq, self.embed_dim)
        attn_output = self.out_proj(attn_output)

        if not self.batch_first:
            attn_output = attn_output.transpose(0, 1)

        if need_weights:

            if average_attn_weights:
                attn_weights = attn_weights.sum(dim=1) / self.num_heads

            return attn_output, attn_weights

        return attn_output, None