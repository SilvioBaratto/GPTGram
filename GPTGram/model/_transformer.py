import math
import inspect
import torch
import torch.nn as nn
from torch.nn import functional as F
from transformers import GPT2LMHeadModel
from ..config import Config as cfg

class LayerNorm(nn.Module):
    """
    Implements a Layer Normalization module with an optional bias.

    PyTorch's LayerNorm does not support `bias=False`, which is why we have this custom implementation.

    Attributes
    ----------
    weight : torch.nn.Parameter
        A learnable weight parameter initialized with ones.
    bias : torch.nn.Parameter
        A learnable bias parameter initialized with zeros if `cfg.gpt.bias` is True.

    Methods
    -------
    forward(input: torch.Tensor) -> torch.Tensor
        Applies layer normalization on the input tensor.
    """

    def __init__(self):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(cfg.gpt.n_embd))
        self.bias = nn.Parameter(torch.zeros(cfg.gpt.n_embd)) if cfg.gpt.bias else None

    def forward(self, input):
        return F.layer_norm(input, self.weight.shape, self.weight, self.bias, 1e-5)

class CausalSelfAttention(nn.Module):
    """
    Implements a causal self-attention module. This module is designed to only attend to earlier positions in the sequence.

    Attributes
    ----------
    c_attn : torch.nn.Linear
        Linear layer for key, query, value projections.
    c_proj : torch.nn.Linear
        Linear layer for output projection.
    attn_dropout : torch.nn.Dropout
        Dropout layer for attention.
    resid_dropout : torch.nn.Dropout
        Dropout layer for residuals.
    flash : bool
        Indicates whether PyTorch's `scaled_dot_product_attention` is available. 
        If True, uses this for efficient attention; otherwise, uses manual implementation.
    bias : torch.Tensor
        Causal mask to ensure attention is only applied to earlier positions in the sequence.

    Methods
    -------
    forward(x: torch.Tensor) -> torch.Tensor
        Applies causal self-attention on the input tensor.
    """

    def __init__(self):
        super().__init__()
        assert cfg.gpt.n_embd % cfg.gpt.n_head == 0
        # key, query, value projections for all heads, but in a batch
        self.c_attn = nn.Linear(cfg.gpt.n_embd, 3 * cfg.gpt.n_embd, bias=cfg.gpt.bias)
        # output projection
        self.c_proj = nn.Linear(cfg.gpt.n_embd, cfg.gpt.n_embd, bias=cfg.gpt.bias)
        # regularization
        self.attn_dropout = nn.Dropout(cfg.gpt.dropout)
        self.resid_dropout = nn.Dropout(cfg.gpt.dropout)

        # flash attention make GPU go brrrrr but support is only in PyTorch >= 2.0
        self.flash = hasattr(torch.nn.functional, 'scaled_dot_product_attention')
        if not self.flash:
            print("WARNING: using slow attention. Flash Attention requires PyTorch >= 2.0")
            # causal mask to ensure that attention is only applied to the left in the input sequence
            self.register_buffer("bias", torch.tril(torch.ones(cfg.gpt.block_size, cfg.gpt.block_size))
                                        .view(1, 1, cfg.gpt.block_size, cfg.gpt.block_size))

    def forward(self, x):
        B, T, C = x.size() # batch size, sequence length, embedding dimensionality (n_embd)

        # calculate query, key, values for all heads in batch and move head forward to be the batch dim
        q, k, v  = self.c_attn(x).split(cfg.gpt.n_embd, dim=2)
        k = k.view(B, T, cfg.gpt.n_head, C // cfg.gpt.n_head).transpose(1, 2) # (B, nh, T, hs)
        q = q.view(B, T, cfg.gpt.n_head, C // cfg.gpt.n_head).transpose(1, 2) # (B, nh, T, hs)
        v = v.view(B, T, cfg.gpt.n_head, C // cfg.gpt.n_head).transpose(1, 2) # (B, nh, T, hs)

        # causal self-attention; Self-attend: (B, nh, T, hs) x (B, nh, hs, T) -> (B, nh, T, T)
        if self.flash:
            # efficient attention using Flash Attention CUDA kernels
            y = torch.nn.functional.scaled_dot_product_attention(q, k, v, 
                                                                 attn_mask=None, 
                                                                 dropout_p=self.dropout if self.training else 0, 
                                                                 is_causal=True)
        else:
            # manual implementation of attention
            att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
            att = att.masked_fill(self.bias[:,:,:T,:T] == 0, float('-inf'))
            att = F.softmax(att, dim=-1)
            att = self.attn_dropout(att)
            y = att @ v # (B, nh, T, T) x (B, nh, T, hs) -> (B, nh, T, hs)
        y = y.transpose(1, 2).contiguous().view(B, T, C) # re-assemble all head outputs side by side

        # output projection
        y = self.resid_dropout(self.c_proj(y))
        return y

class MLP(nn.Module):
    """
    Implements a multi-layer perceptron (MLP) module.

    Attributes
    ----------
    c_fc : torch.nn.Linear
        Linear layer for the first fully connected layer.
    gelu : torch.nn.GELU
        GELU activation function.
    c_proj : torch.nn.Linear
        Linear layer for projection.
    dropout : torch.nn.Dropout
        Dropout layer.

    Methods
    -------
    forward(x: torch.Tensor) -> torch.Tensor
        Applies MLP on the input tensor.
    """

    def __init__(self):
        super().__init__()
        self.c_fc    = nn.Linear(cfg.gpt.n_embd, 4 * cfg.gpt.n_embd, bias=cfg.gpt.bias)
        self.gelu    = nn.GELU()
        self.c_proj  = nn.Linear(4 * cfg.gpt.n_embd, cfg.gpt.n_embd, bias=cfg.gpt.bias)
        self.dropout = nn.Dropout(cfg.gpt.dropout)

    def forward(self, x):
        x = self.c_fc(x)
        x = self.gelu(x)
        x = self.c_proj(x)
        x = self.dropout(x)
        return x

class Block(nn.Module):
    """
    Implements a transformer block module.

    Attributes
    ----------
    ln_1 : LayerNorm
        Layer normalization.
    attn : CausalSelfAttention
        Causal self-attention.
    ln_2 : LayerNorm
        Layer normalization.
    mlp : MLP
        Multi-layer perceptron (MLP).

    Methods
    -------
    forward(x: torch.Tensor) -> torch.Tensor
        Applies transformer block operations on the input tensor.
    """

    def __init__(self):
        super().__init__()
        self.ln_1 = LayerNorm()
        self.attn = CausalSelfAttention()
        self.ln_2 = LayerNorm()
        self.mlp = MLP()

    def forward(self, x):
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x
