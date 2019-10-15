"""
OpenAI's GPT-2 ported to PyTorch.
"""
import math

import attr
import torch
from torch import nn
from torch.nn import functional as F
import torch.utils.checkpoint

@attr.s(auto_attribs=True, frozen=True)
class HParams:
    n_vocab: int
    n_ctx: int
    n_embed: int
    n_hidden: int
    n_head: int
    n_layer: int
    gradient_checkpointing: bool
    share_parameters: bool

class Model(nn.Module):
    def __init__(self, hparams: HParams):
        super().__init__()
        self.hparams = hparams
        self.wpe = nn.Embedding(hparams.n_ctx, hparams.n_embed)
        nn.init.normal_(self.wpe.weight, std=0.01)
        self.wte = nn.Embedding(hparams.n_vocab, hparams.n_embed)
        nn.init.normal_(self.wte.weight, std=0.02)
        if self.hparams.share_parameters:
            b = Block(hparams)
            self.blocks = nn.ModuleList(
                [b for _ in range(hparams.n_layer)])
        else:
            self.blocks = nn.ModuleList(
                [Block(hparams) for _ in range(hparams.n_layer)])

        self.ln_f = Norm(self.hparams.n_hidden)
        if hparams.n_hidden != hparams.n_embed:
            self.in_proj = Conv1D(hparams.n_embed, hparams.n_hidden)
            self.out_proj = Conv1D(hparams.n_hidden, hparams.n_embed)
        else:
            self.in_proj = self.out_proj = None

    def forward(self, x, past=None):

        # Embedding
        past_length = 0 if past is None else past.shape[-2]                     # past_length = 0
        batch_size, n_ctx = x.shape                                             # batch_size=8, n_ctx=128
        position = position_for(batch_size, n_ctx, past_length, x.device)       # position: [batch_size=8, n_ctx=128]
        h = self.wte(x) + self.wpe(position)                                    # h: [batch_size=8, n_ctx=128, n_embed=128]

        assert h.shape == (batch_size, n_ctx, self.hparams.n_embed)
        if self.in_proj:
            h = self.in_proj(h)                                                 # h: [batch_size=8, n_ctx=128, n_hidden=512]
        # Transformer
        presents = []
        for i, block in enumerate(self.blocks):
            if self.hparams.gradient_checkpointing:
                h, present = torch.utils.checkpoint.checkpoint(block, h, past[:, i] if past is not None else None)
            else:
                h, present = block(h, past=past[:, i] if past is not None else None)
                                                                                # h: [batch_size=8, n_ctx=128, n_hidden=512]
                                                                                # present: [batch_size=8, 2[k,v], n_head=8, n_ctx=128, 64]
            presents.append(present)

        # presents: 12 x [batch_size=8, 2[k,v], n_head=8, n_ctx=128, 64]

        h = self.ln_f(h)                                                        # h: [batch_size=8, n_ctx=128, n_hidden=512]
        if self.out_proj:
            h = self.out_proj(h)                                                # h: [batch_size=8, n_ctx=128, n_embed=128]
        # Output logits
        h_flat = h.reshape([batch_size * n_ctx, self.hparams.n_embed])          # [1024, 128]
        logits = torch.matmul(h_flat, self.wte.weight.t())                      # [1024, 50000]
        logits = logits.reshape([batch_size, n_ctx, self.hparams.n_vocab])      # [8, 128, 50000]

        return {
            'presents': torch.stack(tuple(presents), dim=1),
            'logits': logits,
        }


class Block(nn.Module):
    def __init__(self, hparams: HParams):
        super().__init__()
        self.ln_1 = Norm(hparams.n_hidden)
        self.ln_2 = Norm(hparams.n_hidden)
        self.mlp = MLP(hparams.n_hidden, hparams.n_hidden * 4)
        self.attn = Attention(hparams)

    def forward(self, x, past):                                                 # x: [batch_size=8, n_ctx=128, n_hidden=512]
                                                                                # past: None

        a, present = self.attn(self.ln_1(x), past=past)                         # a: [batch_size=8, n_ctx=128, n_hidden=512]
                                                                                # present: [batch_size=8, 2[k,v], n_head=8, n_ctx=128, 64]

        x = x + a                                                               # x: [batch_size=8, n_ctx=128, n_hidden=512]
        m = self.mlp(self.ln_2(x))                                              # m: [batch_size=8, n_ctx=128, n_hidden=512]
        x = x + m                                                               # x: [batch_size=8, n_ctx=128, n_hidden=512]
        return x, present


class Norm(nn.Module):
    """ Normalize to mean = 0, std = 1, then do a diagonal affine transform.
    """
    def __init__(self, n_features, *, dim=-1, epsilon=1e-5):
        super().__init__()
        self.n_features = n_features
        self.dim = dim
        self.epsilon = epsilon
        self.g = nn.Parameter(torch.ones(n_features))
        self.b = nn.Parameter(torch.zeros(n_features))

    def forward(self, x):
        assert x.shape[-1] == self.n_features
        u = torch.mean(x, dim=self.dim, keepdim=True)
        xmu = x - u
        s = torch.mean(xmu * xmu, dim=self.dim, keepdim=True)
        return xmu * torch.rsqrt(s + self.epsilon) * self.g + self.b


class MLP(nn.Module):
    def __init__(self, n_features, n_hidden):
        super().__init__()
        self.c_fc = Conv1D(n_features, n_hidden)
        self.c_proj = Conv1D(n_hidden, n_features)

    def forward(self, x):
        x = gelu(self.c_fc(x))
        x = self.c_proj(x)
        return x


class Attention(nn.Module):
    def __init__(self, hparams: HParams):
        super().__init__()
        assert hparams.n_hidden % hparams.n_head == 0
        self.hparams = hparams
        self.c_attn = Conv1D(hparams.n_hidden, hparams.n_hidden * 3)
        self.c_proj = Conv1D(hparams.n_hidden, hparams.n_hidden)

    def forward(self, x, past):                                                 # x: [batch_size=8, n_ctx=128, n_hidden=512]
                                                                                # past: None

        assert len(x.shape) == 3
        assert x.shape[-1] == self.hparams.n_hidden
        if past is not None:
            # Should be [batch, 2, heads, sequence, features], where 2 is [k, v]
            assert len(past.shape) == 5
            # assert past.shape[-1] == self.hparams.n_hidden
        c = self.c_attn(x)                                                      # c: [batch_size=8, n_ctx=128, 3*n_hidden=1536]
        q, k, v = map(self.split_heads, torch.split(c, x.shape[-1], dim=2))     # q: [batch_size=8, n_head=8, n_ctx=128, 64]
                                                                                # k: [batch_size=8, n_head=8, n_ctx=128, 64]
                                                                                # v: [batch_size=8, n_head=8, n_ctx=128, 64]
        present = torch.stack([k, v], dim=1)                                    # present: [batch_size=8, 2[k,v], n_head=8, n_ctx=128, 64]
        if past is not None:
            pk, pv = past[:, 0], past[:, 1]
            k = torch.cat([pk, k], dim=-2)
            v = torch.cat([pv, v], dim=-2)
        a = self.multihead_attn(q, k, v)                                        # a: [batch_size=8, 2[k,v], n_head=8, n_ctx=128, 64]
        a = self.merge_heads(a)                                                 # a: [batch_size=8, n_ctx=128, n_hidden=512]
        a = self.c_proj(a)                                                      # a: [batch_size=8, n_ctx=128, n_hidden=512]
        return a, present

    def split_heads(self, x):
        """ From [batch, sequence, features] to
        [batch, heads, sequence, features].
        """
        return self.split_states(x, self.hparams.n_head).permute(0, 2, 1, 3)

    @staticmethod
    def split_states(x, n):
        """ Reshape the last dimension of x into [n, x.shape[-1]/n].
        """
        *start, m = x.shape
        return x.reshape(start + [n, m // n])

    def merge_heads(self, x):
        """ Reverse of split_heads.
        """
        return self.merge_states(x.permute(0, 2, 1, 3))

    @staticmethod
    def merge_states(x):
        """ Smash the last two dimensions of x into a single dimension.
        """
        *start, a, b = x.shape
        return x.reshape(start + [a * b])

    def mask_attn_weights(self, w):
        # w has shape [batch, heads, dst_sequence, src_sequence],
        # where information flows from src to dst.
        _, _, nd, ns = w.shape
        b = self.attention_mask(nd, ns, dtype=w.dtype, device=w.device)
        b = b.reshape((1, 1, nd, ns))
        w = w * b - 1e10 * (1 - b)
        return w

    @staticmethod
    def attention_mask(nd, ns, *, dtype, device=None):
        """ 1's in the lower triangle, counting from the lower right corner.
        Same as tf.matrix_band_part(tf.ones([nd, ns]), -1, ns-nd),
        but doesn't produce garbage on TPUs.
        """
        i = torch.arange(0, nd).unsqueeze(1)
        j = torch.arange(ns)
        return (i >= j - ns + nd).to(dtype=dtype, device=device)

    def multihead_attn(self, q, k, v):
        # q, k, v have shape [batch, heads, sequence, features]
        w = torch.matmul(q, k.permute(0, 1, 3, 2))
        w = w / math.sqrt(v.shape[-1])
        w = self.mask_attn_weights(w)
        w = F.softmax(w, dim=-1)
        a = torch.matmul(w, v)
        return a


class Conv1D(nn.Linear):
    def reset_parameters(self):
        nn.init.normal_(self.weight, std=0.02)
        nn.init.zeros_(self.bias)


def gelu(x, c=math.sqrt(2 / math.pi)):
    return 0.5 * x * (1 + torch.tanh(c * (x + 0.044715 * torch.pow(x, 3))))


def position_for(batch_size, n_steps, past_length, device=None):
    return (torch.arange(past_length, n_steps + past_length, device=device)
            .unsqueeze(0).repeat(batch_size, 1))
