from inspect import isfunction
from typing import Callable, Optional, TypeVar, Union

import torch
import torch.nn.functional as F
from einops import rearrange
from einops_exts import rearrange_many
from torch import Tensor, einsum, nn
from typing_extensions import TypeGuard

T = TypeVar("T")

"""
Utils
"""


def exists(val: Optional[T]) -> TypeGuard[T]:
    return val is not None


def default(val: Optional[T], d: Union[Callable[..., T], T]) -> T:
    if exists(val):
        return val
    return d() if isfunction(d) else d


"""
Attention Components
"""


def attention_mask(
    sim: Tensor,
    mask: Tensor,
) -> Tensor:
    mask = rearrange(mask, "b j -> b 1 1 j")
    max_neg_value = -torch.finfo(sim.dtype).max
    sim = sim.masked_fill(~mask, max_neg_value)
    return sim


def causal_mask(sim: Tensor) -> Tensor:
    i, j, device = *sim.shape[-2:], sim.device
    max_neg_value = -torch.finfo(sim.dtype).max
    mask = torch.ones((i, j), dtype=torch.bool, device=device).triu(j - i + 1)
    sim = sim.masked_fill(mask, max_neg_value)
    del mask
    return sim


class LayerNorm(nn.Module):
    def __init__(self, features: int, *, bias: bool = True, eps: float = 1e-5):
        super().__init__()
        self.bias = bias
        self.eps = eps
        self.g = nn.Parameter(torch.ones(features))
        self.b = nn.Parameter(torch.zeros(features)) if bias else None

    def forward(self, x: Tensor) -> Tensor:
        var = torch.var(x, dim=-1, unbiased=False, keepdim=True)
        mean = torch.mean(x, dim=-1, keepdim=True)
        norm = (x - mean) * (var + self.eps).rsqrt() * self.g
        return norm + self.b if self.bias else norm


class AttentionBase(nn.Module):
    def __init__(
        self,
        features: int,
        *,
        head_features: int,
        num_heads: int,
        out_features: Optional[int] = None,
        causal: bool = False,
    ):
        super().__init__()
        self.scale = head_features**-0.5
        self.num_heads = num_heads
        self.causal = causal
        mid_features = head_features * num_heads
        out_features = out_features if exists(out_features) else features

        self.to_out = nn.Sequential(
            nn.Linear(in_features=mid_features, out_features=out_features, bias=False),
            LayerNorm(features=out_features, bias=False),
        )

    def forward(
        self,
        q: Tensor,
        k: Tensor,
        v: Tensor,
        *,
        causal: bool = False,
        mask: Optional[Tensor] = None,
        rel_pos: Optional[nn.Module] = None,
    ) -> Tensor:
        # Split heads
        q, k, v = rearrange_many((q, k, v), "b n (h d) -> b h n d", h=self.num_heads)

        # Compute similarity matrix, add bias and mask
        sim = einsum("... n d, ... m d -> ... n m", q, k) * self.scale
        sim = rel_pos(sim) if exists(rel_pos) else sim
        sim = attention_mask(sim, mask) if exists(mask) else sim
        sim = causal_mask(sim) if (causal or self.causal) else sim

        # Get attention matrix with softmax
        attn = sim.softmax(dim=-1, dtype=torch.float32)

        # Compute values
        out = einsum("... n m, ... m d -> ... n d", attn, v)
        out = rearrange(out, "b h n d -> b n (h d)")
        return self.to_out(out)


class Attention(nn.Module):
    def __init__(
        self,
        features: int,
        *,
        head_features: int,
        num_heads: int,
        out_features: Optional[int] = None,
        **kwargs,
    ):
        super().__init__()
        mid_features = head_features * num_heads

        self.norm = LayerNorm(features, bias=False)
        self.to_qkv = nn.Linear(
            in_features=features, out_features=mid_features * 3, bias=False
        )
        self.attention = AttentionBase(
            features,
            num_heads=num_heads,
            head_features=head_features,
            out_features=out_features,
            **kwargs,
        )

    def forward(self, x: Tensor, **kwargs) -> Tensor:
        x = self.norm(x)
        q, k, v = torch.chunk(self.to_qkv(x), chunks=3, dim=-1)
        x = self.attention(q, k, v, **kwargs)
        return x


class CrossAttention(nn.Module):
    def __init__(
        self,
        features: int,
        *,
        head_features: int,
        num_heads: int,
        context_features: int = None,
        **kwargs,
    ):
        super().__init__()
        mid_features = head_features * num_heads
        context_features = default(context_features, features)

        self.norm_in = LayerNorm(features=features, bias=False)
        self.norm_context = LayerNorm(features=context_features, bias=False)

        self.to_q = nn.Linear(
            in_features=features, out_features=mid_features, bias=False
        )
        self.to_kv = nn.Linear(
            in_features=context_features, out_features=mid_features * 2, bias=False
        )
        self.attention = AttentionBase(
            features,
            num_heads=num_heads,
            head_features=head_features,
            **kwargs,
        )

    def forward(self, x: Tensor, context: Tensor, **kwargs) -> Tensor:
        b, n, d = x.shape
        x = self.norm_in(x)
        context = self.norm_context(context)
        # Queries form x, k and v from context
        q, k, v = (self.to_q(x), *torch.chunk(self.to_kv(context), chunks=2, dim=-1))
        x = self.attention(q, k, v, **kwargs)
        return x


def FeedForward(features: int, multiplier: int) -> nn.Module:
    mid_features = int(features * multiplier)
    return nn.Sequential(
        LayerNorm(features, bias=False),
        nn.Linear(in_features=features, out_features=mid_features, bias=False),
        nn.GELU(),
        LayerNorm(mid_features, bias=False),
        nn.Linear(in_features=mid_features, out_features=features, bias=False),
    )


class AbsolutePositionalEmbedding(nn.Module):
    def __init__(self, features: int, max_length: int):
        super().__init__()
        self.max_length = max_length
        self.embedding = nn.Embedding(max_length, features)

    def forward(self, x: Tensor) -> Tensor:
        length, device = x.shape[1], x.device
        assert_message = "Input sequence length must be <= max_length"
        assert length <= self.max_length, assert_message
        position = torch.arange(length, device=device)
        return self.embedding(position)


"""
Transformer Blocks
"""


class TransformerBlock(nn.Module):
    def __init__(
        self,
        features: int,
        *,
        head_features: int,
        num_heads: int,
        multiplier: int,
        max_length: Optional[int] = None,
        use_positional_embedding: bool = False,
        **kwargs,
    ):
        super().__init__()

        self.use_positional_embedding = use_positional_embedding

        if use_positional_embedding:
            assert_message = "max_length required if use_positional_embedding=True"
            assert exists(max_length), assert_message

            self.positional_embedding = AbsolutePositionalEmbedding(
                max_length=max_length,
                features=features,
            )

        self.attention = Attention(
            features=features,
            head_features=head_features,
            num_heads=num_heads,
            **kwargs,
        )

        self.feed_forward = FeedForward(features=features, multiplier=multiplier)

    def forward(self, x: Tensor, **kwargs) -> Tensor:
        if self.use_positional_embedding:
            x = self.positional_embedding(x) + x
        x = self.attention(x, **kwargs) + x
        x = self.feed_forward(x) + x
        return x


"""
Transformers
"""


class AutoregressiveTransformer(nn.Module):
    def __init__(
        self,
        features: int,
        num_tokens: int,
        max_length: int,
        num_layers: int = 12,
        head_features: int = 64,
        num_heads: int = 12,
        multiplier: int = 4,
        use_positional_embedding: bool = True,
    ):
        super().__init__()

        self.to_embedding = nn.Embedding(num_tokens, features)

        self.transformer = nn.Sequential(
            *[
                TransformerBlock(
                    features=features,
                    head_features=head_features,
                    num_heads=num_heads,
                    multiplier=multiplier,
                    max_length=max_length,
                    use_positional_embedding=use_positional_embedding,
                    causal=True,
                )
                for i in range(num_layers)
            ]
        )

        self.to_logits = nn.Linear(in_features=features, out_features=num_tokens)

    def forward(self, tokens: Tensor, **kwargs) -> Tensor:
        # Pick input embedding and target sequence
        input_tokens = tokens[:, :-1]
        target_tokens = tokens[:, 1:]
        input_embedding = self.to_embedding(input_tokens)
        # Compute output embedding and logits
        output_embedding = self.transformer(input_embedding, **kwargs)
        output_logits = self.to_logits(output_embedding)
        output_logits = rearrange(output_logits, "b n t -> b t n")
        # Compute and return loss
        loss = F.cross_entropy(output_logits, target_tokens)
        return loss
