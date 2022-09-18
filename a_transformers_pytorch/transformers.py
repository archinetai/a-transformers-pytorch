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


def log(val: Tensor, eps: float = 1e-20) -> Tensor:
    return torch.log(val.clamp(min=eps))


def gumbel_sample(val: Tensor, temperature: float, dim: int = -1) -> Tensor:
    noise = torch.zeros_like(val).uniform_(0, 1)
    gumbel_noise = -log(-log(noise))
    return ((val / temperature) + gumbel_noise).argmax(dim=dim)


def top_k(logits: Tensor, threshold: float) -> Tensor:
    num_logits = logits.shape[-1]
    k = max(int((1 - threshold) * num_logits), 1)
    values, indices = torch.topk(logits, k)
    probs = torch.full_like(logits, float("-inf"))
    probs.scatter_(1, indices, values)
    return probs


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
        out_features = default(out_features, features)

        self.to_out = nn.Linear(
            in_features=mid_features, out_features=out_features, bias=False
        )

    def forward(
        self,
        q: Tensor,
        k: Tensor,
        v: Tensor,
        *,
        causal: bool = False,
        mask: Optional[Tensor] = None,
    ) -> Tensor:
        # Split heads
        q, k, v = rearrange_many((q, k, v), "b n (h d) -> b h n d", h=self.num_heads)

        # Compute similarity matrix, add bias and mask
        sim = einsum("... n d, ... m d -> ... n m", q, k) * self.scale
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
        context_features: Optional[int] = None,
        **kwargs,
    ):
        super().__init__()
        self.context_features = context_features
        mid_features = head_features * num_heads
        context_features = default(context_features, features)

        self.norm = nn.LayerNorm(features)
        self.norm_context = nn.LayerNorm(context_features)
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
            out_features=out_features,
            **kwargs,
        )

    def forward(self, x: Tensor, context: Optional[Tensor] = None, **kwargs) -> Tensor:
        assert_message = "You must provide a context when using context_features"
        assert not self.context_features or exists(context), assert_message
        context = default(context, x)
        x, context = self.norm(x), self.norm_context(context)
        q, k, v = (self.to_q(x), *torch.chunk(self.to_kv(context), chunks=2, dim=-1))
        x = self.attention(q, k, v, **kwargs)
        return x


def FeedForward(features: int, multiplier: int) -> nn.Module:
    mid_features = features * multiplier
    return nn.Sequential(
        nn.Linear(in_features=features, out_features=mid_features, bias=False),
        nn.GELU(),
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


class Transformer(nn.Module):
    def __init__(
        self,
        features: int,
        max_length: int,
        num_layers: int,
        head_features: int,
        num_heads: int,
        multiplier: int,
        use_positional_embedding: bool = True,
        causal: bool = False,
        use_cross_attention: bool = False,
    ):
        super().__init__()
        self.features = features
        self.max_length = max_length

        self.blocks = nn.ModuleList(
            [
                TransformerBlock(
                    features=features,
                    head_features=head_features,
                    num_heads=num_heads,
                    multiplier=multiplier,
                    max_length=max_length,
                    use_positional_embedding=use_positional_embedding,
                    causal=causal,
                    context_features=features if use_cross_attention else None,
                )
                for i in range(num_layers)
            ]
        )

    def forward(self, embedding: Tensor, **kwargs) -> Tensor:
        for block in self.blocks:
            embedding = block(embedding, **kwargs)
        return embedding


class Autoregressive(nn.Module):
    def __init__(self, transformer: Transformer, num_tokens: int, **kwargs):
        super().__init__()
        self.features = transformer.features
        self.max_length = transformer.max_length
        self.transformer = transformer

        self.to_embedding = nn.Embedding(num_tokens, self.features)
        self.to_logits = nn.Linear(in_features=self.features, out_features=num_tokens)

    def compute_logits(self, tokens: Tensor, **kwargs) -> Tensor:
        input_embedding = self.to_embedding(tokens)
        # Compute output embedding and logits
        output_embedding = self.transformer(input_embedding, **kwargs)
        output_logits = self.to_logits(output_embedding)
        output_logits = rearrange(output_logits, "b n t -> b t n")
        return output_logits

    def forward(self, tokens: Tensor, **kwargs) -> Tensor:
        input_tokens = tokens[:, :-1]
        target_tokens = tokens[:, 1:]
        logits = self.compute_logits(input_tokens)
        loss = F.cross_entropy(logits, target_tokens)
        return loss

    def generate(
        self,
        start_tokens: Tensor,
        sequence_length: int,
        top_k_threshold: float = 0.9,
        temperature: float = 1.0,
        **kwargs,
    ) -> Tensor:
        t, s = start_tokens.shape[1], self.max_length
        tokens = start_tokens

        for _ in range(sequence_length):
            # Compute last token logits
            logits = self.compute_logits(tokens=tokens[:, -s:], **kwargs)
            logits = logits[:, -1]
            # Gumbel sample from top-k logits
            logits = top_k(logits, threshold=top_k_threshold)
            sample = gumbel_sample(logits, dim=-1, temperature=temperature)
            # Append sampled token
            tokens = torch.cat([tokens, rearrange(sample, "b -> b 1")], dim=-1)

        # Return only generated tokens
        return tokens[:, t:]
