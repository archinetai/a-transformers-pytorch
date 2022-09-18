from functools import reduce
from typing import List, Tuple

import torch
from einops import rearrange
from torch import Tensor, nn

from .transformers import Resampler, Transformer, TransformerShifter, exists


class MonolithBase(nn.Module):
    def __init__(
        self,
        features: int,
        head_features: int,
        num_heads: int,
        multiplier: int,
        hierarchy: List[Tuple[int, int]],
        layer_tokens: List[Tuple[int, int]],
    ):
        super().__init__()
        self.hierarchy = hierarchy
        self.layer_tokens = layer_tokens
        self.num_layers = len(layer_tokens)

        self.resamplers = nn.ModuleList(
            [
                Resampler(
                    features=features,
                    in_tokens=hierarchy[i][0],
                    out_tokens=hierarchy[i][1],
                    num_layers=4,
                    head_features=head_features,
                    num_heads=num_heads,
                    multiplier=multiplier,
                )
                for i in range(self.num_layers - 1)
            ]
        )

        self.transformers = nn.ModuleList(
            [
                TransformerShifter(
                    Transformer(
                        features=features,
                        max_length=layer_tokens[i][0],
                        num_layers=8,
                        head_features=head_features,
                        num_heads=num_heads,
                        multiplier=multiplier,
                        causal=True,
                        use_cross_attention=True,
                    ),
                    num_shift=1,
                )
                for i in range(self.num_layers)
            ]
        )

    def forward(self, embedding: Tensor) -> Tensor:
        batch, device = embedding.shape[0], embedding.device
        layers, context, input_embedding = [embedding], None, embedding

        for i in range(self.num_layers - 1):
            # Compute compressed tokens bottom-up
            hi = self.hierarchy[i][0]
            embedding = rearrange(layers[-1], "b (n hi) d -> (b n) hi d", hi=hi)
            resampled = self.resamplers[i](embedding)
            layers += [rearrange(resampled, "(b n) ho d -> b (n ho) d", b=batch)]

        for i in reversed(range(self.num_layers)):
            # Compute next token autoregressively top-down
            embedding_length, context_length = self.layer_tokens[i]
            hierarchy_in, hierarchy_out = self.hierarchy
            embedding = layers.pop()[:, -embedding_length:] if layers else None  # type: ignore
            context = context[:, -context_length:] if exists(context) else embedding  # type: ignore
            context = self.transformers[i](
                embedding,
                context=context,
                cross_attention_mask=self.get_mask(i, device=device),
            )

        # Shift input embedding once and concatenate predicted output embedding
        output_head = input_embedding[:, 1 : -self.layer_tokens[0][0] + 1]
        output_tail = context
        output_embedding = torch.cat([output_head, output_tail], dim=-2)  # type: ignore
        return output_embedding

    def get_mask(self, layer: int, device: torch.device) -> Tensor:
        n, m, kwargs = *self.layer_tokens[layer], dict(dtype=torch.bool, device=device)

        if layer == self.num_layers - 1:  # Causal mask if top layer
            return ~torch.ones((n, m), **kwargs).triu(1)  # type: ignore

        # Makes sure current level tokens cannot see future by using tokens from top
        mask = torch.zeros((n, m), **kwargs)  # type: ignore
        mask[-1, :] = True  # Only new token as access to last context token
        n1, m1, hi, ho = n - 1, m - 1, *self.hierarchy[layer]
        while n1 >= 0 and m1 >= 0:
            n0, m0 = max(n1 - hi, 0), 0
            shape = (n1 - n0, m1 - m0)
            mask[n0:n1, m0:m1] = torch.ones(shape, **kwargs)  # type: ignore
            n1, m1 = n1 - hi, m1 - ho
        return mask

    def get_info(self):
        factors = [hi / ho for (hi, ho) in self.hierarchy]
        context = reduce(lambda x, y: x * y, factors)
        info = dict(min_tokens=context * self.layer_tokens[-1][0])
        return info


# class Monolith(nn.Module):
#     def __init__(self, monolith: MonolithBase, num_tokens: int):
#         super().__init__()
#         self.monolith = monolith
#         self.features = momolith.features
#         self.num_tokens = num_tokens

#         self.to_embedding = nn.Embedding(num_tokens, features)
#         self.token = nn.Parameter(torch.randn(features))
#         self.to_logits = nn.Linear(in_features=features, out_features=num_tokens)

#     def forward(self, tokens: Tensor, return_logits: bool = False) -> Tensor:
#         input_tokens = tokens if return_logits else tokens[:, :-1]
#         embedding = self.to_embedding(input_tokens)
#         output_embedding = self.compute_embedding(embedding)

#         output_logits = self.to_logits(output_embedding)
#         output_logits = rearrange(output_logits, "b n t -> b t n")

#         if return_logits:
#             return output_logits
#         # Compute and return loss
#         target_tokens = tokens[:, -self.layer_tokens[0][0] :]
#         loss = F.cross_entropy(output_logits, target_tokens)
#         return loss
