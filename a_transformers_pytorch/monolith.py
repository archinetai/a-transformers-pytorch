from functools import reduce
from typing import List, Tuple

import torch
from einops import rearrange
from torch import Tensor, nn

from .transformers import Resampler, Transformer, TransformerRotator, exists


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
                TransformerRotator(
                    Transformer(
                        features=features,
                        max_length=layer_tokens[i][0],
                        num_layers=8,
                        head_features=head_features,
                        num_heads=num_heads,
                        multiplier=multiplier,
                        use_cross_attention=True,
                    ),
                    num_rotations=1,
                )
                for i in range(self.num_layers)
            ]
        )

    def forward(self, embedding: Tensor) -> Tensor:
        batch = embedding.shape[0]
        layers: List[Tensor] = [embedding]
        context, input_embedding = None, embedding

        for i in range(self.num_layers - 1):
            # Compute compressed tokens bottom-up
            hi = self.hierarchy[i][0]
            embedding = rearrange(layers[-1], "b (n hi) d -> (b n) hi d", hi=hi)
            resampled = self.resamplers[i](embedding)
            layers += [rearrange(resampled, "(b n) ho d -> b (n ho) d", b=batch)]

        for i in reversed(range(self.num_layers)):
            # Compute next token autoregressively top-down
            in_tokens, out_tokens = self.layer_tokens[i]
            embedding = layers.pop()[:, -in_tokens:] if layers else None  # type: ignore
            context = context[:, -out_tokens:] if exists(context) else embedding  # type: ignore
            context = self.transformers[i](embedding, context=context)

        output_head = input_embedding[:, 1 : -self.layer_tokens[0][0] + 1]
        output_tail = context
        output_embedding = torch.cat([output_head, output_tail], dim=-2)  # type: ignore
        return output_embedding

    def get_info(self):
        factors = [hi / ho for (hi, ho) in self.hierarchy]
        context = reduce(lambda x, y: x * y, factors)
        info = dict(min_tokens=context * self.layer_tokens[-1][0])
        return info


class Monolith(nn.Module):
    def __init__(
        self,
        num_tokens: int,
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

        self.to_embedding = nn.Embedding(num_tokens, features)
        self.token = nn.Parameter(torch.randn(features))

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
                Transformer(
                    features=features,
                    max_length=layer_tokens[i][0],
                    num_layers=8,
                    head_features=head_features,
                    num_heads=num_heads,
                    multiplier=multiplier,
                    use_cross_attention=True,
                )
                for i in range(self.num_layers)
            ]
        )

        self.to_logits = nn.Linear(in_features=features, out_features=num_tokens)

    def forward(self, tokens: Tensor, return_logits: bool = False) -> Tensor:
        pass
        # input_tokens = tokens if return_logits else tokens[:, :-1]
        # embedding = self.to_embedding(input_tokens)
        # output_embedding = self.compute_embedding(embedding)

        # output_logits = self.to_logits(output_embedding)
        # output_logits = rearrange(output_logits, "b n t -> b t n")

        # if return_logits:
        #     return output_logits
        # # Compute and return loss
        # target_tokens = tokens[:, -self.layer_tokens[0][0] :]
        # loss = F.cross_entropy(output_logits, target_tokens)
        # return loss
