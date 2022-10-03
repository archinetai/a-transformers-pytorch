import torch
import torch.nn.functional as F
from einops import rearrange, reduce
from torch import Tensor, nn

from .transformers import Transformer, gumbel_sample, top_k


class RQTransformer(nn.Module):

    """Inspired by https://arxiv.org/abs/2203.01941"""

    def __init__(
        self,
        features: int,
        max_length: int,
        max_residuals: int,
        num_tokens: int,
        num_layers: int,
        head_features: int,
        num_heads: int,
        multiplier: int,
        shared_codebook: bool,
    ):
        super().__init__()
        self.max_length = max_length
        self.max_residuals = max_residuals

        self.to_embedding = nn.ModuleList(
            [
                nn.Embedding(num_embeddings=num_tokens, embedding_dim=features)
                for _ in range(max_residuals)
            ]
        )

        self.transformer_time = Transformer(
            features=features,
            max_length=max_length,
            num_layers=num_layers,
            head_features=head_features,
            num_heads=num_heads,
            multiplier=multiplier,
            causal=True,
        )

        self.transformer_residual = Transformer(
            features=features,
            max_length=max_residuals + 1,
            num_layers=num_layers // 2,
            head_features=head_features,
            num_heads=num_heads,
            multiplier=multiplier,
            causal=True,
        )

        self.to_logits = nn.ModuleList(
            [
                nn.Linear(in_features=features, out_features=num_tokens)
                for _ in range(max_residuals)
            ]
        )

        if shared_codebook:
            for i in range(max_residuals):
                self.to_embedding[i] = self.to_embedding[0]
                self.to_logits[i] = self.to_logits[0]

    def forward(self, tokens: Tensor) -> Tensor:
        b, _, r = tokens.shape

        # Compute embedding for each residual series
        embeddings_list = [self.to_embedding[i](tokens[:, :, i]) for i in range(r)]
        embeddings = torch.stack(embeddings_list)

        # Compute time embedding autoregressively (hide last and predict it)
        embedding_time = reduce(embeddings, "r b n d -> b n d", "sum")[:, :-1]
        embedding_time = self.transformer_time(embedding_time)
        embedding_time = rearrange(embedding_time, "b n d -> 1 b n d")

        # Compute output embedding by autoregressively predicting residual embeddings
        embeddings_target = embeddings[:, :, 1:]
        embedding_residual = torch.cat([embedding_time, embeddings_target], dim=0)
        embedding_residual = rearrange(embedding_residual, "r b n d -> (b n) r d")
        embedding_output = self.transformer_residual(embedding_residual)
        embedding_output = rearrange(embedding_output, "(b n) r d -> r b n d", b=b)

        # Compute logits for each residual output embedding
        logits_list = [self.to_logits[i](embedding_output[i]) for i in range(r)]
        logits = torch.stack(logits_list)
        logits = rearrange(logits, "r b n t -> b t (n r)")

        # Compute loss
        tokens_target = rearrange(tokens[:, 1:], "b n r -> b (n r)")
        loss = F.cross_entropy(logits, tokens_target)
        return loss

    def generate(
        self,
        start_tokens: Tensor,
        sequence_length: int,
        top_k_threshold: float = 0.9,
        temperature: float = 1.0,
        keep_start: bool = False,
        **kwargs,
    ) -> Tensor:
        _, _, r, s = *start_tokens.shape, self.max_length
        assert r <= self.max_residuals, "more residuals provided than max_residuals"
        tokens = start_tokens

        for _ in range(sequence_length):
            # Compute start time embedding
            embeddings_list = [
                self.to_embedding[i](tokens[:, -s:, i]) for i in range(r)
            ]
            embeddings = torch.stack(embeddings_list)
            embedding_time = reduce(embeddings, "r b n d -> b n d", "sum")

            # Compute next time embedding
            embedding_time = self.transformer_time(embedding_time)
            embedding_residual = rearrange(embedding_time[:, -1], "b d -> b 1 d")

            for _ in range(r):
                # Compute residual autoregressively
                embedding_last = self.transformer_residual(embedding_residual)[:, -1]
                embedding_last = rearrange(embedding_last, "b d -> b 1 d")
                embedding_residual = torch.cat(
                    [embedding_residual, embedding_last], dim=1
                )

            embedding_output = rearrange(embedding_residual[:, 1:], "b r d -> r b d")
            logits_list = [self.to_logits[i](embedding_output[i]) for i in range(r)]
            logits = torch.stack(logits_list)
            logits = rearrange(logits, "r b t -> (r b) t")

            # Gumbel sample from top-k logits
            logits = top_k(logits, threshold=top_k_threshold)
            sample = gumbel_sample(logits, dim=-1, temperature=temperature)
            sample = rearrange(sample, "(r b) -> b 1 r", r=r)

            # Append new sample tokens
            tokens = torch.cat([tokens, sample], dim=1)

            # # Compute sampled token embedding
            # embeddings_list = [self.to_embedding[i](sample[:,:,i]) for i in range(r)]
            # embeddings = torch.stack(embeddings_list)
            # embedding_time_sample = reduce(embeddings, 'r b 1 d -> b 1 d', 'sum')
            # embedding_time = torch.cat([embedding_time, embedding_time_sample], dim=1)
            # embedding_time = embedding_time[:,-s:]

        return tokens if keep_start else tokens[:, sequence_length:]
