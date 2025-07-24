import torch
import torch.nn as nn
from transformers import ViTModel, ViTConfig
from typing import Optional
from dataclasses import dataclass

# ------- Model Output Struct -------
@dataclass
class ModelOutput:
    all_logits: torch.Tensor       # (B, 101)
    number_logits: torch.Tensor    # (B, 100)
    number_probs: torch.Tensor     # (B, 100)
    uncertainty: torch.Tensor      # (B,)

# ------- Tied Digit-Aware Head -------
class TiedDigitAwareClassifier(nn.Module):
    def __init__(self, embed_dim, embedding_type = "multiplicative", per_digit_bias = False):
        super().__init__()
        assert embedding_type in ["additive", "multiplicative"]
        self.embedding_type = embedding_type
        self.per_digit_bias = per_digit_bias

        self.digit_classifier = nn.Linear(embed_dim, 10, bias=False)
        self.absent_classifier = nn.Linear(embed_dim, 1)

        if per_digit_bias:
            self.position_biases = nn.Parameter(torch.zeros(3, 10))
        else:
            self.position_biases = nn.Parameter(torch.zeros(3, 1))

        init = (
            torch.randn(3, embed_dim) if embedding_type == "additive"
            else torch.ones(3, embed_dim)
        )
        self.position_embeddings = nn.Parameter(init)

        tens_idx = torch.div(torch.arange(10, 100), 10, rounding_mode='floor')
        ones_idx = torch.remainder(torch.arange(10, 100), 10)
        self.register_buffer("tens_idx", tens_idx)
        self.register_buffer("ones_idx", ones_idx)

    def forward(self, x):
        B = x.size(0)
        if self.embedding_type == "additive":
            positioned = x.unsqueeze(1) + self.position_embeddings.unsqueeze(0)
        else:
            positioned = x.unsqueeze(1) * self.position_embeddings.unsqueeze(0)

        digit_logits = self.digit_classifier(positioned)
        digit_logits += self.position_biases.unsqueeze(0)

        single_logits = digit_logits[:, 0]
        tens_logits = digit_logits[:, 1]
        ones_logits = digit_logits[:, 2]

        absent_logit = self.absent_classifier(x)  # (B, 1)

        number_logits = torch.empty(B, 100, device=x.device)
        number_logits[:, :10] = single_logits
        number_logits[:, 10:] = tens_logits[:, self.tens_idx] + ones_logits[:, self.ones_idx]

        all_logits = torch.cat([number_logits, absent_logit], dim=1)
        return all_logits

# ------- Light Custom ViT Model -------
class vit(nn.Module):
    def __init__(self, use_time=False, use_size=False, embed_dim=120,hidden_layers=4,attention_heads=3,uncertainty_head="dirichlet",image_size=224):
        super().__init__()
        self.use_time = use_time
        self.use_size = use_size
        self.uncertainty_head = uncertainty_head
        self.image_size = image_size

        config = ViTConfig(
            image_size=self.image_size,
            patch_size=16,
            hidden_size=embed_dim,
            num_hidden_layers=hidden_layers,
            num_attention_heads=attention_heads,
            intermediate_size=embed_dim * 4,
        )
        self.encoder = ViTModel(config)

        if use_time:
            self.time_embed = nn.Linear(1, embed_dim)
            nn.init.uniform_(self.time_embed.weight, -0.001, 0.001)
            nn.init.zeros_(self.time_embed.bias)

        if use_size:
            self.size_embed = nn.Linear(2, embed_dim)
            nn.init.uniform_(self.size_embed.weight, -0.003, 0.003)
            nn.init.zeros_(self.size_embed.bias)

        self.classifier = TiedDigitAwareClassifier(embed_dim)

    def forward(self, x, t= None, size = None):
        x = self.encoder(x).last_hidden_state[:, 0]  # CLS token

        if self.use_time and t is not None:
            x = x + self.time_embed(t)

        if self.use_size and size is not None:
            x = x + self.size_embed(size)

        all_logits = self.classifier(x)
        number_logits = all_logits[:, :100]

        if self.uncertainty_head == "dirichlet":
            alpha = torch.exp(number_logits) + 1.0
            S = alpha.sum(dim=1, keepdim=True)
            number_probs = alpha / S
            uncertainty = 100.0 / S.squeeze()
        else:
            all_probs = torch.softmax(all_logits, dim=1)
            number_probs = all_probs[:, :100]
            uncertainty = all_probs[:, 100]

        return ModelOutput(
            all_logits=all_logits,
            number_logits=number_logits,
            number_probs=number_probs,
            uncertainty=uncertainty,
        )