"""H100-friendly multimodal encoder with autoregressive action decoders."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import torch
from torch import nn
from torch.utils.checkpoint import checkpoint

from .constants import (
    BOARD_CHANNELS,
    GLOBAL_FEATURES,
    ITEMS,
    MARKET_OPS,
    MAX_MARKET_ORDERS,
    MAX_UNITS,
    QUANTITY_BUCKETS,
    UNIT_FEATURES,
    UNIT_OPS,
)


@dataclass
class ModelConfig:
    d_model: int = 512
    board_blocks: int = 6
    transformer_layers: int = 8
    attention_heads: int = 8
    ff_multiplier: int = 4
    dropout: float = 0.10
    gradient_checkpointing: bool = False

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class ResidualConvBlock(nn.Module):
    def __init__(self, channels: int, dropout: float) -> None:
        super().__init__()
        self.norm1 = nn.GroupNorm(16, channels)
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1, bias=False)
        self.norm2 = nn.GroupNorm(16, channels)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1, bias=False)
        self.dropout = nn.Dropout2d(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        x = self.conv1(torch.nn.functional.silu(self.norm1(x)))
        x = self.dropout(x)
        x = self.conv2(torch.nn.functional.silu(self.norm2(x)))
        return x + residual


class BoardEncoder(nn.Module):
    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        channels = config.d_model // 2
        self.stem = nn.Conv2d(BOARD_CHANNELS, channels, 3, padding=1, bias=False)
        self.blocks = nn.ModuleList(
            ResidualConvBlock(channels, config.dropout) for _ in range(config.board_blocks)
        )
        self.projection = nn.Linear(channels, config.d_model)
        self.gradient_checkpointing = config.gradient_checkpointing

    def forward(self, board: torch.Tensor) -> torch.Tensor:
        x = board.to(dtype=self.stem.weight.dtype) / 255.0
        x = self.stem(x)
        for block in self.blocks:
            if self.gradient_checkpointing and self.training:
                x = checkpoint(block, x, use_reentrant=False)
            else:
                x = block(x)
        return self.projection(x.flatten(2).transpose(1, 2))


class AutoregressiveDecoder(nn.Module):
    def __init__(self, d_model: int, op_vocab: int, max_slots: int, dropout: float) -> None:
        super().__init__()
        self.op_vocab = op_vocab
        self.max_slots = max_slots
        self.op_embedding = nn.Embedding(op_vocab, d_model)
        self.item_embedding = nn.Embedding(len(ITEMS), d_model)
        self.quantity_embedding = nn.Embedding(len(QUANTITY_BUCKETS), d_model)
        self.slot_embedding = nn.Embedding(max_slots, d_model)
        self.input_norm = nn.LayerNorm(d_model)
        self.gru = nn.GRUCell(d_model, d_model)
        self.dropout = nn.Dropout(dropout)
        self.op_head = nn.Linear(d_model, op_vocab)
        self.item_head = nn.Linear(d_model, len(ITEMS))
        self.quantity_head = nn.Linear(d_model, len(QUANTITY_BUCKETS))
        self.initial = nn.Linear(d_model, d_model)

    def initial_state(self, global_context: torch.Tensor) -> tuple[torch.Tensor, tuple[torch.Tensor, ...]]:
        batch = global_context.shape[0]
        hidden = torch.tanh(self.initial(global_context))
        zeros = torch.zeros(batch, dtype=torch.long, device=global_context.device)
        previous = (zeros, zeros, zeros)
        return hidden, previous

    def step(
        self,
        slot_context: torch.Tensor,
        slot_index: int,
        hidden: torch.Tensor,
        previous: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    ) -> tuple[dict[str, torch.Tensor], torch.Tensor]:
        previous_op, previous_item, previous_quantity = previous
        slot_ids = torch.full_like(previous_op, slot_index)
        decoder_input = (
            slot_context
            + self.slot_embedding(slot_ids)
            + self.op_embedding(previous_op)
            + self.item_embedding(previous_item)
            + self.quantity_embedding(previous_quantity)
        )
        hidden = self.gru(self.input_norm(decoder_input), hidden)
        output = self.dropout(hidden)
        return {
            "op": self.op_head(output),
            "item": self.item_head(output),
            "quantity": self.quantity_head(output),
        }, hidden

    def forward(
        self,
        contexts: torch.Tensor,
        global_context: torch.Tensor,
        targets: dict[str, torch.Tensor] | None = None,
    ) -> dict[str, torch.Tensor]:
        hidden, previous = self.initial_state(global_context)
        op_logits, item_logits, quantity_logits = [], [], []
        for slot in range(contexts.shape[1]):
            logits, hidden = self.step(contexts[:, slot], slot, hidden, previous)
            op_logits.append(logits["op"])
            item_logits.append(logits["item"])
            quantity_logits.append(logits["quantity"])
            if targets is None:
                previous = (
                    logits["op"].argmax(-1),
                    logits["item"].argmax(-1),
                    logits["quantity"].argmax(-1),
                )
            else:
                previous = (
                    targets["op"][:, slot],
                    targets["item"][:, slot],
                    targets["quantity"][:, slot],
                )
        return {
            "op": torch.stack(op_logits, dim=1),
            "item": torch.stack(item_logits, dim=1),
            "quantity": torch.stack(quantity_logits, dim=1),
        }


class DynamicPolicy(nn.Module):
    def __init__(self, config: ModelConfig | None = None) -> None:
        super().__init__()
        self.config = config or ModelConfig()
        d_model = self.config.d_model
        self.board_encoder = BoardEncoder(self.config)
        self.global_encoder = nn.Sequential(
            nn.LayerNorm(GLOBAL_FEATURES),
            nn.Linear(GLOBAL_FEATURES, d_model * 2),
            nn.SiLU(),
            nn.Dropout(self.config.dropout),
            nn.Linear(d_model * 2, d_model),
        )
        self.unit_encoder = nn.Sequential(
            nn.LayerNorm(UNIT_FEATURES),
            nn.Linear(UNIT_FEATURES, d_model * 2),
            nn.SiLU(),
            nn.Dropout(self.config.dropout),
            nn.Linear(d_model * 2, d_model),
        )
        self.global_type = nn.Parameter(torch.zeros(1, 1, d_model))
        self.board_type = nn.Parameter(torch.zeros(1, 1, d_model))
        self.unit_type = nn.Parameter(torch.zeros(1, 1, d_model))
        layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=self.config.attention_heads,
            dim_feedforward=d_model * self.config.ff_multiplier,
            dropout=self.config.dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.context_encoder = nn.TransformerEncoder(
            layer,
            num_layers=self.config.transformer_layers,
            norm=nn.LayerNorm(d_model),
            enable_nested_tensor=False,
        )
        self.unit_decoder = AutoregressiveDecoder(d_model, len(UNIT_OPS), MAX_UNITS, self.config.dropout)
        self.market_positions = nn.Parameter(torch.randn(1, MAX_MARKET_ORDERS, d_model) * 0.02)
        self.market_decoder = AutoregressiveDecoder(
            d_model, len(MARKET_OPS), MAX_MARKET_ORDERS, self.config.dropout
        )
        self.value_head = nn.Sequential(nn.LayerNorm(d_model), nn.Linear(d_model, d_model), nn.SiLU(), nn.Linear(d_model, 1))
        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(module: nn.Module) -> None:
        if isinstance(module, (nn.Linear, nn.Conv2d)):
            nn.init.trunc_normal_(module.weight, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, std=0.02)

    def encode(
        self, board: torch.Tensor, global_features: torch.Tensor, units: torch.Tensor, unit_mask: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        board_tokens = self.board_encoder(board) + self.board_type
        global_token = self.global_encoder(global_features).unsqueeze(1) + self.global_type
        unit_tokens = self.unit_encoder(units) + self.unit_type
        tokens = torch.cat((global_token, board_tokens, unit_tokens), dim=1)
        prefix = torch.zeros(
            (tokens.shape[0], 1 + board_tokens.shape[1]), dtype=torch.bool, device=tokens.device
        )
        padding_mask = torch.cat((prefix, ~unit_mask.bool()), dim=1)
        encoded = self.context_encoder(tokens, src_key_padding_mask=padding_mask)
        global_context = encoded[:, 0]
        unit_context = encoded[:, -MAX_UNITS:]
        return global_context, unit_context

    def forward(
        self,
        board: torch.Tensor,
        global_features: torch.Tensor,
        units: torch.Tensor,
        unit_mask: torch.Tensor,
        targets: dict[str, torch.Tensor] | None = None,
    ) -> dict[str, torch.Tensor]:
        global_context, unit_context = self.encode(board, global_features, units, unit_mask)
        unit_targets = None
        market_targets = None
        if targets is not None:
            unit_targets = {
                "op": targets["unit_op"],
                "item": targets["unit_item"],
                "quantity": targets["unit_quantity"],
            }
            market_targets = {
                "op": targets["market_op"],
                "item": targets["market_item"],
                "quantity": targets["market_quantity"],
            }
        unit_output = self.unit_decoder(unit_context, global_context, unit_targets)
        market_context = global_context.unsqueeze(1) + self.market_positions
        market_output = self.market_decoder(market_context, global_context, market_targets)
        return {
            "unit_op": unit_output["op"],
            "unit_item": unit_output["item"],
            "unit_quantity": unit_output["quantity"],
            "market_op": market_output["op"],
            "market_item": market_output["item"],
            "market_quantity": market_output["quantity"],
            "value": self.value_head(global_context).squeeze(-1),
        }
