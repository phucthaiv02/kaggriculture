from __future__ import annotations

import torch
import torch.nn.functional as F


def weighted_accuracy(
    logits: torch.Tensor,
    targets: torch.Tensor,
    mask: torch.Tensor,
    sample_weight: torch.Tensor,
    top_k: int = 1,
    slot_weight: torch.Tensor | None = None,
) -> torch.Tensor:
    k = min(top_k, logits.shape[-1])
    predictions = logits.topk(k, dim=-1).indices
    correct = predictions.eq(targets.long().unsqueeze(-1)).any(dim=-1)
    weights = mask.float() * sample_weight[:, None]
    if slot_weight is not None:
        weights = weights * slot_weight
    return (correct.float() * weights).sum() / weights.sum().clamp_min(1.0)


def weighted_exact_action_accuracy(
    outputs: dict[str, torch.Tensor],
    batch: dict[str, torch.Tensor],
    prefix: str,
    active: torch.Tensor,
    sample_weight: torch.Tensor,
    slot_weight: torch.Tensor | None = None,
) -> torch.Tensor:
    correct = outputs[f"{prefix}_op"].argmax(-1).eq(batch[f"{prefix}_op"].long())
    for component in ("item", "quantity"):
        component_mask = batch[f"{prefix}_{component}_mask"].bool()
        component_correct = outputs[f"{prefix}_{component}"].argmax(-1).eq(
            batch[f"{prefix}_{component}"].long()
        )
        correct &= ~component_mask | component_correct
    weights = active.float() * sample_weight[:, None]
    if slot_weight is not None:
        weights = weights * slot_weight
    return (correct.float() * weights).sum() / weights.sum().clamp_min(1.0)


def weighted_ce(
    logits: torch.Tensor,
    targets: torch.Tensor,
    mask: torch.Tensor,
    sample_weight: torch.Tensor,
    slot_weight: torch.Tensor | None = None,
) -> torch.Tensor:
    raw = F.cross_entropy(logits.flatten(0, -2), targets.long().flatten(), reduction="none").view_as(targets)
    weights = mask.float() * sample_weight[:, None]
    if slot_weight is not None:
        weights = weights * slot_weight
    return (raw * weights).sum() / weights.sum().clamp_min(1.0)


def illegal_probability_loss(
    logits: torch.Tensor, legal: torch.Tensor, targets: torch.Tensor, active: torch.Tensor
) -> torch.Tensor:
    # Demonstrations occasionally rely on same-turn prerequisites (DROP then
    # SELL), so always whitelist the target before penalizing illegal mass.
    legal = legal.clone().bool()
    legal.scatter_(-1, targets.long().unsqueeze(-1), True)
    illegal_mass = (logits.softmax(-1) * ~legal).sum(-1)
    return (illegal_mass * active.float()).sum() / active.float().sum().clamp_min(1.0)


def behavior_cloning_loss(outputs: dict[str, torch.Tensor], batch: dict[str, torch.Tensor]):
    weight = batch["sample_weight"].float()
    unit_active = batch["unit_mask"].bool()
    unit_op = weighted_ce(outputs["unit_op"], batch["unit_op"], unit_active, weight)
    unit_item = weighted_ce(
        outputs["unit_item"], batch["unit_item"], batch["unit_item_mask"].bool(), weight
    )
    unit_quantity = weighted_ce(
        outputs["unit_quantity"],
        batch["unit_quantity"],
        batch["unit_quantity_mask"].bool(),
        weight,
    )
    market_active = torch.ones_like(batch["market_op"], dtype=torch.bool)
    market_slot_weight = torch.where(batch["market_op"].long() == 0, 0.15, 1.0)
    market_op = weighted_ce(
        outputs["market_op"], batch["market_op"], market_active, weight, market_slot_weight
    )
    market_item = weighted_ce(
        outputs["market_item"], batch["market_item"], batch["market_item_mask"].bool(), weight
    )
    market_quantity = weighted_ce(
        outputs["market_quantity"],
        batch["market_quantity"],
        batch["market_quantity_mask"].bool(),
        weight,
    )
    value = F.smooth_l1_loss(outputs["value"].float(), batch["value_target"].float())
    unit_illegal = illegal_probability_loss(
        outputs["unit_op"], batch["unit_legal"], batch["unit_op"], unit_active
    )
    market_illegal = illegal_probability_loss(
        outputs["market_op"], batch["market_legal"], batch["market_op"], market_active
    )
    unit_op_accuracy = weighted_accuracy(
        outputs["unit_op"], batch["unit_op"], unit_active, weight
    )
    unit_op_top3_accuracy = weighted_accuracy(
        outputs["unit_op"], batch["unit_op"], unit_active, weight, top_k=3
    )
    unit_item_accuracy = weighted_accuracy(
        outputs["unit_item"], batch["unit_item"], batch["unit_item_mask"].bool(), weight
    )
    unit_quantity_accuracy = weighted_accuracy(
        outputs["unit_quantity"],
        batch["unit_quantity"],
        batch["unit_quantity_mask"].bool(),
        weight,
    )
    market_op_accuracy = weighted_accuracy(
        outputs["market_op"],
        batch["market_op"],
        market_active,
        weight,
        slot_weight=market_slot_weight,
    )
    market_op_top3_accuracy = weighted_accuracy(
        outputs["market_op"],
        batch["market_op"],
        market_active,
        weight,
        top_k=3,
        slot_weight=market_slot_weight,
    )
    market_item_accuracy = weighted_accuracy(
        outputs["market_item"],
        batch["market_item"],
        batch["market_item_mask"].bool(),
        weight,
    )
    market_quantity_accuracy = weighted_accuracy(
        outputs["market_quantity"],
        batch["market_quantity"],
        batch["market_quantity_mask"].bool(),
        weight,
    )
    unit_exact_accuracy = weighted_exact_action_accuracy(
        outputs, batch, "unit", unit_active, weight
    )
    market_exact_accuracy = weighted_exact_action_accuracy(
        outputs, batch, "market", market_active, weight, market_slot_weight
    )
    value_mae = F.l1_loss(outputs["value"].float(), batch["value_target"].float())
    total = (
        unit_op
        + 0.35 * unit_item
        + 0.20 * unit_quantity
        + market_op
        + 0.35 * market_item
        + 0.20 * market_quantity
        + 0.25 * value
        + 0.05 * unit_illegal
        + 0.02 * market_illegal
    )
    metrics = {
        "loss": total.detach(),
        "unit_op": unit_op.detach(),
        "unit_item": unit_item.detach(),
        "unit_quantity": unit_quantity.detach(),
        "market_op": market_op.detach(),
        "market_item": market_item.detach(),
        "market_quantity": market_quantity.detach(),
        "value": value.detach(),
        "unit_illegal": unit_illegal.detach(),
        "market_illegal": market_illegal.detach(),
        "unit_op_accuracy": unit_op_accuracy.detach(),
        "unit_op_top3_accuracy": unit_op_top3_accuracy.detach(),
        "unit_item_accuracy": unit_item_accuracy.detach(),
        "unit_quantity_accuracy": unit_quantity_accuracy.detach(),
        "market_op_accuracy": market_op_accuracy.detach(),
        "market_op_top3_accuracy": market_op_top3_accuracy.detach(),
        "market_item_accuracy": market_item_accuracy.detach(),
        "market_quantity_accuracy": market_quantity_accuracy.detach(),
        "unit_exact_accuracy": unit_exact_accuracy.detach(),
        "market_exact_accuracy": market_exact_accuracy.detach(),
        "value_mae": value_mae.detach(),
    }
    return total, metrics
