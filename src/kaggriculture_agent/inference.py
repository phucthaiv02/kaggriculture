"""Sequential constrained decoding for the dynamic neural policy."""

from __future__ import annotations

import copy
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch

from .codec import encode_observation, legal_unit_op_matrix
from .constants import (
    ANIMALS,
    ANIMAL_COST,
    ANIMAL_STRUCTURE,
    CROPS,
    ITEMS,
    ITEM_TO_ID,
    MARKET_OPS,
    MARKET_OP_TO_ID,
    MAX_MARKET_ORDERS,
    MAX_UNITS,
    PRODUCTS,
    QUANTITY_BUCKETS,
    SEED_COST,
    SHED_ACCESS,
    UNIT_OPS,
    UNIT_OP_TO_ID,
)
from .model import DynamicPolicy, ModelConfig


def _masked_choice(logits: torch.Tensor, legal: np.ndarray, temperature: float) -> int:
    mask = torch.as_tensor(legal, dtype=torch.bool, device=logits.device)
    if not bool(mask.any()):
        return 0
    masked = logits.masked_fill(~mask, -torch.inf)
    if temperature <= 0:
        return int(masked.argmax().item())
    return int(torch.distributions.Categorical(logits=masked / temperature).sample().item())


def _masked_log_probability(
    logits: torch.Tensor, legal: np.ndarray, choice: int, temperature: float
) -> torch.Tensor:
    mask = torch.as_tensor(legal, dtype=torch.bool, device=logits.device)
    scaled = logits if temperature <= 0 else logits / temperature
    scaled = scaled.masked_fill(~mask, -torch.inf)
    return torch.log_softmax(scaled, dim=-1)[choice]


def _fib(index: int) -> int:
    a, b = 1, 1
    for _ in range(index):
        a, b = b, a + b
    return a


@dataclass
class ReservationState:
    money: float
    seeds: dict[str, int]
    shed: dict[str, int]
    inventories: list[dict[str, int]]
    hires_today: int
    unlocked_land: int

    @classmethod
    def from_observation(cls, obs: dict[str, Any]) -> "ReservationState":
        player = int(obs["player"])
        farm = obs["farms"][player]
        private = obs.get("private", {}) or {}
        return cls(
            money=float(farm.get("money", 0)),
            seeds=copy.deepcopy(private.get("seeds", {}) or {}),
            shed=copy.deepcopy(private.get("shed", {}) or {}),
            inventories=copy.deepcopy(private.get("inventories", []) or []),
            hires_today=int(farm.get("hires_today", 0)),
            unlocked_land=len(farm.get("unlocked_quadrants", [])),
        )

    @property
    def shed_occupancy(self) -> int:
        return sum(max(0, int(value)) for value in self.shed.values())


class DynamicAgent:
    def __init__(self, checkpoint_path: str | Path, device: str = "cpu", temperature: float = 0.0) -> None:
        self.device = torch.device(device)
        self.temperature = temperature
        checkpoint_path = Path(checkpoint_path)
        checkpoint = self._load_checkpoint(checkpoint_path)
        config = ModelConfig(**checkpoint["model_config"])
        config.gradient_checkpointing = False
        self.model = DynamicPolicy(config).to(self.device)
        self.model.load_state_dict(checkpoint["model"])
        self.model.eval()
        if self.device.type == "cpu":
            torch.set_num_threads(1)

    def _load_checkpoint(self, path: Path) -> dict[str, Any]:
        try:
            return torch.load(path, map_location=self.device, weights_only=True)
        except TypeError:  # PyTorch versions before weights_only was introduced.
            return torch.load(path, map_location=self.device)

    def _unit_item_mask(
        self, obs: dict[str, Any], state: ReservationState, unit_index: int, operation: str
    ) -> np.ndarray:
        mask = np.zeros(len(ITEMS), dtype=np.bool_)
        player = int(obs["player"])
        farm = obs["farms"][player]
        positions = [farm["farmer"], *farm.get("hands", [])]
        x, y = map(int, positions[unit_index])
        tile = farm["tiles"][y][x]
        inventory = state.inventories[unit_index] if unit_index < len(state.inventories) else {}
        if operation == "PLANT":
            for crop in CROPS:
                mask[ITEM_TO_ID[crop]] = state.seeds.get(crop, 0) > 0
        elif operation == "PICKUP":
            for item, quantity in state.shed.items():
                if item in ITEM_TO_ID and quantity > 0:
                    mask[ITEM_TO_ID[item]] = True
        elif operation == "PLACE":
            if isinstance(tile, dict) and tile.get("kind") in ("COOP", "PASTURE"):
                for animal in ANIMALS:
                    if ANIMAL_STRUCTURE[animal] == tile.get("kind") and inventory.get(animal, 0) > 0:
                        mask[ITEM_TO_ID[animal]] = True
            if (x, y) in SHED_ACCESS and state.shed_occupancy < 100:
                for item, quantity in inventory.items():
                    if item in ITEM_TO_ID and quantity > 0:
                        mask[ITEM_TO_ID[item]] = True
        else:
            mask[0] = True
        return mask

    def _unit_command(
        self,
        obs: dict[str, Any],
        state: ReservationState,
        unit_index: int,
        op_logits: torch.Tensor,
        item_logits: torch.Tensor,
        quantity_logits: torch.Tensor,
        legal_ops: np.ndarray,
    ) -> tuple[list[Any], tuple[int, int, int], np.ndarray, np.ndarray]:
        item_mask = np.zeros(len(ITEMS), dtype=np.bool_)
        quantity_mask = np.zeros(len(QUANTITY_BUCKETS), dtype=np.bool_)
        if not any(state.seeds.get(crop, 0) > 0 for crop in CROPS):
            legal_ops[UNIT_OP_TO_ID["PLANT"]] = False
        for candidate in ("PICKUP", "PLACE", "PLANT"):
            candidate_id = UNIT_OP_TO_ID[candidate]
            if legal_ops[candidate_id] and not self._unit_item_mask(
                obs, state, unit_index, candidate
            ).any():
                legal_ops[candidate_id] = False
        operation_id = _masked_choice(op_logits, legal_ops, self.temperature)
        operation = UNIT_OPS[operation_id]
        item_id, quantity_id = 0, 0
        command: list[Any] = [operation]
        if operation in ("PICKUP", "PLACE", "PLANT"):
            item_mask = self._unit_item_mask(obs, state, unit_index, operation)
            if not item_mask.any():
                return ["PASS"], (UNIT_OP_TO_ID["PASS"], 0, 0), item_mask, quantity_mask
            item_id = _masked_choice(item_logits, item_mask, self.temperature)
            item = ITEMS[item_id]
            command.append(item)
            if operation in ("PICKUP", "PLACE"):
                quantity_mask[1:] = True
                quantity_id = _masked_choice(quantity_logits, quantity_mask, self.temperature)
                requested = max(1, QUANTITY_BUCKETS[quantity_id])
                inventory = state.inventories[unit_index]
                if operation == "PICKUP":
                    quantity = min(requested, int(state.shed.get(item, 0)))
                    state.shed[item] = int(state.shed.get(item, 0)) - quantity
                    inventory[item] = int(inventory.get(item, 0)) + quantity
                else:
                    player = int(obs["player"])
                    position = [obs["farms"][player]["farmer"], *obs["farms"][player].get("hands", [])][unit_index]
                    x, y = map(int, position)
                    tile = obs["farms"][player]["tiles"][y][x]
                    places_animal = (
                        item in ANIMALS
                        and isinstance(tile, dict)
                        and tile.get("kind") == ANIMAL_STRUCTURE[item]
                        and "animal" not in tile
                    )
                    quantity = min(1 if places_animal else requested, int(inventory.get(item, 0)))
                    if tuple(position) in SHED_ACCESS and not places_animal:
                        room = max(0, 100 - state.shed_occupancy)
                        quantity = min(quantity, room)
                        state.shed[item] = int(state.shed.get(item, 0)) + quantity
                    inventory[item] = int(inventory.get(item, 0)) - quantity
                if quantity <= 0:
                    return ["PASS"], (UNIT_OP_TO_ID["PASS"], 0, 0), item_mask, quantity_mask
                command.append(quantity)
            elif operation == "PLANT":
                state.seeds[item] = int(state.seeds.get(item, 0)) - 1
        else:
            player = int(obs["player"])
            farm = obs["farms"][player]
            positions = [farm["farmer"], *farm.get("hands", [])]
            x, y = map(int, positions[unit_index])
            tile = farm["tiles"][y][x]
            inventory = state.inventories[unit_index]
            if operation == "DROP" and (x, y) in SHED_ACCESS:
                room = max(0, 100 - state.shed_occupancy)
                for carried_item, carried_quantity in list(inventory.items()):
                    moved = min(int(carried_quantity), room)
                    if moved > 0:
                        state.shed[carried_item] = int(state.shed.get(carried_item, 0)) + moved
                        room -= moved
                    inventory[carried_item] = 0
            elif operation == "FEED" and inventory.get("WHEAT", 0) > 0:
                inventory["WHEAT"] -= 1
            elif operation == "FERTILIZE" and inventory.get("FERTILIZER", 0) > 0:
                inventory["FERTILIZER"] -= 1
            elif operation == "HARVEST" and isinstance(tile, dict):
                quantity = int(tile.get("yield_units", 0))
                item = tile.get("crop")
                if tile.get("animal") == "GOOSE":
                    item = "EGG"
                elif tile.get("animal") == "COW":
                    item = "MILK"
                elif tile.get("animal") == "SHEEP":
                    item = "WOOL"
                if item in ITEM_TO_ID and quantity > 0:
                    inventory[item] = int(inventory.get(item, 0)) + quantity
        return command, (operation_id, item_id, quantity_id), item_mask, quantity_mask

    def _market_op_mask(self, state: ReservationState, obs: dict[str, Any] | None = None) -> np.ndarray:
        mask = np.zeros(len(MARKET_OPS), dtype=np.bool_)
        mask[MARKET_OP_TO_ID["NONE"]] = True
        mask[MARKET_OP_TO_ID["BUY_SEED"]] = state.money >= min(SEED_COST.values())
        prices = (obs or {}).get("market", {}).get("prices", {})
        affordable_product = any(
            state.money >= max(1, int(prices.get(item, 1))) for item in ("WHEAT", "FERTILIZER")
        )
        mask[MARKET_OP_TO_ID["BUY_PRODUCT"]] = affordable_product and state.shed_occupancy < 100
        mask[MARKET_OP_TO_ID["BUY_ANIMAL"]] = state.money >= min(ANIMAL_COST.values()) and state.shed_occupancy < 100
        mask[MARKET_OP_TO_ID["SELL"]] = any(state.shed.get(item, 0) > 0 for item in PRODUCTS)
        mask[MARKET_OP_TO_ID["HIRE"]] = state.money >= _fib(state.hires_today)
        land_cost = (1000, 2000, 4000)[state.unlocked_land - 1] if state.unlocked_land < 4 else float("inf")
        mask[MARKET_OP_TO_ID["BUY_LAND"]] = state.money >= land_cost
        return mask

    def _market_item_mask(
        self, operation: str, state: ReservationState, obs: dict[str, Any] | None = None
    ) -> np.ndarray:
        mask = np.zeros(len(ITEMS), dtype=np.bool_)
        candidates: tuple[str, ...] = ()
        if operation == "BUY_SEED":
            candidates = tuple(item for item in CROPS if SEED_COST[item] <= state.money)
        elif operation == "BUY_PRODUCT":
            prices = (obs or {}).get("market", {}).get("prices", {})
            candidates = tuple(
                item
                for item in ("WHEAT", "FERTILIZER")
                if max(1, int(prices.get(item, 1))) <= state.money
            )
        elif operation == "BUY_ANIMAL":
            candidates = tuple(item for item in ANIMALS if ANIMAL_COST[item] <= state.money)
        elif operation == "SELL":
            candidates = tuple(item for item in PRODUCTS if state.shed.get(item, 0) > 0)
        for item in candidates:
            mask[ITEM_TO_ID[item]] = True
        return mask

    def _market_order(
        self,
        obs: dict[str, Any],
        state: ReservationState,
        op_logits: torch.Tensor,
        item_logits: torch.Tensor,
        quantity_logits: torch.Tensor,
    ) -> tuple[list[Any] | None, tuple[int, int, int], np.ndarray, np.ndarray]:
        item_mask = np.zeros(len(ITEMS), dtype=np.bool_)
        quantity_mask = np.zeros(len(QUANTITY_BUCKETS), dtype=np.bool_)
        operation_id = _masked_choice(op_logits, self._market_op_mask(state, obs), self.temperature)
        operation = MARKET_OPS[operation_id]
        if operation == "NONE":
            return None, (operation_id, 0, 0), item_mask, quantity_mask
        if operation == "HIRE":
            state.money -= _fib(state.hires_today)
            state.hires_today += 1
            return ["HIRE"], (operation_id, 0, 0), item_mask, quantity_mask
        if operation == "BUY_LAND":
            cost = (1000, 2000, 4000)[state.unlocked_land - 1]
            state.money -= cost
            state.unlocked_land += 1
            return ["BUY_LAND"], (operation_id, 0, 0), item_mask, quantity_mask
        item_mask = self._market_item_mask(operation, state, obs)
        if not item_mask.any():
            return None, (MARKET_OP_TO_ID["NONE"], 0, 0), item_mask, quantity_mask
        item_id = _masked_choice(item_logits, item_mask, self.temperature)
        item = ITEMS[item_id]
        quantity_mask[1:] = True
        quantity_id = _masked_choice(quantity_logits, quantity_mask, self.temperature)
        requested = max(1, QUANTITY_BUCKETS[quantity_id])
        prices = obs.get("market", {}).get("prices", {})
        if operation == "BUY_SEED":
            quantity = min(requested, int(state.money // SEED_COST[item]))
            state.money -= quantity * SEED_COST[item]
            state.seeds[item] = int(state.seeds.get(item, 0)) + quantity
        elif operation == "BUY_ANIMAL":
            quantity = min(requested, int(state.money // ANIMAL_COST[item]), 100 - state.shed_occupancy)
            state.money -= quantity * ANIMAL_COST[item]
            state.shed[item] = int(state.shed.get(item, 0)) + quantity
        elif operation == "BUY_PRODUCT":
            price = max(1, int(prices.get(item, 1)))
            quantity = min(requested, int(state.money // price), 100 - state.shed_occupancy)
            state.money -= quantity * price
            state.shed[item] = int(state.shed.get(item, 0)) + quantity
        else:
            quantity = min(requested, int(state.shed.get(item, 0)))
            state.shed[item] = int(state.shed.get(item, 0)) - quantity
            state.money += quantity * max(1, int(prices.get(item, 1)))
        if quantity <= 0:
            return None, (MARKET_OP_TO_ID["NONE"], 0, 0), item_mask, quantity_mask
        return [operation, item, quantity], (operation_id, item_id, quantity_id), item_mask, quantity_mask

    @torch.inference_mode()
    def _act(self, obs: dict[str, Any], with_trace: bool) -> tuple[dict[str, Any], dict[str, Any] | None]:
        encoded = encode_observation(obs)
        tensors = {
            "board": torch.from_numpy(encoded["board"]).unsqueeze(0).to(self.device),
            "global": torch.from_numpy(encoded["global"]).unsqueeze(0).to(self.device),
            "units": torch.from_numpy(encoded["units"]).unsqueeze(0).to(self.device),
            "unit_mask": torch.from_numpy(encoded["unit_mask"]).unsqueeze(0).to(self.device),
        }
        global_context, unit_context = self.model.encode(
            tensors["board"], tensors["global"], tensors["units"], tensors["unit_mask"]
        )
        state = ReservationState.from_observation(obs)
        unit_legal = legal_unit_op_matrix(obs)
        unit_hidden, previous = self.model.unit_decoder.initial_state(global_context)
        unit_commands = []
        unit_count = int(encoded["unit_mask"].sum())
        unit_choices = np.zeros((MAX_UNITS, 3), dtype=np.int64)
        unit_component_active = np.zeros((MAX_UNITS, 3), dtype=np.bool_)
        unit_op_legal = np.zeros((MAX_UNITS, len(UNIT_OPS)), dtype=np.bool_)
        unit_item_legal = np.zeros((MAX_UNITS, len(ITEMS)), dtype=np.bool_)
        unit_quantity_legal = np.zeros((MAX_UNITS, len(QUANTITY_BUCKETS)), dtype=np.bool_)
        log_probability = torch.zeros((), device=self.device)
        for slot in range(unit_count):
            logits, unit_hidden = self.model.unit_decoder.step(
                unit_context[:, slot], slot, unit_hidden, previous
            )
            operation_mask = unit_legal[slot].copy()
            command, chosen, item_mask, quantity_mask = self._unit_command(
                obs,
                state,
                slot,
                logits["op"][0],
                logits["item"][0],
                logits["quantity"][0],
                operation_mask,
            )
            unit_choices[slot] = chosen
            unit_component_active[slot, 0] = True
            unit_op_legal[slot] = operation_mask
            log_probability += _masked_log_probability(
                logits["op"][0], operation_mask, chosen[0], self.temperature
            )
            if item_mask.any():
                unit_component_active[slot, 1] = True
                unit_item_legal[slot] = item_mask
                log_probability += _masked_log_probability(
                    logits["item"][0], item_mask, chosen[1], self.temperature
                )
            if quantity_mask.any():
                unit_component_active[slot, 2] = True
                unit_quantity_legal[slot] = quantity_mask
                log_probability += _masked_log_probability(
                    logits["quantity"][0], quantity_mask, chosen[2], self.temperature
                )
            previous = tuple(torch.tensor([value], device=self.device) for value in chosen)
            unit_commands.append(command)

        market_hidden, previous = self.model.market_decoder.initial_state(global_context)
        orders = []
        market_choices = np.zeros((MAX_MARKET_ORDERS, 3), dtype=np.int64)
        market_component_active = np.zeros((MAX_MARKET_ORDERS, 3), dtype=np.bool_)
        market_op_legal = np.zeros((MAX_MARKET_ORDERS, len(MARKET_OPS)), dtype=np.bool_)
        market_item_legal = np.zeros((MAX_MARKET_ORDERS, len(ITEMS)), dtype=np.bool_)
        market_quantity_legal = np.zeros(
            (MAX_MARKET_ORDERS, len(QUANTITY_BUCKETS)), dtype=np.bool_
        )
        for slot in range(MAX_MARKET_ORDERS):
            context = global_context + self.model.market_positions[:, slot]
            logits, market_hidden = self.model.market_decoder.step(context, slot, market_hidden, previous)
            operation_mask = self._market_op_mask(state, obs)
            order, chosen, item_mask, quantity_mask = self._market_order(
                obs, state, logits["op"][0], logits["item"][0], logits["quantity"][0]
            )
            market_choices[slot] = chosen
            market_component_active[slot, 0] = True
            market_op_legal[slot] = operation_mask
            log_probability += _masked_log_probability(
                logits["op"][0], operation_mask, chosen[0], self.temperature
            )
            if item_mask.any():
                market_component_active[slot, 1] = True
                market_item_legal[slot] = item_mask
                log_probability += _masked_log_probability(
                    logits["item"][0], item_mask, chosen[1], self.temperature
                )
            if quantity_mask.any():
                market_component_active[slot, 2] = True
                market_quantity_legal[slot] = quantity_mask
                log_probability += _masked_log_probability(
                    logits["quantity"][0], quantity_mask, chosen[2], self.temperature
                )
            previous = tuple(torch.tensor([value], device=self.device) for value in chosen)
            if order is None:
                break
            orders.append(order)
        action = {
            "farmer": unit_commands[0] if unit_commands else ["PASS"],
            "hands": unit_commands[1:],
            "market": orders,
        }
        if not with_trace:
            return action, None
        trace = {
            **encoded,
            "unit_choices": unit_choices,
            "unit_component_active": unit_component_active,
            "unit_op_legal": unit_op_legal,
            "unit_item_legal": unit_item_legal,
            "unit_quantity_legal": unit_quantity_legal,
            "market_choices": market_choices,
            "market_component_active": market_component_active,
            "market_op_legal": market_op_legal,
            "market_item_legal": market_item_legal,
            "market_quantity_legal": market_quantity_legal,
            "old_log_probability": np.float32(log_probability.item()),
            "old_value": np.float32(self.model.value_head(global_context).item()),
            "policy_temperature": np.float32(max(self.temperature, 1e-6)),
        }
        return action, trace

    def act_with_trace(
        self, obs: dict[str, Any], configuration: Any = None
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        action, trace = self._act(obs, with_trace=True)
        assert trace is not None
        return action, trace

    def __call__(self, obs: dict[str, Any], configuration: Any = None) -> dict[str, Any]:
        action, _ = self._act(obs, with_trace=False)
        return action
