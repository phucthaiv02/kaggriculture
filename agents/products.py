"""Shared producer names and purchase costs from the game rules."""
from kaggle_environments.envs.kaggriculture import kaggriculture as game

CROPS = ("WHEAT", "CARROT", "MELON", "TOMATO", "STRAWBERRY")
ANIMALS = ("GOOSE", "COW", "SHEEP")
SEED_COST = {name: game.CROPS[name]["seed"] for name in CROPS}
ANIMAL_COST = {name: game.ANIMALS[name]["cost"] for name in ANIMALS}
ANIMAL_STRUCTURE = {name: game.ANIMALS[name]["structure"] for name in ANIMALS}
BUILD = {name: f"BUILD_{structure}" for name, structure in ANIMAL_STRUCTURE.items()}
