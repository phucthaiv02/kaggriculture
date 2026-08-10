from __future__ import annotations

BOARD_SIZE = 10
MAX_UNITS = 32
MAX_MARKET_ORDERS = 10
BOARD_CHANNELS_PER_PLAYER = 22
BOARD_CHANNELS = BOARD_CHANNELS_PER_PLAYER * 2
GLOBAL_FEATURES = 70
UNIT_FEATURES = 36

CROPS = ("WHEAT", "CARROT", "TOMATO", "STRAWBERRY", "MELON")
ANIMALS = ("GOOSE", "COW", "SHEEP")
PRODUCTS = ("WHEAT", "CARROT", "TOMATO", "STRAWBERRY", "MELON", "EGG", "MILK", "WOOL", "FERTILIZER")
ITEMS = (
    "NONE",
    "WHEAT",
    "CARROT",
    "TOMATO",
    "STRAWBERRY",
    "MELON",
    "EGG",
    "MILK",
    "WOOL",
    "FERTILIZER",
    "GOOSE",
    "COW",
    "SHEEP",
)
SHED_ITEMS = ITEMS[1:]

UNIT_OPS = (
    "PASS",
    "NORTH",
    "SOUTH",
    "EAST",
    "WEST",
    "PICKUP",
    "PLACE",
    "DROP",
    "PLANT",
    "WATER",
    "HARVEST",
    "FERTILIZE",
    "BUILD_COOP",
    "BUILD_PASTURE",
    "FEED",
    "COLLECT_FERTILIZER",
    "CARE",
    "DIG",
)
MARKET_OPS = ("NONE", "BUY_SEED", "BUY_PRODUCT", "BUY_ANIMAL", "SELL", "HIRE", "BUY_LAND")
QUANTITY_BUCKETS = (0, 1, 2, 3, 4, 5, 6, 8, 10, 12, 16, 20, 24, 25, 32, 40, 50, 64, 75, 100)

ITEM_TO_ID = {name: index for index, name in enumerate(ITEMS)}
UNIT_OP_TO_ID = {name: index for index, name in enumerate(UNIT_OPS)}
MARKET_OP_TO_ID = {name: index for index, name in enumerate(MARKET_OPS)}

SHOPS = (
    "BAKERY",
    "PIZZA_SHOP",
    "BRUNCH_SPOT",
    "YARN_STORE",
    "ICE_CREAM_SHOP",
    "PET_CAFE",
    "SMOOTHIE_SHOP",
    "FARMERS_MARKET",
)

CROP_RULES = {
    "WHEAT": {"first": 2, "max": 4, "ongoing": False},
    "CARROT": {"first": 2, "max": 3, "ongoing": False},
    "TOMATO": {"first": 8, "max": 8, "ongoing": True},
    "STRAWBERRY": {"first": 10, "max": 10, "ongoing": True},
    "MELON": {"first": 10, "max": 12, "ongoing": False},
}
SEED_COST = {"WHEAT": 10, "CARROT": 20, "TOMATO": 50, "STRAWBERRY": 100, "MELON": 80}
ANIMAL_COST = {"GOOSE": 300, "COW": 400, "SHEEP": 500}
ANIMAL_STRUCTURE = {"GOOSE": "COOP", "COW": "PASTURE", "SHEEP": "PASTURE"}
PRODUCT_BASE_PRICE = {
    "WHEAT": 25,
    "CARROT": 35,
    "TOMATO": 60,
    "STRAWBERRY": 120,
    "MELON": 250,
    "EGG": 50,
    "MILK": 160,
    "WOOL": 200,
    "FERTILIZER": 100,
}

MOVE_DELTA = {"NORTH": (0, -1), "SOUTH": (0, 1), "EAST": (1, 0), "WEST": (-1, 0)}
SHED_ACCESS = frozenset(((4, 4), (5, 4), (4, 5), (5, 5)))
