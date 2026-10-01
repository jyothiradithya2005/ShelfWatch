from pathlib import Path

BASE_DIR = Path(__file__).parent
MODEL_PATH = BASE_DIR / "best.pt"
COCO_MODEL = "yolo11n.pt"

DEFAULT_CONFIDENCE = 0.35
DEFAULT_THRESHOLD = 5
SMOOTHING_WINDOW = 15
DEFAULT_TRACKED = ["tomato", "onion", "carrot", "potato"]

REORDER_LINKS = {
    "tomato": "https://www.bigbasket.com/ps/?q=tomato",
    "onion": "https://blinkit.com/s/?q=onion",
    "carrot": "https://www.swiggy.com/instamart/search?query=carrot",
    "potato": "https://www.bigbasket.com/ps/?q=potato",
}


def reorder_link(item):
    return REORDER_LINKS.get(item, f"https://www.bigbasket.com/ps/?q={item.replace('_', '%20')}")
