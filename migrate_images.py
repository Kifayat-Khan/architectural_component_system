# migrate_images.py
import json
from pathlib import Path
from glob import glob

ROOT = Path(__file__).parent.resolve()
DATA_DIR = ROOT / "data"
CARDS_PATH = DATA_DIR / "buildings.jsonl"

def load_cards_jsonl(path: Path):
    cards = []
    buf = ""
    if not path.exists():
        return cards
    with path.open("r", encoding="utf-8") as f:
        for raw in f:
            line = raw.strip()
            if not line or line == "data:":
                continue
            buf = f"{buf}{(' ' if buf else '')}{line}"
            try:
                obj = json.loads(buf)
                cards.append(obj)
                buf = ""
            except json.JSONDecodeError:
                pass
    if buf.strip():
        raise ValueError("Incomplete JSON object at end of file")
    return cards

def save_cards_jsonl(path: Path, cards):
    with path.open("w", encoding="utf-8") as f:
        for c in cards:
            f.write(json.dumps(c, ensure_ascii=False) + "\n")

def main():
    cards = load_cards_jsonl(CARDS_PATH)
    updated = 0

    for c in cards:
        cid = c.get("id")
        if not cid:
            continue

        # if images already set, don’t touch it
        if c.get("images"):
            continue

        # assume images live in data/<id>/*
        folder = DATA_DIR / cid
        if not folder.exists():
            continue

        files = []
        for ext in ("*.jpg", "*.jpeg", "*.png", "*.JPG", "*.JPEG", "*.PNG"):
            files.extend(glob(str(folder / ext)))

        files = sorted(files)
        if not files:
            continue

        # store relative paths if you prefer
        c["images"] = files
        updated += 1

    save_cards_jsonl(CARDS_PATH, cards)
    print(f"Updated {updated} cards with images[]")

if __name__ == "__main__":
    main()
