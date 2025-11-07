# generate_all_qr.py
# One-time script to generate QR codes for every facade in data/buildings.jsonl
#
# Requirements:
#   pip install qrcode[pil] pillow

import json
from pathlib import Path
import urllib.parse

import qrcode
from PIL import Image, ImageDraw, ImageFont

# ------------------- Paths & settings -------------------

ROOT = Path(__file__).parent.resolve()
DATA_DIR = ROOT / "data"
CARDS_PATH = DATA_DIR / "buildings.jsonl"
QR_DIR = DATA_DIR / "qr"
QR_DIR.mkdir(parents=True, exist_ok=True)

# 👉 your font is here:
FONTS_DIR = DATA_DIR / "fonts"
FONT_PATH = FONTS_DIR / "NotoSansSC-Regular.ttf"

# <<< CHANGE THIS to your deployed app URL >>>
BASE_URL = "https://facade-analysis-system.zeabur.app/"

# ------------------- Helpers -------------------

def load_cards_jsonl(path: Path):
    """Load JSONL file like in app.py (robust to streaming-style writes)."""
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
                # line belongs to a multi-line JSON block; keep accumulating
                pass
    if buf.strip():
        raise ValueError("Incomplete JSON object at end of file")
    return cards


def _measure_text(draw_obj, text, font_obj):
    """Robust text size measurement (no draw.textsize in recent Pillow)."""
    try:
        bbox = draw_obj.textbbox((0, 0), text, font=font_obj)
        return bbox[2] - bbox[0], bbox[3] - bbox[1]
    except Exception:
        pass
    try:
        bbox = font_obj.getbbox(text)
        return bbox[2] - bbox[0], bbox[3] - bbox[1]
    except Exception:
        pass
    try:
        return font_obj.getsize(text)
    except Exception:
        return (8 * len(text), 18)


def _load_title_font(size: int = 28) -> ImageFont.FreeTypeFont:
    """
    Load the NotoSansSC font from data/fonts/NotoSansSC-Regular.ttf.
    If that fails, raise a clear error.
    """
    if not FONT_PATH.exists():
        raise FileNotFoundError(
            f"CJK font file not found at: {FONT_PATH}\n"
            "Make sure NotoSansSC-Regular.ttf is in data/fonts/"
        )

    try:
        print(f"[QR] Using CJK font: {FONT_PATH}")
        return ImageFont.truetype(str(FONT_PATH), size)
    except Exception as e:
        raise RuntimeError(f"Could not open font {FONT_PATH}: {e}")


def generate_qr_for_building(card: dict, base_url: str) -> str:
    """
    Generate a QR code PNG for one building card.
    Format:
    - URL: <base_url>/?facade_id=<id>
    - QR with a title band (EN / ZH) above it
    - Image upscaled for sharper text
    Returns the path to the saved PNG (string).
    """
    b_id = str(card.get("id", "")).strip()
    if not b_id:
        return ""

    # URL encoded id
    target_url = f"{base_url.rstrip('/')}/?facade_id={urllib.parse.quote(b_id)}"

    # --- 1) Base QR ---
    qr = qrcode.QRCode(version=1, box_size=12, border=4)
    qr.add_data(target_url)
    qr.make(fit=True)
    qr_img = qr.make_image(fill_color="black", back_color="white").convert("RGB")
    W, H = qr_img.size

    # --- 2) Title text (EN / ZH) ---
    name_en = (card.get("name") or "").strip()
    name_zh = (card.get("name_zh") or "").strip()
    if name_en and name_zh:
        title = f"{name_en} / {name_zh}"
    else:
        title = name_en or name_zh

    # No title? Just save the QR
    if not title:
        out_path = QR_DIR / f"{b_id}_qr.png"
        qr_img.save(out_path)
        return str(out_path)

    # Load CJK-capable font
    font = _load_title_font(28)

    title_band_height = 80
    temp_canvas = Image.new("RGB", (W, H + title_band_height), "white")
    temp_draw = ImageDraw.Draw(temp_canvas)
    text_w, text_h = _measure_text(temp_draw, title, font)
    max_width = W - 20

    # Shrink font a bit if text is too wide
    while text_w > max_width and getattr(font, "size", None) and font.size > 16:
        try:
            font = ImageFont.truetype(str(FONT_PATH), font.size - 2)
        except Exception:
            break
        text_w, text_h = _measure_text(temp_draw, title, font)

    # --- 3) Compose final QR image (title + QR) ---
    canvas = Image.new("RGB", (W, H + title_band_height), "white")
    draw = ImageDraw.Draw(canvas)

    text_x = (W - text_w) // 2
    text_y = (title_band_height - text_h) // 2
    draw.text((text_x, text_y), title, fill="black", font=font)

    canvas.paste(qr_img, (0, title_band_height))

    # Upscale for sharpness
    scale = 2
    big = canvas.resize((canvas.width * scale, canvas.height * scale), Image.LANCZOS)

    out_path = QR_DIR / f"{b_id}_qr.png"
    big.save(out_path)
    return str(out_path)


# ------------------- Main batch job -------------------

def main():
    cards = load_cards_jsonl(CARDS_PATH)
    if not cards:
        print(f"No cards found in {CARDS_PATH}")
        return

    print(f"Found {len(cards)} building entries.")
    for i, card in enumerate(cards, start=1):
        b_id = card.get("id", f"#{i}")
        try:
            out_path = generate_qr_for_building(card, BASE_URL)
            if out_path:
                print(f"[{i}/{len(cards)}] QR generated for {b_id}: {out_path}")
            else:
                print(f"[{i}/{len(cards)}] Skipped {b_id} (missing id)")
        except Exception as e:
            print(f"[{i}/{len(cards)}] Failed for {b_id}: {e}")

    print("\nDone. All QR codes are in:", QR_DIR)


if __name__ == "__main__":
    main()
