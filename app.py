# =========================
# app.py — PART 1 of 3
# =========================
# Fast, no local LLMs. Uses Pollinations text API for narrative & explanations.
# Deps:
# pip install -U streamlit pillow numpy scikit-image matplotlib reportlab \
#   opencv-python-headless requests
import os
import io, re, math, time, json, hashlib, random, urllib.parse
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple

import numpy as np
from PIL import Image, ImageDraw, ImageFont
import cv2
import streamlit as st
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from skimage.metrics import structural_similarity as ssim
from skimage.color import rgb2gray
from skimage.filters import sobel

from reportlab.pdfbase.pdfmetrics import stringWidth
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.cidfonts import UnicodeCIDFont  # built-in CJK
from reportlab.lib.pagesizes import A4
from reportlab.pdfgen import canvas
from reportlab.lib.utils import ImageReader

import requests

#--qr code imports
import qrcode
from qrcode.image.pil import PilImage





# --------- Runtime knobs (variables; no env) ----------
NUM_THREADS     = 4
MAX_SIDE        = 1024
CACHE_TTL_MIN   = 240
API_BASE        = "https://text.pollinations.ai"   # free text endpoint
API_MODEL       = "gpt-4o-mini"                    # not used directly; Pollinations picks model
KEY_NS          = "acs_v3p"                        # widget key namespace

REINDEX_IF_EMPTY = True
# ---- UI display toggles ----
SHOW_DB_MATCH_UI   = False   # hide DB match banner/caption
SHOW_OVERLAY_UI    = True    # SHOW overlay image on page
SHOW_HEATMAP_UI    = True    # SHOW heatmap image on page

# If you ever want to speed up by skipping computation (PDF will also lose them if False):
COMPUTE_OVERLAY    = True
COMPUTE_HEATMAP    = True

# --- fonts for PDF (CJK ready)
pdfmetrics.registerFont(UnicodeCIDFont("MSung-Light"))
pdfmetrics.registerFont(UnicodeCIDFont("STSong-Light"))

# --- app title and state
st.set_page_config(page_title="AI Analysis of Facade", layout="wide")
st.title("AI Analysis of Facade / 立面人工智能分析")
st.write("")
if "flash" in st.session_state:
    st.success(st.session_state.pop("flash"))

# ---- absolute paths
ROOT = Path(__file__).parent.resolve()
DATA_DIR = ROOT / "data"
DATA_DIR.mkdir(exist_ok=True)
CARDS_PATH = DATA_DIR / "buildings.jsonl"
IDX_PATH   = DATA_DIR / "index_lab.npz"           # lightweight index
Path("outputs").mkdir(exist_ok=True)
HIST_PATH = Path("outputs/history.json")

# Base URL of your deployed app (update this when you deploy)
BASE_URL = "https://facade-analysis-system.zeabur.app/"

QR_DIR = DATA_DIR / "qr"
QR_DIR.mkdir(exist_ok=True)

#---helper for qr code 
def generate_qr_for_facade(facade_id: str) -> Tuple[str, str]:
    """
    Generate a QR code PNG for this facade id.

    Returns:
      (qr_image_path, qr_target_url)
    """
    # The URL that will open this facade via QR (you already handle this in your QR logic)
    target = f"{APP_BASE_URL}/?facade_id={facade_id}"

    # Save QR PNG under data/qr/{facade_id}.png
    out_path = QR_DIR / f"{facade_id}.png"

    qr = qrcode.QRCode(
        version=1,
        error_correction=qrcode.constants.ERROR_CORRECT_M,
        box_size=8,
        border=2,
    )
    qr.add_data(target)
    qr.make(fit=True)
    img: PilImage = qr.make_image(fill_color="black", back_color="white")
    img.save(out_path)

    return str(out_path), target


def load_index(index_npz: str) -> Tuple[List[str], List[str], List[str], np.ndarray, str]:
    index_npz = str(index_npz)
    if not Path(index_npz).exists():
        # ids, facade_ids, img_paths, vecs, cards_path
        return [], [], [], np.zeros((0, 96), np.float32), str(CARDS_PATH)
    try:
        data = np.load(index_npz, allow_pickle=True)
        vecs = data["vecs"]
        cards_path = str(data["cards_path"])

        # Backward compatible
        ids = list(data["ids"]) if "ids" in data else []
        facade_ids = list(data["facade_ids"]) if "facade_ids" in data else ids
        img_paths = list(data["img_paths"]) if "img_paths" in data else []

        return ids, facade_ids, img_paths, vecs, cards_path
    except Exception:
        return [], [], [], np.zeros((0, 96), np.float32), str(CARDS_PATH)

# ---------------- Sidebar ----------------
with st.sidebar:
    st.header("Settings")
    lang_choice = st.selectbox(
        "Language / 語言",
        ["English", "中文 (简体)"],
        index=0,
        key=f"{KEY_NS}_lang"
    )
    LANG = "zh" if lang_choice.startswith("中文") else "en"

    st.markdown("---")
    page = st.radio(
        "App Section",
        ["Analysis", "Database Manager"],
        index=0,
        key=f"{KEY_NS}_page"
    )
with st.sidebar.expander("Debug: index status"):
    try:
        ids, facade_ids, img_paths, vecs, cards_path = load_index(str(IDX_PATH))
        st.write(f"Vectors: {vecs.shape[0]}")
        st.write(f"Unique facades: {len(set(facade_ids))}")
        # show a few sample rows
        for i in range(min(5, len(ids))):
            st.caption(f"{facade_ids[i]} -> {img_paths[i]}")
    except Exception as e:
        st.error(f"Index error: {e}")

# ---------- DB I/O ----------
def load_cards_jsonl(path: str) -> List[Dict[str, Any]]:
    cards, buf = [], ""
    p = Path(path)
    if not p.exists():
        return cards
    with p.open("r", encoding="utf-8") as f:
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

def _read_image_rgb(path: str):
    try:
        return cv2.cvtColor(cv2.imread(path), cv2.COLOR_BGR2RGB)
    except Exception:
        return None

def _resize_max_side(img: np.ndarray, max_side: int = 720) -> np.ndarray:
    h, w = img.shape[:2]
    if max(h, w) <= max_side:
        return img
    scale = max_side / float(max(h, w))
    return cv2.resize(img, (int(w*scale), int(h*scale)), interpolation=cv2.INTER_AREA)

def _lab_hist_descriptor(img_rgb: np.ndarray, bins: int = 32) -> np.ndarray:
    # Convert to LAB and build concatenated hist (L,a,b) normalized
    lab = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2LAB)
    hL = cv2.calcHist([lab],[0],None,[bins],[0,256]).flatten()
    ha = cv2.calcHist([lab],[1],None,[bins],[0,256]).flatten()
    hb = cv2.calcHist([lab],[2],None,[bins],[0,256]).flatten()
    v = np.concatenate([hL, ha, hb]).astype(np.float32)
    v /= (np.linalg.norm(v) + 1e-8)
    return v

def _empty_index(index_npz: str, cards_path: str, dim: int = 96) -> None:
    Path(index_npz).parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        index_npz,
        ids=np.array([], dtype=object),
        facade_ids=np.array([], dtype=object),
        img_paths=np.array([], dtype=object),
        vecs=np.zeros((0, dim), dtype=np.float32),
        cards_path=str(cards_path),
    )

def build_index(cards_path: str, index_npz: str) -> None:
    cards_path = str(cards_path)
    index_npz = str(index_npz)

    if not Path(cards_path).exists():
        _empty_index(index_npz, cards_path)
        return

    try:
        cards = load_cards_jsonl(cards_path)
    except Exception:
        _empty_index(index_npz, cards_path)
        return

    if not cards:
        _empty_index(index_npz, cards_path)
        return

    vecs, ids, facade_ids, img_paths = [], [], [], []

    for c in cards:
        f_id = c.get("id")
        if not f_id:
            continue

        # new schema: list of images
        imgs = c.get("images") or []

        # optional legacy fallback if some cards still have "image"
        if not imgs and c.get("image"):
            imgs = [c["image"]]

        for p in imgs:
            if not p:
                continue
            if not Path(p).exists():
                continue
            rgb = _read_image_rgb(p)
            if rgb is None:
                continue

            v = _lab_hist_descriptor(_resize_max_side(rgb, 720))

            vecs.append(v)
            ids.append(f"{f_id}::{Path(p).name}")  # per-image id
            facade_ids.append(f_id)
            img_paths.append(p)

    if not vecs:
        _empty_index(index_npz, cards_path)
        return

    arr = np.stack(vecs, axis=0).astype(np.float32)
    np.savez(
        index_npz,
        ids=np.array(ids, dtype=object),
        facade_ids=np.array(facade_ids, dtype=object),
        img_paths=np.array(img_paths, dtype=object),
        vecs=arr,
        cards_path=str(cards_path),
    )

def ensure_index(cards_path: str, index_npz: str) -> None:
    """Rebuild index if missing, empty, or pointing at a different cards file."""
    try:
        if not Path(index_npz).exists():
            build_index(cards_path, index_npz); return
        data = np.load(index_npz, allow_pickle=True)
        vecs = data["vecs"]; saved_cards = str(data["cards_path"])
        if vecs.shape[0] == 0 or Path(saved_cards).resolve() != Path(cards_path).resolve():
            build_index(cards_path, index_npz)
    except Exception:
        build_index(cards_path, index_npz)

#---search by name helper 
def _norm_key(s: str) -> str:
    """Normalize a search string / name for matching (works for English + Chinese)."""
    if not isinstance(s, str):
        return ""
    s = s.strip().lower()
    # remove normal and full-width spaces
    s = s.replace(" ", "").replace("\u3000", "")
    return s

def card_matches_name_query(card: dict, query: str) -> bool:
    q = _norm_key(query)
    if not q:
        return False

    name_en = _norm_key(card.get("name", ""))
    name_zh = _norm_key(card.get("name_zh", ""))

    # allow partial matches
    return (q in name_en) or (q in name_zh)

#search by id helper 
def _norm_id(s: str | None) -> str:
    if not s:
        return ""
    return s.strip().lower()

#----qr code helpers

QR_DIR = DATA_DIR / "qr"
QR_DIR.mkdir(exist_ok=True)

def generate_qr_for_building(
    card: dict,
    base_url: str,
    out_dir: Path = QR_DIR,
) -> str:
    import urllib.parse, qrcode
    from PIL import Image, ImageDraw, ImageFont

    b_id = str(card.get("id", "")).strip()
    if not b_id:
        return ""

    out_dir.mkdir(parents=True, exist_ok=True)

    # URL the QR will open
    target_url = f"{base_url.rstrip('/')}/?facade_id={urllib.parse.quote(b_id)}"

    # --- 1) Base QR (slightly larger) ---
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

    if not title:
        out_path = out_dir / f"{b_id}_qr.png"
        qr_img.save(out_path)
        return str(out_path)

    # --- helper to measure text without draw.textsize ---
    def measure_text(draw_obj, text, font_obj):
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

    # --- 3) Choose a reasonably large font ---
    font = None
    for fname in ["arial.ttf", "DejaVuSans.ttf"]:
        try:
            font = ImageFont.truetype(fname, 28)   # bigger size
            break
        except Exception:
            continue
    if font is None:
        font = ImageFont.load_default()

    title_band_height = 80  # more space for the text
    temp_canvas = Image.new("RGB", (W, H + title_band_height), "white")
    temp_draw = ImageDraw.Draw(temp_canvas)

    text_w, text_h = measure_text(temp_draw, title, font)
    max_width = W - 20

    # Shrink font only if really necessary
    while text_w > max_width and getattr(font, "size", None) and font.size > 16:
        try:
            font = ImageFont.truetype(font.path, font.size - 2)  # type: ignore[attr-defined]
        except Exception:
            break
        text_w, text_h = measure_text(temp_draw, title, font)

    # --- 4) Compose title + QR ---
    canvas = Image.new("RGB", (W, H + title_band_height), "white")
    draw = ImageDraw.Draw(canvas)

    text_x = (W - text_w) // 2
    text_y = (title_band_height - text_h) // 2
    draw.text((text_x, text_y), title, fill="black", font=font)

    canvas.paste(qr_img, (0, title_band_height))

    # --- 5) Upscale for sharper text ---
    scale = 2
    big = canvas.resize((canvas.width * scale, canvas.height * scale), Image.LANCZOS)

    out_path = out_dir / f"{b_id}_qr.png"
    big.save(out_path)
    return str(out_path)



def generate_qr_guide_text(
    db_info: str,
    lang: str = "en",
    image_data_url: Optional[str] = None
) -> str:
    """
    Short, tourist-friendly explanation based ONLY on db_info.
    Used for QR-code info page (no metrics, no image analysis).
    """

    if not db_info:
        return "No detailed information is available for this building yet."

    if lang == "zh":
        system_text = (
            "你是一位在台湾带团的建筑导览员，正在为游客介绍眼前的建筑。"
            "语气要：亲切、清楚、具有教学性，但不要太学术，也不要太幼稚。\n"
            "重要规则：\n"
            "1. 只能根据提供的建筑信息来讲解，禁止编造新的具体事实（例如材料、年代、建筑师、用途等）。\n"
            "2. 不要说“这张图片中你可以看到”“如图所示”等元话语，只假设游客正站在建筑前面。\n"
            "3. 不要打招呼（不要用“大家好”“欢迎各位”之类）。\n"
            "4. 用简单的中文说明建筑在哪里、属于什么年代/风格、有怎样的体量和立面特征，"
            "以及它在城市或文化中的意义，适合普通游客和学生理解。"
        )
        prompt = (
            "以下是数据库中的建筑信息，请你把它转化为现场导览用的说明文字：\n"
            f"{db_info}\n\n"
            "请写 2–3 小段简短文字：\n"
            "第 1 段：介绍建筑的名称、位置、年代和大致用途，以及它为什么重要（例如是地标、文化据点、保存再利用等）。\n"
            "第 2 段：用简单的词，帮游客“看懂”这个立面的主要特点，比如体量、材料、颜色、屋顶或立面构成方式，"
            "以及和历史或当地生活的关系。\n"
            "如有需要，可以加第 3 段，说明游客在这里可以学习到什么建筑或文化概念。"
        )
    else:
        system_text = (
            "You are a friendly on-site tour guide in Taichung, explaining a building to visitors. "
            "Your tone is clear, calm, and educational – suitable for tourists and students.\n"
            "Rules:\n"
            "1. You MUST only use facts from the DB text. Do NOT invent extra details "
            "about materials, dates, designers, height, or functions.\n"
            "2. Do NOT mention images, photos, slides, or QR codes. Assume the visitor is "
            "standing in front of the real building.\n"
            "3. Do NOT start with greetings like 'Welcome' or 'Today I will introduce'. "
            "Start directly with the building.\n"
            "4. Use simple, accessible language. Explain what the place is, where it is, its era and style, "
            "and why it matters for culture or history."
        )
        prompt = (
            "Here is the building DB information:\n"
            f"{db_info}\n\n"
            "Turn this into a short on-site guide explanation.\n"
            "Write 2–3 short paragraphs:\n"
            "Paragraph 1: Introduce what the building is, where in Taichung it is, its era, style, "
            "and why it is important (for example, a landmark, a preserved historic site, or a creative reuse).\n"
            "Paragraph 2: Help visitors 'read' the facade in simple words – mention massing, key elements "
            "(like murals, courtyards, roofs, arcades) that are listed in the DB, and how the place feels.\n"
            "Optional Paragraph 3: Explain what tourists and students can learn here about architecture or local culture."
        )

    return _pollinations_chat(
        prompt,
        system_text=system_text,
        image_data_url=image_data_url,
        api_base=API_BASE,
    )

# --- helpers: image -> data URL (for multimodal chat) ---
import base64

def _bytes_to_data_url(raw_bytes: bytes, mime: str = "image/jpeg") -> str:
    try:
        # ensure JPEG for size + compat
        img = Image.open(io.BytesIO(raw_bytes)).convert("RGB")
        buf = io.BytesIO()
        img.save(buf, format="JPEG", quality=85, optimize=True, subsampling=1)
        b64 = base64.b64encode(buf.getvalue()).decode("ascii")
        return f"data:{mime};base64,{b64}"
    except Exception:
        # fallback: still try to pass what we have
        b64 = base64.b64encode(raw_bytes).decode("ascii")
        return f"data:{mime};base64,{b64}"

def _make_info(card: dict) -> str:
    parts = []
    if card.get("name") and card.get("location"):
        parts.append(f"{card['name']} is in {card['location']}.")
    if card.get("era"): parts.append(f"Era: {card['era']}.")
    if card.get("style"): parts.append(f"Style: {card['style']}.")
    if card.get("massing"): parts.append(f"Form/Massing: {card['massing']}.")
    if card.get("structure"): parts.append(f"Structure: {card['structure']}.")
    if card.get("condition"): parts.append(f"Condition: {card['condition']}.")
    if card.get("materials"): parts.append("Materials: " + ", ".join(card["materials"]) + ".")
    if card.get("elements"): parts.append("Elements: " + ", ".join(card["elements"]) + ".")
    if card.get("intro"): parts.append("Intro: " + card["intro"])
    if card.get("history"): parts.append("History: " + card["history"])
    return " ".join(p.strip() for p in parts if p and p.strip())

def _t_en_zh(en: str, zh: str, lang: str) -> str:
    return zh if lang == "zh" else en

def _t_en_zh(en: str, zh: str, lang: str) -> str:
    return zh if lang == "zh" else en

def _db_labels(lang: str) -> dict:
    return {
        "header":      _t_en_zh("📚 Building Database Manager", "📚 建筑数据库管理", lang),
        "info":        _t_en_zh("Add building entries for retrieval and grounding.",
                                "添加建筑条目用于检索与叙事实据。", lang),
        "name":        _t_en_zh("Building Name", "建筑名称", lang),          # ✅ THIS MUST EXIST
        "location":    _t_en_zh("Location (City, Area)", "位置（城市、区域）", lang),
        "era":         _t_en_zh("Era / Period", "时代 / 时期", lang),
        "style":       _t_en_zh("Style", "风格", lang),
        "massing":     _t_en_zh("Form / Massing", "形体 / 体量", lang),
        "structure":   _t_en_zh("Structure", "结构", lang),
        "condition":   _t_en_zh("Condition", "保存状况", lang),
        "intro":       _t_en_zh("Introduction / Description", "简介 / 描述", lang),
        "history":     _t_en_zh("History / Notes", "历史 / 备注", lang),
        "materials":   _t_en_zh("Materials (comma-separated)", "材料（以英文逗号分隔）", lang),
        "elements":    _t_en_zh("Elements (comma-separated)", "要素（以英文逗号分隔）", lang),
        "upload":      _t_en_zh("Upload main facade image(s)", "上传立面主图（可多张）", lang),
        "add_btn":     _t_en_zh("Add to Database", "添加到数据库", lang),
        "err_name":    _t_en_zh("Name is required.", "请填写名称。", lang),
        "indexing":    _t_en_zh("Indexing (lightweight)…", "正在建立索引（轻量）…", lang),
        "added_ok":    _t_en_zh("Added '{name}' and rebuilt index.",
                                "已添加「{name}」并重建索引。", lang),
    }


# ---------------- Database Manager ----------------
if page == "Database Manager":
    LBL = _db_labels(LANG)

    st.header(LBL["header"])
    st.info(LBL["info"])

    # English + Chinese names
    name_en  = st.text_input(
        LBL["name"] + " (English)",
        key=f"{KEY_NS}_dm_name_en"
    )
    name_zh  = st.text_input(
        "建筑名称（中文，可选）" if LANG == "zh" else "Building Name (Chinese, optional)",
        key=f"{KEY_NS}_dm_name_zh"
    )

    location  = st.text_input(LBL["location"],  key=f"{KEY_NS}_dm_loc")
    era       = st.text_input(LBL["era"],       key=f"{KEY_NS}_dm_era")
    style     = st.text_input(LBL["style"],     key=f"{KEY_NS}_dm_style")
    massing   = st.text_input(LBL["massing"],   key=f"{KEY_NS}_dm_massing")
    structure = st.text_input(LBL["structure"], key=f"{KEY_NS}_dm_struct")
    condition = st.text_input(LBL["condition"], key=f"{KEY_NS}_dm_cond")
    intro     = st.text_area(LBL["intro"],      height=80, key=f"{KEY_NS}_dm_intro")
    history   = st.text_area(LBL["history"],    height=80, key=f"{KEY_NS}_dm_hist")
    materials = st.text_input(LBL["materials"], key=f"{KEY_NS}_dm_mat")
    elements  = st.text_input(LBL["elements"],  key=f"{KEY_NS}_dm_elem")

    image_file = st.file_uploader(
        LBL["upload"],
        type=["jpg","jpeg","png"],
        key=f"{KEY_NS}_dm_up",
        accept_multiple_files=True
    )

    if image_file and st.button(LBL["add_btn"], key=f"{KEY_NS}_dm_add"):
        # 🔴 FIX: check the English name variable, not `name`
        if not name_en:
            st.error(LBL["err_name"])
        else:
            # Create a new ID (you can change this if you use your own IDs)
            card_id = str(int(time.time()))

            img_dir = DATA_DIR / card_id
            img_dir.mkdir(parents=True, exist_ok=True)

            image_paths = []
            for idx, up in enumerate(image_file, start=1):
                pil_img = Image.open(up).convert("RGB")
                out_path = img_dir / f"img_{idx:03d}.jpg"
                pil_img.save(out_path)
                image_paths.append(str(out_path))

            # 🔴 FIX: use `name_en` and `name_zh` here
            card = {
                "id": card_id,
                "name": name_en,
                "name_zh": name_zh,
                "location": location,
                "era": era,
                "style": style,
                "massing": massing,
                "structure": structure,
                "condition": condition,
                "intro": intro,
                "history": history,
                "materials": [m.strip() for m in materials.split(",") if m.strip()],
                "elements": [e.strip() for e in elements.split(",") if e.strip()],
                "images": image_paths,
            }

            # build compact info string for grounding
            card["info"] = _make_info(card)

            # append to JSONL DB
            with open(CARDS_PATH, "a", encoding="utf-8") as f:
                f.write(json.dumps(card, ensure_ascii=False) + "\n")

        # Rebuild index (for visual retrieval)
        with st.spinner(LBL["indexing"]):
            build_index(str(CARDS_PATH), str(IDX_PATH))

        # Generate QR code with building name at the top
        try:
            qr_path = generate_qr_for_building(card, base_url=BASE_URL)

            st.success(LBL["added_ok"].format(name=name_en))

            st.caption(
                "QR code for this building (scan to open guide page):"
                if LANG != "zh" else
                "建筑二维码（扫码打开导览页面）："
            )
            st.image(qr_path, width=180)

            # Show the URL text (optional but useful for debugging/printing)
            qr_url = f"{BASE_URL.rstrip('/')}/?facade_id={card_id}"
            st.caption(qr_url)

        except Exception as e:
            # If QR generation fails, at least confirm DB + index success
            st.success(LBL["added_ok"].format(name=name_en))
            st.warning(f"QR generation failed: {e}")


    st.stop()  # don't run analysis on this page



# =========================
# app.py — PART 2 of 3
# =========================

# ---------- small image helpers ----------
def pil_from_upload(up):
    return Image.open(io.BytesIO(up.read())).convert("RGB")

def show_img(col_like, img, caption):
    if img is None:
        (col_like if hasattr(col_like, "info") else st).info(f"{caption} (not available)")
        return
    if not isinstance(img, Image.Image):
        try:
            img = Image.fromarray(np.array(img))
        except Exception:
            (col_like if hasattr(col_like, "warning") else st).warning(f"Could not render: {caption}")
            return
    col_like.image(img, caption=caption,   width='stretch')

# ---------- Lightweight indexing (LAB histogram) + ORB verification ----------

COLOR_MAP = {"window": (46, 204, 113), "arch": (241, 196, 15)}

def orb_inlier_ratio(query_path: str, cand_path: str, max_side: int = 720) -> float:
    q = _read_image_rgb(query_path); c = _read_image_rgb(cand_path)
    if q is None or c is None: return 0.0
    q = _resize_max_side(q, max_side); c = _resize_max_side(c, max_side)
    qg = cv2.cvtColor(q, cv2.COLOR_RGB2GRAY); cg = cv2.cvtColor(c, cv2.COLOR_RGB2GRAY)
    orb = cv2.ORB_create(1000)
    kq, dq = orb.detectAndCompute(qg, None)
    kc, dc = orb.detectAndCompute(cg, None)
    if dq is None or dc is None or len(kq) < 20 or len(kc) < 20: return 0.0
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
    matches = bf.knnMatch(dq, dc, k=2)
    good = [m for m, n in matches if m.distance < 0.75 * n.distance]
    if len(good) < 12: return 0.0
    src = np.float32([kq[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
    dst = np.float32([kc[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)
    H, mask = cv2.findHomography(src, dst, cv2.RANSAC, 3.0)
    if H is None or mask is None: return 0.0
    inliers = int(mask.sum())
    denom = max(1, min(len(kq), len(kc)))
    return float(inliers) / float(denom)

def nms_boxes(dets, iou_thr=0.30):
    if not dets:
        return []
    boxes = np.array([d["box"] for d in dets], dtype=np.float32)
    scores = np.array([d["score"] for d in dets], dtype=np.float32)
    x1, y1, x2, y2 = boxes[:,0], boxes[:,1], boxes[:,2], boxes[:,3]
    x2 = np.maximum(x2, x1); y2 = np.maximum(y2, y1)
    areas = (x2 - x1) * (y2 - y1)
    order = scores.argsort()[::-1].astype(np.int64)
    keep = []
    eps = 1e-6
    while order.size > 0:
        i = order[0]; keep.append(i)
        if order.size == 1: break
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])
        w = np.maximum(0.0, xx2 - xx1)
        h = np.maximum(0.0, yy2 - yy1)
        inter = w * h
        iou = inter / (areas[i] + areas[order[1:]] - inter + eps)
        inds = np.where(iou <= iou_thr)[0]
        order = order[inds + 1]
    return [dets[i] for i in keep]

def heuristic_components(pil_img):
    im = np.array(pil_img)
    g = cv2.cvtColor(im, cv2.COLOR_RGB2GRAY)
    g = cv2.equalizeHist(g)
    edges = cv2.Canny(g, 60, 160)
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    H, W = g.shape
    dets = []
    for c in contours:
        x,y,w,h = cv2.boundingRect(c)
        if w*h < 0.001*W*H: continue
        ar = float(w) / float(max(1,h))
        if 0.5 <= ar <= 2.0 and h > 12 and w > 12:
            dets.append({"label":"window","score":0.55,"box":[x,y,x+w,y+h]})
        roi = edges[y:y+h, x:x+w]
        if roi.size == 0: continue
        top = roi[:max(1,h//2),:]
        arc_ratio = float(top.sum()) / (float(roi.sum()) + 1e-6)
        if arc_ratio > 0.6 and h > 20 and w > 20:
            dets.append({"label":"arch","score":0.45,"box":[x,y,x+w,y+h]})
    return nms_boxes(dets, 0.30)[:150]

def draw_semantic_overlay(pil_img, dets, alpha=0.35):
    base = pil_img.copy().convert("RGBA")
    overlay = Image.new("RGBA", base.size, (0,0,0,0))
    draw = ImageDraw.Draw(overlay, "RGBA")
    for d in dets:
        x1,y1,x2,y2 = map(float, d["box"])
        color = COLOR_MAP.get(d["label"], (255,255,255))
        draw.rectangle([x1,y1,x2,y2], fill=(color[0], color[1], color[2], int(255*alpha)))
        draw.rectangle([x1,y1,x2,y2], outline=(color[0], color[1], color[2], 255), width=2)
    return Image.alpha_composite(base, overlay).convert("RGB")

def retrieve_verified_multiview(
    image_path: str,
    index_npz: str,
    base_threshold: float = 0.28,
    inlier_floor: float = 0.05,
    inlier_strong: float = 0.14,
    alpha: float = 0.80,
    k_per_facade: int = 3,
    k_global: int = 30,
) -> Tuple[Optional[Dict[str, Any]], float, Dict[str, Any]]:
    ids, facade_ids, img_paths, vecs, cards_path = load_index(index_npz)
    if vecs.shape[0] == 0:
        return None, 0.0, {"reason": "empty_index"}

    # query descriptor
    qrgb = _read_image_rgb(image_path)
    if qrgb is None:
        return None, 0.0, {"reason": "bad_query_image"}

    q = _lab_hist_descriptor(_resize_max_side(qrgb, 720))
    sims = vecs @ q
    if sims.size == 0:
        return None, 0.0, {"reason": "no_vectors"}

    # take top-k images globally
    top_idx = np.argsort(-sims)[:max(1, k_global)]

    # group candidate images by facade
    buckets: Dict[str, List[int]] = {}
    for i in top_idx:
        f = facade_ids[i]
        buckets.setdefault(f, []).append(i)

    # keep only the best few images per facade
    for f in list(buckets.keys()):
        idxs = buckets[f]
        idxs_sorted = sorted(idxs, key=lambda j: float(sims[j]), reverse=True)
        buckets[f] = idxs_sorted[:k_per_facade]

    best_f = None
    best_combined = 0.0
    best_dbg: Dict[str, Any] = {"reason": "no_verified"}

    for f, idxs in buckets.items():
        for j in idxs:
            s = float(sims[j])
            if s < base_threshold:
                continue

            cand_path = img_paths[j]
            inl = orb_inlier_ratio(image_path, cand_path)

            if inl < inlier_floor:
                continue

            combined = alpha * s + (1.0 - alpha) * inl

            # require at least some minimum inlier or good sim
            if combined > best_combined and (inl >= inlier_strong or s >= base_threshold):
                best_f = f
                best_combined = combined
                best_dbg = {
                    "reason": "ok",
                    "facade_id": f,
                    "img_path": cand_path,
                    "s": s,
                    "inliers": inl,
                    "combined": combined,
                }

    if best_f is None:
        return None, float(np.max(sims)), best_dbg

    # load card by best facade id
    cards = load_cards_jsonl(cards_path)
    id2card = {c.get("id"): c for c in cards if c.get("id")}
    card = id2card.get(best_f)
    if not card:
        best_dbg["reason"] = "missing_card"
        return None, best_combined, best_dbg

    return card, best_combined, best_dbg

# ---------- metrics & viz ----------
def compute_symmetry_scores(pil_img):
    im = np.array(pil_img)
    g = cv2.cvtColor(im, cv2.COLOR_RGB2GRAY)
    max_w = 640
    if g.shape[1] > max_w:
        h2 = int(g.shape[0] * max_w / g.shape[1])
        g = cv2.resize(g, (max_w, h2))
    v = ssim(g, np.fliplr(g), data_range=255)
    r = ssim(g, np.rot90(g, 2), data_range=255)
    return float(v), float(r)

def compute_proportions(pil_img, dets):
    w, h = pil_img.size
    w = float(max(1, w)); h = float(max(1, h))
    facade_ratio = h / w
    total = 0.0
    for d in dets:
        if d["label"] != "window": continue
        x1, y1, x2, y2 = map(float, d["box"])
        x1 = min(max(0.0, x1), w); x2 = min(max(0.0, x2), w)
        y1 = min(max(0.0, y1), h); y2 = min(max(0.0, y2), h)
        total += max(0.0, x2-x1) * max(0.0, y2-y1)
    w2w = total / (w*h)
    w2w = float(max(0.0, min(w2w, 0.95)))
    return float(facade_ratio), float(w2w)

def compute_rhythm_fft(pil_img):
    g = rgb2gray(np.array(pil_img))
    edges = sobel(g)
    F = np.fft.fftshift(np.fft.fft2(edges))
    mag = np.log1p(np.abs(F))
    center = np.array(mag.shape)/2
    ys, xs = np.indices(mag.shape)
    r = np.hypot(xs-center[1], ys-center[0])
    ring = (r>20) & (r<120)
    if ring.sum() == 0:
        return 0.0
    return float(np.quantile(mag[ring], 0.98) / (np.mean(mag[ring]) + 1e-6))

def box_count_fractal_dimension(pil_img, min_box=4, max_box=128):
    g = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2GRAY)
    edges = cv2.Canny(g, 100, 200)
    sizes, counts = [], []
    min_pow = int(math.log2(min_box)); max_pow = int(math.log2(max_box))
    for i in range(min_pow, max_pow + 1):
        k = 2**i
        S = (edges.shape[0] // k) * k, (edges.shape[1] // k) * k
        if S[0] <= 0 or S[1] <= 0: continue
        e = edges[:S[0], :S[1]].reshape(S[0]//k, k, S[1]//k, k).max(axis=(1,3))
        N = np.count_nonzero(e)
        if N > 0:
            sizes.append(1.0/k); counts.append(N)
    if len(sizes) < 2:
        return 1.0
    slope, _ = np.polyfit(np.log(sizes), np.log(counts), 1)
    return float(slope)

def fig_to_pil(fig, dpi=120):
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=dpi, bbox_inches="tight", pad_inches=0.1,
                facecolor="white", edgecolor="none")
    plt.close(fig)
    buf.seek(0)
    return Image.open(buf).convert("RGB")

def _safe_norm01(x, lo, hi):
    return float(np.clip((x - lo) / (hi - lo + 1e-6), 0.0, 1.0))

def _edge_map(pil_img):
    g = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2GRAY)
    g = cv2.GaussianBlur(g, (3,3), 0)
    e = cv2.Canny(g, 80, 180).astype(np.float32)
    return g.astype(np.float32), e

# Ten Principles
def principle_symmetry(sym_v, sym_r):
    return float(np.clip(0.6*sym_v + 0.4*sym_r, 0, 1))

def principle_balance(pil_img):
    g, e = _edge_map(pil_img)
    H, W = e.shape
    yy, xx = np.mgrid[0:H, 0:W]
    m = e + 1e-6
    cx = float((xx*m).sum()/m.sum()); cy = float((yy*m).sum()/m.sum())
    dx = abs(cx - (W-1)/2.0) / ((W-1)/2.0 + 1e-6)
    dy = abs(cy - (H-1)/2.0) / ((H-1)/2.0 + 1e-6)
    centroid_term = 1.0 - np.clip((dx+dy)/2.0, 0, 1)
    left = e[:, :W//2].sum(); right = e[:, W//2:].sum()
    lr = 1.0 - (abs(left-right) / (left+right+1e-6))
    return float(np.clip(0.5*centroid_term + 0.5*lr, 0, 1))

def _lab(pil_img):
    rgb = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
    lab = cv2.cvtColor(rgb, cv2.COLOR_BGR2LAB).astype(np.float32)
    L, a, b = lab[:,:,0], lab[:,:,1], lab[:,:,2]
    return L, a, b

def principle_harmony(pil_img):
    """
    Harmony = how smoothly the color mood changes across the facade.
    If different vertical slices have very different chroma, harmony drops.
    """
    L, a, b = _lab(pil_img)
    H, W = L.shape
    cols = 6
    widths = np.array_split(np.arange(W), cols)

    slice_means = []
    for idx in widths:
        if idx.size == 0:
            continue
        aa = a[:, idx].mean()
        bb = b[:, idx].mean()
        slice_means.append([aa, bb])

    slice_means = np.array(slice_means) if slice_means else np.zeros((1, 2), dtype=np.float32)

    # Average distance of slices from the overall mean in (a,b) space
    disp = float(
        np.linalg.norm(slice_means - slice_means.mean(axis=0), axis=1).mean()
    )

    # Typical facades sit somewhere in ~[4, 25] in this distance.
    # 0 => perfectly uniform color; very large => patchy / clashing.
    harmony = 1.0 - _safe_norm01(disp, 4.0, 25.0)
    return float(np.clip(harmony, 0.0, 1.0))



def principle_contrast(pil_img):
    """
    Contrast = mix of global light–dark spread and local texture.
    Tuned so most facade photos fall somewhere in the middle,
    instead of saturating at 1.0.
    """
    L, _, _ = _lab(pil_img)           # L in 0–255 from OpenCV Lab
    L = L.astype(np.float32) / 255.0  # work in 0–1

    # 1) global contrast (how wide the brightness range is)
    rms = float(L.std())

    # 2) local texture contrast (edges / fine detail)
    lap = cv2.Laplacian(L, cv2.CV_32F, ksize=3)
    tex = float(lap.std())

    # Map into 0–1 but with softer ranges so we don't always hit 1.0
    rms_norm = _safe_norm01(rms, 0.03, 0.20)   # 3%–20% std
    tex_norm = _safe_norm01(tex, 0.01, 0.10)   # small–medium texture

    # Smooth compression so very strong contrast doesn't all become 1.0
    rms_smooth = float(np.tanh(2.0 * rms_norm))
    tex_smooth = float(np.tanh(2.0 * tex_norm))

    raw = 0.6 * rms_smooth + 0.4 * tex_smooth
    return float(np.clip(raw, 0.0, 1.0))


def principle_proportion(aspect_ratio, w2w):
    """
    Proportion = mix of whole–facade shape (height/width)
    and window-to-wall ratio. We prefer comfortable, mid-range
    values instead of a single “perfect” golden value.
    """
    # Typical facades: H/W roughly 0.7–2.5
    # Reward moderate verticality around ~1.6, but not too sharply.
    ar_term = np.exp(-((aspect_ratio - 1.6) ** 2) / (2 * 0.50 ** 2))

    # Window-to-wall: we like a comfortable middle band ~0.18–0.35
    wwr_term = np.exp(-((w2w - 0.26) ** 2) / (2 * 0.14 ** 2))

    score = 0.5 * ar_term + 0.5 * wwr_term
    return float(np.clip(score, 0.0, 1.0))

def _column_autocorr(signal):
    sig = signal - signal.mean()
    ac = np.correlate(sig, sig, mode='full')[len(sig)-1:]
    if ac[0] <= 0: 
        return 0.0
    ac /= ac[0]
    if len(ac) < 3: 
        return 0.0
    k = int(np.argmax(ac[1:len(ac)//2])) + 1
    return float(max(0.0, ac[k]))

def principle_rhythm(pil_img):
    """
    Rhythm = how strongly the facade suggests repeated bays / beats.
    Combines a frequency-domain measure and column auto-correlation.
    """
    g = rgb2gray(np.array(pil_img))
    edges = sobel(g)

    # Frequency energy on a mid-distance ring (repetition scale)
    F = np.fft.fftshift(np.fft.fft2(edges))
    mag = np.log1p(np.abs(F))
    center = np.array(mag.shape) / 2.0
    ys, xs = np.indices(mag.shape)
    r = np.hypot(xs - center[1], ys - center[0])
    ring = (r > 20) & (r < 120)

    if ring.sum() == 0:
        fft_raw = 0.0
    else:
        fft_raw = float(np.quantile(mag[ring], 0.98) / (np.mean(mag[ring]) + 1e-6))

    # Map raw FFT measure into 0–1 with a softer range
    fft_norm = _safe_norm01(fft_raw, 0.8, 3.0)

    # Column auto-correlation of edge density
    e = cv2.Canny((g * 255).astype(np.uint8), 80, 180).astype(np.float32)
    col_sig = e.sum(axis=0)
    ac = _column_autocorr(col_sig)
    ac_norm = _safe_norm01(ac, 0.15, 0.85)

    score = 0.6 * fft_norm + 0.4 * ac_norm
    return float(np.clip(score, 0.0, 1.0))



def principle_repetition(pil_img):
    return principle_rhythm(pil_img)

def principle_simplicity(pil_img):
    """
    Simplicity = fewer edges + less “busy” edge distribution.
    High simplicity means the facade reads clean and calm.
    """
    _, e = _edge_map(pil_img)  # Canny edges, values 0 or 255

    # Fraction of pixels that are edges (0–1)
    edge_fraction = float((e > 0).mean())

    # Entropy of edge distribution: higher = more complex/busy
    hist, _ = np.histogram(e, bins=16, range=(0, 255), density=True)
    p = hist + 1e-8
    p /= p.sum()
    entropy = float(-(p * np.log(p)).sum())

    # We want: small edge_fraction => high simplicity.
    # Typical facades maybe 3%–40% pixels as edges.
    d_term = 1.0 - _safe_norm01(edge_fraction, 0.03, 0.40)

    # And lower entropy => higher simplicity.
    # For binary-ish edges, entropy is usually in ~[0.2, 1.5].
    h_term = 1.0 - _safe_norm01(entropy, 0.2, 1.5)

    score = 0.6 * d_term + 0.4 * h_term
    return float(np.clip(score, 0.0, 1.0))



def principle_unity(pil_img, dets):
    L, a, b = _lab(pil_img); H, W = L.shape
    cols = 6
    idxs = np.array_split(np.arange(W), cols)
    hue_means = []
    for idc in idxs:
        if idc.size == 0: continue
        hue_means.append([a[:, idc].mean(), b[:, idc].mean()])
    hue_means = np.array(hue_means) if hue_means else np.zeros((1,2))
    hue_var = float(hue_means.var(axis=0).mean())
    hue_term = 1.0 - _safe_norm01(hue_var, 2.0, 18.0)
    ws = []
    for d in dets:
        if d.get("label") == "window":
            x1,y1,x2,y2 = d["box"]
            ws.append((x2-x1)*(y2-y1))
    if len(ws) >= 3:
        ws = np.array(ws, dtype=np.float32)
        wcv = float(ws.std()/(ws.mean()+1e-6))
        size_term = 1.0 - np.clip(wcv/1.0, 0, 1)
    else:
        size_term = 0.5
    return float(np.clip(0.5*hue_term + 0.5*size_term, 0, 1))

def principle_gradation(pil_img):
    L,_,_ = _lab(pil_img)
    prof = L.mean(axis=1)
    diffs = np.diff(prof)
    if len(diffs) == 0:
        return 0.5
    signs = np.sign(diffs)
    same = np.sum(signs[:-1]*signs[1:] >= 0)
    mono = same / max(1, len(signs)-1)
    return float(np.clip(mono, 0, 1))

def compute_ten_principles(pil_img, dets, sym_v, sym_r, ratio, w2w):
    scores = {}
    scores["symmetry"]   = principle_symmetry(sym_v, sym_r)
    scores["balance"]    = principle_balance(pil_img)
    scores["harmony"]    = principle_harmony(pil_img)
    scores["contrast"]   = principle_contrast(pil_img)
    scores["proportion"] = principle_proportion(ratio, w2w)
    scores["rhythm"]     = principle_rhythm(pil_img)
    scores["repetition"] = principle_repetition(pil_img)
    scores["simplicity"] = principle_simplicity(pil_img)
    scores["unity"]      = principle_unity(pil_img, dets)
    scores["gradation"]  = principle_gradation(pil_img)
    for k in list(scores.keys()):
        scores[k] = float(np.clip(scores[k], 0, 1))
    return scores

def composite_beauty_score(principles: Dict[str, float], weights: Optional[Dict[str, float]] = None) -> float:
    order = ["repetition","gradation","symmetry","balance","harmony","contrast","proportion","rhythm","simplicity","unity"]
    vals = np.array([float(principles.get(k, 0.0)) for k in order], dtype=float)
    if vals.size == 0:
        return 0.0
    if weights:
        w = np.array([float(weights.get(k, 1.0)) for k in order], dtype=float)
        w = np.clip(w, 1e-8, None)
        return float(np.clip(np.sum(vals * w) / np.sum(w), 0.0, 1.0))
    return float(np.clip(vals.mean(), 0.0, 1.0))

def build_principles_viz(principles: Dict[str, float], figsize=(10,5)):
    labels = ["repetition","gradation","symmetry","balance","harmony","contrast","proportion","rhythm","simplicity","unity"]
    vals = [float(principles.get(k, 0.0)) for k in labels]
    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(1,1,1)
    ax.bar(range(len(labels)), vals)
    ax.set_ylim(0, 1.0)
    ax.set_ylabel("Score (0–1)")
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.set_title("Ten Principles of Beauty")
    fig.tight_layout()
    return fig_to_pil(fig, dpi=300)

def build_overall_beauty_line(score: float, figsize=(10, 4), lang: str = "en") -> Image.Image:
    score = float(np.clip(score, 0.0, 1.0))
    x = np.linspace(0.0, 1.0, 60)
    y = np.linspace(0.0, score, 60)
    fig, ax = plt.subplots(figsize=figsize)
    ax.set_ylim(0, 1); ax.set_xlim(0, 1)
    ax.plot(x, y, linewidth=3)
    ax.plot([1.0], [score], marker="o", markersize=8)
    ax.fill_between(x, 0, y, alpha=0.15)
    ax.axhline(score, linestyle="--", linewidth=1, alpha=0.5)
    title = "Overall Facade Beauty (0–1)" if lang != "zh" else "立面总体美度（0–1）"
    ax.set_title(title); ax.set_xlabel("Overall index"); ax.set_ylabel("Beauty score")
    ax.set_xticks([0.0, 0.5, 1.0]); ax.set_yticks([0.0, 0.25, 0.5, 0.75, 1.0]); ax.grid(True, linestyle=":", alpha=0.6)
    ax.text(1.0, score, f"{score:.2f}", va="bottom", ha="right", fontsize=10, weight="bold")
    fig.tight_layout()
    return fig_to_pil(fig, dpi=300)

def build_aesthetic_viz(norms, composite_index_0_5, figsize=(8, 6)):
    """
    Only show the radar chart of 6 core features (0–1).
    The overall index is already shown elsewhere, so we don't draw a second bar here.
    """
    labels = ["Vert Sym", "Rot Sym", "Proportion", "Win/Wall", "Rhythm", "Fractal"]

    # close the radar polygon
    vals = norms + [norms[0]]
    angles = np.linspace(0, 2 * np.pi, len(labels), endpoint=False)
    angles = np.concatenate([angles, angles[:1]])

    fig = plt.figure(figsize=figsize)
    ax = plt.subplot(1, 1, 1, polar=True)

    ax.plot(angles, vals, linewidth=3)
    ax.fill(angles, vals, alpha=0.30)
    ax.set_xticks(np.linspace(0, 2 * np.pi, len(labels), endpoint=False))
    ax.set_xticklabels(labels, fontsize=12)
    ax.set_ylim(0, 1)
    ax.grid(True, linewidth=0.8)
    ax.set_title("Aesthetic feature profile (0–1)", fontsize=13)

    fig.tight_layout()
    return fig_to_pil(fig, dpi=300)



def normalize_features(sym_v, sym_r, ratio, w2w, rhythm, fractal):
    nv  = np.clip((sym_v - 0.6) / 0.4, 0, 1)
    nr  = np.clip((sym_r - 0.5) / 0.5, 0, 1)
    nrx = np.exp(-((ratio - 1.5) ** 2) / (2 * 0.4 ** 2))
    nww = np.exp(-((w2w - 0.22) ** 2) / (2 * 0.12 ** 2))
    nry = np.clip(rhythm / 4.0, 0, 1)
    nfr = np.exp(-((fractal - 1.4) ** 2) / (2 * 0.15 ** 2))
    return [float(v) for v in (nv, nr, nrx, nww, nry, nfr)]

# ---------- Pollinations client (text only) ----------
def _pollinations_chat(
    prompt: str,
    system_text: str | None = None,
    image_data_url: str | None = None,
    *,
    model: str | None = None,         # accepted but intentionally unused
    timeout_s: float = 30.0,
    api_base: str = "https://text.pollinations.ai",
    **_
) -> str:
    """
    Pollinations client aligned with their docs:

    - Advanced: POST https://text.pollinations.ai/
      Sends system + user as OpenAI-style messages, no explicit model.
    - Simple fallback: GET https://text.pollinations.ai/{prompt}

    No local LLM; all text comes from Pollinations.
    """
    import requests
    from urllib.parse import quote
    import urllib.parse

    # ---- Build prompt, including image ref if any ----
    if image_data_url:
        prompt = (
            "You are given an image reference below. Use it in your analysis.\n"
            f"[IMAGE]: {image_data_url}\n\n{prompt}"
        )

    def _truncate(s: str | None, hard: int) -> str:
        if not s:
            return ""
        return (s[:hard] + " …[truncated]") if len(s) > hard else s

    user_text   = _truncate(prompt, 3500) 
    system_text = _truncate(system_text, 800)

    last_err = None

    # ---------- 1) Advanced: POST https://text.pollinations.ai/ ----------
    try:
        url = api_base.rstrip("/") + "/"
        payload = {
            "messages": [
                {"role": "system", "content": system_text or ""},
                {"role": "user",   "content": user_text or ""},
            ]
        }
        # Do NOT send "model" – let Pollinations choose a default.

        r = requests.post(
            url,
            json=payload,
            timeout=timeout_s,
            headers={"accept": "text/plain"}  # ask for plain text
        )

        if r.status_code == 200:
            ct = (r.headers.get("content-type") or "").lower()
            txt = (r.text or "").strip()

            # If server gives us plain text, just return it
            if txt and (ct.startswith("text/") or not ct):
                return txt

            # If JSON, try to extract a string-ish field
            try:
                js = r.json()
                if isinstance(js, str):
                    return js.strip()
                for k in ("response", "text", "output", "message"):
                    v = js.get(k)
                    if isinstance(v, str) and v.strip():
                        return v.strip()
            except Exception:
                pass

            last_err = f"unexpected POST body (ct={ct}, len={len(r.text)})"
        else:
            last_err = f"POST / status {r.status_code}: {r.text[:200]}"
    except Exception as e:
        last_err = f"POST / exception: {e}"

    # ---------- 2) Simple fallback: GET https://text.pollinations.ai/{prompt} ----------
    try:
        # Combine system + user into one prompt for the simple GET API
        full_prompt = (
            (f"System:\n{system_text}\n\nUser:\n{user_text}")
            if system_text else user_text
        ) or ""

        url = f"{api_base.rstrip('/')}/{quote(full_prompt)}"
        r = requests.get(
            url,
            timeout=timeout_s,
            headers={"accept": "text/plain"}
        )
        txt = (r.text or "").strip()

        if r.status_code == 200 and txt:
            low = txt.lstrip().lower()
            # If the body is HTML (e.g. a docs page), don’t dump it into UI
            if low.startswith("<!doctype html") or low.startswith("<html"):
                return "(Pollinations error: received HTML instead of plain text)"
            return txt

        return f"(Pollinations error {r.status_code}: {txt[:200]})"
    except Exception as e:
        return f"(Pollinations exception: {e}; last error: {last_err})"

# ---------- small JSON helper for safe-sized prompts ----------
def _safe_json(obj: dict, max_chars: int = 8000) -> str:
    s = json.dumps(obj, ensure_ascii=False, indent=2)
    return s if len(s) <= max_chars else (s[:max_chars] + " …[truncated]")

# ---------- Narrative & Chart explanation via Pollinations ----------
def compose_metrics_context(metrics: Dict[str, Any], dets: list) -> str:
    """Compact, human-readable summary of metrics (no big JSON)."""
    win_count = sum(1 for d in dets if d.get("label") == "window")
    arch_count = sum(1 for d in dets if d.get("label") == "arch")

    # Core numeric metrics
    vs  = metrics.get("symmetry_vertical")
    rs  = metrics.get("symmetry_rotational")
    ratio = metrics.get("facade_ratio_H_W")
    w2w = metrics.get("window_to_wall_ratio")
    rhy = metrics.get("rhythm_fft_peak")
    fr  = metrics.get("fractal_dimension")

    # Ten principles: only top 3 and bottom 2
    ten = {k.replace("principle_", ""): float(v)
           for k, v in metrics.items() if k.startswith("principle_")}
    sorted_ten = sorted(ten.items(), key=lambda x: x[1], reverse=True)
    top3 = sorted_ten[:3]
    low2 = sorted_ten[-2:] if len(sorted_ten) >= 2 else []

    lines = []
    lines.append(
        "Core metrics: "
        f"vertical_symmetry={vs:.3f}, "
        f"rotational_symmetry={rs:.3f}, "
        f"facade_ratio_H_W={ratio:.3f}, "
        f"window_to_wall_ratio={w2w:.3f}, "
        f"rhythm_index={rhy:.3f}, "
        f"fractal_dimension={fr:.3f}."
    )
    lines.append(
        f"Detected components: windows={win_count}, arches={arch_count}."
    )

    if top3:
        lines.append(
            "Ten principles – strongest: " +
            ", ".join(f"{k}({v:.2f})" for k, v in top3) + "."
        )
    if low2:
        lines.append(
            "Ten principles – weakest: " +
            ", ".join(f"{k}({v:.2f})" for k, v in low2) + "."
        )

    return "\n".join(lines)

def generate_facade_narrative_pollinations(
    metrics: Dict[str, Any],
    dets: list,
    lang: str = "en",
    db_info: Optional[str] = None,
    image_data_url: Optional[str] = None  # kept for signature, NOT used
) -> str:
    """
    Generate a tourist-friendly facade narrative that is STRICTLY grounded
    in db_info + high-level metrics. We do NOT send the image to avoid the
    model recognizing famous landmarks and ignoring the DB.
    """
    # --- DB facts block (ensure it's never empty) ---
    db_text = (db_info or "").strip()
    if not db_text:
        db_text = (
            "No database facts are available. You must NOT invent a specific "
            "building name, location, or detailed history."
        )

    # Short metrics context (we allow numbers here, but tell the model not to repeat them)
    ctx = compose_metrics_context(metrics, dets)

    if lang == "zh":
        system_text = (
            "你是一位在城市中为游客讲解的资深建筑师兼导览员。"
            "风格：专业、清晰、有教学性，但不要像论文。\n"
            "【硬性规则】\n"
            "1. 你只能根据 [建筑事实] 中提供的信息来给出具体描述。"
            "   不得编造新的建筑名称、城市、年代、设计师、用途或材料。\n"
            "2. 如果 [建筑事实] 中有建筑名称，你必须使用该名称；如果没有，只能称为“这座建筑”或“该建筑”。\n"
            "3. 不要使用“欢迎大家”“今天我要介绍”等开场问候，也不要说“如图所示”“这张图片里”。\n"
            "4. [立面指标概况] 只用于帮助你判断：例如“比较对称 / 不太对称”“窗洞偏多 / 偏少”“节奏感强 / 较弱”。"
            "   不要在答案中写出任何具体数字。\n"
            "5. 只描述立面和紧邻的室外空间，不要描述室内。"
        )

        user_prompt = (
            "下面是这座建筑可用的事实信息：\n"
            f"[建筑事实]\n{db_text}\n\n"
            "下面是立面指标的简要概况（包含数字，只供你内部参考，回答时不要写出具体数字）：\n"
            f"[立面指标概况]\n{ctx}\n\n"
            "任务：写出 3 段面向游客和学生的导览讲解：\n"
            "第 1 段：基于 [建筑事实] 介绍建筑的名称、所在城市/区域、年代和风格，如果有 intro 或 history，"
            "请用 1–2 句提到至少一个具体事实（例如设计者、开放年份、改造背景等）。\n"
            "第 2 段：结合风格、体量（massing）、[建筑事实] 中列出的材料和要素，再加上 [立面指标概况]，"
            "说明立面的组织方式：大致是否对称、窗洞大致多还是少、节奏感和重复感如何、整体感觉偏开放还是扎实。\n"
            "第 3 段：从游客和学生的角度，总结站在这座建筑前可以学到哪些建筑概念，例如对称、节奏、比例、"
            "城市地标性、历史与当代的叠加或再利用等。\n"
            "注意：不要使用任何欢迎语，不要提到“图像、照片、这张图”等字眼。"
        )

    else:
        system_text = (
            "You are a senior architect and on-site tour guide explaining a facade "
            "to tourists and students. Tone: calm, clear, and educational.\n"
            "HARD RULES:\n"
            "1. You may ONLY use concrete facts that appear in the [FACTS] block. "
            "   Do NOT invent a different building name, city, year, architect, use, or materials.\n"
            "2. If [FACTS] contains a building name, you MUST use that exact name. "
            "   If it does not, refer only to “this building” or “the building”.\n"
            "3. Do NOT greet the audience (no “Welcome, everyone”, “Today I will introduce”, etc.).\n"
            "4. Do NOT mention images, photos, or slides.\n"
            "5. The [METRICS] block is only for internal guidance. You may use it to say things like "
            "   “reads as fairly symmetrical” or “window area feels generous”, but you MUST NOT quote "
            "   the raw numbers or write approximate values like “0.84” or “63%”.\n"
            "6. Talk only about the facade and immediate exterior, not the interior."
        )

        user_prompt = (
            "Here are the factual building details you MUST treat as true:\n"
            f"[FACTS]\n{db_text}\n\n"
            "Here is a compact description of facade metrics, with numbers. "
            "These are only for your internal reasoning – do NOT repeat the numbers in your answer:\n"
            f"[METRICS]\n{ctx}\n\n"
            "Now, using ONLY the information in [FACTS] plus high-level impressions from [METRICS], "
            "write exactly three paragraphs:\n"
            "Paragraph 1 – Tourist context: introduce the building using the DB facts: name (if present), "
            "location in Taichung, era/period, main use, and style. If intro/history contain a designer, "
            "opening year, or reuse story, include at least one such concrete detail in this paragraph.\n"
            "Paragraph 2 – How the facade is organized: using style, massing, the listed materials/elements in [FACTS], "
            "and only qualitative cues from [METRICS], explain the composition: sense of symmetry or freedom, "
            "strength of rhythm and repetition, and whether the facade feels more open or solid.\n"
            "Paragraph 3 – Educational takeaway: explain what visitors or students can learn from this facade "
            "about architectural ideas like symmetry, rhythm, proportion, adaptive reuse, or the role of contemporary "
            "design in the city.\n"
            "Remember: do NOT introduce any new factual data that is not supported by [FACTS]."
        )

    return _pollinations_chat(
        user_prompt,
        system_text=system_text,
        image_data_url=None,   # IMPORTANT: no image for narrative
        api_base=API_BASE,
    )

 
def chart_explainer_pollinations(
    metrics: Dict[str, Any],
    lang: str = "en",
    db_info: Optional[str] = None,
    image_data_url: Optional[str] = None  # kept for signature, NOT used
) -> str:
    """
    Explain the chart in simple language for tourists and kids.
    No raw numbers in the answer; we still send numbers in the prompt
    but clearly tell the model not to repeat them.
    """
    db_text = (db_info or "").strip()
    if not db_text:
        db_text = "No building database facts are available. Keep the explanation very general."

    ctx = compose_metrics_context(metrics, dets=[])

    if lang == "zh":
        system_text = (
            "你是一位在博物馆里讲解建筑图表的老师，听众包括家庭游客和小学生。\n"
            "要求：\n"
            "1. 语言非常简单、口语化，每条尽量 1–2 句。\n"
            "2. 不要在回答中写出任何数字或百分比，也不要用“指数、FFT、分形”等术语。\n"
            "3. 不要编造新的建筑名称或历史，只能在 [建筑事实] 的范围内概括。\n"
            "4. 每一点都要告诉孩子和游客：可以看立面的哪一部分来观察这个特点。"
        )

        user_prompt = (
            "【建筑事实】\n"
            f"{db_text}\n\n"
            "【立面指标概况】（包含数字，只给你参考，请不要在回答中写出数字）\n"
            f"{ctx}\n\n"
            "请写出 4–6 条项目，每条用 “- ” 开头。\n"
            "每条：\n"
            "• 概括一个简单的观察点（例如：左右看起来比较平衡、窗户排成有规律的节奏、立面比较简单、细节比较丰富等）。\n"
            "• 加一句很短的提示，让游客/孩子知道应该看哪里（比如“看看两边的窗户是不是差不多高”）。"
        )

    else:
        system_text = (
            "You are a museum educator explaining a simple facade score chart to families and children.\n"
            "Rules:\n"
            "1. Use very simple words and short sentences.\n"
            "2. Do NOT show any numbers or percentages in your answer.\n"
            "3. Do NOT use technical terms like 'index', 'FFT', or 'fractal'.\n"
            "4. Stay consistent with the building facts and do not invent a new name or history.\n"
            "5. Each bullet should say what people can actually look at on the facade."
        )

        user_prompt = (
            "Here are the building facts you should keep in mind:\n"
            f"[FACTS]\n{db_text}\n\n"
            "Here is a compact metrics summary with numbers (for your understanding only; "
            "DO NOT repeat the numbers in your answer):\n"
            f"[METRICS]\n{ctx}\n\n"
            "Write 4–6 short bullet points, each starting with '- ':\n"
            "• Each bullet explains ONE simple idea, such as 'the two sides feel balanced', "
            "'windows repeat in a clear pattern', 'the facade feels calm and simple', "
            "or 'there are many small details to explore'.\n"
            "• For each bullet, add a tiny suggestion of what kids and visitors can look at "
            "on the building to notice this.\n"
            "• Do NOT include any numbers or percentages."
        )

    return _pollinations_chat(
        user_prompt,
        system_text=system_text,
        image_data_url=None,   # IMPORTANT: no image here either
        api_base=API_BASE,
    )

# ---------- text-only fallback (simple & kid-friendly) ----------
def chart_explainer_text_only(metrics: Dict[str, Any], lang="en") -> str:
    """
    Local fallback: still explain for general users / kids.
    No formulas, just intuitive language.
    """
    vs  = metrics.get("symmetry_vertical", 0.5)
    rs  = metrics.get("symmetry_rotational", 0.5)
    prop = metrics.get("facade_ratio_H_W", 1.5)
    w2w = metrics.get("window_to_wall_ratio", 0.22)
    rhy = metrics.get("rhythm_fft_peak", 1.0)
    fr  = metrics.get("fractal_dimension", 1.4)

    # helper levels
    def level(x, lo, hi):
        if x <= lo: return "low"
        if x >= hi: return "high"
        return "mid"

    lev_sym = level(vs, 0.45, 0.75)
    lev_prop = level(prop, 1.1, 1.9)
    lev_w2w = level(w2w, 0.10, 0.40)
    lev_rhy = level(rhy, 0.7, 2.0)
    lev_detail = level(fr, 1.25, 1.55)

    bullets_en = []

    # Symmetry / balance
    if lev_sym == "high":
        bullets_en.append(
            "- The two sides of the facade feel very similar, so the building looks calm and stable. "
            "You can notice this by comparing the left and right sides: windows and shapes line up in a similar way."
        )
    elif lev_sym == "low":
        bullets_en.append(
            "- The two sides of the facade are quite different, which makes it feel more playful and irregular. "
            "You can look for details that change from one side to the other."
        )
    else:
        bullets_en.append(
            "- The facade is somewhat balanced: the two sides are not exactly the same, but they do not feel random either. "
            "Look at the main opening or center line and see how the parts on each side relate."
        )

    # Proportion (tall vs wide)
    if lev_prop == "high":
        bullets_en.append(
            "- The building looks tall and slim, so your eyes are gently pulled upward. "
            "Try tracing the outline of the building from bottom to top."
        )
    elif lev_prop == "low":
        bullets_en.append(
            "- The building looks wider than it is tall, so it feels very grounded and stable. "
            "You can feel this by looking at how long the facade stretches from left to right."
        )
    else:
        bullets_en.append(
            "- The building feels quite balanced between tall and wide. "
            "It doesn’t stretch too far in either direction, which gives a comfortable overall shape."
        )

    # Openness (window-to-wall)
    if lev_w2w == "high":
        bullets_en.append(
            "- There is a lot of window area compared to wall, so the facade feels bright and open. "
            "Count how many window surfaces you see compared with solid wall parts."
        )
    elif lev_w2w == "low":
        bullets_en.append(
            "- There is more solid wall than window, so the facade feels heavier and more closed. "
            "Notice how much plain wall you see between the openings."
        )
    else:
        bullets_en.append(
            "- The amount of wall and window feels balanced, so the facade is neither too closed nor too open. "
            "Look at how windows and solid wall pieces take turns."
        )

    # Rhythm / repetition
    if lev_rhy == "high":
        bullets_en.append(
            "- There is a strong rhythm in the facade: many parts repeat again and again, like a beat in music. "
            "You can see this by following rows or columns of windows and other repeating shapes."
        )
    elif lev_rhy == "low":
        bullets_en.append(
            "- The rhythm of the facade is soft, with fewer exact repeats. "
            "Instead of perfect rows, look for gentle changes from one part to the next."
        )
    else:
        bullets_en.append(
            "- The facade has some repetition, but also some variety. "
            "Notice how certain shapes come back, but not always in a strict pattern."
        )

    # Detail / complexity
    if lev_detail == "high":
        bullets_en.append(
            "- There are many small details to discover, so the facade feels rich and busy. "
            "If you stand close, you can keep finding new lines, edges, or decorations."
        )
    elif lev_detail == "low":
        bullets_en.append(
            "- The facade is quite simple, with fewer small details. "
            "This makes it easy to understand the main shapes from far away."
        )
    else:
        bullets_en.append(
            "- The amount of detail is in the middle: not too plain, not too busy. "
            "You can enjoy both the big outline and some smaller features when you look closer."
        )

    result_en = "\n".join(bullets_en)

    if lang == "zh":
        # Translate to Simplified Chinese while keeping list formatting
        zh = _pollinations_chat(
            f"Translate to Simplified Chinese. Keep '- ' bullet formatting:\n{result_en}",
            system_text="You are a precise translator."
        )
        return zh

    return result_en
# =========================
# app.py — PART 3 of 3
# =========================

# ---------- cache helpers ----------
from functools import lru_cache

@st.cache_data(show_spinner=False, ttl=CACHE_TTL_MIN*60)
def get_all_cards() -> list[dict]:
    return load_cards_jsonl(str(CARDS_PATH))


def pick_best_image(card: dict) -> Optional[str]:
    """
    Choose one 'clear' facade image for a card.
    For now: pick the image with largest pixel area.
    """
    imgs = card.get("images") or []
    if not imgs and card.get("image"):
        imgs = [card["image"]]

    best_path = None
    best_area = 0
    for p in imgs:
        if not p or not Path(p).exists():
            continue
        try:
            im = Image.open(p)
            w, h = im.size
            area = w * h
            if area > best_area:
                best_area = area
                best_path = p
        except Exception:
            continue
    return best_path


@st.cache_data(show_spinner=False, ttl=CACHE_TTL_MIN*60)
def _cached_components_overlay(raw_bytes: bytes, max_side: int):
    pil = Image.open(io.BytesIO(raw_bytes)).convert("RGB")
    pil_disp = pil.copy()
    pil_disp.thumbnail((MAX_SIDE, MAX_SIDE*10000), Image.LANCZOS)
    dets = heuristic_components(pil_disp)
    overlay = draw_semantic_overlay(pil_disp, dets)
    return pil_disp, dets, overlay

@st.cache_data(show_spinner=False, ttl=CACHE_TTL_MIN*60)
def _cached_metrics(raw_bytes: bytes, dets: list, max_side: int):
    pil = Image.open(io.BytesIO(raw_bytes)).convert("RGB")
    pil.thumbnail((MAX_SIDE, MAX_SIDE*10000), Image.LANCZOS)
    ratio, w2w = compute_proportions(pil, dets)
    sym_v, sym_r = compute_symmetry_scores(pil)
    rhythm = compute_rhythm_fft(pil)
    fractal = box_count_fractal_dimension(pil)
    principles = compute_ten_principles(pil, dets, sym_v, sym_r, ratio, w2w)
    principles_img = build_principles_viz(principles)
    return ratio, w2w, sym_v, sym_r, rhythm, fractal, principles, principles_img

@st.cache_data(show_spinner=False, ttl=CACHE_TTL_MIN*60)
def _cached_heatmap(raw_bytes: bytes, max_side: int):
    pil = Image.open(io.BytesIO(raw_bytes)).convert("RGB")
    im = np.array(pil)
    if im.ndim == 2:
        im = np.stack([im]*3, axis=-1)
    g = cv2.cvtColor(im, cv2.COLOR_RGB2GRAY).astype(np.float32)
    def _norm(x):
        x = x.astype(np.float32); x -= x.min()
        return x / (x.max() - x.min() + 1e-6)
    e1 = cv2.Canny(g.astype(np.uint8), 50, 150).astype(np.float32)
    e2 = cv2.Canny(g.astype(np.uint8), 80, 180).astype(np.float32)
    e3 = cv2.Canny(g.astype(np.uint8), 120, 240).astype(np.float32)
    edges = _norm(e1 + e2 + e3)
    lap = _norm(np.abs(cv2.Laplacian(g, cv2.CV_32F, ksize=3)))
    G = g / (g.max() + 1e-6)
    F = np.fft.fft2(G); A = np.abs(F)
    L = np.log(A + 1e-6)
    ker = np.ones((3,3), np.float32) / 9.0
    L_avg = cv2.filter2D(L, -1, ker, borderType=cv2.BORDER_REFLECT)
    SR = L - L_avg
    sal = np.abs(np.fft.ifft2(np.exp(SR + 1j*np.angle(F))))
    sal = _norm(cv2.GaussianBlur(np.real(sal).astype(np.float32), (5,5), 0))
    heat = _norm(0.5 * sal + 0.3 * edges + 0.2 * lap)
    cmap = plt.get_cmap("jet")
    heat_rgb = (cmap(heat)[..., :3] * 255).astype(np.uint8)
    heat_img = Image.fromarray(heat_rgb).resize(pil.size, Image.BILINEAR).convert("RGBA")
    base = pil.convert("RGBA"); heat_img.putalpha(int(255*0.45))
    return Image.alpha_composite(base, heat_img).convert("RGB")

@st.cache_data(show_spinner=False, ttl=CACHE_TTL_MIN*60)
def _cached_norms(sym_v, sym_r, ratio, w2w, rhythm, fractal):
    return normalize_features(sym_v, sym_r, ratio, w2w, rhythm, fractal)

@st.cache_data(show_spinner=False, ttl=CACHE_TTL_MIN*60)
def _cached_viz(norms, composite_index_0_5):
    return build_aesthetic_viz(norms, composite_index_0_5)

# ---------- PDF builder ----------
def make_report(
    orig_img: Image.Image,
    overlay_img: Image.Image,
    heat_img: Image.Image,
    viz_img: Image.Image,
    metrics: Dict[str, Any],
    ranking: Tuple[int,int,float],
    story_text: str,
    lang: str = "en",
    interiors: Optional[list] = None,
    chart_explanation_text: Optional[str] = None,
    principles_img: Optional[Image.Image] = None,
    principles_scores: Optional[dict] = None,
    overall_img: Optional[Image.Image] = None,
    overall_score: Optional[float] = None,
):
    import io as _io, re as _re, time as _t
    from reportlab import rl_config
    rl_config.defaultCompression = 1

    def _pil_to_jpeg_bytes(pil_img, target_width_pt, dpi=150, quality=80):
        target_px = max(1, int(target_width_pt * dpi / 72.0))
        img = pil_img.convert("RGB").copy()
        img.thumbnail((target_px, target_px * 10000), Image.LANCZOS)
        bio = _io.BytesIO()
        img.save(bio, format="JPEG", quality=80, optimize=True, subsampling=1)
        return bio.getvalue()

    def draw_img_fit_top(c, pil, x_left, y_top, max_w_pt):
        if pil is None: return 0.0
        jpeg = _pil_to_jpeg_bytes(pil, target_width_pt=max_w_pt, dpi=150, quality=80)
        rdr = ImageReader(_io.BytesIO(jpeg))
        img = Image.open(_io.BytesIO(jpeg))
        w_draw = float(max_w_pt)
        h_draw = w_draw * (img.height / float(img.width))
        c.drawImage(rdr, x_left, y_top - h_draw, width=w_draw, height=h_draw, preserveAspectRatio=True, mask='auto')
        return h_draw

    def wrap_text_to_width(text: str, max_width_pt: float, font_name: str, font_size: int, is_cjk=False) -> list[str]:
        sw = pdfmetrics.stringWidth
        lines = []
        if is_cjk:
            cur = ""
            for ch in text or "":
                if ch == "\n":
                    if cur: lines.append(cur)
                    cur = ""
                    continue
                if sw(cur + ch, font_name, font_size) <= max_width_pt:
                    cur += ch
                else:
                    if cur: lines.append(cur)
                    cur = ch
            if cur: lines.append(cur)
            return lines
        else:
            words = re.split(r"\s+", (text or "").strip())
            cur = ""
            for w in words:
                add = (w if not cur else " " + w)
                if sw(cur + add, font_name, font_size) <= max_width_pt:
                    cur += add
                else:
                    if cur: lines.append(cur)
                    cur = w
            if cur: lines.append(cur)
            return lines

    buf = _io.BytesIO()
    c = canvas.Canvas(buf, pagesize=A4)
    c.setPageCompression(1)
    W, H = A4
    TOP, BOT, LM, RM = 40, 50, 40, 40
    y = H - TOP

    # Header
    c.setFont("STSong-Light", 16)
    c.drawString(LM, y, "AI Analysis of Facade / 立面人工智能分析")
    y -= 22
    c.setFont("STSong-Light", 10)
    c.drawString(LM, y, _t.strftime("Generated on / 生成于 %Y-%m-%d %H:%M:%S"))
    y -= 16

    # Row of images
    COL_W = (W - LM - RM - 20) / 3.0
    row_top = y
    h1 = draw_img_fit_top(c, orig_img, LM, row_top, COL_W)
    h2 = draw_img_fit_top(c, overlay_img, LM + COL_W + 10, row_top, COL_W)
    h3 = draw_img_fit_top(c, heat_img, LM + 2*(COL_W + 10), row_top, COL_W)
    y = row_top - max(h1, h2, h3) - 14

    # Story
    if y < BOT + 140:
        c.showPage(); y = H - TOP
    c.setFont("STSong-Light", 12)
    c.drawString(LM, y, "Design Narrative / 设计叙事")
    y -= 14
    body_font = "STSong-Light" if lang == "zh" else "Helvetica"
    font_size = 10 if lang == "en" else 11
    leading = 13 if lang == "en" else 14
    c.setFont(body_font, font_size)

    paragraphs = [p.strip() for p in re.split(r'\n\s*\n', (story_text or "")) if p.strip()]
    max_w = W - LM - RM
    for para in paragraphs:
        for ln in wrap_text_to_width(para, max_w, body_font, font_size, is_cjk=(lang=="zh")):
            if y < BOT:
                c.showPage(); y = H - TOP
                c.setFont(body_font, font_size)
            c.drawString(LM, y, ln)
            y -= leading
        y -= int(leading * 0.5)

    # Viz page
    c.showPage(); y = H - TOP
    c.setFont("STSong-Light", 12)
    c.drawString(LM, y, "Aesthetic Visualization / 美学可视化")
    y -= 8
    if viz_img is not None:
        w_draw = W - LM - RM
        used_h = draw_img_fit_top(c, viz_img, LM, y, w_draw)
        y -= used_h + 16

        c.setFont("STSong-Light", 12)
        c.drawString(LM, y, "Explanation / 解释")
        y -= 14
        body_font = "STSong-Light" if lang=="zh" else "Helvetica"
        c.setFont(body_font, 10 if lang=="en" else 11)
        leading = 13 if lang=="en" else 14
        max_w = W - LM - RM
        for para in (chart_explanation_text or "").split("\n"):
            for ln in wrap_text_to_width(para, max_w, body_font, 10 if lang=="en" else 11, is_cjk=(lang=="zh")):
                if y < BOT:
                    c.showPage(); y = H - TOP
                    c.setFont(body_font, 10 if lang=="en" else 11)
                c.drawString(LM, y, ln)
                y -= leading
            y -= int(leading * 0.5)

    # Ten Principles
    if principles_img is not None or principles_scores is not None:
        if y < BOT + 280:
            c.showPage(); y = H - TOP
        c.setFont("STSong-Light", 12)
        c.drawString(LM, y, "Ten Principles of Beauty / 十大美学原则")
        y -= 8
        if principles_img is not None:
            used_h = draw_img_fit_top(c, principles_img, LM, y, W - LM - RM)
            y -= used_h + 12

    # Overall beauty
    if overall_img is not None:
        if y < BOT + 200:
            c.showPage(); y = H - TOP
            c.setFont("STSong-Light", 12)
            c.drawString(LM, y, "Overall Beauty / 总体美度")
            y -= 8
        used_h = draw_img_fit_top(c, overall_img, LM, y, W - LM - RM)
        y -= used_h + 10
        if overall_score is not None:
            c.setFont("Helvetica", 10)
            c.drawString(LM, y, f"Overall Facade Beauty (0–1): {overall_score:.2f}")
            y -= 14

    # Ranking
    c.setFont("Helvetica", 11)
    try:
        rank, N, perc = ranking
        c.drawString(LM, y, f"Comparative ranking: {rank} / {N}  (~{perc:.1f}th percentile)")
        y -= 16
    except Exception:
        pass

    c.showPage()
    c.save()
    pdf_bytes = buf.getvalue()
    buf.close()
    return pdf_bytes

# ---------- ranking store ----------
def update_and_rank(score, name, hist_path=HIST_PATH):
    try:
        data = json.loads(hist_path.read_text(encoding="utf-8")) if hist_path.exists() else []
    except Exception:
        data = []
    data.append({"name": name, "score": float(score), "ts": int(time.time())})
    data_sorted = sorted(data, key=lambda x: x["score"], reverse=True)
    N = len(data_sorted)
    rank = 1 + [d["name"] for d in data_sorted].index(name)
    percentile = 100.0 * (N - rank) / max(1, N - 1) if N > 1 else 100.0
    try:
        hist_path.write_text(json.dumps(data, indent=2), encoding="utf-8")
    except Exception:
        pass
    return rank, N, percentile


#--- analysis part method 

def run_facade_analysis(
    raw: bytes,
    label: str,
    *,
    known_card: Optional[dict] = None,
    source_id: str = "facade"
):
    """
    Run the full analysis pipeline for ONE image.

    - raw: image bytes
    - label: title shown in the UI
    - known_card: if provided, we ALREADY know which DB card this belongs to,
      so we skip image retrieval. If None, we run retrieve_verified_multiview.
    - source_id: short id for ranking / PDF / QA keys.
    """
    from pathlib import Path

    st.subheader(label)

    # Turn into data URL for Pollinations
    img_data_url = _bytes_to_data_url(raw, mime="image/jpeg")

    # 1) Components + overlay (cached)
    with st.spinner("Detecting components..."):
        pil, dets, overlay = _cached_components_overlay(raw, MAX_SIDE)

    # 2) Metrics + principles (cached)
    ratio, w2w, sym_v, sym_r, rhythm, fractal, principles, principles_img = _cached_metrics(raw, dets, MAX_SIDE)

    # 3) Heatmap (cached)
    with st.spinner("Computing heatmap..."):
        heat = _cached_heatmap(raw, MAX_SIDE)

    # 4) Normalize + viz (cached)
    norms = _cached_norms(sym_v, sym_r, ratio, w2w, rhythm, fractal)

    overall_beauty = composite_beauty_score(principles)      # 0–1
    composite_index_0_5 = 5.0 * overall_beauty               # 0–5 scale for bar
    viz_img = _cached_viz(norms, composite_index_0_5)

    # ---------- show images ----------
    cols = st.columns(3)
    cols[0].image(pil, caption="Original", width='stretch')
    cols[1].image(overlay, caption="Semantic overlay (heuristic)", width='stretch')
    cols[2].image(heat, caption="Explainability heatmap", width='stretch')

    # Metrics dict (for narrative & PDF)
    metrics_dict = {
        "symmetry_vertical": round(sym_v, 3),
        "symmetry_rotational": round(sym_r, 3),
        "facade_ratio_H_W": round(ratio, 3),
        "window_to_wall_ratio": round(w2w, 3),
        "rhythm_fft_peak": round(rhythm, 3),
        "fractal_dimension": round(fractal, 3),
        **{f"principle_{k}": round(v, 3) for k, v in principles.items()}
    }

    # ---------- DB retrieval / info ----------
    db_info = None
    verified_card = None

    if known_card is not None:
        # We already know which building this is (search-by-name path)
        verified_card = known_card
        db_info = _make_info(verified_card)
        st.caption(f"DB facts used for narrative: {db_info}")
    else:
        # Upload path: we still need to find the matching building
        tmp_image_path = str(Path("outputs") / f"_tmp_{source_id}.png")
        Image.open(io.BytesIO(raw)).convert("RGB").save(tmp_image_path)

        try:
            matched_card, combined, dbg = retrieve_verified_multiview(
                tmp_image_path,
                str(IDX_PATH),
                base_threshold=0.30,
            )
            if matched_card:
                verified_card = matched_card
                db_info = _make_info(verified_card)
                st.caption(f"DB facts used for narrative: {db_info}")
            else:
                st.caption(
                    f"No reliable DB match ({dbg.get('reason', '?')})"
                )
        except Exception as e:
            st.caption(f"DB match error: {e}")

    # ---------- narrative ----------
    with st.spinner("Generating facade narrative..."):
        story = generate_facade_narrative_pollinations(
            metrics_dict,
            dets,
            lang=LANG,
            db_info=db_info,
            image_data_url=img_data_url,
        )

    st.markdown("### Design Narrative / 設計敘事")
    st.write(story)

    # ---------- aesthetic viz + explanation ----------
    st.markdown("### Aesthetic Visualization / 美學視覺化")
    st.image(viz_img, caption="Feature profile and score makeup", width='stretch')

    with st.spinner("Explaining the chart..."):
        chart_explanation = chart_explainer_pollinations(
            metrics_dict,
            lang=LANG,
            db_info=db_info,
            image_data_url=img_data_url,
        )
        if not chart_explanation or "(Pollinations" in chart_explanation:
            chart_explanation = chart_explainer_text_only(metrics_dict, lang=LANG)

    st.markdown("#### Explanation / 解释")
    st.markdown(chart_explanation)

    st.markdown("### Ten Principles of Beauty / 十大美学原则")
    st.image(principles_img, caption="Normalized 0–1 scores per principle", width='stretch')

    st.markdown("### Overall Facade Beauty / 立面总体美度")
    overall_img = build_overall_beauty_line(overall_beauty, lang=LANG)
    st.image(overall_img, caption=f"Overall beauty = {overall_beauty:.2f}", width='stretch')
    st.info(f"Overall Beauty (0–1): **{overall_beauty:.2f}**")

    # ---------- ranking ----------
    rank, N, perc = update_and_rank(overall_beauty, source_id, HIST_PATH)
    st.info(f"Comparative ranking / 對比排名: {rank} / {N}  (~{perc:.1f}th percentile)")

    # ---------- Q&A ----------
    st.markdown("### Ask / 问")
    user_q = st.text_input(
        "Ask a question about this building" if LANG != "zh" else "请就此建筑提问",
        key=f"{KEY_NS}_qa_{source_id}",
    )

    if user_q:
        info_text = _make_info(verified_card) if verified_card else ""
        if info_text:
            sys = "Answer ONLY using the provided Information. If not present, reply EXACTLY: NOTFOUND."
            if LANG == "zh":
                sys = "仅根据提供的信息回答。如果信息中没有，请严格回复：NOTFOUND。"
            ans = _pollinations_chat(
                f"Information:\n{info_text}\n\nQuestion:\n{user_q}\nAnswer:",
                system_text=sys,
            )
            if ans.strip().upper().startswith("NOTFOUND"):
                st.warning(
                    "Sorry, not found in the current building information."
                    if LANG != "zh" else
                    "抱歉，在当前建筑信息中未找到。"
                )
            else:
                st.success(ans)
        else:
            st.warning("No building info available for Q&A.")

    # ---------- PDF ----------
    pdf_bytes = make_report(
        pil, overlay, heat, viz_img,
        metrics=metrics_dict,
        ranking=(rank, N, perc),
        story_text=story,
        lang=LANG,
        interiors=(verified_card.get("interiors") if verified_card else None),
        chart_explanation_text=chart_explanation,
        principles_img=principles_img,
        principles_scores=principles,
        overall_img=overall_img,
        overall_score=overall_beauty,
    )

    st.download_button(
        "Download report PDF / 下載報告 PDF",
        data=pdf_bytes,
        file_name=f"{source_id}_report.pdf",
        mime="application/pdf",
        key=f"{KEY_NS}_dl_{source_id}",
    )


#--analysis part method end 
# ---------------- Main Analysis ----------------

if page == "Analysis":
    ensure_index(str(CARDS_PATH), str(IDX_PATH))

    # ---- 1) QR / direct ID lookup via URL query params ----
    # Example QR URL:
    #   https://your-app-url/?id=tch-001
    #   https://your-app-url/?facade=tch-001
    #   https://your-app-url/?facade_id=tch-001
    params = st.query_params  # ✅ new API, replaces st.experimental_get_query_params()

    qr_id = None
    for key in ("id", "facade", "facade_id"):
        if key in params:
            v = params[key]
            if isinstance(v, (list, tuple)):
                qr_id = (v[0] or "").strip()
            else:
                qr_id = str(v).strip()
            break

    if qr_id:
        # 1) Load DB and find the matching card by id / English name / Chinese name
        cards = load_cards_jsonl(str(CARDS_PATH))
        card = None
        for c in cards:
            cid     = str(c.get("id", "")).strip()
            name_en = (c.get("name") or "").strip()
            name_zh = (c.get("name_zh") or "").strip()
            if qr_id == cid or qr_id == name_en or (name_zh and qr_id == name_zh):
                card = c
                break

        if not card:
            st.error(
                "This building is not in our database yet."
                if LANG != "zh" else
                "此建筑尚未收录在数据库中。"
            )
            st.stop()

        # 2) Title: EN + ZH if available
        title_line = card.get("name", "")
        if card.get("name_zh"):
            title_line += f" / {card['name_zh']}"
        st.header(title_line)

        # 3) Show ALL available images as a small thumbnail gallery
        imgs = card.get("images") or []
        if not imgs and card.get("image"):  # legacy single image
            imgs = [card["image"]]

        valid_paths = [str(Path(p)) for p in imgs if p and Path(p).exists()]

        if valid_paths:
            st.markdown(
                "### Facade views / 立面视图"
                if LANG != "zh" else
                "### 立面视图"
            )

            # thumbnails in a grid; each image is clickable to enlarge
            num_cols = min(4, len(valid_paths))  # up to 4 per row
            cols = st.columns(num_cols)

            for i, img_path in enumerate(valid_paths):
                col = cols[i % num_cols]
                # smaller display; click opens larger preview
                col.image(
                    img_path,
                    width='stretch',
                    caption=f"View {i+1}" if LANG != "zh" else f"视角 {i+1}",
                )

        # 4) Build DB info text and generate *guide-style* narrative
        db_info = _make_info(card)

        guide_text = generate_qr_guide_text(
            db_info=db_info,
            lang=LANG,
            image_data_url=None  # no need to send image for this
        )

        st.markdown(
            "### Building Guide / 建筑导览"
            if LANG != "zh" else
            "### 建筑导览说明"
        )
        st.write(guide_text)

        # 5) Q&A — based ONLY on db_info (no hallucinated facts)
        st.markdown(
            "### Ask a question / 提问"
            if LANG != "zh" else
            "### 提问"
        )
        qa_key = f"{KEY_NS}_qr_qa_{card.get('id','')}"
        user_q = st.text_input(
            "Ask something about this building."
            if LANG != "zh" else
            "可以就这座建筑问一个问题。",
            key=qa_key
        )

        if user_q:
            if db_info:
                if LANG == "zh":
                    sys = (
                        "你是这座建筑的导览员，只能根据提供的信息回答问题。"
                        "如果问题中涉及的信息在资料里不存在，请简短说明“资料中没有相关信息”。"
                        "不要编造具体事实。"
                    )
                else:
                    sys = (
                        "You are the guide for this building. "
                        "Answer ONLY using the provided information. "
                        "If the question asks about something not in the info, briefly say "
                        "'That detail is not in the current information.' Do NOT invent facts."
                    )

                ans = _pollinations_chat(
                    f"Information:\n{db_info}\n\nQuestion:\n{user_q}\nAnswer:",
                    system_text=sys,
                    api_base=API_BASE,
                )
                st.success(ans)
            else:
                st.warning(
                    "No building info is available yet for Q&A."
                    if LANG != "zh" else
                    "目前没有可用于问答的建筑信息。"
                )

        # 6) Stop: QR page should NOT fall through to upload UI
        st.stop()

        
   
    # --- 1) normal upload path (can reuse same function) ---
    uploaded = st.file_uploader(
        "Upload Building Facade Image / 上传建筑立面图片",
        type=["jpg", "jpeg", "png"],
        accept_multiple_files=True,
        key=f"{KEY_NS}_uploader",
    )
    st.markdown("---")
    # --- 2) optional search-by-name ---
    st.markdown("#### Search by building name (optional) / 按建筑物名称搜索（可选）")
    name_query = st.text_input(
        "Type building name (e.g. 'National Taichung Theater') / 建筑物名称（例如“台中国家剧院”）",
        key=f"{KEY_NS}_name_search",
    )

    if name_query:
        cards = load_cards_jsonl(str(CARDS_PATH))
        matches = [c for c in cards if card_matches_name_query(c, name_query)]

        if not matches:
            st.warning("This building name does not exist in our database.")
        else:
            if len(matches) == 1:
                selected_card = matches[0]
            else:
                options = [c["name"] for c in matches]
                chosen = st.selectbox(
                    "Multiple matches found – choose one:",
                    options,
                    key=f"{KEY_NS}_name_choice",
                )
                selected_card = next(c for c in matches if c["name"] == chosen)

            img_path = pick_best_image(selected_card)
            if not img_path:
                st.warning("This building is in the database but has no usable images.")
            else:
                with open(img_path, "rb") as f:
                    raw = f.read()
                # ONE call, same pipeline as upload
                run_facade_analysis(
                    raw,
                    label=f"{selected_card.get('name','(from DB)')} (from database)",
                    known_card=selected_card,
                    source_id=selected_card.get("id", "db_facade"),
                )

   

    if uploaded:
        for up in uploaded:
            raw = up.read()
            # here we DON'T know the card, so known_card=None
            run_facade_analysis(
                raw,
                label=up.name,
                known_card=None,
                source_id=Path(up.name).stem,
            )
