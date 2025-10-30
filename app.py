# =========================
# app.py — PART 1 of 3
# =========================
# Fast, no local LLMs. Uses Pollinations text API for narrative & explanations.
# Deps:
# pip install -U streamlit pillow numpy scikit-image matplotlib reportlab \
#   opencv-python-headless requests

import io, re, math, time, json, hashlib, random, urllib.parse
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple

import numpy as np
from PIL import Image, ImageDraw
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

# --------- Runtime knobs (variables; no env) ----------
NUM_THREADS     = 4
MAX_SIDE        = 1024
CACHE_TTL_MIN   = 240
API_BASE        = "https://text.pollinations.ai"   # free text endpoint
API_MODEL       = "openai"                         # openai-compatible text model name
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
st.set_page_config(page_title="AI Analysis of Historic Architecture", layout="wide")
st.title("AI Analysis of Historic Architecture / 历史建筑的人工智能分析")
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

def build_index(cards_path: str, index_npz: str) -> None:
    cards_path = str(cards_path); index_npz = str(index_npz)
    if not Path(cards_path).exists():
        _empty_index(index_npz, cards_path); return
    try:
        cards = load_cards_jsonl(cards_path)
    except Exception:
        _empty_index(index_npz, cards_path); return
    if not cards:
        _empty_index(index_npz, cards_path); return

    vecs, ids = [], []
    for c in cards:
        img_path = c.get("image")
        if not img_path or not Path(img_path).exists(): continue
        rgb = _read_image_rgb(img_path)
        if rgb is None: continue
        v = _lab_hist_descriptor(_resize_max_side(rgb, 720))
        vecs.append(v); ids.append(c.get("id", str(int(time.time()))))

    if not vecs:
        _empty_index(index_npz, cards_path); return

    arr = np.stack(vecs, axis=0).astype(np.float32)
    np.savez(index_npz, ids=np.array(ids, dtype=object), vecs=arr, cards_path=str(cards_path))


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

def _db_labels(lang: str) -> dict:
    return {
        "header":      _t_en_zh("📚 Building Database Manager", "📚 建筑数据库管理", lang),
        "info":        _t_en_zh("Add building entries for retrieval and grounding.",
                                "添加建筑条目用于检索与叙事实据。", lang),
        "name":        _t_en_zh("Building Name", "建筑名称", lang),
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
        "upload":      _t_en_zh("Upload main facade image", "上传立面主图", lang),
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

    name      = st.text_input(LBL["name"],      key=f"{KEY_NS}_dm_name")
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
    image_file= st.file_uploader(LBL["upload"], type=["jpg","jpeg","png"], key=f"{KEY_NS}_dm_up")

    if image_file and st.button(LBL["add_btn"], key=f"{KEY_NS}_dm_add"):
        if not name:
            st.error(LBL["err_name"])
        else:
            img_path = DATA_DIR / f"{name.replace(' ', '_')}.jpg"
            Image.open(image_file).convert("RGB").save(img_path)
            card = {
                "id": str(int(time.time())),
                "name": name, "location": location, "era": era, "style": style,
                "massing": massing, "structure": structure, "condition": condition,
                "intro": intro, "history": history,
                "materials": [m.strip() for m in materials.split(",") if m.strip()],
                "elements": [e.strip() for e in elements.split(",") if e.strip()],
                "image": str(img_path)
            }
            card["info"] = _make_info(card)
            with open(CARDS_PATH, "a", encoding="utf-8") as f:
                f.write(json.dumps(card, ensure_ascii=False) + "\n")

            with st.spinner(LBL["indexing"]):
                build_index(str(CARDS_PATH), str(IDX_PATH))

            st.success(LBL["added_ok"].format(name=name))
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




def _empty_index(index_npz: str, cards_path: str, dim: int = 96) -> None:
    Path(index_npz).parent.mkdir(parents=True, exist_ok=True)
    np.savez(index_npz, ids=np.array([], dtype=object),
             vecs=np.zeros((0, dim), dtype=np.float32),
             cards_path=str(cards_path))


def load_index(index_npz: str) -> Tuple[List[str], np.ndarray, str]:
    index_npz = str(index_npz)
    if not Path(index_npz).exists():
        return [], np.zeros((0, 96), np.float32), str(CARDS_PATH)
    try:
        data = np.load(index_npz, allow_pickle=True)
        return list(data["ids"]), data["vecs"], str(data["cards_path"])
    except Exception:
        return [], np.zeros((0, 96), np.float32), str(CARDS_PATH)

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

def retrieve_verified(
    image_path: str,
    index_npz: str,
    base_threshold: float = 0.30,   # slightly lower: allow ORB to decide
    margin: float = 0.05,
    inlier_floor: float = 0.05,
    inlier_strong: float = 0.18,
    alpha: float = 0.85,
    k: int = 6,
) -> Tuple[Optional[Dict[str, Any]], float, Dict[str, float]]:
    ids, vecs, cards_path = load_index(index_npz)
    if vecs.shape[0] == 0:
        return None, 0.0, {"reason": "empty_index"}

    # query descriptor
    qrgb = _read_image_rgb(image_path)
    if qrgb is None:
        return None, 0.0, {"reason": "bad_query_image"}
    q = _lab_hist_descriptor(_resize_max_side(qrgb, 720))

    sims = vecs @ q
    top_idx = np.argsort(-sims)[:max(1, k)]
    cards = load_cards_jsonl(cards_path)
    id2card = {c.get("id"): c for c in cards if c.get("id")}

    # group by (name, location) to avoid duplicates of same building
    def _key(c: dict) -> str:
        nm = (c.get("name") or "").strip().lower()
        loc = (c.get("location") or "").strip().lower()
        return f"{nm}|||{loc}"

    grouped = {}
    for i in top_idx:
        cid = ids[i]
        card = id2card.get(cid)
        if not card:
            continue
        key = _key(card)
        s = float(sims[i])
        if key not in grouped or s > grouped[key]["score"]:
            grouped[key] = {"card": card, "score": s}

    if not grouped:
        return None, 0.0, {"reason": "no_candidates_after_group"}

    uniq = sorted(grouped.values(), key=lambda x: -x["score"])
    s1 = uniq[0]["score"]
    s2 = uniq[1]["score"] if len(uniq) > 1 else -1.0
    best_card = uniq[0]["card"]

    n_unique = len(uniq)
    dyn_margin = min(max(0.02, margin * (0.6 if n_unique < 4 else 1.0)), 0.12 * (1.0 - s1) + 0.03)

    if s1 < base_threshold:
        return None, s1, {"reason": "below_threshold", "s1": s1, "thr": base_threshold}

    cand_img_path = best_card.get("image")
    if not cand_img_path or not Path(cand_img_path).exists():
        return None, s1, {"reason": "missing_candidate_image", "s1": s1}

    inliers = orb_inlier_ratio(image_path, cand_img_path)
    if inliers >= inlier_strong:
        combined = alpha * s1 + (1.0 - alpha) * inliers
        return best_card, combined, {"reason": "ok_strong_orb", "s1": s1, "s2": s2, "inliers": inliers, "combined": combined, "dyn_margin": dyn_margin, "n_unique": n_unique}

    if s2 >= 0 and (s1 - s2) < dyn_margin:
        return None, s1, {"reason": "low_margin", "s1": s1, "s2": s2, "dyn_margin": dyn_margin, "inliers": inliers, "n_unique": n_unique}

    if inliers < inlier_floor:
        return None, s1, {"reason": "low_inliers", "s1": s1, "inliers": inliers, "floor": inlier_floor, "n_unique": n_unique}

    combined = alpha * s1 + (1.0 - alpha) * inliers
    return best_card, combined, {"reason": "ok", "s1": s1, "s2": s2, "inliers": inliers, "combined": combined, "dyn_margin": dyn_margin, "n_unique": n_unique}



# ---------- heuristic components ----------
COLOR_MAP = {"window": (46, 204, 113), "arch": (241, 196, 15)}
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
    L, a, b = _lab(pil_img)
    H, W = L.shape
    cols = 6
    widths = np.array_split(np.arange(W), cols)
    slice_means = []
    for idx in widths:
        if idx.size == 0: continue
        aa = a[:, idx].mean(); bb = b[:, idx].mean()
        slice_means.append([aa, bb])
    slice_means = np.array(slice_means) if slice_means else np.zeros((1,2))
    disp = float(np.linalg.norm(slice_means - slice_means.mean(axis=0), axis=1).mean())
    return float(np.clip(1.0 - _safe_norm01(disp, 2.0, 12.0), 0, 1))

def principle_contrast(pil_img):
    L, _, _ = _lab(pil_img)
    rms = float(L.std())
    tex = float(cv2.Laplacian(L, cv2.CV_32F, ksize=3).var()**0.5)
    raw = 0.6 * _safe_norm01(rms, 5.0, 35.0) + 0.4 * _safe_norm01(tex, 2.0, 25.0)
    return float(np.clip(raw, 0, 1))

def principle_proportion(aspect_ratio, w2w):
    ar_term = np.exp(-((aspect_ratio - 1.5)**2)/(2*0.4**2))
    wwr_term = np.exp(-((w2w - 0.22)**2)/(2*0.12**2))
    return float(np.clip(0.6*ar_term + 0.4*wwr_term, 0, 1))

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
    g = rgb2gray(np.array(pil_img))
    edges = sobel(g)
    F = np.fft.fftshift(np.fft.fft2(edges))
    mag = np.log1p(np.abs(F))
    center = np.array(mag.shape)/2
    ys, xs = np.indices(mag.shape)
    r = np.hypot(xs-center[1], ys-center[0])
    ring = (r>20) & (r<120)
    fft_term = float(np.quantile(mag[ring], 0.98) / (np.mean(mag[ring]) + 1e-6)) if ring.sum() else 0.0
    fft_term = float(np.clip(fft_term/4.0, 0, 1))
    e = cv2.Canny((g*255).astype(np.uint8), 80, 180).astype(np.float32)
    col_sig = e.sum(axis=0)
    ac = _column_autocorr(col_sig)
    ac_term = float(np.clip(ac, 0, 1))
    return float(np.clip(0.6*fft_term + 0.4*ac_term, 0, 1))

def principle_repetition(pil_img):
    return principle_rhythm(pil_img)

def principle_simplicity(pil_img):
    _, e = _edge_map(pil_img)
    density = float(e.mean())
    hist, _ = np.histogram(e, bins=16, range=(0,255), density=True)
    p = hist + 1e-8; p /= p.sum()
    entropy = float(-(p*np.log(p)).sum())
    d_term = 1.0 - _safe_norm01(density*255.0, 5.0, 35.0)
    h_term = 1.0 - _safe_norm01(entropy, 1.0, 2.8)
    return float(np.clip(0.6*d_term + 0.4*h_term, 0, 1))

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

def build_aesthetic_viz(norms, final, clip_like, figsize=(18,7)):
    labels = ["Vert Sym", "Rot Sym", "Proportion", "Win/Wall", "Rhythm", "Fractal"]
    vals = norms + [norms[0]]
    angles = np.linspace(0, 2*np.pi, len(labels), endpoint=False)
    angles = np.concatenate([angles, angles[:1]])

    fig = plt.figure(figsize=figsize)
    ax1 = plt.subplot(1,2,1, polar=True)
    ax1.plot(angles, vals, linewidth=3)
    ax1.fill(angles, vals, alpha=0.30)
    ax1.set_xticks(np.linspace(0, 2*np.pi, len(labels), endpoint=False))
    ax1.set_xticklabels(labels, fontsize=12)
    ax1.set_ylim(0,1)
    ax1.grid(True, linewidth=0.8)
    ax1.set_title("Aesthetic feature profile (0–1)", fontsize=13)

    ax2 = plt.subplot(1,2,2)
    ax2.grid(axis="x", linewidth=0.5)
    ax2.barh(["Composite index (0–5)"], [float(clip_like)], height=0.6)
    ax2.set_xlabel("Contribution")
    ax2.set_xlim(0, max(6.0, float(clip_like) + 0.5))
    ax2.tick_params(labelsize=12)
    ax2.set_title("Score makeup", fontsize=13)

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
def _pollinations_chat(user_text: str, system_text: str = "", model: str = API_MODEL, json_mode: bool = False) -> str:
    payload = {
        "messages": [
            {"role": "system", "content": system_text},
            {"role": "user", "content": user_text}
        ],
        "model": model,
        "seed": random.randint(1, 999_999_999),
        "jsonMode": bool(json_mode),
        "private": True,
        "stream": False
    }
    try:
        r = requests.post(API_BASE, json=payload, timeout=60)
        if r.status_code == 200:
            return r.text.strip()
        return f"(Pollinations error {r.status_code})"
    except Exception as e:
        return f"(Pollinations exception: {e})"

# ---------- Narrative & Chart explanation via Pollinations ----------
def compose_metrics_context(metrics: Dict[str, Any], dets: list) -> str:
    win_count = sum(1 for d in dets if d.get("label") == "window")
    arch_count = sum(1 for d in dets if d.get("label") == "arch")
    ctx = {
        "vertical_symmetry": metrics.get("symmetry_vertical"),
        "rotational_symmetry": metrics.get("symmetry_rotational"),
        "facade_ratio_H_over_W": metrics.get("facade_ratio_H_W"),
        "window_to_wall_ratio": metrics.get("window_to_wall_ratio"),
        "rhythm_fft_peak": metrics.get("rhythm_fft_peak"),
        "fractal_dimension": metrics.get("fractal_dimension"),
        "windows_detected": win_count,
        "arches_detected": arch_count,
        "ten_principles": {k.replace("principle_",""): float(v) for k,v in metrics.items() if k.startswith("principle_")}
    }
    return json.dumps(ctx, ensure_ascii=False, indent=2)

def generate_facade_narrative_pollinations(metrics: Dict[str, Any], dets: list, lang: str = "en", db_info: Optional[str] = None) -> str:
    ctx = compose_metrics_context(metrics, dets)
    sys = "You are an architectural critic. Be precise, visual, and professional."
    if lang == "zh":
        sys = "你是一名建筑评论家。请语言准确、具象、专业，用简体中文回答。"

    db_block = ""
    if db_info:
        # Firm rule: facts from DB must be integrated; no inventions.
        if lang == "zh":
            db_block = f"\n\n【已知事实（来自数据库，必须遵循）】\n{db_info}\n\n"
        else:
            db_block = f"\n\n[KNOWN FACTS from DB — you MUST respect them and do not invent details]\n{db_info}\n\n"

    prompt = (
        f"{db_block}"
        f"Use ONLY the observed metrics/components (and the KNOWN FACTS if present) to write a 3-paragraph architectural critique.\n"
        f"Metrics JSON:\n{ctx}\n\n"
        f"Paragraph 1: composition, form, massing, how the facade occupies space.\n"
        f"Paragraph 2: organization — symmetry/asymmetry, rhythm, repetition, proportions, openings/materials implied.\n"
        f"Paragraph 3: atmosphere & aesthetic impression — light, texture, detail, perception.\n"
        f"Rules: No meta language (e.g., 'this image shows'), no surroundings/interiors, no invented history. If a detail is not in KNOWN FACTS, do not assert it as fact."
    )
    return _pollinations_chat(prompt, system_text=sys)


def chart_explainer_pollinations(metrics: Dict[str, Any], lang: str = "en") -> str:
    sys = "You explain charts succinctly for architects."
    if lang == "zh":
        sys = "你是一名为建筑师简洁解释图表的讲解者。请用简体中文。"
    prompt = (
        "You are given normalized aesthetic metrics extracted from a facade image.\n"
        f"Metrics JSON:\n{json.dumps(metrics, ensure_ascii=False, indent=2)}\n\n"
        "Write 5–7 concise bullet points (each starting with '- ') explaining the chart implications: relative levels, balance, trends, repetition, dominant peaks/gaps. "
        "Do NOT quote exact numbers. Keep it practical and insight-driven."
    )
    return _pollinations_chat(prompt, system_text=sys)

# ---------- text-only fallback (kept for resilience) ----------
def chart_explainer_text_only(metrics: Dict[str, Any], lang="en") -> str:
    def lvl(x, lo, hi):
        if x is None: return "mid"
        r = (x - lo) / (hi - lo + 1e-6)
        return "low" if r < 0.33 else ("high" if r > 0.67 else "mid")

    vs  = metrics.get("symmetry_vertical", 0.5)
    rs  = metrics.get("symmetry_rotational", 0.5)
    prop = metrics.get("facade_ratio_H_W", 1.5)
    w2w = metrics.get("window_to_wall_ratio", 0.22)
    rhy = metrics.get("rhythm_fft_peak", 1.0)
    fr  = metrics.get("fractal_dimension", 1.4)

    bullets_en = [
        f"- Left-right symmetry is {lvl(vs, 0.4, 0.85)}; the order feels {'balanced' if vs>=0.65 else 'relaxed'}.",
        f"- Rotational symmetry reads {lvl(rs, 0.35, 0.8)}, affecting perceived centering.",
        f"- Aspect (H/W≈{prop:.2f}) suggests a {'slender vertical' if prop>1.9 else ('grounded horizontal' if prop<1.1 else 'balanced')} stance.",
        f"- Window-to-wall ratio is {lvl(w2w, 0.08, 0.45)}, implying a {'transparent' if w2w>0.45 else ('solid' if w2w<0.10 else 'comfortable')} facade.",
        f"- Repetition strength is {lvl(rhy, 0.6, 2.4)}; bays {'read clearly' if rhy>=1.2 else 'are softer and less pronounced'}.",
        f"- Detail across scales is {lvl(fr, 1.2, 1.6)}, trending {'ornate' if fr>=1.55 else ('plain' if fr<=1.25 else 'measured')}."
    ]
    if lang == "zh":
        # quick inline translation prompt through Pollinations
        joined = "\n".join(bullets_en)
        zh = _pollinations_chat(
            f"Translate to Simplified Chinese, keep list formatting:\n{joined}",
            system_text="You are a precise translator."
        )
        return zh
    return "\n".join(bullets_en)


# =========================
# app.py — PART 3 of 3
# =========================

# ---------- cache helpers ----------
from functools import lru_cache

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
    return build_aesthetic_viz(norms, final=composite_index_0_5, clip_like=composite_index_0_5)

@st.cache_data(show_spinner=False, ttl=CACHE_TTL_MIN*60)
def _cached_pollinations_story(metrics: Dict[str,Any], dets: list, lang: str):
    return generate_facade_narrative_pollinations(metrics, dets, lang=lang)

@st.cache_data(show_spinner=False, ttl=CACHE_TTL_MIN*60)
def _cached_pollinations_chart(metrics: Dict[str,Any], lang: str):
    txt = chart_explainer_pollinations(metrics, lang=lang)
    if not txt or "(Pollinations" in txt:
        return chart_explainer_text_only(metrics, lang=lang)
    return txt

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
    c.drawString(LM, y, "AI Analysis of Historic Architecture / 历史建筑的人工智能分析")
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

# ---------------- Main Analysis ----------------

if page == "Analysis":
    # Make sure index is present/valid before any retrieval
    ensure_index(str(CARDS_PATH), str(IDX_PATH))

    uploaded = st.file_uploader(
        "Upload Building Facade Image / 上传建筑立面图片",
        type=["jpg", "jpeg", "png"], accept_multiple_files=True, key=f"{KEY_NS}_uploader"
    )

    if uploaded:
        for up in uploaded:
            st.subheader(up.name)
            raw = up.read()
            up.seek(0)

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

            # Composite indices
            overall_beauty = composite_beauty_score(principles)            # 0–1
            composite_index_0_5 = 5.0 * overall_beauty                     # 0–5 scale for bar
            viz_img = _cached_viz(norms, composite_index_0_5)

            # UI render — always show original, overlay, heatmap as requested
            cols = st.columns(3)
            cols[0].image(pil, caption="Original", use_container_width=True)

            if SHOW_OVERLAY_UI and overlay is not None:
                cols[1].image(overlay, caption="Semantic overlay (heuristic)", use_container_width=True)
            else:
                cols[1].empty()

            if SHOW_HEATMAP_UI and heat is not None:
                cols[2].image(heat, caption="Explainability heatmap", use_container_width=True)
            else:
                cols[2].empty()


            # Metrics dict (for text + PDF)
            metrics_dict = {
                "symmetry_vertical": round(sym_v, 3),
                "symmetry_rotational": round(sym_r, 3),
                "facade_ratio_H_W": round(ratio, 3),
                "window_to_wall_ratio": round(w2w, 3),
                "rhythm_fft_peak": round(rhythm, 3),
                "fractal_dimension": round(fractal, 3),
                **{f"principle_{k}": round(v, 3) for k, v in principles.items()}
            }

            # save temp for retrieval check
            tmp_image_path = str(Path("outputs") / f"_tmp_{Path(up.name).stem}.png")
            Image.open(io.BytesIO(raw)).convert("RGB").save(tmp_image_path)

            # retrieval verification (lightweight index)
            verified_card = None
            db_info = None
            try:
                verified_card, combined, dbg = retrieve_verified(
                    tmp_image_path, str(IDX_PATH),
                    base_threshold=0.30, margin=0.05, inlier_floor=0.05, inlier_strong=0.18, alpha=0.85
                )
                if verified_card:
                    db_info = verified_card.get("info") or _make_info(verified_card)
                    badge = f"DB match: **{verified_card.get('name','?')}**"
                    sub = []
                    if verified_card.get("location"): sub.append(verified_card["location"])
                    if verified_card.get("era"): sub.append(verified_card["era"])
                    if verified_card.get("style"): sub.append(verified_card["style"])
                    if sub:
                        badge += " — " + ", ".join(sub)
                    # st.success(badge)
                    # st.caption(
                    #     f"(combined={dbg.get('combined',0):.3f}, s1={dbg.get('s1',0):.3f}, "
                    #     f"inliers={dbg.get('inliers',0):.3f}, reason={dbg.get('reason','ok')})"
                    # )
                else:
                    st.caption(f"No reliable DB match ({dbg.get('reason','?')}, s1={dbg.get('s1',0):.3f})")
            except Exception as _e:
                st.caption(f"DB match error: {_e}")

            # Narrative (Pollinations), now **grounded** when db_info is present
            with st.spinner("Generating facade narrative..."):
                story = _cached_pollinations_story.__wrapped__(  # bypass cache to include db_info
                    metrics_dict, dets, LANG
                ) if db_info is None else generate_facade_narrative_pollinations(metrics_dict, dets, lang=LANG, db_info=db_info)


            # add highlights from principles
            top3 = sorted(principles.items(), key=lambda x: -x[1])[:3]
            low2 = sorted(principles.items(), key=lambda x: x[1])[:2]
            if LANG == "zh":
                extra = f"\n\n美学要点：优势在 {', '.join([k for k,_ in top3])}；较弱在 {', '.join([k for k,_ in low2])}。"
            else:
                extra = f"\n\nAesthetic highlights: strengths in {', '.join([k for k,_ in top3])}; weaker in {', '.join([k for k,_ in low2])}."
            story = (story or "").strip() + extra

            st.markdown("### Design Narrative / 設計敘事")
            st.write(story)

            # Viz + Chart Explanation (Pollinations)
            st.markdown("### Aesthetic Visualization / 美學視覺化")
            st.image(viz_img, caption="Feature profile and score makeup",   width='stretch')

            with st.spinner("Explaining the chart..."):
                chart_explanation = _cached_pollinations_chart(metrics_dict, lang=LANG)

            st.markdown("#### Explanation / 解释")
            st.markdown(chart_explanation)

            st.markdown("### Ten Principles of Beauty / 十大美学原则")
            st.image(principles_img, caption="Normalized 0–1 scores per principle",   width='stretch')

            st.markdown("### Overall Facade Beauty / 立面总体美度")
            overall_img = build_overall_beauty_line(overall_beauty, lang=LANG)
            st.image(overall_img, caption=f"Overall beauty = {overall_beauty:.2f}",   width='stretch')
            st.info(f"Overall Beauty (0–1): **{overall_beauty:.2f}**")

            # Ranking uses overall_beauty
            rank, N, perc = update_and_rank(overall_beauty, Path(up.name).stem, HIST_PATH)
            st.info(f"Comparative ranking / 對比排名: {rank} / {N}  (~{perc:.1f}th percentile)")

            # Ask (grounded) — only if matched
            st.markdown("### Ask / 问")
            qa_key = f"{KEY_NS}_qa_{Path(up.name).stem}"
            user_q = st.text_input("Ask a question about this building" if LANG!="zh" else "请就此建筑提问", key=qa_key)
            if user_q:
                info_text = (verified_card or {}).get("info", "")
                if info_text:
                    # Answer using Pollinations with info-only constraint
                    sys = "Answer ONLY using the provided Information. If not present, reply EXACTLY: NOTFOUND."
                    if LANG == "zh":
                        sys = "仅根据提供的信息回答。如果信息中没有，请严格回复：NOTFOUND。"
                    ans = _pollinations_chat(f"Information:\n{info_text}\n\nQuestion:\n{user_q}\nAnswer:", system_text=sys)
                    if ans.strip().upper().startswith("NOTFOUND"):
                        st.warning("Sorry, not found in the current building information." if LANG!="zh" else "抱歉，在当前建筑信息中未找到。")
                    else:
                        st.success(ans)
                else:
                    st.warning("No building info available for Q&A.")

            # PDF
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
                overall_score=overall_beauty
            )
            st.session_state["last_pdf"] = pdf_bytes
            st.session_state["last_name"] = Path(up.name).stem
            st.download_button(
                "Download report PDF / 下載報告 PDF",
                data=st.session_state["last_pdf"],
                file_name=st.session_state["last_name"] + "_report.pdf",
                mime="application/pdf",
                key=f"{KEY_NS}_dl_{st.session_state['last_name']}"
            )
