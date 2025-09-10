# app.py
# Works in GitHub Codespaces with local, open-source pretrained models (no external APIs).
# Deps (install once in Codespaces terminal):
# pip install -U streamlit pillow numpy scikit-image matplotlib reportlab opencv-python-headless \
#             transformers accelerate sentencepiece safetensors torch
import re
import os, io, math, time, json
from pathlib import Path
import numpy as np
import streamlit as st
from PIL import Image, ImageDraw
import cv2
from skimage.metrics import structural_similarity as ssim
from skimage.color import rgb2gray
from skimage.filters import sobel
from reportlab.pdfbase.pdfmetrics import stringWidth
# PDF
from reportlab.lib.pagesizes import A4
from reportlab.pdfgen import canvas
from reportlab.lib.utils import ImageReader
#//
import json
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple
#//
# Charts
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

#---- pdf imports
# PDF
from reportlab.lib.pagesizes import A4
from reportlab.pdfgen import canvas
from reportlab.lib.utils import ImageReader
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.cidfonts import UnicodeCIDFont  # ← built-in CJK
pdfmetrics.registerFont(UnicodeCIDFont("MSung-Light")) 
# register built-in Simplified Chinese font
pdfmetrics.registerFont(UnicodeCIDFont("STSong-Light"))

# ---------------- Setup ----------------
Path("outputs").mkdir(exist_ok=True)
HIST_PATH = Path("outputs/history.json")

# Model IDs (local downloads via transformers)
LLM_MODEL_ID = "google/flan-t5-base"            # small, stable, text2text
CLIP_MODEL_ID = "openai/clip-vit-base-patch32"  # image encoder
TRANS_MODEL_ID = "Helsinki-NLP/opus-mt-en-zh" 

# ---------------- Streamlit UI ----------------
st.set_page_config(page_title="AI Analysis of Taichung Historic Architecture", layout="wide")
st.title("AI Analysis of Taichung Historic Architecture / 台中历史建筑的人工智能分析")
st.title('')


# --- Sidebar: report title + language ---
with st.sidebar:
    st.header("Settings")
    
    lang_choice = st.selectbox("Language / 語言", ["English", "中文 (简体)"], index=0)

LANG = "zh" if lang_choice.startswith("中文") else "en"


uploaded = st.file_uploader(
    "Upload Building FaCade Image / 上传建筑立面图片", type=["jpg", "jpeg", "png"], accept_multiple_files=True
)

# ---------------- Device & Models ----------------
import torch
from transformers import (
    AutoTokenizer, AutoModelForSeq2SeqLM,
    CLIPProcessor, CLIPModel
)

def _resolve_device():
    if torch.cuda.is_available():
        return "cuda"
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return "mps"
    return "cpu"

# Words/symbols we don't want in the story
BAD_WORDS_EN = [
    "modern", "rustic", "italian", "victorian", "gothic", "baroque", "neoclassical",
    "brutalist", "art deco", "colonial", "romanesque", "minimalist", "industrial",
    "farmhouse", "mediterranean", "contemporary", "scandinavian",
    "[", "]", "_", "|", "—", "---", "___", "||", "|||", "_P1_", "_P2_", "_P3_"
]
BAD_WORDS_ZH = [
    "现代", "意大利", "哥特", "巴洛克", "新古典", "粗野主义", "装饰艺术", "殖民地", "罗曼式",
    "极简", "工业风", "农舍", "地中海", "当代", "北欧", "风格",
    "[", "]", "_", "|", "—", "---", "___", "||", "|||", "_P1_", "_P2_", "_P3_"
]

def build_bad_words(t5_tok):
    PHRASES = [
        "AI Analysis of Historic Architecture",
        "This article", "this article", "overview", "intended",
        "introduction", "conclusion", "paragraph", "sentence",
        "bullet", "label", "list", "outline", "prompt", "process",
        "reader", "topic", "perspective",
        # the repeated lines you saw:
        "The rhythm stays even and welcoming",
        "Light softens edges and lifts the surface",
        "Small crafted touches reward a second look",
        "The balance feels calm and sure",
    ]
    return [t5_tok(p, add_special_tokens=False).input_ids for p in PHRASES]


# ---------- DB I/O ----------
def load_cards_jsonl(path: str) -> List[Dict[str, Any]]:
    cards, buf = [], ""
    with open(path, "r", encoding="utf-8") as f:
        for raw in f:
            line = raw.strip()
            if not line or line == "data:":
                continue
            buf = f"{buf}{(' ' if buf else '')}{line}"
            try:
                obj = json.loads(buf)
                cards.append(obj)
                buf = ""  # reset for next object
            except json.JSONDecodeError:
                # not complete yet; keep buffering
                pass
    if buf.strip():
        raise ValueError("Incomplete JSON object at end of file")
    return cards

# ---------- CLIP setup ----------
_CLIP = {"model": None, "proc": None, "device": None}

def get_clip(device: str):
    if _CLIP["model"] is None:
        _CLIP["model"] = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device)
        _CLIP["proc"]  = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
        _CLIP["device"] = device
    return _CLIP["model"], _CLIP["proc"], _CLIP["device"]


@torch.inference_mode()
def embed_image(image_path: str, device: str) -> np.ndarray:
    model, proc, _ = get_clip(device)
    img = Image.open(image_path).convert("RGB")
    inputs = proc(images=img, return_tensors="pt").to(device)
    feats = model.get_image_features(**inputs)
    v = feats[0].detach().cpu().numpy().astype(np.float32)
    v /= (np.linalg.norm(v) + 1e-8)
    return v

def build_index(cards_path: str, index_npz: str, device: str) -> None:
    cards = load_cards_jsonl(cards_path)
    vecs, ids = [], []
    for c in cards:
        v = embed_image(c["image"], device)
        vecs.append(v); ids.append(c["id"])
    arr = np.stack(vecs, axis=0)  # [N, D]
    np.savez(index_npz, ids=np.array(ids, dtype=object), vecs=arr, cards_path=cards_path)

@st.cache_resource(show_spinner=True)
def load_llm_and_clip():
    device = _resolve_device()

    # FLAN-T5 for story (English)
    t5_tok = AutoTokenizer.from_pretrained(LLM_MODEL_ID)
    t5 = AutoModelForSeq2SeqLM.from_pretrained(
        LLM_MODEL_ID,
        torch_dtype=torch.float16 if device in ("cuda", "mps") else torch.float32
    ).to(device).eval()

    # CLIP
    clip_proc = CLIPProcessor.from_pretrained(CLIP_MODEL_ID)
    clip = CLIPModel.from_pretrained(CLIP_MODEL_ID).to(device).eval()

    # Translator EN->ZH
    zh_tok = AutoTokenizer.from_pretrained(TRANS_MODEL_ID)
    zh_mt = AutoModelForSeq2SeqLM.from_pretrained(
        TRANS_MODEL_ID,
        torch_dtype=torch.float16 if device in ("cuda", "mps") else torch.float32
    ).to(device).eval()
    #---load
    device = str(next(t5.parameters()).device)
    build_index("data/buildings.jsonl", "data/index_clip.npz", device)

    return device, t5_tok, t5, clip_proc, clip, zh_tok, zh_mt



try:
    device, t5_tok, t5_model, clip_proc, clip_model, zh_tok, zh_model = load_llm_and_clip()
    #st.caption(f"Device: {device} | LLM: {LLM_MODEL_ID} | CLIP: {CLIP_MODEL_ID}")
except Exception as e:
    st.error(f"Model loading failed: {e}")
    st.stop()
# BAD_WORD_IDS_EN = build_bad_words_ids(t5_tok, BAD_WORDS_EN)
# BAD_WORD_IDS_ZH = build_bad_words_ids(t5_tok, BAD_WORDS_ZH)

#------show interior images
def show_interior_gallery(card: dict):
    interiors = card.get("interiors") or []
    if not interiors:
        return
    st.markdown("### Interior views")
    cols = st.columns(3)
    for i, item in enumerate(interiors):
        path = item.get("image")
        cap  = item.get("caption", "Interior view")
        if not path:
            continue
        try:
            img = Image.open(path).convert("RGB")
            with cols[i % 3]:
                st.image(img, caption=cap, width='stretch')
        except Exception as e:
            st.warning(f"Could not load interior image: {path} ({e})")

# ---------------- Visualization helpers ----------------
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
    try:
        col_like.image(img, caption=caption, width='stretch')
    except TypeError:
        col_like.image(img, caption=caption, use_column_width=True)

def pil_from_upload(up):
    return Image.open(io.BytesIO(up.read())).convert("RGB")
def sanitize_story_text(text: str) -> str:
    """
    Removes leftover tags / prompt echoes and produces clean 3-paragraph text.
    Keeps double newlines between paragraphs; collapses internal whitespace.
    """
    # remove bracketed tags like [P1], [/P1], [something]
    text = re.sub(r"\[[^\]]+\]", " ", text)

    # remove stray tag-like leftovers: _P1_, P2:, (P3), etc.
    text = re.sub(r"[_()*#\-\s]*P\s*\d+[_()*#\-\s]*", " ", text, flags=re.I)

    # remove obvious instruction echoes
    text = re.sub(r"(?i)(now write.*|write exactly.*|you are an architecture narrator.*|hints:.*)", " ", text)

    # collapse spaces
    text = re.sub(r"\s+", " ", text).strip()

    # heuristically split into paragraphs: prefer existing blank lines, else sentence regrouping
    # try to restore paragraph breaks around natural pauses
    sents = re.split(r'(?<=[.!?])\s+', text)
    sents = [s.strip() for s in sents if len(s.strip().split()) > 2]

    if len(sents) < 4:
        # too short; just return the cleaned line
        return text

    # target ~3 paragraphs of ~4–6 sentences each
    per = max(4, min(6, math.ceil(len(sents) / 3)))
    paras = [' '.join(sents[i:i+per]) for i in range(0, len(sents), per)]
    paras = (paras + ["", "", ""])[:3]
    return ("\n\n".join([p for p in paras if p])).strip()


def wrap_text_to_width(text: str, max_width_pt: float,
                       font_name: str = "Helvetica", font_size: int = 10) -> list[str]:
    """
    Simple word-wrapping for ReportLab canvas.
    Returns a list of lines that fit within max_width_pt.
    """
    words = text.split()
    lines, current = [], ""
    for w in words:
        trial = (current + " " + w).strip() if current else w
        if stringWidth(trial, font_name, font_size) <= max_width_pt:
            current = trial
        else:
            if current:
                lines.append(current)
            current = w
    if current:
        lines.append(current)
    return lines

# ---------------- NMS (for heuristic boxes) ----------------
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

# ---------------- Simple heuristic component detection ----------------
COMPONENT_QUERIES = ["window","arch"]  # heuristic guessers
COLOR_MAP = {"window": (46, 204, 113), "arch": (241, 196, 15)}
min_det_score = 0.30

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
        if w*h < 0.001*W*H:
            continue
        ar = float(w) / float(max(1,h))
        # "window-like" rectangles
        if 0.5 <= ar <= 2.0 and h > 12 and w > 12:
            dets.append({"label":"window","score":0.55,"box":[x,y,x+w,y+h]})
        # crude "arch-like" cue: more edges in top half
        roi = edges[y:y+h, x:x+w]
        if roi.size == 0: 
            continue
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

# ---------------- Classical features ----------------
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
    center = np.array(mag.shape) / 2
    ys, xs = np.indices(mag.shape)
    r = np.hypot(xs - center[1], ys - center[0])
    ring = (r > 20) & (r < 120)
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
        if S[0] <= 0 or S[1] <= 0:
            continue
        e = edges[:S[0], :S[1]].reshape(S[0]//k, k, S[1]//k, k).max(axis=(1,3))
        N = np.count_nonzero(e)
        if N > 0:
            sizes.append(1.0/k); counts.append(N)
    if len(sizes) < 2:
        return 1.0
    slope, _ = np.polyfit(np.log(sizes), np.log(counts), 1)
    return float(slope)

# ---------------- Feature normalization + viz ----------------
def normalize_features(sym_v, sym_r, ratio, w2w, rhythm, fractal):
    nv  = np.clip((sym_v - 0.6) / 0.4, 0, 1)
    nr  = np.clip((sym_r - 0.5) / 0.5, 0, 1)
    nrx = np.exp(-((ratio - 1.5) ** 2) / (2 * 0.4 ** 2))
    nww = np.exp(-((w2w - 0.22) ** 2) / (2 * 0.12 ** 2))
    nry = np.clip(rhythm / 4.0, 0, 1)
    nfr = np.exp(-((fractal - 1.4) ** 2) / (2 * 0.15 ** 2))
    return [float(v) for v in (nv, nr, nrx, nww, nry, nfr)]

def fig_to_pil(fig, dpi=400):
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=dpi, bbox_inches="tight", pad_inches=0.1,
                facecolor="white", edgecolor="none",
                pil_kwargs={"optimize": False, "compress_level": 0})
    plt.close(fig); buf.seek(0)
    return Image.open(buf).convert("RGB")

def build_aesthetic_viz(norms, final, clip, figsize=(18,7)):
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
    ax2.barh(["CLIP aesthetic proxy"], [float(clip)], height=0.6)
    ax2.set_xlabel("Contribution")
    ax2.set_xlim(0, max(6.0, float(clip) + 0.5))
    ax2.tick_params(labelsize=12)
    ax2.set_title("Score makeup", fontsize=13)

    fig.tight_layout()
    return fig_to_pil(fig, dpi=400)

# ---------------- Local saliency heatmap ----------------
def build_saliency_heatmap(pil_img, alpha=0.45):
    im = np.array(pil_img)
    if im.ndim == 2:
        im = np.stack([im]*3, axis=-1)
    g = cv2.cvtColor(im, cv2.COLOR_RGB2GRAY).astype(np.float32)

    def _norm(x):
        x = x.astype(np.float32)
        x -= x.min()
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
    heat_rgba = Image.fromarray(heat_rgb).resize(pil_img.size, Image.BILINEAR).convert("RGBA")
    base = pil_img.convert("RGBA"); heat_rgba.putalpha(int(255*alpha))
    return Image.alpha_composite(base, heat_rgba).convert("RGB")

# ---------------- CLIP aesthetic proxy ----------------
@torch.inference_mode()
def clip_aesthetic_score(pil_img):
    inputs = clip_proc(images=pil_img, return_tensors="pt")
    inputs = {k: v.to(device) for k, v in inputs.items()}
    feats = clip_model.get_image_features(**inputs)  # [1, D]
    feats = torch.nn.functional.normalize(feats, p=2, dim=-1)[0]
    v = feats.detach().cpu().numpy()
    # stable 0..10 mapping from embedding stats
    return float(5.0 * (np.tanh(3.0 * (float(v.mean() + 0.5 * v.max()) - 0.5)) + 1.0))

# ---------------- Story writer (FLAN-T5) and helper for translation----------------
import re
import torch

# --- helpers ---
def _sent_tokenize_en(text: str) -> list[str]:
    # split on . ! ? (and Chinese 。！？ if any appear)
    parts = re.split(r'(?<=[。！？.!?])\s+', text)
    # keep only sentences with >= 3 tokens
    return [s.strip() for s in parts if len(s.strip().split()) >= 3]

def _join_paragraphs(paras: list[str]) -> str:
    return "\n\n".join(p.strip() for p in paras).strip()


@torch.inference_mode()
def translate_en_to_zh(paragraphs_en: list[str]) -> list[str]:
    outs = []
    for para in paragraphs_en:
        enc = zh_tok(para, return_tensors="pt", truncation=True, max_length=1024).to(device)
        gen = zh_model.generate(
            **enc,
            max_new_tokens=512,
            num_beams=4,
            length_penalty=1.0,
            no_repeat_ngram_size=3,
        )
        zh = zh_tok.decode(gen[0], skip_special_tokens=True).strip()
        # polish punctuation for nicer Chinese output
        zh = (zh.replace(", ", "，")
               .replace(" ,", "，")
               .replace(" .", "。")
               .replace(".", "。")
               .replace(": ", "：")
               .replace(";", "；"))
        outs.append(zh)
    return outs

# --- main generator ---
#---- story write helpers-------

# ---------- Build or update index ----------
def load_index(index_npz: str) -> Tuple[List[str], np.ndarray, str]:
    data = np.load(index_npz, allow_pickle=True)
    return list(data["ids"]), data["vecs"], str(data["cards_path"])

@torch.inference_mode()
def retrieve(image_path: str, index_npz: str, device: str) -> Tuple[Optional[Dict[str,Any]], float]:
    ids, vecs, cards_path = load_index(index_npz)
    q = embed_image(image_path, device)                       # [D]
    sims = vecs @ q                                          # cosine (both normalized)
    i = int(np.argmax(sims)); score = float(sims[i])
    cards = load_cards_jsonl(cards_path)
    id2card = {c["id"]: c for c in cards}
    return id2card.get(ids[i]), score

def build_hints(metrics: Dict[str, float]) -> List[str]:
    r  = metrics.get("facade_ratio_H_W", 1.5)
    v  = metrics.get("symmetry_vertical", 0.5)
    ry = metrics.get("rhythm_fft_peak", 1.0)
    fr = metrics.get("fractal_dimension", 1.4)
    w2w = metrics.get("window_to_wall_ratio", 0.22)

    hints = []
    hints.append("balanced overall" if 1.1 <= r <= 1.9 else ("slender vertical feel" if r > 2.0 else "grounded horizontal feel"))
    hints.append("strong symmetry" if v >= 0.65 else "relaxed symmetry")
    hints.append("clear repeating bays" if ry >= 1.2 else "soft rhythm")
    hints.append("rich ornament" if fr >= 1.55 else ("quiet detailing" if fr <= 1.25 else "measured detailing"))
    hints.append("glassy frontage" if w2w >= 0.45 else ("solid masonry presence" if w2w <= 0.10 else "comfortable window pattern"))
    return hints

def build_context_from_card(card: Dict[str,Any], hints: List[str]) -> str:
    parts = [
        f"Name: {card['name']} in {card['location']}. Era: {card.get('era','')}. Style: {card.get('style','')}.",
        f"Form: {card.get('massing','')}. Structure: {card.get('structure','')}.",
        f"Elements: {', '.join(card.get('elements', []))}. Materials: {', '.join(card.get('materials', []))}.",
        f"Order: {card.get('symmetry','')}. Rhythm: {card.get('rhythm','')}. Condition: {card.get('condition','')}.",
        f"Intro: {card.get('intro','')}. History: {card.get('history','')}.",
        f"Hints: {', '.join(hints)}."
    ]
    return " ".join([p for p in parts if p and p.strip()])

# --- prompts (no seeding) ---
# ---------- 1) helpers ----------

def build_hints(metrics: dict) -> list[str]:
    r  = metrics.get("facade_ratio_H_W", 1.5)
    v  = metrics.get("symmetry_vertical", 0.5)
    ry = metrics.get("rhythm_fft_peak", 1.0)
    fr = metrics.get("fractal_dimension", 1.4)
    w2w = metrics.get("window_to_wall_ratio", 0.22)
    return [
        "balanced overall" if 1.1 <= r <= 1.9 else ("slender vertical feel" if r > 2.0 else "grounded horizontal feel"),
        "strong symmetry" if v >= 0.65 else "relaxed symmetry",
        "clear repeating bays" if ry >= 1.2 else "soft rhythm",
        "rich ornament" if fr >= 1.55 else ("quiet detailing" if fr <= 1.25 else "measured detailing"),
        "glassy frontage" if w2w >= 0.45 else ("solid masonry presence" if w2w <= 0.10 else "comfortable window pattern"),
    ]

def build_facts_line(card: dict | None) -> str:
    if not card:
        return ""
    parts = [
        f"Name: {card.get('name','')} in {card.get('location','')}.",
        f"Style: {card.get('style','')}. Era: {card.get('era','')}.",
        f"Form: {card.get('massing','')}. Structure: {card.get('structure','')}.",
        f"Order: {card.get('symmetry','')}. Rhythm: {card.get('rhythm','')}.",
        f"Materials: {', '.join(card.get('materials', []))}.",
        f"Condition: {card.get('condition','')}.",
    ]
    return " ".join(p for p in parts if p.strip())


def build_bad_words(t5_tok):
    banned = [
        # meta/structure words the model keeps echoing
        "article","this article","analysis","study","survey","review","guide","overview","research",
        "paragraph","section","passage","label","list","reader",
        "Describe","describe","Explain","explain",
        # stale filler you keep seeing
        "The rhythm stays even and welcoming",
        "Light softens edges and lifts the surface",
        "Small crafted touches reward a second look",
        "The balance feels calm and sure",
        "These passages are intended",
        "This text is intended",
    ]
    return [t5_tok(b, add_special_tokens=False).input_ids for b in banned]

def get_gen_kwargs(t5_tok):
    return dict(
        max_new_tokens=240,
        do_sample=False,                 # deterministic, less drift
        num_beams=8,
        length_penalty=1.08,
        no_repeat_ngram_size=6,
        encoder_no_repeat_ngram_size=6,  # prevents copying from the prompt
        repetition_penalty=1.4,
        early_stopping=True,
        bad_words_ids=build_bad_words(t5_tok),
    )
#-- give answer 
# ---------- LLM Q&A using `info` as grounding ----------
@torch.inference_mode()
def guide_answer_from_info(question: str, info_text: str, LANG: str) -> str:
    """
    Use FLAN-T5 to answer conversationally, but ONLY from the given info_text.
    If not found, return a polite apology.
    """
    if not question or not info_text:
        return "NOTFOUND"

    # Instruction in English and Chinese
    inst_en = (
        "You are a knowledgeable guide for historic architecture. "
        "Answer the visitor's question in details using the following Information. "
        "Modify the information and make it intersting for user."
        "If the answer is not in the Information, reply exactly with: NOTFOUND.\n\n"
        f"Information:\n{info_text}\n\n"
        f"Question: {question}\nAnswer:"
    )
    inst_zh = (
        "你是历史建筑的导览员。请只根据以下信息回答游客的问题。"
        "如果答案不在信息中，请只回复：NOTFOUND。\n\n"
        f"信息：\n{info_text}\n\n"
        f"问题：{question}\n答案："
    )
    prompt = inst_zh if LANG == "zh" else inst_en

    enc = t5_tok(prompt, return_tensors="pt", truncation=True, max_length=1024).to(device)
    out = t5_model.generate(
        **enc,
        max_new_tokens=5000,
        min_new_tokens=30,
        num_beams=6,
        do_sample=False,
        length_penalty=1.0,
        no_repeat_ngram_size=4,
        encoder_no_repeat_ngram_size=4,
        early_stopping=True,
    )
    ans = t5_tok.decode(out[0], skip_special_tokens=True).strip()

    if ans.upper().startswith("NOTFOUND"):
        return "NOTFOUND"
    return ans
# -------- Language helper --------
def _lang_text(en: str, zh: str, LANG: str) -> str:
    """Return English or Chinese text based on LANG flag."""
    return zh if LANG == "zh" else en

# ---------- 2) writer (simple and grounded if image matches) ----------


@torch.inference_mode()
def write_story(
    metrics: dict,
    lang: str = "en",
    image_path: Optional[str] = None,
    index_npz: str = "data/index_clip.npz",
    match_threshold: float = 0.22,
    fewshot: Optional[List[Tuple[str, str]]] = None,
    k_examples: int = 2,
) -> str:
    """
    Generic 3-paragraph narrative with DB-first priority:

      P1 — history & city context, then one clean form line.
      P2 — exterior composition/structure/materials, rhythm vs. human movement.
      P3 — ground-level experience (texture, light/shadow), concise significance.

    Exterior-only. Only commas and periods. No meta language.
    """
    import re

    device = str(next(t5_model.parameters()).device)

    # ---------- retrieve nearest card (for grounding) ----------
    card: Optional[dict] = None
    if image_path and Path(index_npz).exists():
        try:
            c, s = retrieve(image_path, index_npz, device)
            if c and s >= match_threshold:
                card = c
        except Exception:
            card = None

    # ---------- quick helpers ----------
    def _norm(s: Optional[str]) -> str:
        return re.sub(r"\s+", " ", s or "").strip()

    def _norm_era(e: Optional[str]) -> str:
        e2 = _norm(e)
        if not e2: return ""
        if "century" in e2.lower() and "ad" not in e2.lower():
            return e2 + " AD"
        return e2

    def _uniq(seq):
        out, seen = [], set()
        for x in seq:
            k = x.lower().strip()
            if k and k not in seen:
                seen.add(k); out.append(x)
        return out

    def _join(items: list[str]) -> str:
        items = _uniq([i for i in items if i])
        if not items: return ""
        if len(items) == 1: return items[0]
        return ", ".join(items[:-1]) + ", and " + items[-1]

    def _clean_text(t: str) -> str:
        t = re.sub(r"\[[^\]]+\]", " ", t)
        t = re.sub(r"[;:–—_•\[\]\(\)]", " ", t)   # commas/periods only
        # filter banned / vague
        bans = [
            r"\bporch(es)?\b", r"\binterior(s)?\b", r"\blobby\b",
            r"\bsense of scale\b", r"\brectangular\s+oval\b",
            r"\bround\s+tower\b", r"\bcircular\s+shaft\b", r"\bcauldron\b"
        ]
        for rx in bans:
            t = re.sub(rx, "", t, flags=re.I)
        t = re.sub(r"\s{2,}", " ", t).strip()
        return t

    def _sentences(text: str) -> list[str]:
        return [s.strip() for s in re.split(r'(?<=[.!?])\s+', text or "") if len(s.strip().split()) >= 3]

    # ---------- DB fields ----------
    name      = _norm((card or {}).get("name"))
    loc       = _norm((card or {}).get("location"))
    era       = _norm_era((card or {}).get("era"))
    intro     = _norm((card or {}).get("intro"))
    history   = _norm((card or {}).get("history"))
    massing   = _norm((card or {}).get("massing"))
    structure = _norm((card or {}).get("structure"))
    style     = _norm((card or {}).get("style"))
    condition = _norm((card or {}).get("condition"))

    # Prefer curated 3-paragraph narrative if present
    narrative = (card or {}).get("narrative")  # {"p1": "...","p2":"...","p3":"..."}
    if isinstance(narrative, dict) and all(isinstance(narrative.get(k), str) and narrative.get(k).strip() for k in ("p1","p2","p3")):
        p1 = _clean_text(narrative["p1"])
        p2 = _clean_text(narrative["p2"])
        p3 = _clean_text(narrative["p3"])
        result_en = "\n\n".join([p for p in (p1, p2, p3) if p]).strip()
        if lang == "zh":
            parts = [p for p in result_en.split("\n\n") if p.strip()]
            zh_parts = translate_en_to_zh(parts)
            return "\n\n".join(zh_parts).strip()
        return result_en

    # ---------- normalize elements/materials (exterior-only) ----------
    raw_elements  = [e for e in (card or {}).get("elements", []) if isinstance(e, str)]
    raw_materials = [m for m in (card or {}).get("materials", []) if isinstance(m, str)]

    banned_interior_noise = {
        "interior","room","rooms","hall","halls","dining","nave","apse","altar",
        "chapel","corridor","atrium","lobby","gallery","parliament chambers"
    }
    allowed_exterior = {
        "tower","towers","pinnacle","pinnacles","tracery","buttress","buttresses",
        "arcade","arcades","arch","arches","archway","bay","bays","pier","piers",
        "pilaster","pilasters","capital","cornice","cornices","stringcourse","molding",
        "keystone","voussoir","spandrel","colonnade","opening","openings",
        "parapet","attic","balustrade","lintel","jamb","threshold",
        "vault","vaults","barrel vaults","rib vaults","radial walls","truss","frame",
        "elevation","facade","wall","walls","window","windows","door","doors"
    }
    mat_map = {
        "roman concrete":"roman concrete","concrete":"concrete",
        "brick":"brick","brickwork":"brick","stone":"stone","stonework":"stone",
        "travertine":"travertine","tuff":"tuff","marble":"marble","granite":"granite",
        "iron":"iron","steel":"steel","wood":"wood","timber":"wood","terracotta":"terracotta","clay":"terracotta",
        "anston limestone":"anston limestone","limestone":"limestone"
    }

    elements = []
    for e in raw_elements:
        e2 = e.strip().lower()
        if e2 in banned_interior_noise:
            continue
        if e2 in allowed_exterior:
            elements.append(e2)
    elements = _uniq(elements)

    materials = []
    for m in raw_materials:
        m2 = mat_map.get(m.strip().lower())
        if m2: materials.append(m2)
    if "roman concrete" in materials and "concrete" in materials:
        materials = [m for m in materials if m != "concrete"]
    materials = _uniq(materials)

    # ---------- paragraph builders (deterministic) ----------
    def build_p1() -> str:
        bits = []
        if name and loc:
            bits.append(f"{name} is in {loc}.")
        elif loc:
            bits.append(f"The building is in {loc}.")
        if era:
            bits.append(f"It was built in the {era}.")
        # one or two short factual lines from history/intro
        picked = []
        for src in (history, intro):
            for s in _sentences(src):
                if 4 <= len(s.split()) <= 25:
                    picked.append(s)
            if len(picked) >= 2: break
        bits += picked[:2]

        # concise form line from massing/elements
        m = massing.lower()
        if any(w in m for w in ["oval","elliptic","elliptical"]):
            bits.append("The oval form rises in layered tiers.")
        elif any(w in m for w in ["circle","circular","rotunda"]):
            bits.append("The circular form rises in layered tiers.")
        elif any(w in m for w in ["long","riverfront","bays"]) or "bays" in elements:
            bits.append("A long front steps in measured bays.")
        elif "towers" in elements or "pinnacles" in elements:
            bits.append("Towers and pinnacles mark the skyline.")
        elif any(w in m for w in ["rectangular","block","massive"]):
            bits.append("A rectangular mass rises in layered levels.")
        elif "arches" in elements or "arcades" in elements:
            bits.append("A ring of openings stacks in layered tiers.")
        return _clean_text(" ".join(bits))

    def build_p2() -> str:
        bits = []
        # composition / rhythm
        if "arcades" in elements or "arches" in elements:
            bits.append("Stacked arcades march around the exterior.")
        elif "bays" in elements:
            bits.append("Regular bays set a steady rhythm across the facade.")
        else:
            bits.append("Repeated openings set a steady rhythm across the facade.")
        # structure
        if "barrel vaults" in elements or "vaults" in elements or "rib vaults" in elements:
            if "radial walls" in elements:
                bits.append("Barrel vaults and radial walls carry the upper levels.")
            else:
                bits.append("Vaults carry the upper levels.")
        elif structure:
            bits.append(f"Structure is {structure}.")
        # articulation
        if "tracery" in elements:
            bits.append("Window tracery gives fine vertical pattern.")
        if "buttresses" in elements or "buttress" in elements:
            bits.append("Buttresses brace the walls and break the mass into bays.")
        if "engaged columns" in elements:
            bits.append("Engaged columns articulate the openings.")
        if "piers" in elements:
            bits.append("Strong piers frame the openings.")
        if "towers" in elements or "pinnacles" in elements:
            bits.append("Towers and pinnacles punctuate the roofline.")
        # materials
        if materials:
            bits.append(f"The structure was built with {_join(materials)}, showing both strength and adaptability.")
        return _clean_text(" ".join(bits))

    def build_p3() -> str:
        # pick a feature noun for light/shadow
        feature = "arches" if ("arches" in elements or "arcades" in elements) else ("windows" if "tracery" in elements else "openings")
        # material texture line
        tex = []
        if "travertine" in materials:
            tex.append("Travertine blocks feel smooth and pale")
        if "anston limestone" in materials or "limestone" in materials:
            tex.append("limestone shows crisp carving and weathering")
        if "brick" in materials and ("roman concrete" in materials or "concrete" in materials):
            tex.append("brick and concrete add warmth and variation")
        elif "brick" in materials:
            tex.append("brick adds warmth and variation")
        elif "roman concrete" in materials or "concrete" in materials:
            tex.append("concrete adds variation in tone")

        bits = []
        bits.append("At ground level, the surfaces reveal textures of stone and masonry.")
        if tex:
            bits.append(tex[0] + ".")
        bits.append(f"Light and shadow shift across the {feature}, giving the exterior both grandeur and intimacy.")
        # closing significance
        if intro:
            bits.append("Even today, the building remains a clear marker of civic life and identity.")
        else:
            bits.append("Even today, the building remains a clear marker of its city and public life.")
        return _clean_text(" ".join(bits))

    # ---------- assemble ----------
    p1 = build_p1()
    p2 = build_p2()
    p3 = build_p3()

    result_en = "\n\n".join([p for p in (p1, p2, p3) if p]).strip()

    # ---------- Chinese option ----------
    if lang == "zh":
        parts = [p for p in result_en.split("\n\n") if p.strip()]
        zh_parts = translate_en_to_zh(parts)
        return "\n\n".join(zh_parts).strip()

    return result_en


#---story with database
# @torch.inference_mode()
# def write_story(
#     metrics: dict,
#     lang: str = "en",
#     image_path: Optional[str] = None,
#     index_npz: str = "data/index_clip.npz",
#     match_threshold: float = 0.22,
#     fewshot: Optional[List[Tuple[str, str]]] = None,  # optional, keeps compatibility
#     k_examples: int = 2
# ) -> str:
#     """
#     Extended, grounded architectural narrative.
#     - If CLIP finds a match in data/buildings.jsonl, we:
#         * switch to 4 paragraphs (longer story),
#         * inject real facts from the card (name, location, era, style, intro, history, materials, elements),
#         * optionally include the card's 'story' as style guidance (not copied).
#     - No labels or meta commentary in the output.
#     """
#     import re

#     device = str(next(t5_model.parameters()).device)

#     # ---- cues from metrics (concise, non-meta) ----
#     r  = metrics.get("facade_ratio_H_W", 1.5)
#     v  = metrics.get("symmetry_vertical", 0.5)
#     ry = metrics.get("rhythm_fft_peak", 1.0)
#     fr = metrics.get("fractal_dimension", 1.4)
#     w2w = metrics.get("window_to_wall_ratio", 0.22)
#     cues = [
#         ("balanced overall" if 1.1 <= r <= 1.9 else ("slender vertical feel" if r > 2.0 else "grounded horizontal feel")),
#         ("strong symmetry" if v >= 0.65 else "relaxed symmetry"),
#         ("clear repeating bays" if ry >= 1.2 else "soft rhythm"),
#         ("rich ornament" if fr >= 1.55 else ("quiet detailing" if fr <= 1.25 else "measured detailing")),
#         ("glassy frontage" if w2w >= 0.45 else ("solid masonry presence" if w2w <= 0.10 else "comfortable window pattern")),
#     ]

#     # ---- try to ground with CLIP index ----
#     card, score = (None, 0.0)
#     if image_path and Path(index_npz).exists():
#         try:
#             c, s = retrieve(image_path, index_npz, device)
#             if c and s >= match_threshold:
#                 card, score = c, s
#         except Exception:
#             card, score = None, 0.0

#     # ---- build rich facts (prefer exterior + history) ----
#     def _norm_era(e: str) -> str:
#         if not e: return ""
#         if "century" in e.lower() and "ad" not in e.lower():
#             return e.strip() + " AD"
#         return e.strip()

#     def _facts_lines(c: Optional[dict]) -> str:
#         if not c: return ""
#         parts = []
#         if c.get("name"):      parts.append(f"{c['name']}")
#         if c.get("location"):  parts.append(f"in {c['location']}")
#         if c.get("era"):       parts.append(f"built in the {_norm_era(c['era'])}")
#         # style as context, not a label dump
#         if c.get("style"):     parts.append(f"showing {c['style'].lower()} traits")
#         base = ", ".join(parts) + "." if parts else ""

#         extras = []
#         if c.get("intro"):     extras.append(c["intro"])
#         if c.get("history"):   extras.append(c["history"])
#         if c.get("materials"): extras.append("Materials include " + ", ".join(c["materials"]) + ".")
#         if c.get("elements"):  extras.append("Key parts include " + ", ".join(c["elements"]) + ".")
#         return " ".join([base] + extras).strip()

#     facts = _facts_lines(card)
#     example_story = (card.get("story", "").strip() if card else "")
#     if example_story:
#         example_story = re.sub(r"\s+", " ", example_story)
#         if len(example_story) > 1000:
#             example_story = example_story[:1000].rsplit(" ", 1)[0] + " ..."

#     # ---- optional few-shot (by similarity) to steer tone; not required ----
#     fewshot_block = ""
#     if fewshot and image_path:
#         try:
#             qv = embed_image(image_path, device)
#             picked = []
#             for ex_img, ex_story in fewshot:
#                 try:
#                     vv = embed_image(ex_img, device)
#                     sim = float(np.dot(qv, vv) / (np.linalg.norm(qv) * np.linalg.norm(vv) + 1e-8))
#                     s_clean = re.sub(r"\s+", " ", ex_story).strip()
#                     picked.append((sim, s_clean))
#                 except Exception:
#                     continue
#             picked.sort(key=lambda x: x[0], reverse=True)
#             picked = [s for _, s in picked[:max(1, min(k_examples, len(picked)))]]
#             if picked:
#                 fewshot_block = "\n\n".join([f"Example:\n{p}" for p in picked]) + "\n\n"
#         except Exception:
#             pass

#     # also include the DB story as a soft style cue (if present)
#     if example_story:
#         fewshot_block = (fewshot_block + f"Example (database):\n{example_story}\n\n")

#     # ---- bad-words: block meta/labels and known fillers ----
#     def _bad_words_ids(tok):
#         ban = [
#             "Paragraph one", "Paragraph two", "Paragraph three",
#             "paragraph one", "paragraph two", "paragraph three",
#             "Cues:", "Hints:", "This is a", "This text", "This passage", "This description",
#             "It begins with a distant view", "then describes composition",
#             "The rhythm stays even and welcoming",
#             "Light softens edges and lifts the surface",
#             "Small crafted touches reward a second look",
#             "The balance feels calm and sure",
#             "interior", "Interior",  # steer away from interior talk
#         ]
#         return [tok(b, add_special_tokens=False).input_ids for b in ban]

#     # ---- choose length: extend when grounded ----
#     num_paras = 4 if card else 3
#     min_sents = 5 if card else 4
#     max_sents = 7 if card else 6

#     # ---- build prompt (no labels, exterior focus, facts woven in) ----
#     facts_block = f"Facts to use accurately: {facts}\n\n" if facts else ""
#     prompt = (
#         f"{fewshot_block}"
#         "Write a cohesive architectural narrative about the exterior of the building in the image.\n"
#         f"Produce {num_paras} paragraphs. Each paragraph has {min_sents} to {max_sents} short sentences.\n"
#         "Plain English, present tense, active voice. Use only commas and periods. No headings or bullets. No meta commentary.\n"
#         "Begin with the distant view and light. Then composition, balance, symmetry, proportion, rhythm using parts such as arches, bays, columns, cornices, joints, openings. "
#         "Include relevant historical context if known. Finish with materials, textures, crafted details, signs of age or repair, and the feeling at human scale.\n"
#         "Do not discuss interiors. Do not fabricate names or dates.\n\n"
#         f"{facts_block}"
#         f"Cues: {', '.join(cues)}.\n"
#         "Return only prose."
#     )

#     # ---- generate deterministically ----
#     enc = t5_tok(prompt, return_tensors="pt", truncation=True, max_length=1024).to(device)
#     out = t5_model.generate(
#         **enc,
#         max_new_tokens=480,
          
#         num_beams=8,
#         do_sample=False,
#         length_penalty=1.08,
#         no_repeat_ngram_size=6,
#         encoder_no_repeat_ngram_size=6,
#         repetition_penalty=1.38,
#         early_stopping=True,
#         bad_words_ids=_bad_words_ids(t5_tok),
#     )
#     text = t5_tok.decode(out[0], skip_special_tokens=True).strip()

#     # ---- clean and arrange into the requested number of paragraphs ----
#     text = re.sub(r"\[[^\]]+\]", " ", text)
#     text = re.sub(r"(?i)(cues?|hints?)\s*:.*", " ", text)
#     text = re.sub(r"(?i)paragraph\s*(one|two|three|four)\s*:?", " ", text)
#     text = re.sub(r"[|_—\-]{2,}", " ", text)
#     text = re.sub(r"\s+", " ", text).strip()

#     # sentence split
#     sents = [s.strip() for s in re.split(r'(?<=[.!?])\s+', text) if len(s.strip().split()) >= 3]
#     # drop any residual banned lines verbatim
#     banned = {
#         "The rhythm stays even and welcoming.",
#         "Light softens edges and lifts the surface.",
#         "Small crafted touches reward a second look.",
#         "The balance feels calm and sure.",
#     }
#     sents = [s for s in sents if s not in banned]
#     # cap length to something sane
#     sents = sents[: num_paras * (max_sents + 1)]

#     # chunk evenly across num_paras without inventing filler
#     if not sents:
#         result_en = ""
#     else:
#         approx = max(min_sents, min(max_sents, len(sents) // num_paras or min_sents))
#         paras = []
#         i = 0
#         for p in range(num_paras - 1):
#             j = min(len(sents), i + approx)
#             paras.append(" ".join(sents[i:j]).strip()); i = j
#         paras.append(" ".join(sents[i:]).strip())
#         result_en = "\n\n".join([p for p in paras if p]).strip()

#     if lang == "zh":
#         parts = [p.strip() for p in result_en.split("\n\n") if p.strip()]
#         zh_parts = translate_en_to_zh(parts)
#         return "\n\n".join(zh_parts).strip()

#     return result_en

#story general 
# @torch.inference_mode()
# def write_story(metrics: dict, lang: str = "en") -> str:
#     # Build human hints from metrics
#     r  = metrics.get("facade_ratio_H_W", 1.5)
#     v  = metrics.get("symmetry_vertical", 0.5)
#     ry = metrics.get("rhythm_fft_peak", 1.0)
#     fr = metrics.get("fractal_dimension", 1.4)
#     w2w = metrics.get("window_to_wall_ratio", 0.22)

#     hints = []
#     hints.append("balanced overall" if 1.1 <= r <= 1.9 else ("slender vertical feel" if r > 2.0 else "grounded horizontal feel"))
#     hints.append("strong symmetry" if v >= 0.65 else "relaxed symmetry")
#     hints.append("clear repeating bays" if ry >= 1.2 else "soft rhythm")
#     hints.append("rich ornament" if fr >= 1.55 else ("quiet detailing" if fr <= 1.25 else "measured detailing"))
#     if w2w >= 0.45: hints.append("glassy frontage")
#     elif w2w <= 0.10: hints.append("solid masonry presence")
#     else: hints.append("comfortable window pattern")

#     # prompt = (
#     #     "You are an architecture narrator.\n"
#     #     "Write EXACTLY three paragraphs, each 4 to 6 short sentences.\n"
#     #     "Keep language simple. No style labels. No numbers. No bullet points.\n"
#     #     "Focus on balance, symmetry, rhythm, light, and crafted details.\n"
#     #     "Describe what the eye notices first, then what rewards a closer look.\n"
#     #     f"Hints: {', '.join(hints)}\n"
#     #     "Return only prose paragraphs separated by a blank line."
#     # )
#     prompt = (
#     "Write exactly three paragraphs of 4–6 short sentences each.\n"
#     "Plain English, present tense, active voice. Use only commas and periods.\n"
#     "No headings, lists, or commentary about writing, readers, or process.\n"
#     "Never use these words: article, analysis, study, survey, review, guide, overview, tutorial, introduction, conclusion, paragraph, page, section, passage, purpose, function, story, post.\n\n"
#     f"Cues: {', '.join(hints)}.\n\n"
#     "Paragraph one: distant view and light.\n"
#     "Paragraph two: composition with balance, symmetry, proportion, rhythm using parts such as arches, bays, columns, cornices, joints, openings, and include style and era if known.\n"
#     "Paragraph three: materials, craft, textures, signs of age or repair, and close human-scale feel.\n"
#     "Return only the three paragraphs separated by one blank line."
#     )


#     enc = t5_tok(prompt, return_tensors="pt", truncation=True, max_length=1024).to(device)
#     gen_kwargs = dict(
#         max_new_tokens=320,
#         min_new_tokens=220,           # encourage 3 paras
#         do_sample=True,
#         temperature=0.7,
#         top_p=0.9,
#         no_repeat_ngram_size=3,
#         repetition_penalty=1.12,
#     )
#     # (optional) bad words list if you defined BAD_WORD_IDS_EN elsewhere
#     bad_ids = globals().get("BAD_WORD_IDS_EN")
#     if bad_ids is not None:
#         gen_kwargs["bad_words_ids"] = bad_ids

#     out = t5_model.generate(**enc, **gen_kwargs)
#     text_en = t5_tok.decode(out[0], skip_special_tokens=True).strip()

#     # enforce 3 paragraphs
#     paras_en = _group_into_three_paragraphs(text_en)

#     if lang == "zh":
#         paras_zh = translate_en_to_zh(paras_en)
#         return _join_paragraphs(paras_zh)
#     return _join_paragraphs(paras_en)

# ---------------- Ranking ----------------
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
# ---------------- PDF ----------------
def make_report(
    orig_img,
    overlay_img,
    heat_img,
    viz_img,
    metrics,
    ranking,
    story_text,
    lang: str = "en",
    interiors: Optional[list] = None,   # <-- NEW
):
    import io, re, time as _t
    from reportlab.lib.pagesizes import A4
    from reportlab.pdfgen import canvas
    from reportlab.lib.utils import ImageReader
    from reportlab.pdfbase import pdfmetrics
    from reportlab.pdfbase.cidfonts import UnicodeCIDFont
    from PIL import Image

    # ensure CJK font available
    try:
        pdfmetrics.getFont("STSong-Light")
    except KeyError:
        pdfmetrics.registerFont(UnicodeCIDFont("STSong-Light"))

    buf = io.BytesIO()
    c = canvas.Canvas(buf, pagesize=A4)
    W, H = A4
    TOP, BOT, LM, RM = 40, 50, 40, 40
    y = H - TOP

    def draw_img_fit_top(pil, x_left, y_top, max_w):
        if pil is None:
            return 0.0
        try:
            w_px, h_px = pil.size
        except Exception:
            return 0.0
        if w_px <= 0 or h_px <= 0:
            return 0.0
        w = float(max_w)
        h = w * (h_px / float(w_px))
        c.drawImage(ImageReader(pil), x_left, y_top - h, width=w, height=h,
                    preserveAspectRatio=True, mask='auto')
        return h

    def wrap_lines(text, max_width_pt, font_name, font_size, is_cjk=False):
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

    def clean_story(t: str) -> str:
        t = re.sub(r"\[[^\]]+\]", " ", t)
        t = re.sub(r"[|_—\-]{2,}", " ", t)
        t = re.sub(r"\s+", " ", t).strip()
        parts = [p.strip() for p in re.split(r'\n\s*\n', t) if p.strip()]
        if not parts:
            sents = re.split(r'(?<=[.!?])\s+', t)
            sents = [s.strip() for s in sents if len(s.strip().split()) >= 3]
            per = max(4, min(6, (len(sents) // 3) or 4))
            parts = [' '.join(sents[i:i+per]) for i in range(0, len(sents), per)]
        return "\n\n".join(parts[:6])

    def elide(text: str, max_w: float, font_name: str, font_size: int) -> str:
        """Single-line ellipsis to fit width."""
        sw = pdfmetrics.stringWidth
        if sw(text, font_name, font_size) <= max_w:
            return text
        ell = "…"
        out = ""
        for ch in text:
            if sw(out + ch + ell, font_name, font_size) <= max_w:
                out += ch
            else:
                break
        return out + ell

    # Header
    c.setFont("STSong-Light", 16)
    c.drawString(LM, y, "AI Analysis of Taichung Historic Architecture / 台中历史建筑的人工智能分析")
    y -= 22
    c.setFont("STSong-Light", 10)
    c.drawString(LM, y, _t.strftime("Generated on / 生成于 %Y-%m-%d %H:%M:%S"))
    y -= 16

    # Row of three images (original, overlay, heat)
    COL_W = (W - LM - RM - 20) / 3.0
    row_top = y
    h1 = draw_img_fit_top(orig_img, LM, row_top, COL_W)
    h2 = draw_img_fit_top(overlay_img, LM + COL_W + 10, row_top, COL_W)
    h3 = draw_img_fit_top(heat_img, LM + 2*(COL_W + 10), row_top, COL_W)
    y = row_top - max(h1, h2, h3) - 14

    # Story block
    if y < BOT + 140:
        c.showPage(); y = H - TOP
    c.setFont("STSong-Light", 12)
    c.drawString(LM, y, "Design Narrative / 设计叙事")
    y -= 14
    body_font = "STSong-Light" if lang == "zh" else "Helvetica"
    font_size = 10 if lang == "en" else 11
    leading = 13 if lang == "en" else 14
    c.setFont(body_font, font_size)

    story_clean = clean_story(story_text or "")
    paragraphs = [p.strip() for p in story_clean.split("\n\n") if p.strip()]
    max_w = W - LM - RM
    for para in paragraphs:
        for ln in wrap_lines(para, max_w, body_font, font_size, is_cjk=(lang=="zh")):
            if y < BOT:
                c.showPage(); y = H - TOP
                c.setFont(body_font, font_size)
            c.drawString(LM, y, ln)
            y -= leading
        y -= int(leading * 0.5)

    # -------- Interior Views (NEW) --------
    if interiors:
        # new page if not enough room for a grid row
        if y < BOT + 160:
            c.showPage(); y = H - TOP

        c.setFont("STSong-Light", 12)
        c.drawString(LM, y, "Interior Views / 室内影像")
        y -= 10

        cols = 3
        gutter = 10.0
        cell_w = (W - LM - RM - gutter * (cols - 1)) / cols
        caption_font = "STSong-Light"
        caption_size = 9
        c.setFont(caption_font, caption_size)

        col_i = 0
        row_max_h = 0.0
        x = LM

        for item in interiors:
            path = (item or {}).get("image")
            cap  = (item or {}).get("caption", "Interior view")

            # load safely
            try:
                im = Image.open(path).convert("RGB")
            except Exception:
                # skip unreadable images
                continue

            # ensure space for image + caption; if not, new row / page
            min_cell_h = cell_w * 0.6 + 18  # conservative estimate
            if y - min_cell_h < BOT:
                # flush remaining columns as a new page row
                c.showPage(); y = H - TOP
                c.setFont("STSong-Light", 12)
                c.drawString(LM, y, "Interior Views / 室内影像")
                y -= 10
                c.setFont(caption_font, caption_size)
                x = LM
                col_i = 0
                row_max_h = 0.0

            # draw image
            img_h = draw_img_fit_top(im, x, y, cell_w)

            # draw caption (single line, ellipsized)
            cap_y = y - img_h - 11
            c.setFont(caption_font, caption_size)
            c.drawString(x, cap_y, elide(cap, cell_w, caption_font, caption_size))

            # track row height
            used_h = img_h + 18  # image + space + caption
            row_max_h = max(row_max_h, used_h)

            # move to next column or wrap to next row
            col_i += 1
            if col_i < cols:
                x += cell_w + gutter
            else:
                # next row
                y = y - row_max_h - 12
                x = LM
                col_i = 0
                row_max_h = 0.0

        # if last row not closed, move y below it so later content won't overlap
        if col_i != 0:
            y = y - row_max_h - 12

    # Viz page
    c.showPage()
    y = H - TOP
    c.setFont("STSong-Light", 12)
    c.drawString(LM, y, "Aesthetic Visualization / 美学可视化")
    y -= 8
    if viz_img is not None:
        _ = draw_img_fit_top(viz_img, LM, y, W - LM - RM)
        w_px, h_px = viz_img.size
        y -= (W - LM - RM) * (h_px / float(w_px)) + 16

    # Ranking + metrics
    c.setFont("Helvetica", 11)
    try:
        rank, N, perc = ranking
        c.drawString(LM, y, f"Comparative ranking: {rank} / {N}  (~{perc:.1f}th percentile)")
        y -= 16
    except Exception:
        pass

    c.setFont("Helvetica", 10)
    for k, v in (metrics or {}).items():
        if y < BOT:
            c.showPage(); y = H - TOP; c.setFont("Helvetica", 10)
        c.drawString(LM, y, f"{k}: {v}")
        y -= 12

    c.showPage()
    c.save()
    pdf_bytes = buf.getvalue()
    buf.close()
    return pdf_bytes

# ---------------- Main ----------------
if uploaded:
    for up in uploaded:
        st.subheader(up.name)
        pil = pil_from_upload(up)
        col1, col2, col3 = st.columns(3)

        # Heuristic components
        with st.spinner("Detecting simple components..."):
            dets = heuristic_components(pil)
        overlay = draw_semantic_overlay(pil, dets)

        # Window-to-wall uses heuristic windows
        ratio, w2w = compute_proportions(pil, dets)

        # Features
        sym_v, sym_r = compute_symmetry_scores(pil)
        rhythm = compute_rhythm_fft(pil)
        fractal = box_count_fractal_dimension(pil)

        # Heatmap
        with st.spinner("Computing heatmap..."):
            heat = build_saliency_heatmap(pil)

        # CLIP aesthetic
        with st.spinner("Computing aesthetic proxy..."):
            clip_score = clip_aesthetic_score(pil)

        # Norms + chart
        norms = normalize_features(sym_v, sym_r, ratio, w2w, rhythm, fractal)
        viz_img = build_aesthetic_viz(norms, final=clip_score, clip=clip_score)

        # Story
        metrics_dict = {
            "symmetry_vertical": round(sym_v, 3),
            "symmetry_rotational": round(sym_r, 3),
            "facade_ratio_H_W": round(ratio, 3),
            "window_to_wall_ratio": round(w2w, 3),
            "rhythm_fft_peak": round(rhythm, 3),
            "fractal_dimension": round(fractal, 3),
            "aesthetic_proxy": round(clip_score, 2)
        }
        
        # Save current upload for CLIP retrieval
        tmp_image_path = str(Path("outputs") / f"_tmp_{Path(up.name).stem}.png")
        pil.save(tmp_image_path)

        with st.spinner("Writing the story..."):
            story = write_story(
                metrics_dict,
                lang=LANG,
                image_path=tmp_image_path,
            )
        story = sanitize_story_text(story)

        # try to retrieve the same card your writer used
        matched_card = None
        try:
            c, s = retrieve(tmp_image_path, "data/index_clip.npz", device)
            if c and s >= 0.22:
                matched_card = c
        except Exception:
            pass

        st.markdown("### Design Narrative / 設計敘事")
        st.write(story)
        
        #----question anser ui
        # ---------- Ask (LLM guide using `info`) ----------
        st.markdown("### Ask / 问")
        qa_key = f"qa-{Path(up.name).stem}"
        user_q = st.text_input(_lang_text("Ask a question about this building",
                                          "请就此建筑提问",LANG),
                               key=qa_key)

        if user_q:
            info_text = (matched_card or {}).get("info", "")
            ans = guide_answer_from_info(user_q, info_text, LANG)
            if ans == "NOTFOUND":
                st.warning(_lang_text(
                    "Sorry, I can’t find that in the current building information.",
                    "抱歉，我在当前建筑信息中找不到这个答案。",LANG
                ))
            else:
                st.success(ans)

        #---interiour designs 
        if matched_card and matched_card.get("interiors"):
            show_interior_gallery(matched_card)


        # with st.spinner("Writing the story..."):
        #     story = write_story(metrics_dict)

        # --- UI render ---
        show_img(col1, pil, "Original")
        show_img(col2, overlay, "Semantic overlay (heuristic)")
        show_img(col3, heat, "Explainability heatmap")

        # st.markdown("### Design Narrative / 設計敘事")
        # st.write(story)

        from collections import Counter
        label_counts = Counter([d["label"] for d in dets])
        #st.caption(f"Detections (heuristic): {dict(label_counts)}  — total={len(dets)}")

        st.markdown("### Aesthetic Visualization / 美學視覺化")
        try:
            st.image(viz_img, caption="Feature profile and score makeup", width="stretch")
        except TypeError:
            st.image(viz_img, caption="Feature profile and score makeup", use_column_width=True)

        # Ranking
        rank, N, perc = update_and_rank(clip_score, Path(up.name).stem, HIST_PATH)
        st.info(f"Comparative ranking / 對比排名: {rank} / {N}  (~{perc:.1f}th percentile)")

        # PDF
        pdf_bytes = make_report(
            pil, overlay, heat, viz_img,
            metrics=metrics_dict,
            ranking=(rank, N, perc),
            story_text=story,
            lang=LANG,
            interiors=(matched_card.get("interiors") if matched_card else None)
            )
        if not isinstance(pdf_bytes, (bytes, bytearray)):
            st.error("PDF generation failed. No binary data returned.")            
        else:
            st.download_button(
                label="Download report PDF / 下載報告 PDF",
                data=pdf_bytes,
                file_name=Path(up.name).stem + "_report.pdf",
                mime="application/pdf",
                key=f"dl-{Path(up.name).stem}"
            )

            

   
        # st.download_button(
        #     label="Download report PDF / 下載報告 PDF",
        #     data=pdf_bytes,
        #     file_name=Path(up.name).stem + "_report.pdf",
        #     mime="application/pdf",
        #     key="dl-{}".format(Path(up.name).stem)
        # )
