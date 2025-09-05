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
st.set_page_config(page_title="AI Analysis of Historic Architecture", layout="wide")
st.title("AI Analysis of Historic Architecture / 歷史建築的人工智慧分析")
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

def build_bad_words_ids(tokenizer, words):
    ids = []
    for w in words:
        toks = tokenizer.encode(w, add_special_tokens=False)
        if toks:
            ids.append(toks)
    return ids


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

    return device, t5_tok, t5, clip_proc, clip, zh_tok, zh_mt



try:
    device, t5_tok, t5_model, clip_proc, clip_model, zh_tok, zh_model = load_llm_and_clip()
    #st.caption(f"Device: {device} | LLM: {LLM_MODEL_ID} | CLIP: {CLIP_MODEL_ID}")
except Exception as e:
    st.error(f"Model loading failed: {e}")
    st.stop()
BAD_WORD_IDS_EN = build_bad_words_ids(t5_tok, BAD_WORDS_EN)
BAD_WORD_IDS_ZH = build_bad_words_ids(t5_tok, BAD_WORDS_ZH)

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
        col_like.image(img, caption=caption, use_container_width=True)
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

def _clean_generated(text: str) -> str:
    # remove tags, instructions, separators
    text = re.sub(r"\[[^\]]+\]", " ", text)                  # [P1] etc.
    text = re.sub(r"(?i)\b(hints?:.*|now write.*)\b", " ", text)
    text = re.sub(r"[|_—\-]{2,}", " ", text)                 # junk separators
    text = re.sub(r"\s+", " ", text).strip()
    return text

def _join_paragraphs(paras: list[str]) -> str:
    return "\n\n".join(p.strip() for p in paras).strip()

def _group_into_three_paragraphs(text_en: str,
                                 min_per: int = 4,
                                 max_per: int = 6) -> list[str]:
    txt = _clean_generated(text_en)
    sents = _sent_tokenize_en(txt)

    # ensure we have enough sentences
    filler_cycle = [
        "The rhythm stays even and welcoming.",
        "Light softens edges and lifts the surface.",
        "Small crafted touches reward a second look.",
        "The balance feels calm and sure."
    ]
    k = 0
    while len(sents) < 12:
        sents.append(filler_cycle[k % len(filler_cycle)])
        k += 1

    # allocate 3 blocks within [min_per, max_per]
    n1 = min(max_per, max(min_per, len(sents) // 3))
    n2 = min(max_per, max(min_per, (len(sents) - n1) // 2))
    n3 = max(min_per, min(max_per, len(sents) - n1 - n2))

    # if we still have extras, spill into the last paragraph up to max_per
    extra = len(sents) - (n1 + n2 + n3)
    n3 = min(max_per, n3 + max(0, extra))

    p1 = " ".join(sents[:n1])
    p2 = " ".join(sents[n1:n1+n2])
    p3 = " ".join(sents[n1+n2:n1+n2+n3])

    # last guardrails
    if not p3:
        p3 = "The scene settles with easy balance. Light and shade trade places through the day. Small crafted touches reward a second look. It feels made for people, not for show."
    return [p1, p2, p3]

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
@torch.inference_mode()
def write_story(metrics: dict, lang: str = "en") -> str:
    # Build human hints from metrics
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
    if w2w >= 0.45: hints.append("glassy frontage")
    elif w2w <= 0.10: hints.append("solid masonry presence")
    else: hints.append("comfortable window pattern")

    # prompt = (
    #     "You are an architecture narrator.\n"
    #     "Write EXACTLY three paragraphs, each 4 to 6 short sentences.\n"
    #     "Keep language simple. No style labels. No numbers. No bullet points.\n"
    #     "Focus on balance, symmetry, rhythm, light, and crafted details.\n"
    #     "Describe what the eye notices first, then what rewards a closer look.\n"
    #     f"Hints: {', '.join(hints)}\n"
    #     "Return only prose paragraphs separated by a blank line."
    # )
    prompt = (
    "Write analysis for the article title 'AI Analysis of Historic Architecture'.\n"
    "Produce EXACTLY three paragraphs. Each paragraph has 4–6 short, complete sentences.\n"
    "Plain English. Present tense. Active voice. Only commas and periods.\n"
    "Do NOT mention AI, titles, prompts, paragraphs, sentences, or your process.\n"
    "Do NOT use labels, numbers, lists, or single-word sentences.\n"
    "Begin with what the eye notices first from a distance.\n"
    "Then explain balance, symmetry, proportion, and rhythm using concrete parts such as arches, bays, columns, cornices, joints, and openings.\n"
    "Finish with crafted details, materials, light and shadow, and signs of wear or repair at human scale.\n"
    f"Integrate these hints smoothly without listing them: {', '.join(hints)}\n"
    "Avoid repeating the same word at the start of adjacent sentences.\n"
    "Return only the three paragraphs separated by one blank line."
    )


    enc = t5_tok(prompt, return_tensors="pt", truncation=True, max_length=1024).to(device)
    gen_kwargs = dict(
        max_new_tokens=320,
        min_new_tokens=220,           # encourage 3 paras
        do_sample=True,
        temperature=0.7,
        top_p=0.9,
        no_repeat_ngram_size=3,
        repetition_penalty=1.12,
    )
    # (optional) bad words list if you defined BAD_WORD_IDS_EN elsewhere
    bad_ids = globals().get("BAD_WORD_IDS_EN")
    if bad_ids is not None:
        gen_kwargs["bad_words_ids"] = bad_ids

    out = t5_model.generate(**enc, **gen_kwargs)
    text_en = t5_tok.decode(out[0], skip_special_tokens=True).strip()

    # enforce 3 paragraphs
    paras_en = _group_into_three_paragraphs(text_en)

    if lang == "zh":
        paras_zh = translate_en_to_zh(paras_en)
        return _join_paragraphs(paras_zh)
    return _join_paragraphs(paras_en)

# @torch.inference_mode()
# def write_story(metrics: dict, language: str = "en") -> str:
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

    
#     example_hints = "balanced overall, strong symmetry, clear repeating bays, comfortable window pattern"
#     example = (
#         "You are an architecture narrator for everyday readers.\n"
#         "Write in **English**, EXACTLY three paragraphs, each **4–6 sentences**. Use present tense, simple language.\n"
#         "Do not use style labels. No numbers or dates.\n"
#         "Focus on what the eye notices: balance, symmetry, rhythm, light, craft, human scale.\n"
#         "Use the tag format exactly as shown.\n"
#         f"Hints: {example_hints}\n"
#         "[P1] The facade meets the street with calm order. Openings line up from left to right. "
#         "A clear middle bay frames the entry. Sun picks out the frames and sills. "
#         "Materials look steady, well placed, and inviting. [/P1]\n"
#         "[P2] Up close, windows repeat with even spacing. Vertical strips read like quiet supports. "
#         "Shadow lines at the reveals add depth. A simple band marks the roof edge. "
#         "Corners stay crisp and patient. [/P2]\n"
#         "[P3] Craft shows in the small touches. Joints sit tight and true. "
#         "Handles and railings meet the hand without fuss. Evening light warms the surface and softens the edges. "
#         "The building feels welcoming and sure. [/P3]\n"
#         "\n"
#         "Now write the next description using the same format and tags only.\n"
#         f"Hints: {', '.join(hints)}\n"
#         "[P1]"
#    )
#     bad_ids = BAD_WORD_IDS_EN

#     inputs = t5_tok(example, return_tensors="pt").to(device)
#     out = t5_model.generate(
#         **inputs,
#         max_new_tokens=420,
#         min_new_tokens=260,
#         do_sample=True,
#         temperature=0.85 if language=="zh" else 0.8,
#         top_p=0.92,
#         top_k=60,
#         repetition_penalty=1.12,
#         no_repeat_ngram_size=4,
#         bad_words_ids=bad_ids
#     )
#     raw = t5_tok.decode(out[0], skip_special_tokens=True).strip()

#     # --- sanitize to exactly three paragraphs ---
#     def clean(text: str) -> str:
#         text = re.sub(r"\[[^\]]+\]", " ", text)   # remove any leftover [tags]
#         text = re.sub(r"[|_—\-]{2,}", " ", text)  # remove junk separators
#         text = re.sub(r"(?i)\b(now write.*|write exactly.*|hints:.*)\b", " ", text)
#         text = re.sub(r"\s+", " ", text).strip()
#         return text

#     txt = clean(raw)
#     sents = re.split(r'(?<=[。！？.!?])\s+', txt)
#     sents = [s.strip() for s in sents if len(s.strip().split()) > 2]
#     while len(sents) < 12:  # ensure content
#         txt += " The rhythm stays even and welcoming." if language=="en" else " 节奏平稳而友好。"
#         sents = re.split(r'(?<=[。！？.!?])\s+', clean(txt))

#     per = 5
#     p1 = " ".join(sents[0:per])
#     p2 = " ".join(sents[per:2*per])
#     p3_src = " ".join(sents[2*per:3*per])
#     if not p3_src:
#         p3_src = ("The scene settles with easy balance. Light and shade trade places through the day. "
#                   "Small crafted touches reward a second look. It feels made for people, not for show."
#                   if language=="en" else
#                   "画面以从容的均衡收束。光与影在一天里交替。小小的手工细节值得回望。它是为人而做，而不是为炫耀。")
#     return "\n\n".join([p1, p2, p3_src]).strip()

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
def make_report(orig_img, overlay_img, heat_img, viz_img, metrics, ranking, story_text, lang: str = "en"):
    import io, re
    from reportlab.lib.pagesizes import A4
    from reportlab.pdfgen import canvas
    from reportlab.lib.utils import ImageReader
    from reportlab.pdfbase import pdfmetrics
    from reportlab.pdfbase.cidfonts import UnicodeCIDFont

    # --- ensure CJK font is available (no local files required) ---
    try:
        pdfmetrics.getFont("STSong-Light")
    except KeyError:
        pdfmetrics.registerFont(UnicodeCIDFont("STSong-Light"))

    buf = io.BytesIO()
    c = canvas.Canvas(buf, pagesize=A4)
    W, H = A4
    TOP, BOT, LM, RM = 40, 60, 40, 40
    y = H - TOP
    COL_W = (W - LM - RM - 20) / 3.0   # three columns + 10pt gutters

    # ---------- header (static title) ----------
    c.setFont("STSong-Light", 16)
    c.drawString(LM, y, "AI Analysis of Historic Architecture / 历史建筑的人工智能分析")
    y -= 24
    c.setFont("STSong-Light", 10)
    import time as _t
    c.drawString(LM, y, _t.strftime("Generated on / 生成于 %Y-%m-%d %H:%M:%S"))
    y -= 12

    # ---------- helpers ----------
    def draw_img_bottom(pil, x, y_bottom, maxw_pt):
        if pil is None:
            return 0.0
        w_px, h_px = pil.size
        if w_px <= 0:
            return 0.0
        target_w = float(maxw_pt)
        target_h = target_w * (h_px / float(w_px))
        c.drawImage(ImageReader(pil), x, y_bottom, width=target_w, height=target_h,
                    preserveAspectRatio=True, mask='auto')
        return target_h

    def draw_img_top(pil, x, y_top, maxw_pt):
        if pil is None:
            return 0.0
        w_px, h_px = pil.size
        if w_px <= 0:
            return 0.0
        target_w = float(maxw_pt)
        target_h = target_w * (h_px / float(w_px))
        c.drawImage(ImageReader(pil), x, y_top - target_h, width=target_w, height=target_h,
                    preserveAspectRatio=True, mask='auto')
        return target_h

    # robust wrap: English by words, Chinese by characters
    def wrap_lines(text, max_width_pt, font_name, font_size, is_cjk=False):
        sw = pdfmetrics.stringWidth
        lines = []
        if is_cjk:
            cur = ""
            for ch in text:
                if ch == "\n":
                    lines.append(cur); cur = ""; continue
                w = sw(cur + ch, font_name, font_size)
                if w <= max_width_pt:
                    cur += ch
                else:
                    if cur:
                        lines.append(cur)
                    cur = ch
            if cur:
                lines.append(cur)
            return lines
        else:
            words = re.split(r"\s+", text.strip())
            cur = ""
            for w in words:
                add = (w if not cur else " " + w)
                if sw(cur + add, font_name, font_size) <= max_width_pt:
                    cur += add
                else:
                    if cur:
                        lines.append(cur)
                    cur = w
            if cur:
                lines.append(cur)
            return lines

    def sanitize_story_text(t: str) -> str:
        t = re.sub(r"\[[^\]]+\]", " ", t)               # remove stray [tags]
        t = re.sub(r"(Now write.*|Hints:.*)$", " ", t, flags=re.I | re.M)
        t = re.sub(r"[|_—\-]{2,}", " ", t)              # junk separators
        t = re.sub(r"[ \t]+", " ", t)
        t = re.sub(r"(\s*\n\s*){3,}", "\n\n", t)
        return t.strip()

    # ---------- Row 1: three images ----------
    row1_bottom = y - 200  # anchor the bottom of the image row
    h1 = draw_img_bottom(orig_img, LM,                     row1_bottom, COL_W)
    h2 = draw_img_bottom(overlay_img, LM + COL_W + 10,    row1_bottom, COL_W)
    h3 = draw_img_bottom(heat_img,    LM + 2*(COL_W + 10),row1_bottom, COL_W)
    tallest = max(h1, h2, h3)

    # move cursor safely below images
    y = row1_bottom - tallest - 18
    if y < BOT + 40:
        c.showPage()
        y = H - TOP

    # ---------- Story block ----------
    c.setFont("STSong-Light", 12)
    c.drawString(LM, y, "Design Narrative / 设计叙事")   
    y -= 16

    body_font = "STSong-Light" if lang == "zh" else "Helvetica"
    font_size = 11 if lang == "zh" else 10
    leading   = 14 if lang == "zh" else 12
    c.setFont(body_font, font_size)
    col_width = W - LM - RM - 8

    clean_story = sanitize_story_text(story_text)
    paragraphs = [p.strip() for p in clean_story.split("\n\n") if p.strip()]

    for para in paragraphs:
        lines = wrap_lines(para, col_width, body_font, font_size, is_cjk=(lang == "zh"))
        for ln in lines:
            if y < BOT:
                c.showPage()
                y = H - TOP
                c.setFont("Helvetica-Bold", 12)
                c.drawString(LM, y, "Story (cont.)")
                y -= 16
                c.setFont(body_font, font_size)
            c.drawString(LM + 8, y, ln)
            y -= leading
        y -= leading // 2  # small gap between paragraphs

    # ---------- Page 2: Visualization + Ranking ----------
    c.showPage()
    y = H - TOP

    c.setFont("STSong-Light", 12)
    c.drawString(LM, y, "Aesthetic Visualization / 美学可视化")
    y -= 12

    y_top_for_viz = y - 8
    _ = draw_img_top(viz_img, LM, y_top_for_viz, W - LM - RM)

    if viz_img is not None:
        w_px, h_px = viz_img.size
        viz_h = (W - LM - RM) * (h_px / float(w_px))
    else:
        viz_h = 0.0
    y = (y_top_for_viz - viz_h) - 18

    if y < BOT + 40:
        c.showPage(); y = H - TOP

    c.setFont("STSong-Light", 12)
    c.drawString(LM, y, "Ranking / 排行")
    y -= 14
    c.setFont("Helvetica", 10)
    c.drawString(
        LM + 8, y,
        "Comparative ranking: {} / {}  (~{:.1f}th percentile)".format(
            ranking[0], ranking[1], ranking[2]
        )
    )

    # ---------- finish ----------
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
        with st.spinner("Writing the story..."):

            story = write_story(metrics_dict, lang=LANG)
            story = sanitize_story_text(story)
        # with st.spinner("Writing the story..."):
        #     story = write_story(metrics_dict)

        # --- UI render ---
        show_img(col1, pil, "Original")
        show_img(col2, overlay, "Semantic overlay (heuristic)")
        show_img(col3, heat, "Explainability heatmap")

        st.markdown("### Design Narrative / 設計敘事")
        st.write(story)

        from collections import Counter
        label_counts = Counter([d["label"] for d in dets])
        #st.caption(f"Detections (heuristic): {dict(label_counts)}  — total={len(dets)}")

        st.markdown("### Aesthetic Visualization / 美學視覺化")
        try:
            st.image(viz_img, caption="Feature profile and score makeup", use_container_width=True)
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
            lang=LANG 
        )
   
        st.download_button(
            label="Download report PDF / 下載報告 PDF",
            data=pdf_bytes,
            file_name=Path(up.name).stem + "_report.pdf",
            mime="application/pdf",
            key="dl-{}".format(Path(up.name).stem)
        )
