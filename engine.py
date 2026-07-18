"""
engine.py

The AI engine — model loading, FAISS index, SQLite access, metadata,
query preprocessing, and search_similar(). This is the single source
of truth for all search logic. Both app.py (production/Render) and
colab_launcher.py (Google Colab) import from here unchanged.

NOTHING in this file was rewritten or optimized — it is the exact same
logic that was already working in the notebook, only moved out of
notebook cells into a plain importable module.
"""

import os
import io
import json
import time
import sqlite3
import base64
from pathlib import Path
from typing import List, Tuple, Dict, Optional

import numpy as np
from PIL import Image, ImageOps
import torch
import faiss
import cv2
from skimage.feature import local_binary_pattern
from transformers import AutoModel, AutoProcessor

# ---------------------------------------------------------------------------
# Data directory
# ---------------------------------------------------------------------------
# In Colab, colab_launcher.py sets DATA_DIR to the Google Drive project
# folder before this module is imported. On Render, DATA_DIR is set via
# an environment variable pointing wherever the pre-built database/
# FAISS/metadata files are made available to the deployed service.
# Defaulting to "./data" keeps local/Render runs simple: put
# embeddings.db, index.faiss, metadata.json, config.json, statistics.json
# in a "data" folder next to this file if DATA_DIR isn't set.
PROJECT_ROOT = Path(os.environ.get('DATA_DIR', './data'))

DB_PATH         = PROJECT_ROOT / 'embeddings.db'
FAISS_PATH      = PROJECT_ROOT / 'index.faiss'
METADATA_PATH   = PROJECT_ROOT / 'metadata.json'
CONFIG_PATH     = PROJECT_ROOT / 'config.json'
STATISTICS_PATH = PROJECT_ROOT / 'statistics.json'

assert DB_PATH.exists(), f'Database not found at {DB_PATH}. Set DATA_DIR correctly.'
assert FAISS_PATH.exists(), f'FAISS index not found at {FAISS_PATH}. Set DATA_DIR correctly.'
assert METADATA_PATH.exists(), f'metadata.json not found at {METADATA_PATH}. Set DATA_DIR correctly.'

print('DB       :', DB_PATH, '| exists:', DB_PATH.exists())
print('FAISS    :', FAISS_PATH, '| exists:', FAISS_PATH.exists())
print('Metadata :', METADATA_PATH, '| exists:', METADATA_PATH.exists())

# ---------------------------------------------------------------------------
# GPU
# ---------------------------------------------------------------------------
def detect_gpu():
    if not torch.cuda.is_available():
        return 'cpu', 0.0
    name = torch.cuda.get_device_name(0)
    total_mem_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)
    return name, total_mem_gb

GPU_NAME, GPU_MEM_GB = detect_gpu()
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
USE_AMP = DEVICE == 'cuda'
print(f'Device: {DEVICE} | GPU: {GPU_NAME} | Memory: {GPU_MEM_GB:.1f} GB')

# ---------------------------------------------------------------------------
# Model (must match the model used during indexing — read from config.json)
# ---------------------------------------------------------------------------
with open(CONFIG_PATH, 'r') as f:
    _config = json.load(f)
MODEL_NAME = _config['model_name']
TOPK_RERANK_POOL = _config.get('topk_rerank_pool', 50)

print(f'Loading {MODEL_NAME} (same model used to build the existing index) ...')
processor = AutoProcessor.from_pretrained(MODEL_NAME)
model = AutoModel.from_pretrained(
    MODEL_NAME,
    torch_dtype=torch.float16 if DEVICE == 'cuda' else torch.float32,
).to(DEVICE)
model.eval()
print('Model loaded.')

# ---------------------------------------------------------------------------
# DB
# ---------------------------------------------------------------------------
def get_connection() -> sqlite3.Connection:
    conn = sqlite3.connect(str(DB_PATH))
    conn.execute('PRAGMA journal_mode=WAL;')
    return conn

# ---------------------------------------------------------------------------
# FAISS (load existing index only, never create/train)
# ---------------------------------------------------------------------------
FAISS_INDEX = faiss.read_index(str(FAISS_PATH))
print('FAISS index loaded. Total vectors:', FAISS_INDEX.ntotal)

# ---------------------------------------------------------------------------
# Metadata (load existing file only)
# ---------------------------------------------------------------------------
with open(METADATA_PATH, 'r') as f:
    METADATA = json.load(f)
print('Metadata loaded. Total entries:', len(METADATA))

# ---------------------------------------------------------------------------
# Shared preprocessing / embedding helpers
# ---------------------------------------------------------------------------
def remove_white_borders(img: Image.Image, threshold: int = 245, pad: int = 4) -> Image.Image:
    gray = ImageOps.grayscale(img)
    arr = np.array(gray)
    mask = arr < threshold
    if not mask.any():
        return img
    rows = np.any(mask, axis=1)
    cols = np.any(mask, axis=0)
    top, bottom = np.where(rows)[0][[0, -1]]
    left, right = np.where(cols)[0][[0, -1]]
    top = max(0, top - pad)
    left = max(0, left - pad)
    bottom = min(arr.shape[0] - 1, bottom + pad)
    right = min(arr.shape[1] - 1, right + pad)
    return img.crop((left, top, right + 1, bottom + 1))

def load_and_preprocess_from_pil(img: Image.Image) -> Optional[Image.Image]:
    try:
        img = ImageOps.exif_transpose(img)
        if img.mode != 'RGB':
            img = img.convert('RGB')
        img = remove_white_borders(img)
        return img
    except Exception:
        return None

def load_and_preprocess(path: Path) -> Optional[Image.Image]:
    try:
        img = Image.open(path)
        img.load()
    except Exception:
        return None
    return load_and_preprocess_from_pil(img)

@torch.no_grad()
def embed_images(images: List[Image.Image]) -> np.ndarray:
    inputs = processor(images=images, return_tensors='pt').to(DEVICE)
    with torch.autocast(device_type='cuda', dtype=torch.float16, enabled=USE_AMP):
        _out = model.get_image_features(**inputs)
        feats = _out.pooler_output if hasattr(_out, 'pooler_output') else _out
    feats = feats.float()
    feats = feats / feats.norm(p=2, dim=-1, keepdim=True).clamp_min(1e-12)
    return feats.cpu().numpy().astype('float32')

def embedding_to_blob(vec: np.ndarray) -> bytes:
    return vec.astype('float16').tobytes()

def blob_to_embedding(blob: bytes) -> np.ndarray:
    return np.frombuffer(blob, dtype='float16').astype('float32')

def get_stored_embedding(fid: int) -> Optional[np.ndarray]:
    conn = get_connection()
    row = conn.execute('SELECT embedding FROM images WHERE faiss_id = ?', (fid,)).fetchone()
    conn.close()
    if row is None:
        return None
    return blob_to_embedding(row[0])

# ---------------------------------------------------------------------------
# Query preprocessing
# ---------------------------------------------------------------------------
def _gray_world_white_balance(img: Image.Image) -> Image.Image:
    arr = np.array(img).astype(np.float32)
    channel_means = arr.reshape(-1, 3).mean(axis=0)
    gray_mean = channel_means.mean()
    gains = gray_mean / (channel_means + 1e-6)
    gains = np.clip(gains, 0.7, 1.4)
    arr = np.clip(arr * gains, 0, 255).astype(np.uint8)
    return Image.fromarray(arr)

def preprocess_query_image(img: Image.Image) -> Image.Image:
    img = remove_white_borders(img)
    img = _gray_world_white_balance(img)
    img = ImageOps.autocontrast(img, cutoff=1)
    return img

def generate_query_variants(img: Image.Image) -> List[Image.Image]:
    w, h = img.size
    variants = [img]
    cx0, cy0 = int(w * 0.075), int(h * 0.075)
    cx1, cy1 = int(w * 0.925), int(h * 0.925)
    variants.append(img.crop((cx0, cy0, cx1, cy1)))
    zx0, zy0 = int(w * 0.15), int(h * 0.15)
    zx1, zy1 = int(w * 0.85), int(h * 0.85)
    variants.append(img.crop((zx0, zy0, zx1, zy1)))
    variants.append(ImageOps.autocontrast(img, cutoff=3))
    variants.append(ImageOps.equalize(img))
    variants.append(ImageOps.mirror(img))
    return variants

def multi_embedding_query(raw_img: Image.Image) -> np.ndarray:
    pre = preprocess_query_image(raw_img)
    variants = generate_query_variants(pre)
    embs = embed_images(variants)
    avg = embs.mean(axis=0)
    return avg / (np.linalg.norm(avg) + 1e-12)

# ---------------------------------------------------------------------------
# Reranking signals
# ---------------------------------------------------------------------------
def color_histogram(img: Image.Image, bins: int = 16) -> np.ndarray:
    hsv = np.array(img.convert('HSV'))
    hist, _ = np.histogram(hsv[..., 0], bins=bins, range=(0, 255), density=True)
    return hist.astype('float32')

def color_similarity(img_a: Image.Image, img_b: Image.Image) -> float:
    ha, hb = color_histogram(img_a), color_histogram(img_b)
    denom = np.linalg.norm(ha) * np.linalg.norm(hb) + 1e-6
    return float(np.dot(ha, hb) / denom)

_orb = cv2.ORB_create(nfeatures=500)
_bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

def _to_gray(img: Image.Image, size: int) -> np.ndarray:
    return cv2.cvtColor(np.array(img.resize((size, size))), cv2.COLOR_RGB2GRAY)

def orb_similarity(img_a: Image.Image, img_b: Image.Image) -> float:
    a, b = _to_gray(img_a, 256), _to_gray(img_b, 256)
    kp1, des1 = _orb.detectAndCompute(a, None)
    kp2, des2 = _orb.detectAndCompute(b, None)
    if des1 is None or des2 is None or len(kp1) == 0 or len(kp2) == 0:
        return 0.0
    matches = _bf.match(des1, des2)
    good = [m for m in matches if m.distance < 50]
    denom = max(1, min(len(kp1), len(kp2)))
    return float(min(1.0, len(good) / denom))

def edge_similarity(img_a: Image.Image, img_b: Image.Image, size: int = 128) -> float:
    a, b = _to_gray(img_a, size), _to_gray(img_b, size)
    ea = cv2.Canny(a, 50, 150).astype(np.float32) / 255.0
    eb = cv2.Canny(b, 50, 150).astype(np.float32) / 255.0
    num = np.sum(ea * eb)
    denom = np.sqrt(np.sum(ea ** 2) * np.sum(eb ** 2)) + 1e-6
    return float(num / denom)

def texture_similarity(img_a: Image.Image, img_b: Image.Image, size: int = 128) -> float:
    a, b = _to_gray(img_a, size), _to_gray(img_b, size)
    lbp_a = local_binary_pattern(a, P=8, R=1, method='uniform')
    lbp_b = local_binary_pattern(b, P=8, R=1, method='uniform')
    hist_a, _ = np.histogram(lbp_a, bins=10, range=(0, 10), density=True)
    hist_b, _ = np.histogram(lbp_b, bins=10, range=(0, 10), density=True)
    denom = np.linalg.norm(hist_a) * np.linalg.norm(hist_b) + 1e-6
    return float(np.dot(hist_a, hist_b) / denom)

def _largest_contour(gray: np.ndarray):
    _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
    return max(contours, key=cv2.contourArea)

def shape_similarity(img_a: Image.Image, img_b: Image.Image, size: int = 256) -> float:
    a, b = _to_gray(img_a, size), _to_gray(img_b, size)
    ca, cb = _largest_contour(a), _largest_contour(b)
    if ca is None or cb is None:
        return 0.0
    d = cv2.matchShapes(ca, cb, cv2.CONTOURS_MATCH_I1, 0.0)
    return float(1.0 / (1.0 + d))

RERANK_WEIGHTS = {
    'embedding': 0.45, 'orb': 0.15, 'edge': 0.10,
    'texture': 0.10, 'shape': 0.10, 'color': 0.10,
}

def rerank_candidates(query_variant_img: Image.Image, query_embedding: np.ndarray,
                       candidates: List[Dict]) -> List[Dict]:
    reranked = []
    for cand in candidates:
        cand_img = load_and_preprocess(Path(cand['path']))
        stored_emb = get_stored_embedding(cand['faiss_id'])
        if cand_img is None or stored_emb is None:
            reranked.append({**cand, 'final_score': cand['score']})
            continue
        embed_sim = float(np.dot(query_embedding, stored_emb))
        scores = {
            'embedding': embed_sim,
            'orb': orb_similarity(query_variant_img, cand_img),
            'edge': edge_similarity(query_variant_img, cand_img),
            'texture': texture_similarity(query_variant_img, cand_img),
            'shape': shape_similarity(query_variant_img, cand_img),
            'color': color_similarity(query_variant_img, cand_img),
        }
        final_score = sum(RERANK_WEIGHTS[k] * v for k, v in scores.items())
        reranked.append({**cand, 'final_score': final_score, 'score_breakdown': scores})
    reranked.sort(key=lambda r: r['final_score'], reverse=True)
    return reranked

# ---------------------------------------------------------------------------
# search_similar() — unchanged logic. Accepts either a path or a PIL image.
# ---------------------------------------------------------------------------
def search_similar(query_image, top_k: int = 20, use_reranking: bool = True,
                    ground_truth_path: Optional[str] = None) -> Tuple[List[Dict], float]:
    t0 = time.time()

    if isinstance(query_image, (str, Path)):
        raw_img = load_and_preprocess(Path(query_image))
    else:
        raw_img = load_and_preprocess_from_pil(query_image)

    if raw_img is None:
        raise ValueError('Could not read/preprocess query image.')

    query_pre = preprocess_query_image(raw_img)
    query_embedding = multi_embedding_query(raw_img)

    pool_k = max(top_k, TOPK_RERANK_POOL) if use_reranking else top_k
    scores, ids = FAISS_INDEX.search(query_embedding[None, :], pool_k)
    scores, ids = scores[0], ids[0]

    candidates = []
    for score, fid in zip(scores, ids):
        if fid == -1:
            continue
        meta = METADATA.get(str(fid))
        if meta is None:
            continue
        candidates.append({'faiss_id': int(fid), 'score': float(score), **meta})

    results_full = rerank_candidates(query_pre, query_embedding, candidates) if use_reranking else candidates
    results = results_full[:top_k]
    query_time = time.time() - t0
    return results, query_time

print('Engine fully loaded — model, FAISS, metadata, DB helper, search_similar() ready.')

def db_health_check() -> bool:
    try:
        conn = get_connection()
        conn.execute('SELECT 1 FROM images LIMIT 1')
        conn.close()
        return True
    except Exception:
        return False

def make_thumbnail_base64(path: str, size: int = 200) -> str:
    """Small base64-encoded JPEG data URI for a catalog image, so the
    frontend can render a preview without needing direct filesystem
    access."""
    try:
        img = Image.open(path).convert('RGB')
        img.thumbnail((size, size))
        buf = io.BytesIO()
        img.save(buf, format='JPEG', quality=80)
        b64 = base64.b64encode(buf.getvalue()).decode('utf-8')
        return f'data:image/jpeg;base64,{b64}'
    except Exception:
        return ''
