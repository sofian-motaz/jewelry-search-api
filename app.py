"""
app.py

Production FastAPI application. Same routes, same request/response
schema as the Colab version. All AI logic is imported unchanged from
engine.py — nothing here re-implements or alters search behavior.

Run with:
    uvicorn app:app --host 0.0.0.0 --port $PORT
"""

import os
from pathlib import Path
from typing import List, Optional

from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from PIL import Image
import io
import json

from engine import (
    DB_PATH, FAISS_PATH, METADATA_PATH, STATISTICS_PATH,
    MODEL_NAME, DEVICE, FAISS_INDEX, _config,
    get_connection, db_health_check, make_thumbnail_base64, search_similar,
)

app = FastAPI(title='Jewelry AI Search API', version='1.0.0')

app.add_middleware(
    CORSMiddleware,
    allow_origins=os.environ.get('CORS_ALLOW_ORIGINS', '*').split(','),
    allow_credentials=True,
    allow_methods=['*'],
    allow_headers=['*'],
)

# ---------------------------------------------------------------------------
# Pydantic response models — identical to the Colab version
# ---------------------------------------------------------------------------
class RootResponse(BaseModel):
    name: str
    status: str

class HealthResponse(BaseModel):
    status: str
    model: str
    gpu: bool
    database: str
    faiss: str

class StatsResponse(BaseModel):
    indexed_images: int
    embedding_dimension: int
    database_size_mb: float
    categories: List[str]
    metals: List[str]
    last_update: Optional[str]

class SearchResultItem(BaseModel):
    score: float
    category: str
    metal: str
    filename: str
    path: str
    thumbnail: str

class SearchResponse(BaseModel):
    query_time: float
    results: List[SearchResultItem]

# ---------------------------------------------------------------------------
# Endpoints — identical names, behavior, and response shape
# ---------------------------------------------------------------------------
@app.get('/', response_model=RootResponse)
def root():
    return {'name': 'Jewelry AI Search API', 'status': 'running'}

@app.get('/health', response_model=HealthResponse)
def health():
    return {
        'status': 'online',
        'model': MODEL_NAME,
        'gpu': DEVICE == 'cuda',
        'database': 'connected' if db_health_check() else 'unavailable',
        'faiss': 'loaded' if FAISS_INDEX is not None and FAISS_INDEX.ntotal > 0 else 'unavailable',
    }

@app.get('/stats', response_model=StatsResponse)
def stats():
    if not DB_PATH.exists():
        raise HTTPException(status_code=503, detail='Database not found on disk.')
    try:
        conn = get_connection()
        total = conn.execute('SELECT COUNT(*) FROM images').fetchone()[0]
        categories = [r[0] for r in conn.execute('SELECT DISTINCT category FROM images ORDER BY category').fetchall()]
        metals = [r[0] for r in conn.execute('SELECT DISTINCT metal FROM images ORDER BY metal').fetchall()]
        conn.close()
    except Exception as e:
        raise HTTPException(status_code=503, detail=f'Database read failed: {e}')

    db_size_mb = round(os.path.getsize(DB_PATH) / (1024 * 1024), 2)

    last_update = None
    if STATISTICS_PATH.exists():
        with open(STATISTICS_PATH, 'r') as f:
            last_update = json.load(f).get('generated')

    return {
        'indexed_images': total,
        'embedding_dimension': int(_config.get('embedding_dim', 0)),
        'database_size_mb': db_size_mb,
        'categories': categories,
        'metals': metals,
        'last_update': last_update,
    }

@app.get('/categories')
def categories():
    if not DB_PATH.exists():
        raise HTTPException(status_code=503, detail='Database not found on disk.')
    conn = get_connection()
    rows = [r[0] for r in conn.execute('SELECT DISTINCT category FROM images ORDER BY category').fetchall()]
    conn.close()
    return {'categories': rows}

@app.get('/metals')
def metals():
    if not DB_PATH.exists():
        raise HTTPException(status_code=503, detail='Database not found on disk.')
    conn = get_connection()
    rows = [r[0] for r in conn.execute('SELECT DISTINCT metal FROM images ORDER BY metal').fetchall()]
    conn.close()
    return {'metals': rows}

@app.post('/search', response_model=SearchResponse)
async def search(
    image: UploadFile = File(...),
    top_k: int = Form(20),
    threshold: float = Form(0.0),
):
    if FAISS_INDEX is None or FAISS_INDEX.ntotal == 0:
        raise HTTPException(status_code=503, detail='FAISS index not loaded.')
    if not db_health_check():
        raise HTTPException(status_code=503, detail='Database not available.')

    if image is None:
        raise HTTPException(status_code=400, detail='No image provided.')

    try:
        raw_bytes = await image.read()
        pil_image = Image.open(io.BytesIO(raw_bytes))
        pil_image.load()
    except Exception:
        raise HTTPException(status_code=400, detail='Invalid or corrupted image file.')

    try:
        results, query_time = search_similar(pil_image, top_k=top_k, use_reranking=True)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f'Search failed: {e}')

    formatted = []
    for r in results:
        score = r.get('final_score', r['score'])
        if score < threshold:
            continue
        formatted.append({
            'score': round(float(score), 4),
            'category': r.get('category', 'Unknown'),
            'metal': r.get('metal', 'Unknown'),
            'filename': Path(r['path']).name,
            'path': r['path'],
            'thumbnail': make_thumbnail_base64(r['path']),
        })

    return {'query_time': round(query_time, 3), 'results': formatted}
