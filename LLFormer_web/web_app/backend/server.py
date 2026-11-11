"""
server.py — LLFormer Low-Light Image Enhancement API Server
===========================================================
- 4 chế độ tăng sáng trực quan cho người dùng phổ thông
- Không hiển thị các thuật ngữ chuyên sâu (PSNR, HSV, Lab, Joint, Loss...)
"""

import io
import os
import sys
import time
import base64
import logging
import asyncio
from pathlib import Path
from typing import Dict, Optional
from contextlib import asynccontextmanager

import numpy as np
import torch
from PIL import Image
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware

torch.set_num_threads(os.cpu_count() or 4)

BASE_DIR = Path(__file__).parent.parent          # web_app/
FRONTEND_DIR = BASE_DIR / "frontend"
RESULTS_DIR = BASE_DIR.parent / "LLFormer_Colab" / "results"

sys.path.insert(0, str(Path(__file__).parent))
from inference import (
    load_model_from_checkpoint,
    run_enhance,
    assess_image_quality,
    compute_enhancement_metrics,
    CONFIG_SPECS,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("LLFormerServer")

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

CONFIG_REGISTRY: Dict[str, dict] = {}
MODEL_CACHE: Dict[str, object] = {}
ACTIVE_CONFIG: str = "B"
LOAD_LOCK = asyncio.Lock()


def build_registry():
    global CONFIG_REGISTRY
    CONFIG_REGISTRY = {}
    for cfg_id, spec in CONFIG_SPECS.items():
        ckpt_path = RESULTS_DIR / spec["folder"] / "checkpoint_latest.pth"
        entry = spec.copy()
        if ckpt_path.exists():
            entry["checkpoint_path"] = str(ckpt_path)
            entry["loaded"] = False
            CONFIG_REGISTRY[cfg_id] = entry
        else:
            logger.warning(f"Checkpoint not found: {ckpt_path}")


def _load_and_cache_config(cfg_id: str):
    if cfg_id in MODEL_CACHE:
        return MODEL_CACHE[cfg_id]

    info = CONFIG_REGISTRY[cfg_id]
    logger.info(f"Loading {info['name']} on {DEVICE}...")
    t0 = time.time()

    model, _ = load_model_from_checkpoint(info["checkpoint_path"], DEVICE)
    elapsed = time.time() - t0

    MODEL_CACHE[cfg_id] = model
    CONFIG_REGISTRY[cfg_id]["loaded"] = True
    CONFIG_REGISTRY[cfg_id]["load_time_s"] = round(elapsed, 2)
    logger.info(f"  → Loaded in {elapsed:.2f}s")
    return model


@asynccontextmanager
async def lifespan(app: FastAPI):
    global ACTIVE_CONFIG
    logger.info("Initializing LLFormer Server...")
    build_registry()

    ACTIVE_CONFIG = "B" if "B" in CONFIG_REGISTRY else next(iter(CONFIG_REGISTRY.keys()))
    _load_and_cache_config(ACTIVE_CONFIG)
    logger.info(f"Ready. Default active: {ACTIVE_CONFIG} on {DEVICE}")
    yield
    logger.info("Server shutting down.")


app = FastAPI(title="LLFormer Enhancement API", version="3.2.0", lifespan=lifespan)
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])


@app.get("/api/health")
async def health():
    return {
        "status": "online",
        "device": str(DEVICE),
        "cuda": torch.cuda.is_available(),
        "active_config": ACTIVE_CONFIG,
    }


@app.get("/api/configs")
async def get_configs():
    result = []
    for cfg_id, info in CONFIG_REGISTRY.items():
        result.append({
            "id": cfg_id,
            "name": info["name"],
            "title": info["title"],
            "tag": info["tag"],
            "desc": info["desc"],
            "feature": info["feature"],
            "color": info["color"],
            "is_active": cfg_id == ACTIVE_CONFIG,
            "loaded": info["loaded"],
        })
    return {"configs": result, "active": ACTIVE_CONFIG}


@app.post("/api/switch/{cfg_id}")
async def switch_config(cfg_id: str):
    global ACTIVE_CONFIG
    cfg_id = cfg_id.upper()
    if cfg_id not in CONFIG_REGISTRY:
        raise HTTPException(404, f"Cấu hình '{cfg_id}' không hợp lệ. Chọn: {list(CONFIG_REGISTRY.keys())}")

    if cfg_id == ACTIVE_CONFIG:
        return {"status": "ok", "active": ACTIVE_CONFIG, "name": CONFIG_REGISTRY[cfg_id]["name"]}

    async with LOAD_LOCK:
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(None, _load_and_cache_config, cfg_id)
        ACTIVE_CONFIG = cfg_id

    return {
        "status": "ok",
        "active": ACTIVE_CONFIG,
        "name": CONFIG_REGISTRY[cfg_id]["name"],
        "title": CONFIG_REGISTRY[cfg_id]["title"],
    }


@app.post("/api/enhance")
async def enhance_image(file: UploadFile = File(...)):
    model = MODEL_CACHE.get(ACTIVE_CONFIG)
    if model is None:
        model = _load_and_cache_config(ACTIVE_CONFIG)

    allowed = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tiff", ".tif"}
    ext = Path(file.filename or "photo.jpg").suffix.lower()
    if ext not in allowed:
        raise HTTPException(400, f"Định dạng file không hỗ trợ: {ext}")

    content = await file.read()
    if len(content) > 25 * 1024 * 1024:
        raise HTTPException(413, "Dung lượng ảnh tối đa 25MB")

    try:
        input_img = Image.open(io.BytesIO(content)).convert("RGB")
    except Exception as e:
        raise HTTPException(400, f"Không thể giải mã file ảnh: {e}")

    w, h = input_img.size

    t_start = time.perf_counter()
    quality = assess_image_quality(input_img)

    loop = asyncio.get_event_loop()
    enhanced_img, infer_ms = await loop.run_in_executor(
        None, run_enhance, model, input_img, DEVICE
    )
    total_ms = round((time.perf_counter() - t_start) * 1000.0, 1)

    enh_metrics = compute_enhancement_metrics(input_img, enhanced_img)

    def _to_b64(im: Image.Image) -> str:
        buf = io.BytesIO()
        im.save(buf, format="PNG", optimize=False)
        return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()

    input_b64 = _to_b64(input_img)
    output_b64 = _to_b64(enhanced_img)

    cfg_info = CONFIG_REGISTRY.get(ACTIVE_CONFIG, {})

    sec = infer_ms / 1000.0 if infer_ms > 0 else 1.0
    px_total = w * h
    px_per_s = round(px_total / sec, 1) if sec > 0 else 0
    mpx_per_s = round((px_total / 1e6) / sec, 4) if sec > 0 else 0

    return JSONResponse({
        "success": True,
        "filename": file.filename,
        "image_size": f"{w} × {h}",
        "megapixels": round(w * h / 1e6, 2),
        "input_image": input_b64,
        "output_image": output_b64,
        "timing": {
            "inference_ms": infer_ms,
            "total_ms": total_ms,
            "device": str(DEVICE).upper(),
            "pixels_per_s": px_per_s,
            "throughput_mpx_s": mpx_per_s,
        },
        "input_quality": quality,
        "enhancement": enh_metrics,
        "config": {
            "id": ACTIVE_CONFIG,
            "name": cfg_info.get("name", "Chế độ Sống động"),
            "title": cfg_info.get("title", ""),
            "feature": cfg_info.get("feature", "Màu sắc tối ưu"),
            "color": cfg_info.get("color", "#10b981"),
        },
    })


@app.get("/", response_class=HTMLResponse)
async def serve_index():
    index_file = FRONTEND_DIR / "index.html"
    return HTMLResponse(content=index_file.read_text(encoding="utf-8"))


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=False)
