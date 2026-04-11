#!/usr/bin/env python3
"""
api.py - Production FastAPI inference server

Install:
  pip install fastapi uvicorn python-multipart pillow

Run:
  uvicorn api:app --host 0.0.0.0 --port 8080 --workers 1

Test:
  curl -X POST http://localhost:8080/predict -F "file=@satellite.jpg" --output map.png

Docs:
  http://localhost:8080/docs
"""

import io
import os
import time
import logging
from pathlib import Path
from contextlib import asynccontextmanager

import numpy as np
import tensorflow as tf
from PIL import Image
from fastapi import FastAPI, File, UploadFile, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse, JSONResponse

from src.inference import load_generator, preprocess_pil, postprocess, postprocess_with_comparison


# Logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


# Config
MODEL_DIR = os.getenv("MODEL_DIR", "saved_models/best_generator")
MAX_FILE_MB = int(os.getenv("MAX_FILE_MB", "10"))
MAX_BYTES = MAX_FILE_MB * 1024 * 1024
ALLOWED_TYPES = {"image/jpeg", "image/png", "image/jpg"}


# Model state
_generator = None


def get_generator():
    global _generator
    if _generator is None:
        raise RuntimeError("Model not loaded. Check startup logs.")
    return _generator


def _run_generator(generator, inp_batch: tf.Tensor) -> tf.Tensor:
    """Run inference for Keras models and tf.saved_model modules."""
    if isinstance(generator, tf.keras.Model):
        out = generator(inp_batch, training=False)
    else:
        try:
            out = generator(inp_batch, training=False)
        except TypeError:
            out = None

        if out is None and hasattr(generator, "signatures") and "serving_default" in generator.signatures:
            out = generator.signatures["serving_default"](inp_batch)

        if out is None:
            out = generator(inp_batch)

        if isinstance(out, dict):
            out = next(iter(out.values()))

    if not isinstance(out, tf.Tensor):
        out = tf.convert_to_tensor(out)
    return out


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load model once on startup."""
    global _generator
    model_path = Path(MODEL_DIR)
    if not model_path.exists():
        logger.error("Model path not found: %s", MODEL_DIR)
        raise FileNotFoundError(f"Model path not found: {MODEL_DIR}")

    logger.info("Loading generator from %s ...", MODEL_DIR)
    _generator = load_generator(MODEL_DIR)
    logger.info("Model loaded successfully.")
    yield
    logger.info("Server shutting down.")


# App
app = FastAPI(
    title="Satellite to Map API",
    description="Converts satellite imagery to map-style images using Pix2Pix GAN",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)


@app.middleware("http")
async def log_requests(request: Request, call_next):
    t0 = time.time()
    response = await call_next(request)
    elapsed_ms = (time.time() - t0) * 1000.0
    logger.info("%s %s -> %s (%.0fms)", request.method, request.url.path, response.status_code, elapsed_ms)
    return response


def _pil_to_input_batch(pil_image: Image.Image) -> tf.Tensor:
    arr = preprocess_pil(pil_image)
    return tf.convert_to_tensor(arr[np.newaxis], dtype=tf.float32)


def _image_to_buffer(img: Image.Image, fmt: str = "PNG") -> io.BytesIO:
    buf = io.BytesIO()
    img.save(buf, format=fmt)
    buf.seek(0)
    return buf


def _validate_upload(file: UploadFile, data: bytes):
    if file.content_type not in ALLOWED_TYPES:
        raise HTTPException(
            status_code=415,
            detail=f"Unsupported media type: {file.content_type}. Use JPEG or PNG.",
        )

    if len(data) > MAX_BYTES:
        raise HTTPException(
            status_code=413,
            detail=f"File too large ({len(data) // 1024}KB). Max is {MAX_FILE_MB}MB.",
        )


@app.get("/")
def root():
    return {"message": "Satellite to Map API. POST /predict with a satellite image."}


@app.get("/health")
def health():
    loaded = _generator is not None
    gpus = tf.config.list_physical_devices("GPU")
    return JSONResponse(
        status_code=200 if loaded else 503,
        content={
            "status": "ok" if loaded else "loading",
            "model": MODEL_DIR,
            "gpu_available": len(gpus) > 0,
            "gpus": len(gpus),
        },
    )


@app.get("/model/info")
def model_info():
    gen = get_generator()
    try:
        params = gen.count_params() if hasattr(gen, "count_params") else "N/A"
    except Exception:
        params = "N/A"

    return {
        "model_dir": MODEL_DIR,
        "input_shape": [256, 256, 3],
        "output_shape": [256, 256, 3],
        "parameters": params,
        "framework": f"TensorFlow {tf.__version__}",
    }


@app.post("/predict")
async def predict(
    file: UploadFile = File(..., description="Satellite image (JPEG or PNG, max 10MB)"),
):
    data = await file.read()
    _validate_upload(file, data)

    try:
        pil_img = Image.open(io.BytesIO(data)).convert("RGB")
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Could not decode image: {exc}") from exc

    try:
        inp_batch = _pil_to_input_batch(pil_img)
        out_batch = _run_generator(get_generator(), inp_batch)
        result = postprocess(out_batch)
    except Exception as exc:
        logger.exception("Inference failed")
        raise HTTPException(status_code=500, detail=f"Inference failed: {exc}") from exc

    return StreamingResponse(
        _image_to_buffer(result, "PNG"),
        media_type="image/png",
        headers={"Content-Disposition": "attachment; filename=generated_map.png"},
    )


@app.post("/predict/compare")
async def predict_compare(
    file: UploadFile = File(..., description="Satellite image (JPEG or PNG, max 10MB)"),
):
    data = await file.read()
    _validate_upload(file, data)

    try:
        pil_img = Image.open(io.BytesIO(data)).convert("RGB")
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Could not decode image: {exc}") from exc

    try:
        inp_batch = _pil_to_input_batch(pil_img)
        out_batch = _run_generator(get_generator(), inp_batch)
        comparison = postprocess_with_comparison(inp_batch, out_batch)
    except Exception as exc:
        logger.exception("Comparison inference failed")
        raise HTTPException(status_code=500, detail=f"Inference failed: {exc}") from exc

    return StreamingResponse(
        _image_to_buffer(comparison, "PNG"),
        media_type="image/png",
        headers={"Content-Disposition": "attachment; filename=comparison.png"},
    )


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("api:app", host="0.0.0.0", port=8080, reload=False, workers=1)
