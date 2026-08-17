"""FastAPI serving layer for the prompt injection classifier."""

import asyncio
import logging
import time
from contextlib import asynccontextmanager
from pathlib import Path

import duckdb
import numpy as np
from dotenv import load_dotenv
from fastapi import FastAPI
from optimum.onnxruntime import ORTModelForSequenceClassification
from pydantic import BaseModel, field_validator
from scipy.special import softmax
from transformers import AutoTokenizer

load_dotenv()

logger = logging.getLogger(__name__)

MAX_BATCH_SIZE = 128
MODEL_PATH = "alvi42/prompt-injection-guard-v1"
ONNX_DIR = Path("checkpoints/v1-onnx-int8")
DB_PATH = "data/unified.duckdb"
MAX_LENGTH = 256
LABELS = ["benign", "injection"]


class ClassifyRequest(BaseModel):
    text: str


class ClassifyResponse(BaseModel):
    label: str
    confidence: float
    scores: dict[str, float]
    latency_ms: float


class BatchClassifyRequest(BaseModel):
    texts: list[str]

    @field_validator("texts")
    @classmethod
    def validate_batch_size(cls, v):
        if len(v) > MAX_BATCH_SIZE:
            raise ValueError(f"Batch size {len(v)} exceeds maximum {MAX_BATCH_SIZE}")
        if len(v) == 0:
            raise ValueError("Batch must contain at least one text")
        return v


class BatchClassifyResponse(BaseModel):
    results: list[ClassifyResponse]
    total_texts: int
    total_latency_ms: float


def init_logging_table():
    con = duckdb.connect(DB_PATH)
    con.execute("""
        CREATE TABLE IF NOT EXISTS request_log (
            id INTEGER,
            input_text TEXT,
            prediction TEXT,
            confidence FLOAT,
            latency_ms FLOAT,
            timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)
    con.close()


def log_request(input_text: str, prediction: str, confidence: float, latency_ms: float):
    try:
        con = duckdb.connect(DB_PATH)
        con.execute(
            """
            INSERT INTO request_log (input_text, prediction, confidence, latency_ms)
            VALUES (?, ?, ?, ?)
        """,
            [input_text, prediction, confidence, latency_ms],
        )
        con.close()
    except Exception:
        logger.exception("Failed to log request")


def log_batch(texts: list[str], responses: list) -> None:
    for i, text in enumerate(texts):
        log_request(
            text, responses[i].label, responses[i].confidence, responses[i].latency_ms
        )


@asynccontextmanager
async def lifespan(app: FastAPI):
    print(f"Loading ONNX model from {ONNX_DIR}")
    app.state.tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
    app.state.model = ORTModelForSequenceClassification.from_pretrained(ONNX_DIR)
    init_logging_table()
    print("ONNX model loaded.")
    yield
    print("Shutting down.")


app = FastAPI(
    title="Prompt Injection Guard",
    description="Real-time prompt injection classifier. DeBERTa-v3-base fine-tuned on 11,690 examples.",
    version="1.0.0",
    lifespan=lifespan,
)


@app.get("/health")
def health():
    return {"status": "ok"}


def _run_inference(text: str) -> tuple[str, float, dict, float]:
    t0 = time.perf_counter()
    encoded = app.state.tokenizer(
        text,
        truncation=True,
        padding="max_length",
        max_length=MAX_LENGTH,
        return_tensors="np",
    )
    logits = app.state.model(**encoded).logits
    probs = softmax(logits, axis=1)
    probs = probs.squeeze()
    pred_idx = int(np.argmax(probs))
    latency_ms = (time.perf_counter() - t0) * 1000
    label = LABELS[pred_idx]
    confidence = float(probs[pred_idx])
    scores = {LABELS[i]: float(probs[i]) for i in range(len(LABELS))}
    return label, confidence, scores, latency_ms


def _run_batch_inference(texts: list[str]) -> list[tuple[str, float, dict, float]]:
    t0 = time.perf_counter()
    encoded = app.state.tokenizer(
        texts,
        truncation=True,
        padding="max_length",
        max_length=MAX_LENGTH,
        return_tensors="np",
    )
    logits = app.state.model(**encoded).logits
    probs = softmax(logits, axis=1)
    total_ms = (time.perf_counter() - t0) * 1000
    per_item_ms = total_ms / len(texts)
    results = []
    for i in range(len(texts)):
        p = probs[i]
        pred_idx = int(np.argmax(p))
        results.append(
            (
                LABELS[pred_idx],
                float(p[pred_idx]),
                {LABELS[j]: float(p[j]) for j in range(len(LABELS))},
                round(per_item_ms, 2),
            )
        )
    return results


@app.post("/classify", response_model=ClassifyResponse)
async def classify(request: ClassifyRequest):
    loop = asyncio.get_event_loop()
    label, confidence, scores, latency_ms = await loop.run_in_executor(
        None, _run_inference, request.text
    )
    loop.run_in_executor(None, log_request, request.text, label, confidence, latency_ms)
    return ClassifyResponse(
        label=label,
        confidence=confidence,
        scores=scores,
        latency_ms=round(latency_ms, 2),
    )


@app.post("/classify-batch", response_model=BatchClassifyResponse)
async def classify_batch(request: BatchClassifyRequest):
    t0 = time.perf_counter()
    loop = asyncio.get_event_loop()
    results = await loop.run_in_executor(None, _run_batch_inference, request.texts)
    total_latency_ms = (time.perf_counter() - t0) * 1000
    responses = [
        ClassifyResponse(
            label=label,
            confidence=confidence,
            scores=scores,
            latency_ms=latency_ms,
        )
        for label, confidence, scores, latency_ms in results
    ]
    loop.run_in_executor(None, log_batch, request.texts, responses)
    return BatchClassifyResponse(
        results=responses,
        total_texts=len(responses),
        total_latency_ms=round(total_latency_ms, 2),
    )
