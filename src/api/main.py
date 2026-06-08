"""FastAPI serving layer for the prompt injection classifier."""

import asyncio
import time
from contextlib import asynccontextmanager
from pathlib import Path

import duckdb
import numpy as np
from fastapi import FastAPI
from optimum.onnxruntime import ORTModelForSequenceClassification
from pydantic import BaseModel
from transformers import AutoTokenizer
from dotenv import load_dotenv


load_dotenv()

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


def init_logging_table():
    con = duckdb.connect(DB_PATH)
    con.execute(
        """
        CREATE TABLE IF NOT EXISTS request_log (
            id INTEGER,
            input_text TEXT,
            prediction TEXT,
            confidence FLOAT,
            latency_ms FLOAT,
            timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """
    )
    con.close()


def log_request(input_text: str, prediction: str, confidence: float, latency_ms: float):
    con = duckdb.connect(DB_PATH)
    con.execute(
        """
        INSERT INTO request_log (input_text, prediction, confidence, latency_ms)
        VALUES (?, ?, ?, ?)
    """,
        [input_text, prediction, confidence, latency_ms],
    )
    con.close()


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
    probs = np.exp(logits) / np.sum(np.exp(logits), axis=1, keepdims=True)
    probs = probs.squeeze()
    pred_idx = int(np.argmax(probs))
    latency_ms = (time.perf_counter() - t0) * 1000
    label = LABELS[pred_idx]
    confidence = float(probs[pred_idx])
    scores = {LABELS[i]: float(probs[i]) for i in range(len(LABELS))}
    return label, confidence, scores, latency_ms


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
