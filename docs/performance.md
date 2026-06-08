# Performance Report

## Test Environment

- Hardware: MacBook Air M5
- Inference backend: ONNX Runtime 1.26.0, int8 dynamic quantization
- Model: DeBERTa-v3-base fine-tuned (alvi42/prompt-injection-guard-v1)
- Server: uvicorn, single worker, single process
- Test tool: locust 2.44.1

## Risk Gate Result: PASSED

Gate criterion: p99 latency < 100ms on a single CPU core.

| Percentile | Latency (ms) |
|---|---|
| p50 | 85 |
| p66 | 86 |
| p75 | 86 |
| p80 | 87 |
| p90 | 88 |
| p95 | 90 |
| p99 | 91 |
| p100 | 120 |

Test: 1 concurrent user, 60 seconds, 60 requests, 0 failures.

## Latency Progression

| Backend | Single request (warm) | p99 (1 user, 60s) |
|---|---|---|
| PyTorch CPU (Day 15) | 116ms | not measured |
| ONNX int8 (Day 16) | 66ms | 91ms |

ONNX int8 quantization reduced single-request latency by 43% versus PyTorch.

## Concurrency Behavior

At 50 concurrent users, average latency rises to ~1500ms due to CPU saturation
from parallel ONNX inference threads on a single machine. This is expected
behavior for a CPU-only single-process server. The gate criterion is single-core
sequential latency, not concurrent throughput.

## Known Issues Fixed During Profiling

**DuckDB PRIMARY KEY contention**: The original log_request function computed
MAX(id)+1 inside each request. Under concurrent load, multiple requests read
the same MAX value simultaneously, causing duplicate key violations (500 errors,
28% failure rate). Fixed by removing the PRIMARY KEY constraint and dropping
the manual id computation.

**Blocking inference in async handler**: Running ONNX inference directly inside
an async def endpoint blocked the event loop under concurrent load, serializing
all requests through a single thread. Fixed by offloading inference to a thread
pool executor via asyncio.get_event_loop().run_in_executor().