.PHONY: data train eval export serve test synth lint

data:
	uv run python -m src.data.unify
	uv run python -m src.data.dedup
	uv run python -m src.data.split

train:
	uv run python -m src.models.train --output-dir checkpoints/v1

train-v2:
	uv run python -m src.models.train --output-dir checkpoints/v2 --db-split split_train_v2

eval:
	uv run python -m src.eval.evaluate --model-path checkpoints/v1 --output docs/eval_results.json

eval-baseline:
	uv run python -m src.eval.baseline

export:
	uv run python scripts/export_onnx.py

serve:
	uv run uvicorn src.api.main:app --host 0.0.0.0 --port 8000 --reload

synth:
	uv run python -m src.synth.generate
	uv run python scripts/dedup_synthetic.py
	uv run python scripts/crossval_label.py
	uv run python scripts/finalize_crossval.py

test:
	uv run pytest tests/ -v

lint:
	uv run ruff check .
	uv run black --check .
