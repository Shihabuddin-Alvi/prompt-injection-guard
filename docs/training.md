# Classifier v1 Training Notes

## Environment
- GPU: Colab T4 (free tier)
- Framework: PyTorch via HF Trainer
- transformers version: 4.47.0 (pinned — 5.0.0 breaks DeBERTa v3 LayerNorm loading)

## Hyperparameters
- Model: microsoft/deberta-v3-base
- Learning rate: 2e-5
- Batch size: 16
- Epochs: 3
- Weight decay: 0.01
- Max sequence length: 256
- fp16: False (required for DeBERTa v3 on T4)

## Results
| Epoch | eval_macro_f1 | eval_loss |
|-------|--------------|-----------|
| 1     | 0.9913       | 0.0363    |
| 2     | 0.9938       | 0.0430    |
| 3     | 0.9938       | 0.0360    |

Best model: epoch 3 (tied F1 with epoch 2, lower loss)

## Training Time
- Total: ~28 minutes on T4
- Steps: 1536
- Throughput: ~1.15 it/s

## GPU Cost
- Colab free tier T4: $0

## Known Issues
- transformers 5.0.0 causes grad_norm=nan and loss=0 due to DeBERTa v3 LayerNorm gamma/beta mapping bug
- Fix: pin to transformers==4.47.0
- fp16=True raises FP16 gradient unscaling error on T4 — use fp16=False

## Model Location
- HF Hub: alvi42/prompt-injection-guard-v1 (private)
- Google Drive: prompt-injection-guard/checkpoints/v1
