"""Export classifier v1 from HF Hub to ONNX with int8 dynamic quantization."""

from pathlib import Path

from dotenv import load_dotenv
from optimum.onnxruntime import ORTModelForSequenceClassification
from optimum.onnxruntime.configuration import AutoQuantizationConfig
from optimum.onnxruntime import ORTQuantizer

load_dotenv()

MODEL_ID = "alvi42/prompt-injection-guard-v1"
ONNX_DIR = Path("checkpoints/v1-onnx")
QUANTIZED_DIR = Path("checkpoints/v1-onnx-int8")


def export_to_onnx() -> None:
    print(f"Exporting {MODEL_ID} to ONNX...")
    ONNX_DIR.mkdir(parents=True, exist_ok=True)

    model = ORTModelForSequenceClassification.from_pretrained(
        MODEL_ID,
        export=True,
    )
    model.save_pretrained(ONNX_DIR)
    print(f"ONNX model saved to {ONNX_DIR}")


def quantize_to_int8() -> None:
    print("Applying int8 dynamic quantization...")
    QUANTIZED_DIR.mkdir(parents=True, exist_ok=True)

    quantizer = ORTQuantizer.from_pretrained(ONNX_DIR)
    qconfig = AutoQuantizationConfig.avx512_vnni(is_static=False, per_channel=False)

    quantizer.quantize(
        save_dir=QUANTIZED_DIR,
        quantization_config=qconfig,
    )
    print(f"Quantized model saved to {QUANTIZED_DIR}")


if __name__ == "__main__":
    export_to_onnx()
    quantize_to_int8()
    print("Done. Files in checkpoints/:")
    for f in sorted(Path("checkpoints").rglob("*")):
        print(f"  {f}")
