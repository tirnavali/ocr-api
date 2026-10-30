"""Quantize a BERT token-classification ONNX model to INT8 for nlp_mode=onnx.

Usage:
    python scripts/prepare_ner_model.py <model_dir>

<model_dir> must contain model.onnx, config.json and tokenizer.json, e.g. the files of
https://huggingface.co/akdeniz27/bert-base-turkish-cased-ner (MIT). Writes
<model_dir>/model_int8.onnx (~110 MB from ~440 MB) and prints a FP32-vs-INT8 tag check.
Requires the "ner" and "ner-tools" extras (onnxruntime, tokenizers, onnx).
"""

from __future__ import annotations

import sys
from pathlib import Path

from onnxruntime.quantization import QuantType, quantize_dynamic

from ocr_service.pipeline.ner_onnx import OnnxBertNer

SAMPLE = (
    "Türkiye Büyük Millet Meclisi Başkanı Kemal Güven , Ankara'da Milliyet Gazetesi muhabiri "
    "Hasan Pulur'a Bandırma ve Balıkesir hakkında bilgi verdi ."
).split()


def main() -> None:
    model_dir = Path(sys.argv[1])
    src, dst = model_dir / "model.onnx", model_dir / "model_int8.onnx"
    if not src.exists():
        sys.exit(f"{src} not found")
    quantize_dynamic(str(src), str(dst), weight_type=QuantType.QInt8)
    print(f"{dst.name}: {dst.stat().st_size / 1e6:.1f} MB (from {src.stat().st_size / 1e6:.1f} MB)")

    fp32 = [t.tag for t in OnnxBertNer(model_dir, "model.onnx").tag_words(SAMPLE)]
    int8 = [t.tag for t in OnnxBertNer(model_dir, "model_int8.onnx").tag_words(SAMPLE)]
    diff = [(w, a, b) for w, a, b in zip(SAMPLE, fp32, int8, strict=True) if a != b]
    print("tags identical to FP32" if not diff else f"tag differences: {diff}")


if __name__ == "__main__":
    main()
