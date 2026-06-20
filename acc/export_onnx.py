#!/usr/bin/env python3
"""导出 SenseVoice ONNX（若模型目录尚无 model.onnx）。"""

import os
import sys

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_LIB_DIR = os.path.join(_ROOT, "lib")
if os.path.isdir(_LIB_DIR):
    os.environ["LD_LIBRARY_PATH"] = _LIB_DIR + (
        ":" + os.environ.get("LD_LIBRARY_PATH", "")
        if os.environ.get("LD_LIBRARY_PATH")
        else ""
    )

sys.path.insert(0, _ROOT)
from torchaudio_stub import install as install_torchaudio_stub

install_torchaudio_stub()

from funasr import AutoModel

DEFAULT_MODEL_DIR = "/home/admin/stephen/02-weight/SenseVoiceSmall"


def main():
    model_dir = os.environ.get("SENSEVOICE_MODEL_DIR", DEFAULT_MODEL_DIR)
    onnx_path = os.path.join(model_dir, "model.onnx")
    if os.path.isfile(onnx_path):
        print(f"ONNX 已存在，跳过导出: {onnx_path}")
        return

    model = AutoModel(
        model=model_dir,
        device="cuda:0",
        disable_update=True,
    )
    export_dir = model.export(type="onnx", quantize=False, device="cuda:0")
    print(f"ONNX exported to: {export_dir}")


if __name__ == "__main__":
    main()
