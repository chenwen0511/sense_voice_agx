#!/usr/bin/env python3
"""使用 TensorRT Python API 从 ONNX 构建引擎（需系统 python3-libnvinfer 10.3）。"""

import argparse
import os
import sys

# 使用 Jetson 系统自带的 TensorRT 10.3 Python 绑定
sys.path.insert(0, "/usr/lib/python3.10/dist-packages")

import tensorrt as trt

DEFAULT_MODEL_DIR = "/home/admin/stephen/02-weight/SenseVoiceSmall"
DEFAULT_ENGINE = os.path.join(os.path.dirname(__file__), "model_fp16.plan")


def build_engine(onnx_path: str, engine_path: str, fp16: bool = True) -> None:
    logger = trt.Logger(trt.Logger.INFO)
    builder = trt.Builder(logger)
    network = builder.create_network()
    parser = trt.OnnxParser(network, logger)

    with open(onnx_path, "rb") as f:
        onnx_bytes = f.read()

    onnx_dir = os.path.dirname(os.path.abspath(onnx_path))
    prev_cwd = os.getcwd()
    os.chdir(onnx_dir)
    try:
        if not parser.parse(onnx_bytes):
            for i in range(parser.num_errors):
                print(parser.get_error(i))
            raise RuntimeError(f"Failed to parse ONNX: {onnx_path}")
    finally:
        os.chdir(prev_cwd)

    config = builder.create_builder_config()
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 1 << 30)
    if fp16:
        config.set_flag(trt.BuilderFlag.FP16)

    profile = builder.create_optimization_profile()
    profile.set_shape("speech", (1, 1, 560), (1, 100, 560), (16, 3000, 560))
    profile.set_shape("speech_lengths", (1,), (1,), (16,))
    profile.set_shape("language", (1,), (1,), (16,))
    profile.set_shape("textnorm", (1,), (1,), (16,))
    config.add_optimization_profile(profile)

    serialized = builder.build_serialized_network(network, config)
    if serialized is None:
        raise RuntimeError("Failed to build TensorRT engine")

    with open(engine_path, "wb") as f:
        f.write(serialized)
    print(f"TensorRT {trt.__version__} engine saved to: {engine_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model-dir",
        default=os.environ.get("SENSEVOICE_MODEL_DIR", DEFAULT_MODEL_DIR),
    )
    parser.add_argument("--engine", default=DEFAULT_ENGINE)
    parser.add_argument("--fp16", action="store_true", default=True)
    args = parser.parse_args()

    onnx_path = os.path.join(args.model_dir, "model.onnx")
    if not os.path.isfile(onnx_path):
        raise FileNotFoundError(
            f"ONNX not found: {onnx_path}\n请先运行: python acc/export_onnx.py"
        )

    build_engine(onnx_path, args.engine, fp16=args.fp16)


if __name__ == "__main__":
    main()
