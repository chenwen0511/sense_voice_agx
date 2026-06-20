#!/usr/bin/env bash
# Orin AGX: 使用系统 TensorRT 10.3 trtexec 构建引擎
set -euo pipefail

TRTEXEC="${TRTEXEC:-/usr/src/tensorrt/bin/trtexec}"
MODEL_DIR="${SENSEVOICE_MODEL_DIR:-/home/admin/stephen/02-weight/SenseVoiceSmall}"
ENGINE_PATH="${1:-$(dirname "$0")/model_fp16.plan}"

if [[ ! -x "${TRTEXEC}" ]]; then
  echo "trtexec 未找到: ${TRTEXEC}"
  echo "请安装 TensorRT 或设置 TRTEXEC 环境变量"
  exit 1
fi

if [[ ! -f "${MODEL_DIR}/model.onnx" ]]; then
  echo "ONNX 不存在: ${MODEL_DIR}/model.onnx"
  echo "请先运行: python acc/export_onnx.py"
  exit 1
fi

"${TRTEXEC}" \
  --onnx="${MODEL_DIR}/model.onnx" \
  --saveEngine="${ENGINE_PATH}" \
  --minShapes=speech:1x1x560,speech_lengths:1,language:1,textnorm:1 \
  --optShapes=speech:1x100x560,speech_lengths:1,language:1,textnorm:1 \
  --maxShapes=speech:16x3000x560,speech_lengths:16,language:16,textnorm:16 \
  --fp16

echo "TensorRT engine saved to: ${ENGINE_PATH}"
