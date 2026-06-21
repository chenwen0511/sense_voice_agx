#!/usr/bin/env bash
# 目标 AGX 一键安装（部署人员手动放置权重包后执行）
#
# 前置条件：
#   1. 已将 SenseVoiceSmall.tar.gz 放到 /home/ubuntu/stephen/02-weight/
#   2. 项目代码已在 /home/ubuntu/stephen/01-code/sense_voice_agx/
#
# 用法：
#   cd /home/ubuntu/stephen/01-code/sense_voice_agx
#   bash scripts/install.sh
#
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$ROOT"

WEIGHT_BASE="${WEIGHT_BASE:-/home/ubuntu/stephen/02-weight}"
ARCHIVE="${WEIGHT_ARCHIVE:-$WEIGHT_BASE/SenseVoiceSmall.tar.gz}"
MODEL_DIR="${SENSEVOICE_MODEL_DIR:-$WEIGHT_BASE/SenseVoiceSmall}"
ENGINE_PATH="${SENSEVOICE_ENGINE:-$ROOT/acc/model_fp16.plan}"

export SENSEVOICE_MODEL_DIR="$MODEL_DIR"
export SENSEVOICE_ENGINE="$ENGINE_PATH"

echo "=========================================="
echo " SenseVoice AGX 一键安装"
echo "=========================================="
echo "权重包   : $ARCHIVE"
echo "模型目录 : $MODEL_DIR"
echo "TRT 引擎 : $ENGINE_PATH"
echo "项目目录 : $ROOT"
echo "=========================================="

# 1. 解压权重（若尚未解压）
if [[ ! -d "$MODEL_DIR" ]] || [[ ! -f "$MODEL_DIR/model.onnx" ]]; then
  if [[ ! -f "$ARCHIVE" ]]; then
    echo "[ERROR] 模型目录不存在，且未找到权重包。"
    echo "请将 SenseVoiceSmall.tar.gz 放到: $ARCHIVE"
    exit 1
  fi
  echo ""
  echo "[步骤 1/4] 解压权重包..."
  bash "$SCRIPT_DIR/unpack_weights.sh" "$ARCHIVE"
else
  echo ""
  echo "[步骤 1/4] 模型目录已存在，跳过解压"
fi

if [[ ! -f "$MODEL_DIR/model.onnx" ]]; then
  echo "[ERROR] 缺少 model.onnx: $MODEL_DIR/model.onnx"
  exit 1
fi

echo ""
echo "[步骤 2/4] 安装 Python 依赖..."
bash "$ROOT/setup_env.sh"

echo ""
echo "[步骤 3/4] 构建 TensorRT FP16 引擎（约数分钟）..."
bash "$ROOT/acc/build_trt.sh" "$ENGINE_PATH"

echo ""
echo "[步骤 4/4] 冒烟测试（acc2 GPU 特征 + TRT）..."
export LD_LIBRARY_PATH="$ROOT/lib:${LD_LIBRARY_PATH:-}"
# shellcheck disable=SC1091
source "$ROOT/venv/bin/activate"

TEST_AUDIO="${MODEL_DIR}/example/en.mp3"
if [[ ! -f "$TEST_AUDIO" ]]; then
  TEST_AUDIO="${MODEL_DIR}/example/zh.mp3"
fi

python "$ROOT/acc2/infer_trt_feat.py" --once \
  --engine "$ENGINE_PATH" \
  --model-dir "$MODEL_DIR" \
  --audio "$TEST_AUDIO" \
  --language en

echo ""
echo "=========================================="
echo " 安装完成"
echo "=========================================="
echo "日常运行："
echo "  cd $ROOT"
echo "  export LD_LIBRARY_PATH=$ROOT/lib:\$LD_LIBRARY_PATH"
echo "  export SENSEVOICE_MODEL_DIR=$MODEL_DIR"
echo "  export SENSEVOICE_ENGINE=$ENGINE_PATH"
echo "  source venv/bin/activate"
echo "  python acc2/infer_trt_feat.py --engine $ENGINE_PATH --device cuda:0"
