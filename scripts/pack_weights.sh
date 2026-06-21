#!/usr/bin/env bash
# 打包 SenseVoiceSmall 权重，供目标 AGX 解压到 02-weight 目录
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

WEIGHT_SRC="${WEIGHT_SRC:-/home/admin/stephen/02-weight/SenseVoiceSmall}"
OUTPUT_DIR="${OUTPUT_DIR:-$PROJECT_ROOT/dist}"
ARCHIVE_NAME="${ARCHIVE_NAME:-SenseVoiceSmall.tar.gz}"

if [[ ! -d "$WEIGHT_SRC" ]]; then
  echo "[ERROR] 权重目录不存在: $WEIGHT_SRC"
  echo "可通过环境变量指定: WEIGHT_SRC=/path/to/SenseVoiceSmall"
  exit 1
fi

mkdir -p "$OUTPUT_DIR"
OUTPUT_PATH="$OUTPUT_DIR/$ARCHIVE_NAME"

echo "源目录 : $WEIGHT_SRC"
echo "输出   : $OUTPUT_PATH"
echo "打包中（约 1.8G，需数分钟）..."

tar -czf "$OUTPUT_PATH" -C "$(dirname "$WEIGHT_SRC")" "$(basename "$WEIGHT_SRC")"

ls -lh "$OUTPUT_PATH"
echo ""
echo "打包完成。目标板解压示例："
echo "  mkdir -p /home/ubuntu/stephen/02-weight"
echo "  tar -xzf $ARCHIVE_NAME -C /home/ubuntu/stephen/02-weight"
echo "  # 得到 /home/ubuntu/stephen/02-weight/SenseVoiceSmall"
