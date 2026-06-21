#!/usr/bin/env bash
# 在目标 AGX 上解压权重包（手动拷贝 tarball 后使用）
set -euo pipefail

WEIGHT_BASE="${WEIGHT_BASE:-/home/ubuntu/stephen/02-weight}"
ARCHIVE="${1:-/home/ubuntu/stephen/02-weight/SenseVoiceSmall.tar.gz}"

if [[ ! -f "$ARCHIVE" ]]; then
  echo "[ERROR] 找不到压缩包: $ARCHIVE"
  echo "用法: bash scripts/unpack_weights.sh [tar.gz路径]"
  exit 1
fi

mkdir -p "$WEIGHT_BASE"
echo "解压 $ARCHIVE -> $WEIGHT_BASE"
tar -xzf "$ARCHIVE" -C "$WEIGHT_BASE"
echo "完成: $WEIGHT_BASE/SenseVoiceSmall"
