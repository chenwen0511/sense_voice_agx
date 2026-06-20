#!/bin/bash
# SenseVoice AGX 环境安装脚本（Orin JetPack 6.x）
set -euo pipefail

ROOT="$(cd "$(dirname "$0")" && pwd)"
cd "$ROOT"

TORCH_WHL="https://developer.download.nvidia.com/compute/redist/jp/v61/pytorch/torch-2.5.0a0+872d972e41.nv24.08.17622132-cp310-cp310-linux_aarch64.whl"
CUSPARSELT_URL="https://developer.download.nvidia.com/compute/cusparselt/redist/libcusparse_lt/linux-sbsa/libcusparse_lt-linux-sbsa-0.5.2.1-archive.tar.xz"

if [ ! -d "venv" ]; then
  if python3 -m venv venv 2>/dev/null; then
    :
  else
    pip3 install --user virtualenv
    python3 -m virtualenv venv
  fi
fi

source venv/bin/activate
pip install --upgrade pip
pip install 'numpy<2'
pip install --no-deps --force-reinstall "$TORCH_WHL"

if [ ! -f lib/libcusparseLt.so.0 ]; then
  mkdir -p lib /tmp/cusparselt
  curl -OLs "$CUSPARSELT_URL" -o /tmp/cusparselt/archive.tar.xz
  tar xf /tmp/cusparselt/archive.tar.xz -C /tmp/cusparselt
  cp -a /tmp/cusparselt/libcusparse_lt-linux-sbsa-0.5.2.1-archive/lib/* lib/
fi

pip install -r requirements.txt

echo "安装完成。运行推理："
echo "  export LD_LIBRARY_PATH=$ROOT/lib:\$LD_LIBRARY_PATH"
echo "  source venv/bin/activate"
echo "  python infer.py --device cuda:0"
