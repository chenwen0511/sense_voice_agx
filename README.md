# SenseVoice AGX

在 NVIDIA Orin AGX 端侧算力上进行 **SenseVoice-Small** 语音推理的项目。

## 环境要求

- JetPack 6.x（本机：R36.4 / CUDA 12.6）
- Python 3.10
- 模型权重目录：`/home/admin/stephen/02-weight/SenseVoiceSmall`

可通过环境变量覆盖默认配置：

```bash
export SENSEVOICE_MODEL_DIR=/path/to/SenseVoiceSmall
export SENSEVOICE_DEVICE=cuda:0
```

## 快速开始

### 1. 创建虚拟环境

系统若无 `python3-venv`，可先安装 `virtualenv`：

```bash
pip3 install --user virtualenv
python3 -m virtualenv venv
```

或使用一键脚本：

```bash
bash setup_env.sh
```

### 2. 安装依赖

**PyTorch 必须使用 NVIDIA Jetson 专用 wheel**，不能用普通 `pip install torch`。

```bash
source venv/bin/activate
pip install 'numpy<2'

# JetPack 6.1+ PyTorch 2.5（Orin AGX）
pip install --no-deps \
  https://developer.download.nvidia.com/compute/redist/jp/v61/pytorch/torch-2.5.0a0+872d972e41.nv24.08.17622132-cp310-cp310-linux_aarch64.whl

# cuSPARSELt（PyTorch 依赖，下载到项目 lib/）
mkdir -p lib
curl -OLs https://developer.download.nvidia.com/compute/cusparselt/redist/libcusparse_lt/linux-sbsa/libcusparse_lt-linux-sbsa-0.5.2.1-archive.tar.xz
tar xf libcusparse_lt-linux-sbsa-0.5.2.1-archive.tar.xz
cp -a libcusparse_lt-linux-sbsa-0.5.2.1-archive/lib/* lib/

pip install -r requirements.txt
```

> **注意**：Jetson 定制 PyTorch 与 pip 版 `torchaudio` ABI 不兼容。本项目通过 `torchaudio_stub` 提供兼容层，无需安装 `torchaudio`。

### 3. 运行推理（PyTorch 基线）

每次运行前设置运行时库路径：

```bash
export LD_LIBRARY_PATH=$PWD/lib:$LD_LIBRARY_PATH
source venv/bin/activate
```

#### 默认模式：预热 + 批量推理 + 时长统计

```bash
# 默认：预热 10 次 + 推理 30 次 + 统计（zh.mp3）
python infer.py --device cuda:0

# 英文 benchmark 日志
python infer.py --device cuda:0 --warmup 10 --runs 30 \
  --audio /home/admin/stephen/02-weight/SenseVoiceSmall/example/en.mp3 \
  --language en 2>&1 | tee infer_torch.log
```

#### 单次推理

```bash
python infer.py --once --device cuda:0 --audio /path/zh.mp3
```

## 推理脚本参数（infer.py）

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `--model-dir` | 本地模型目录 | `/home/admin/stephen/02-weight/SenseVoiceSmall` |
| `--audio` | 输入音频路径（可多个） | 模型目录下 `example/zh.mp3` |
| `--language` | `auto/zh/en/yue/ja/ko/nospeech` | `auto` |
| `--device` | `cuda:0` 或 `cpu` | 自动检测 |
| `--batch-size` | 静态 batch size | `64` |
| `--warmup` | 预热推理次数（不计入统计） | `10` |
| `--runs` | 计时推理次数 | `30` |
| `--once` | 仅单次推理 | - |
| `--no-itn` | 禁用标点与逆文本正则化 | - |

---

## TensorRT 加速（acc/）

将 SenseVoice **Encoder** 编译为 TensorRT FP16 引擎，特征提取与 CTC 解码仍在 Python。详细步骤见 [acc/acc.md](acc/acc.md)。

### 三步流程

| 步骤 | 命令 | 产出 |
|------|------|------|
| 1. 准备 ONNX | `python acc/export_onnx.py`（已有可跳过） | `model.onnx` |
| 2. 构建引擎 | `bash acc/build_trt.sh` | `acc/model_fp16.plan` |
| 3. TRT 推理 | `python acc/infer_trt.py --engine acc/model_fp16.plan` | `infer_trt.log` |

```bash
export LD_LIBRARY_PATH=$PWD/lib:$LD_LIBRARY_PATH
source venv/bin/activate
cd ~/stephen/01-code/sense_voice_agx

# 构建 FP16 引擎（首次，约数分钟）
bash acc/build_trt.sh

# benchmark（预热 10 + 推理 30）
python acc/infer_trt.py \
  --engine acc/model_fp16.plan \
  --audio /home/admin/stephen/02-weight/SenseVoiceSmall/example/en.mp3 \
  --language en --warmup 10 --runs 30 \
  2>&1 | tee infer_trt.log
```

> Orin 使用系统 TensorRT **10.3.0**，**不要** `pip install tensorrt`（会装 11.x 导致引擎无法加载）。

---

## 加速效果对比（Orin AGX 实测）

测试音频：`example/en.mp3`（约 7.2 s）、`cuda:0`、预热 10 次、计时 30 次、`language=en`。

| 指标 | PyTorch `infer.py` | TensorRT `acc/infer_trt.py` | 变化 |
|------|-------------------|----------------------------|------|
| 运行时加载 | 3.87 s | 11.56 s（引擎） | TRT 首载更慢 |
| 特征提取 | 每次推理内含 | 一次性 76.40 ms | 预热后复用 |
| **端到端 avg** | **180.27 ms** | **129.35 ms** | **↑ 1.39×（-28%）** |
| 端到端 p50 | 178.46 ms | 129.36 ms | |
| 端到端 stdev | 6.83 ms | 1.47 ms | TRT 更稳定 |
| Encoder 段 | 含在端到端内 | **45.11 ms**（TRT） | 可单独观测 |
| CTC 解码 | 含在端到端内 | 7.84 ms | |
| 30 次总耗时 | 5.408 s | TRT 段 1.353 s | |
| 识别结果 | The tribal chieftain called for the boy and presented him with 50 pieces of gold. | 一致 | ✓ |

```
链路说明：
  PyTorch  ：每次 = 特征 + PyTorch Encoder + 解码  →  avg 180 ms
  TensorRT ：一次特征 76 ms + 预热后每次 TRT 45 ms + 解码 8 ms  →  avg 129 ms
```

完整日志：[infer_torch.log](infer_torch.log)、[infer_trt.log](infer_trt.log)。

### PyTorch 基线（infer_torch.log）

| 指标 | 数值 |
|------|------|
| 单次推理 avg / p50 | 180.27 / 178.46 ms |
| min / max | 175.43 / 212.66 ms |
| stdev | 6.83 ms |

### TensorRT（infer_trt.log）

| 指标 | 数值 |
|------|------|
| 端到端 avg / p50 | 129.35 / 129.36 ms |
| TRT encoder avg / p50 | 45.11 / 45.37 ms |
| CTC 解码 avg | 7.84 ms |
| min / max（端到端） | 127.22 / 134.21 ms |

### 中文样本 `zh.mp3`（PyTorch 参考）

| 样本 | 设备 | 推理耗时 | 识别结果 |
|------|------|----------|----------|
| `zh.mp3` | cuda:0 | ~175 ms（预热后） | 开饭时间早上9点至下午五点。 |

---

## 项目结构

```
sense_voice_agx/
├── infer.py              # PyTorch 基线推理
├── infer_torch.log       # PyTorch benchmark 日志
├── infer_trt.log         # TensorRT benchmark 日志
├── setup_env.sh
├── torchaudio_stub/     # Jetson torchaudio 兼容层
├── lib/                  # cuSPARSELt
├── acc/
│   ├── acc.md            # TensorRT 加速文档
│   ├── export_onnx.py    # ONNX 导出
│   ├── build_trt.sh      # 引擎构建
│   ├── infer_trt.py      # TRT 推理 + benchmark
│   └── model_fp16.plan   # TRT 引擎（构建产物）
├── requirements.txt
└── venv/
```

## torchaudio_stub 说明

FunASR 依赖 `torchaudio`，但 Jetson 定制 PyTorch 与 pip 版 `torchaudio` ABI 不兼容。`torchaudio_stub/` 在导入 FunASR 前注入最小兼容实现，详见 `torchaudio_stub/__init__.py`。
