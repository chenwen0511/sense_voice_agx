# SenseVoice TensorRT 推理加速（Orin AGX）

将 SenseVoice-Small 的 **Encoder** 从 PyTorch 替换为 TensorRT FP16 引擎，在 Orin AGX 上验证端侧加速效果。

---

## 方案概述

SenseVoice 推理链路分为三段，本方案仅对 **Encoder** 做 TensorRT 加速：

```
音频文件
   │
   ▼
┌─────────────────────────────┐
│ ① 特征提取（Python）         │  FunASR WavFrontend：读音频 + Fbank + CMVN
│    infer.py / infer_trt.py  │  两端方案相同，均在 CPU/GPU Python 侧
└──────────────┬──────────────┘
               │ speech [B,T,560]
               ▼
┌─────────────────────────────┐
│ ② Encoder 前向              │  PyTorch：infer.py（~180ms 端到端含前后处理）
│                             │  TensorRT：acc/infer_trt.py（~45ms encoder）
└──────────────┬──────────────┘
               │ ctc_logits
               ▼
┌─────────────────────────────┐
│ ③ CTC 解码 + 分词（Python）  │  argmax + SentencePiece
└──────────────┬──────────────┘
               ▼
           识别文本
```

**整体流程（三步）**：

| 步骤 | 操作 | 产出 |
|------|------|------|
| **1. 准备 ONNX** | FunASR 导出或复用已有 `model.onnx` | `model.onnx` + `model.onnx.data` |
| **2. 构建 TRT 引擎** | `trtexec` 指定动态 shape + FP16 | `acc/model_fp16.plan` |
| **3. TRT 推理验证** | `acc/infer_trt.py` 预热 + benchmark | `infer_trt.log` |

---

## 加速效果对比（Orin AGX 实测）

测试条件：`example/en.mp3`（约 7.2 s）、`cuda:0`、预热 10 次、计时 30 次、`language=en`。

| 指标 | PyTorch（`infer_torch.log`） | TensorRT（`infer_trt.log`） | 对比 |
|------|------------------------------|------------------------------|------|
| 运行时加载 | 模型 3.87 s | 引擎 11.56 s | TRT 引擎较大，首载更慢 |
| 特征提取 | 含在每次推理内 | 一次性 76.40 ms | TRT 预热后复用特征 |
| **端到端 avg** | **180.27 ms** | **129.35 ms** | **约 1.39× 加速（-28%）** |
| 端到端 p50 | 178.46 ms | 129.36 ms | |
| 端到端 min / max | 175.43 / 212.66 ms | 127.22 / 134.21 ms | TRT 更稳定（stdev 1.47 vs 6.83） |
| **Encoder（TRT 段）** | 含在端到端内 | **45.11 ms avg** | 可单独观测 |
| CTC 解码 | 含在端到端内 | 7.84 ms avg | |
| 30 次总耗时 | 5.408 s | TRT 段 1.353 s | |
| 识别结果 | The tribal chieftain called for the boy and presented him with 50 pieces of gold. | 一致 | ✓ |

> **说明**：PyTorch 基线为 FunASR 全链路（特征 + PyTorch Encoder + 解码）；TensorRT 端到端 = 一次性特征提取 + TRT Encoder + 解码。  
> `trtexec` 在 opt shape `1×100×560` 下纯 GPU 延迟约 **27 ms**；实际 `en.mp3` 因序列长度与 Python 绑定开销，TRT 段约 **45 ms**。

### PyTorch 基线（预热后 30 次，ms）

```
avg=180.27  p50=178.46  min=175.43  max=212.66  stdev=6.83
```

### TensorRT（预热后 30 次）

```
端到端 avg=129.35 ms  |  TRT encoder avg=45.11 ms  |  解码 avg=7.84 ms
```

---

## 环境要求

| 组件 | 本机（Orin AGX） |
|------|------------------|
| JetPack | R36.4 / CUDA 12.6 |
| TensorRT | **10.3.0**（`libnvinfer` 10.3.0.30） |
| trtexec | `/usr/src/tensorrt/bin/trtexec` |
| Python TRT | 系统包 `python3-libnvinfer`（**勿 pip 安装 tensorrt 11.x**） |
| 模型 | `/home/admin/stephen/02-weight/SenseVoiceSmall` |

```bash
export LD_LIBRARY_PATH=$PWD/lib:$LD_LIBRARY_PATH
source venv/bin/activate
pip install -r requirements.txt

# 若需重新导出 ONNX
pip install onnx onnxscript

# 系统包（JetPack 通常已带）
# sudo apt install python3-libnvinfer libnvinfer-bin
```

> **重要**：venv 内 **不要** `pip install tensorrt`。pip 会装 11.x，与系统 `trtexec` 10.3 不兼容，加载 `.plan` 报 `Version tag does not match`。  
> `infer_trt.py` 通过 `sys.path` 使用 `/usr/lib/python3.10/dist-packages` 中的 TensorRT 10.3。

---

## 步骤 1：准备 ONNX

**目的**：将 PyTorch Encoder 导出为 ONNX，供 TensorRT 解析。

权重目录若已有 `model.onnx` + `model.onnx.data`，**跳过本步**。

```bash
cd ~/stephen/01-code/sense_voice_agx
export LD_LIBRARY_PATH=$PWD/lib:$LD_LIBRARY_PATH
source venv/bin/activate

python acc/export_onnx.py
```

产出目录：

```
/home/admin/stephen/02-weight/SenseVoiceSmall/
├── model.onnx          # 图结构
├── model.onnx.data     # 权重（外部数据，约 936 MB）
├── am.mvn
├── chn_jpn_yue_eng_ko_spectok.bpe.model
└── config.yaml
```

### ONNX 输入 / 输出

| 名称 | 类型 | 形状 | 说明 |
|------|------|------|------|
| `speech` | FP32 | `[B, T, 560]` | Fbank 特征 |
| `speech_lengths` | INT32 | `[B]` | 帧数 |
| `language` | INT32 | `[B]` | 语言 id（0=auto, 3=zh, 4=en …） |
| `textnorm` | INT32 | `[B]` | 14=with ITN, 15=woitn |
| `ctc_logits` | FP32 | `[B, T', 25055]` | CTC 输出 |
| `encoder_out_lens` | INT32 | `[B]` | 编码器输出长度 |

---

## 步骤 2：构建 TensorRT 引擎

**目的**：将 ONNX 编译为 Orin 上的 FP16 `.plan` 引擎，针对动态序列长度做 shape profile。

SenseVoice 支持动态 batch 与动态时间维，必须指定 min / opt / max shape。

### 方式 A：`trtexec`（推荐）

```bash
bash acc/build_trt.sh
# 默认输出 acc/model_fp16.plan
```

等价命令：

```bash
/usr/src/tensorrt/bin/trtexec \
  --onnx=/home/admin/stephen/02-weight/SenseVoiceSmall/model.onnx \
  --saveEngine=/home/admin/stephen/01-code/sense_voice_agx/acc/model_fp16.plan \
  --minShapes=speech:1x1x560,speech_lengths:1,language:1,textnorm:1 \
  --optShapes=speech:1x100x560,speech_lengths:1,language:1,textnorm:1 \
  --maxShapes=speech:16x3000x560,speech_lengths:16,language:16,textnorm:16 \
  --fp16
```

成功标志：终端末尾 `PASSED TensorRT.trtexec`。

### 方式 B：Python API

```bash
python acc/build_trt.py
```

### Shape 参数

| 参数 | 值 | 含义 |
|------|-----|------|
| `minShapes` | `speech:1×1×560` | 最短音频 |
| `optShapes` | `speech:1×100×560` | Builder 优化 profile（约 1 s 音频） |
| `maxShapes` | `speech:16×3000×560` | 最长约 30 s、batch 16 |
| `--fp16` | 开启 | Orin 推荐 |

> `.plan` 与 GPU 架构、TensorRT 版本绑定，换机需重新 build。

---

## 步骤 3：TensorRT 推理与 benchmark

**目的**：加载 `.plan`，执行与 `infer.py` 相同风格的预热 + 批量计时，对比 PyTorch 基线。

```bash
export LD_LIBRARY_PATH=$PWD/lib:$LD_LIBRARY_PATH
source venv/bin/activate
cd ~/stephen/01-code/sense_voice_agx
```

### 默认：预热 + 批量推理 + 统计

```bash
python acc/infer_trt.py \
  --engine acc/model_fp16.plan \
  --audio /home/admin/stephen/02-weight/SenseVoiceSmall/example/en.mp3 \
  --language en \
  --warmup 10 --runs 30 \
  2>&1 | tee infer_trt.log
```

### 单次推理

```bash
python acc/infer_trt.py --once \
  --engine acc/model_fp16.plan \
  --audio /home/admin/stephen/02-weight/SenseVoiceSmall/example/en.mp3 \
  --language en
```

### 与 PyTorch 基线对比命令

```bash
# PyTorch
python infer.py --device cuda:0 --warmup 10 --runs 30 \
  --audio /home/admin/stephen/02-weight/SenseVoiceSmall/example/en.mp3 \
  --language en 2>&1 | tee infer_torch.log

# TensorRT
python acc/infer_trt.py --engine acc/model_fp16.plan --warmup 10 --runs 30 \
  --audio /home/admin/stephen/02-weight/SenseVoiceSmall/example/en.mp3 \
  --language en 2>&1 | tee infer_trt.log
```

---

## infer_trt.py 参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--engine` | `acc/model_fp16.plan` | TensorRT 引擎路径 |
| `--model-dir` | 本地 SenseVoice 目录 | frontend / tokenizer |
| `--audio` | `example/zh.mp3` | 支持多个路径（benchmark 用第一个） |
| `--language` | `auto` | 语言 |
| `--warmup` | `10` | 预热次数 |
| `--runs` | `30` | 计时推理次数 |
| `--once` | - | 单次推理 |
| `--no-itn` | - | 禁用 ITN |

环境变量：`SENSEVOICE_MODEL_DIR`、`SENSEVOICE_ENGINE`

---

## 常见问题

### `trtexec: command not found`

```bash
export TRTEXEC=/usr/src/tensorrt/bin/trtexec
bash acc/build_trt.sh
```

### `Version tag does not match`

```bash
pip uninstall -y tensorrt tensorrt_cu13 tensorrt_cu13_bindings tensorrt_cu13_libs
python -c "import sys; sys.path.insert(0,'/usr/lib/python3.10/dist-packages'); import tensorrt as trt; print(trt.__version__)"
# 应输出 10.3.0
bash acc/build_trt.sh
```

### 导出 ONNX 报 `No module named 'onnxscript'`

```bash
pip install onnx onnxscript
```

### trtexec shape 不匹配

输入名必须为 `speech`, `speech_lengths`, `language`, `textnorm`；超长音频需增大 `maxShapes` 的 T 维。

### 端到端为何没达到 trtexec 的 27 ms？

`trtexec` 仅测纯 GPU kernel；实际推理还包括 Python 绑定、动态 shape 设置、H2D/D2H。benchmark 中 `trt_ms` 为可观测的 Encoder 段耗时。

---

## 文件说明

```
acc/
├── acc.md           # 本文档
├── export_onnx.py   # 步骤 1：导出 ONNX
├── build_trt.sh     # 步骤 2：trtexec 构建
├── build_trt.py     # 步骤 2：Python API 构建
├── infer_trt.py     # 步骤 3：TRT 推理 + benchmark
└── model_fp16.plan  # 构建产物（勿提交）
```

## 参考

- [SenseVoiceInfer/acc](../SenseVoiceInfer/acc)（4090 参考实现）
- [FunASR Triton GPU](https://github.com/modelscope/FunASR/tree/main/runtime/triton_gpu)
- [NVIDIA trtexec 文档](https://docs.nvidia.com/deeplearning/tensorrt/latest/reference/command-line-programs.html#trtexec)
