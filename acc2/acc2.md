# 特征提取加速（第二步）

在 `acc/` 已完成 **Encoder TensorRT 加速** 的基础上，本阶段针对链路中占比最高的 **特征提取**（约 59% 端到端耗时）做 GPU 化优化。

与 `acc/acc.md` 的关系：

| 阶段 | 目录 | 优化对象 | 状态 |
|------|------|----------|------|
| 第一步 | `acc/` | Encoder → TensorRT FP16 | 已验证 |
| **第二步** | `acc2/` | 特征提取 → GPU Fbank + LFR + CMVN | **已验证** |

---

## 现状与瓶颈

### 三阶段耗时（`infer_trt.log`，`en.mp3`）

TensorRT 端到端 avg **129.35 ms** 构成：

| 阶段 | 耗时 (avg) | 占端到端 | 当前实现 |
|------|------------|----------|----------|
| ① 特征提取 | **76.40 ms** | **59.1%** | Python CPU |
| ② TRT Encoder | 45.11 ms | 34.9% | `acc/model_fp16.plan` |
| ③ CTC 解码 | 7.84 ms | 6.1% | Python CPU |

预热后单次循环（特征已缓存）：TRT **45 ms** + 解码 **8 ms** ≈ 53 ms。

### 76 ms 里到底是什么？

`infer_trt.py` 记录的 76.40 ms 为 benchmark **首次** `load_data + extract_feat`（含冷启动）。稳态 profiling（`en.mp3`，预热后）：

| 子步骤 | 耗时 | 占比 |
|--------|------|------|
| 读音频 + mp3 解码 | ~11 ms | ~28% |
| Fbank + LFR + CMVN | ~29 ms | ~72% |
| **稳态合计** | **~40 ms** | |

当前 `acc/infer_trt.py` 特征路径：

```
mp3/wav
  → load_audio_text_image_video()     # soundfile / ffmpeg
  → torchaudio_stub kaldi.fbank()     # CPU
  → WavFrontend LFR (m=7, n=6)        # CPU PyTorch
  → CMVN (am.mvn)                     # CPU PyTorch
  → numpy float32 [1, T, 560]
  → 拷贝到 GPU 喂 TRT
```

瓶颈：**Fbank 在 CPU 上跑**，且特征与 TRT 之间存在 **CPU→GPU 额外拷贝**。

---

## 加速思路：kaldifeat GPU（对齐 FunASR Triton）

### 为什么选这条路线？

FunASR 官方 Triton 部署（`runtime/triton_gpu/model_repo_sense_voice_small/feature_extractor`）已验证：

- 特征提取 **不是纯 C++**，而是 **Triton Python Backend + kaldifeat GPU**
- 流程：`kaldifeat.Fbank (CUDA)` → `LFR (PyTorch CUDA)` → `CMVN (GPU)` → 输出 `speech [B,T,560]`

本方案 **借鉴 Triton 的算子栈，但不部署 Triton Server**——直接在 `infer_trt.py` 侧替换特征前端，**保留已有 TRT 引擎**。

### 目标架构

```
音频文件 / PCM
   │
   ▼
┌─────────────────────────────┐
│ ① 读音频（仍可 Python）      │  ~11 ms → 可优化为 wav 输入 / 内存 PCM
└──────────────┬──────────────┘
               │ waveform (GPU float tensor)
               ▼
┌─────────────────────────────┐
│ ② kaldifeat.Fbank (CUDA)    │  替代 CPU kaldi.fbank
└──────────────┬──────────────┘
               │ [T, 80] mel
               ▼
┌─────────────────────────────┐
│ ③ LFR + CMVN (CUDA)         │  对齐 Triton model.py 的 LFR 模块
└──────────────┬──────────────┘
               │ speech [1, T', 560] 已在 GPU
               ▼
┌─────────────────────────────┐
│ ④ TRT Encoder（不变）        │  acc/model_fp16.plan
└──────────────┬──────────────┘
               ▼
           CTC 解码（暂 Python）
```

### 与 Triton `feature_extractor` 的对应关系

| 组件 | Triton `model.py` | acc2 落地 |
|------|-------------------|-----------|
| Fbank | `kaldifeat.Fbank(opts)`，`opts.device=cuda` | 相同 |
| LFR | `WavFrontend.lfr`（PyTorch CUDA） | 复用 Triton 中 `LFR` 类逻辑 |
| CMVN | `apply_cmvn_batch`（GPU） | 加载 `am.mvn`，在 GPU 上应用 |
| 输出 | `speech` FP32/FP16 `[B,T,560]` | 直接 `set_tensor_address` 喂 TRT |

参考源码：

- https://github.com/modelscope/FunASR/blob/main/runtime/triton_gpu/model_repo_sense_voice_small/feature_extractor/1/model.py

---

## 预期收益

| 指标 | 现状 (`infer_trt.py`) | 目标 (kaldifeat GPU) |
|------|------------------------|----------------------|
| 特征提取（冷） | 76 ms | ~25~40 ms |
| 特征提取（热） | ~40 ms | ~15~25 ms |
| Fbank 段 | ~29 ms (CPU) | ~5~15 ms (GPU) |
| 端到端 avg | 129 ms | **~85~95 ms**（估） |
| TRT Encoder | 45 ms | 不变 |
| 识别结果 | 已验证 | 需数值对齐验证 |

> 收益主要来自：Fbank GPU 化 + 减少 CPU↔GPU 拷贝。mp3 解码仍走 Python，若改为 wav/pcm 可再省 ~5~10 ms。

## 实施结果（Orin AGX 实测）

测试：`en.mp3`、预热 10、推理 30、`acc/model_fp16.plan`。

| 指标 | acc `infer_trt.py`（CPU 特征） | acc2 `infer_trt_feat.py`（GPU 特征） | 变化 |
|------|-------------------------------|--------------------------------------|------|
| 特征提取（一次性） | 76.40 ms | **18.12 ms** | **~4.2×** |
| 端到端 avg | 129.35 ms | **69.12 ms** | **~1.87×** |
| TRT encoder avg | 45.11 ms | 42.46 ms | 相近 |
| CTC 解码 avg | 7.84 ms | 8.54 ms | 相近 |
| 识别结果 | 一致 | 一致 | ✓ |

特征后端：Orin 上 `kaldifeat` pip 编译失败，自动回退 **`torch-kaldi-gpu`**（`torchaudio_stub` kaldi.fbank on CUDA + LFR/CMVN on GPU）。

```bash
# 验证数值
python acc2/verify_feat.py --audio .../en.mp3

# benchmark
python acc2/infer_trt_feat.py --engine acc/model_fp16.plan \
  --audio .../en.mp3 --language en --warmup 10 --runs 30 \
  2>&1 | tee infer_trt_feat.log
```

---

## 实施步骤

### 已实现文件

```
acc2/
├── acc2.md              # 本文档
├── feat_gpu.py          # GpuFeatureExtractor（kaldifeat 或 torch-kaldi-gpu）
├── infer_trt_feat.py    # TRT + GPU 特征推理 + benchmark
└── verify_feat.py       # CPU vs GPU 特征对比
```

### 步骤 1：安装 kaldifeat（可选）

```bash
export LD_LIBRARY_PATH=$PWD/lib:$LD_LIBRARY_PATH
source venv/bin/activate

# 需在 Jetson 上编译，耗时较长
pip install kaldifeat

# 验证
python -c "import kaldifeat; import torch; print('ok', torch.cuda.is_available())"
```

若 pip 编译失败（Orin 常见），**无需手动处理**——`feat_gpu.py` 自动回退 `torch-kaldi-gpu`。

> **不要** 为此路径单独 `pip install tensorrt 11.x`；TRT 仍用系统 10.3（见 `acc/acc.md`）。

### 步骤 2：GPU 特征模块（已完成）

见 `acc2/feat_gpu.py`：`GpuFeatureExtractor` 实现 Fbank → LFR → CMVN，输出 GPU `[1,T,560]`。

### 步骤 3：数值对齐（已完成）

```bash
python acc2/verify_feat.py \
  --audio /home/admin/stephen/02-weight/SenseVoiceSmall/example/en.mp3
```

`dither=1.0` 时特征存在随机差异（max diff ~1.2），**帧长一致、识别文本一致**即可。

### 步骤 4：benchmark（已完成）

```bash
# 基线（CPU 特征）
python acc/infer_trt.py --engine acc/model_fp16.plan ...

# GPU 特征
python acc2/infer_trt_feat.py --engine acc/model_fp16.plan ...
```

---

## 其他方案对比（为何不优先）

| 方案 | 说明 | Orin 端侧评价 |
|------|------|---------------|
| **kaldifeat GPU（本方案）** | 改特征前端，保留 TRT | **推荐**：改动集中、与官方 Triton 一致 |
| Triton ensemble 全链路 | `feature_extractor` 仍是 Python+kaldifeat | 部署重，适合服务器多并发 |
| FunASR C++ Runtime | 主要面向 Paraformer | SenseVoice 支持弱 |
| sherpa-onnx C++ 全链路 | 完整 C++ 推理 | 需放弃自研 TRT 引擎，工程切换大 |
| kaldi-native-fbank 自研 C++ | sherpa 同源 Fbank | 长期最优，但 LFR/CMVN 需手写对齐 |
| TensorRT 编译特征图 | Fbank 非大矩阵乘 | 不推荐 |

---

## 风险与注意事项

1. **kaldifeat aarch64 编译**：Orin 上 pip 失败时已自动回退 `torch-kaldi-gpu`。
2. **数值对齐**：`dither`、`LFR` 边界处理必须与 `WavFrontend` / Triton 一致，否则识别结果漂移。
3. **音频格式**：mp3 解码仍在 Python；生产环境建议 **wav/pcm 直喂** 或 C++ 解码。
4. **内存**：特征留在 GPU 可减少拷贝，但长音频需注意 Orin 显存。
5. **与 acc 引擎兼容**：输出仍为 `speech [B,T,560]` float32，**无需重新 build TRT 引擎**。

---

## 后续（第三步可选）

特征 GPU 化之后，若仍需压缩剩余耗时：

| 阶段 | 现状 | 可选优化 |
|------|------|----------|
| 解码 ~8 ms | Python CTC + SentencePiece | C++ / GPU argmax（收益约数 ms） |
| 读音频 ~11 ms | mp3 解码 | wav 输入、流式 PCM、C++ 解码 |

优先级仍低于特征提取：解码仅占端到端 ~6%。

---

## 参考

- FunASR Triton feature_extractor：https://github.com/modelscope/FunASR/tree/main/runtime/triton_gpu/model_repo_sense_voice_small/feature_extractor
- kaldifeat：https://github.com/csukuangfj/kaldi-native-fbank（kaldifeat 底层）
- 第一步 TRT 加速：`acc/acc.md`
- 基线日志：`infer_trt.log`、`infer_torch.log`
