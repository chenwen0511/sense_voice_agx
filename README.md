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

### 3. 运行推理

每次运行前设置运行时库路径：

```bash
export LD_LIBRARY_PATH=$PWD/lib:$LD_LIBRARY_PATH
source venv/bin/activate
```

#### 默认模式：预热 + 批量推理 + 时长统计

`infer.py` 默认流程：

1. 加载模型并打印加载耗时
2. **预热** `--warmup` 次（默认 10，不计入统计）
3. **批量计时推理** `--runs` 次（默认 30，CUDA 下使用 `synchronize` 精确计时）
4. 输出 min / max / avg / p50 / stdev 及每次耗时

```bash
# 默认：预热 10 次 + 推理 30 次 + 统计（zh.mp3）
python infer.py --device cuda:0

# 自定义预热与推理次数
python infer.py --device cuda:0 --warmup 10 --runs 30

# 英文示例
python infer.py --device cuda:0 \
  --audio /home/admin/stephen/02-weight/SenseVoiceSmall/example/en.mp3 \
  --language en

# 多文件批量推理（一次 generate 传入多个音频）
python infer.py --device cuda:0 \
  --audio /path/zh.mp3 /path/en.mp3

# 保存完整日志
python infer.py --device cuda:0 \
  --audio /home/admin/stephen/02-weight/SenseVoiceSmall/example/en.mp3 \
  --language en 2>&1 | tee infer_torch.log
```

#### 单次推理模式

跳过预热与统计，适合快速验证：

```bash
python infer.py --once --device cuda:0 --audio /path/zh.mp3
```

## 推理脚本参数

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `--model-dir` | 本地模型目录 | `/home/admin/stephen/02-weight/SenseVoiceSmall` |
| `--audio` | 输入音频路径（可多个） | 模型目录下 `example/zh.mp3` |
| `--language` | `auto/zh/en/yue/ja/ko/nospeech` | `auto` |
| `--device` | `cuda:0` 或 `cpu` | 自动检测 |
| `--batch-size` | 静态 batch size | `64` |
| `--warmup` | 预热推理次数（不计入统计） | `10` |
| `--runs` | 计时推理次数 | `30` |
| `--once` | 仅单次推理，跳过预热与统计 | - |
| `--no-itn` | 禁用标点与逆文本正则化 | - |

## 验证结果（Orin AGX / PyTorch）

以下为 `infer_torch.log` 实测数据（FunASR 1.3.10 + PyTorch 2.5 Jetson wheel，`cuda:0`）。

### 英文样本 `en.mp3`（预热 10 + 推理 30）

| 指标 | 数值 |
|------|------|
| 模型加载 | 3.87 s |
| 预热次数 | 10 |
| 有效推理次数 | 30 |
| 单次推理 min | 175.43 ms |
| 单次推理 max | 212.66 ms |
| 单次推理 avg | 180.27 ms |
| 单次推理 p50 | 178.46 ms |
| 单次推理 stdev | 6.83 ms |
| 总推理时长 | 5.408 s |
| 识别结果 | The tribal chieftain called for the boy and presented him with 50 pieces of gold. |

各次推理时长（ms）：

```
[01] 186.47  [02] 178.43  [03] 178.81  [04] 177.42  [05] 177.26
[06] 176.83  [07] 175.98  [08] 185.24  [09] 183.63  [10] 178.25
[11] 177.90  [12] 175.89  [13] 181.35  [14] 178.50  [15] 176.00
[16] 176.22  [17] 177.56  [18] 178.48  [19] 176.44  [20] 175.81
[21] 179.04  [22] 175.43  [23] 182.10  [24] 182.10  [25] 178.68
[26] 182.71  [27] 212.66  [28] 177.80  [29] 180.63  [30] 184.45
```

预热后单次推理约 **180 ms**，RTF 约 **0.03**（音频约 7.2 s），满足实时推理。

### 中文样本 `zh.mp3`（单次冷启动参考）

| 样本 | 设备 | 推理耗时 | 识别结果 |
|------|------|----------|----------|
| `zh.mp3` | cuda:0 | ~0.5 s（冷启动） / ~175 ms（预热后） | 开饭时间早上9点至下午五点。 |

## 项目结构

```
sense_voice_agx/
├── infer.py           # 推理入口（预热 / 批量计时 / 统计）
├── infer_torch.log    # PyTorch 基线 benchmark 日志
├── setup_env.sh       # 环境安装脚本
├── torchaudio_stub/   # Jetson 兼容的 torchaudio 替代实现
├── lib/               # cuSPARSELt 运行时库（需手动下载）
├── requirements.txt
└── venv/
```

## torchaudio_stub 说明

FunASR 依赖 `torchaudio`（读音频、`kaldi.fbank` 等），但 Jetson 定制 PyTorch 与 pip 版 `torchaudio` ABI 不兼容。`torchaudio_stub/` 在导入 FunASR 前注入最小兼容实现，详见代码内 `torchaudio_stub/__init__.py`。

## 后续：推理加速

`acc/` 目录预留用于 TensorRT / ONNX 等端侧加速方案（当前使用 FunASR + PyTorch GPU 基线推理）。
