#!/usr/bin/env python3
"""SenseVoice TensorRT + GPU 特征提取推理（acc2）。"""

import argparse
import os
import statistics
import sys
import time
from dataclasses import dataclass, field

import numpy as np
import torch

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_LIB_DIR = os.path.join(_PROJECT_ROOT, "lib")
if os.path.isdir(_LIB_DIR):
    os.environ["LD_LIBRARY_PATH"] = _LIB_DIR + (
        ":" + os.environ.get("LD_LIBRARY_PATH", "")
        if os.environ.get("LD_LIBRARY_PATH")
        else ""
    )
sys.path.insert(0, "/usr/lib/python3.10/dist-packages")
sys.path.insert(0, _PROJECT_ROOT)

from torchaudio_stub import install as install_torchaudio_stub

install_torchaudio_stub()

import tensorrt as trt
from funasr.tokenizer.sentencepiece_tokenizer import SentencepiecesTokenizer
from funasr.utils.load_utils import load_audio_text_image_video
from funasr.utils.postprocess_utils import rich_transcription_postprocess

from acc2.feat_gpu import GpuFeatureExtractor

DEFAULT_MODEL_DIR = "/home/admin/stephen/02-weight/SenseVoiceSmall"
DEFAULT_ENGINE = os.path.join(_PROJECT_ROOT, "acc", "model_fp16.plan")
DEFAULT_AUDIO = os.path.join(DEFAULT_MODEL_DIR, "example", "zh.mp3")

LANGUAGE_MAP = {
    "auto": 0,
    "zh": 3,
    "en": 4,
    "yue": 7,
    "ja": 11,
    "ko": 12,
    "nospeech": 13,
}
TEXTNORM_MAP = {"withitn": 14, "woitn": 15}


@dataclass
class TimingBreakdown:
    load_data_ms: float = 0.0
    extract_feat_ms: float = 0.0
    trt_ms: float = 0.0
    decode_ms: float = 0.0
    total_ms: float = 0.0


@dataclass
class BenchmarkResult:
    total_ms: list[float] = field(default_factory=list)
    trt_ms: list[float] = field(default_factory=list)
    feat_once_ms: float = 0.0
    decode_ms: list[float] = field(default_factory=list)
    last_text: str = ""


class TrtEngineGpu:
    """TensorRT 引擎，speech 输入直接在 GPU。"""

    def __init__(self, engine_path: str):
        self.logger = trt.Logger(trt.Logger.WARNING)
        with open(engine_path, "rb") as f:
            runtime = trt.Runtime(self.logger)
            self.engine = runtime.deserialize_cuda_engine(f.read())
        if self.engine is None:
            raise RuntimeError(f"Failed to load TensorRT engine: {engine_path}")

        self.context = self.engine.create_execution_context()
        self.stream = torch.cuda.Stream()
        self.output_names = ["ctc_logits", "encoder_out_lens"]
        self.device_outputs: dict[str, torch.Tensor] = {}

    def _prepare_outputs(self):
        for name in self.output_names:
            shape = tuple(self.context.get_tensor_shape(name))
            if any(dim < 0 for dim in shape):
                raise RuntimeError(f"Invalid output shape for {name}: {shape}")
            dtype = trt.nptype(self.engine.get_tensor_dtype(name))
            torch_dtype = torch.from_numpy(np.empty((), dtype=dtype)).dtype
            if name not in self.device_outputs or self.device_outputs[name].shape != shape:
                self.device_outputs[name] = torch.empty(
                    shape, dtype=torch_dtype, device="cuda"
                )

    def infer(
        self,
        speech: torch.Tensor,
        speech_lengths: torch.Tensor,
        language: torch.Tensor,
        textnorm: torch.Tensor,
    ) -> tuple[np.ndarray, np.ndarray]:
        self.context.set_input_shape("speech", tuple(speech.shape))
        self.context.set_input_shape("speech_lengths", tuple(speech_lengths.shape))
        self.context.set_input_shape("language", tuple(language.shape))
        self.context.set_input_shape("textnorm", tuple(textnorm.shape))
        self._prepare_outputs()

        self.context.set_tensor_address("speech", int(speech.data_ptr()))
        self.context.set_tensor_address("speech_lengths", int(speech_lengths.data_ptr()))
        self.context.set_tensor_address("language", int(language.data_ptr()))
        self.context.set_tensor_address("textnorm", int(textnorm.data_ptr()))
        for name in self.output_names:
            self.context.set_tensor_address(
                name, int(self.device_outputs[name].data_ptr())
            )

        self.context.execute_async_v3(self.stream.cuda_stream)
        self.stream.synchronize()

        host_out = {}
        for name in self.output_names:
            host_out[name] = self.device_outputs[name].cpu().numpy()
        return host_out["ctc_logits"], host_out["encoder_out_lens"]


class SenseVoiceTrtFeatInfer:
    def __init__(self, model_dir: str, engine_path: str, device: str = "cuda:0"):
        self.model_dir = model_dir
        self.device = device
        self.engine = TrtEngineGpu(engine_path)
        self.feature_extractor = GpuFeatureExtractor(model_dir, device=device)
        self.tokenizer = SentencepiecesTokenizer(
            bpemodel=os.path.join(model_dir, "chn_jpn_yue_eng_ko_spectok.bpe.model")
        )
        self.blank_id = 0
        self._feat_cache: tuple[torch.Tensor, torch.Tensor] | None = None

        self._lang_buf = torch.zeros(1, dtype=torch.int32, device=device)
        self._textnorm_buf = torch.zeros(1, dtype=torch.int32, device=device)

    def _extract_features(
        self, audio_path: str
    ) -> tuple[torch.Tensor, torch.Tensor, float, float]:
        return self.feature_extractor.extract_from_audio_file(
            audio_path, load_audio_text_image_video
        )

    def _decode(
        self, ctc_logits: np.ndarray, encoder_out_lens: np.ndarray
    ) -> tuple[str, float]:
        t0 = time.perf_counter()
        x = torch.from_numpy(ctc_logits[0, : encoder_out_lens[0].item(), :]).float()
        yseq = x.argmax(dim=-1)
        yseq = torch.unique_consecutive(yseq, dim=-1)
        token_int = yseq[yseq != self.blank_id].tolist()
        text = self.tokenizer.decode(token_int)
        return text, (time.perf_counter() - t0) * 1000

    def infer(
        self,
        audio_path: str,
        language: str = "auto",
        use_itn: bool = True,
        cache_features: bool = False,
    ) -> tuple[str, TimingBreakdown]:
        timing = TimingBreakdown()

        if cache_features and self._feat_cache is not None:
            feats, feats_len = self._feat_cache
            timing.load_data_ms = 0.0
            timing.extract_feat_ms = 0.0
        else:
            feats, feats_len, load_ms, feat_ms = self._extract_features(audio_path)
            timing.load_data_ms = load_ms
            timing.extract_feat_ms = feat_ms
            if cache_features:
                self._feat_cache = (feats, feats_len)

        self._lang_buf[0] = LANGUAGE_MAP.get(language, 0)
        self._textnorm_buf[0] = TEXTNORM_MAP["withitn" if use_itn else "woitn"]

        torch.cuda.synchronize()
        t0 = time.perf_counter()
        ctc_logits, encoder_out_lens = self.engine.infer(
            feats, feats_len, self._lang_buf, self._textnorm_buf
        )
        torch.cuda.synchronize()
        timing.trt_ms = (time.perf_counter() - t0) * 1000

        text, decode_ms = self._decode(ctc_logits, encoder_out_lens)
        timing.decode_ms = decode_ms
        timing.total_ms = (
            timing.load_data_ms + timing.extract_feat_ms + timing.trt_ms + timing.decode_ms
        )
        return text, timing

    def warmup(
        self, audio_path: str, language: str, use_itn: bool, runs: int
    ) -> None:
        for i in range(runs):
            self.infer(audio_path, language=language, use_itn=use_itn, cache_features=True)
            print(f"  warmup {i + 1}/{runs} done")

    def benchmark(
        self,
        audio_path: str,
        language: str,
        use_itn: bool,
        warmup: int,
        runs: int,
    ) -> BenchmarkResult:
        self._feat_cache = None
        # 特征路径预热（避免首次 CUDA 冷启动计入 feat_once）
        for _ in range(3):
            self._extract_features(audio_path)
        torch.cuda.synchronize()

        _, feat_timing = self.infer(
            audio_path, language=language, use_itn=use_itn, cache_features=False
        )
        feat_once_ms = feat_timing.load_data_ms + feat_timing.extract_feat_ms
        print(
            f"特征提取（一次性）: {feat_once_ms:.2f} ms "
            f"(load={feat_timing.load_data_ms:.2f}, feat={feat_timing.extract_feat_ms:.2f})"
        )

        print(f"预热推理 {warmup} 次...")
        self.warmup(audio_path, language, use_itn, warmup)

        print(f"\n批量推理 {runs} 次（计时）...")
        result = BenchmarkResult()
        for i in range(runs):
            text, timing = self.infer(
                audio_path, language=language, use_itn=use_itn, cache_features=True
            )
            infer_ms = timing.trt_ms + timing.decode_ms
            e2e_ms = feat_once_ms + infer_ms
            result.total_ms.append(e2e_ms)
            result.trt_ms.append(timing.trt_ms)
            result.decode_ms.append(timing.decode_ms)
            result.last_text = text
            print(
                f"  run {i + 1}/{runs}: e2e={e2e_ms:.2f} ms, "
                f"trt={timing.trt_ms:.2f} ms, decode={timing.decode_ms:.2f} ms"
            )
        result.feat_once_ms = feat_once_ms
        return result


def print_benchmark_stats(
    result: BenchmarkResult,
    audio_path: str,
    engine_path: str,
    feat_backend: str,
    tensorrt_version: str,
) -> None:
    def stats(values: list[float]) -> str:
        return (
            f"min={min(values):.2f}, max={max(values):.2f}, "
            f"avg={statistics.mean(values):.2f}, p50={statistics.median(values):.2f}"
        )

    print("\n=== TensorRT + GPU 特征 推理时长统计 ===")
    print(f"音频: {audio_path}")
    print(f"引擎: {engine_path}")
    print(f"特征后端: {feat_backend}")
    print(f"TensorRT: {tensorrt_version}")
    print(f"有效推理次数: {len(result.total_ms)}")
    print(f"特征提取（一次性）: {result.feat_once_ms:.2f} ms")
    print(f"端到端估算 (ms): {stats(result.total_ms)}")
    print(f"TRT encoder (ms): {stats(result.trt_ms)}")
    print(f"CTC 解码 (ms): {stats(result.decode_ms)}")
    if len(result.total_ms) >= 2:
        print(f"端到端 stdev (ms): {statistics.stdev(result.total_ms):.2f}")
    print(f"{len(result.trt_ms)} 次 TRT 总时长: {sum(result.trt_ms) / 1000:.3f} s")
    print(f"\n最后一次识别结果: {rich_transcription_postprocess(result.last_text)}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="SenseVoice TRT + GPU features")
    parser.add_argument(
        "--model-dir",
        default=os.environ.get("SENSEVOICE_MODEL_DIR", DEFAULT_MODEL_DIR),
    )
    parser.add_argument(
        "--engine",
        default=os.environ.get("SENSEVOICE_ENGINE", DEFAULT_ENGINE),
    )
    parser.add_argument("--audio", nargs="+", default=[DEFAULT_AUDIO])
    parser.add_argument("--language", default="auto", choices=list(LANGUAGE_MAP.keys()))
    parser.add_argument("--no-itn", action="store_true")
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--runs", type=int, default=30)
    parser.add_argument("--once", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    audio_path = os.path.abspath(args.audio[0])
    use_itn = not args.no_itn

    if not os.path.isfile(audio_path):
        print(f"[ERROR] 音频不存在: {audio_path}", file=sys.stderr)
        return 1
    if not os.path.isfile(args.engine):
        print(
            f"[ERROR] TensorRT 引擎不存在: {args.engine}\n请先运行: bash acc/build_trt.sh",
            file=sys.stderr,
        )
        return 1

    print(f"模型目录 : {args.model_dir}")
    print(f"引擎路径 : {args.engine}")
    print(f"输入音频 : {audio_path}")
    print(f"语言     : {args.language}")
    print(f"ITN      : {use_itn}")
    print(f"TensorRT : {trt.__version__}")
    if not args.once:
        print(f"预热次数 : {args.warmup}")
        print(f"推理次数 : {args.runs}")
    print("-" * 50)

    load_start = time.perf_counter()
    inferer = SenseVoiceTrtFeatInfer(args.model_dir, args.engine)
    load_elapsed = time.perf_counter() - load_start
    print(f"引擎加载耗时: {load_elapsed:.2f}s")
    print(f"特征后端   : {inferer.feature_extractor.backend}")

    if args.once:
        text, timing = inferer.infer(audio_path, language=args.language, use_itn=use_itn)
        print(rich_transcription_postprocess(text))
        print(
            f"\n耗时: total={timing.total_ms:.2f} ms, load={timing.load_data_ms:.2f} ms, "
            f"feat={timing.extract_feat_ms:.2f} ms, trt={timing.trt_ms:.2f} ms, "
            f"decode={timing.decode_ms:.2f} ms"
        )
        return 0

    result = inferer.benchmark(
        audio_path, args.language, use_itn, args.warmup, args.runs
    )
    print_benchmark_stats(
        result,
        audio_path,
        args.engine,
        inferer.feature_extractor.backend,
        trt.__version__,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
