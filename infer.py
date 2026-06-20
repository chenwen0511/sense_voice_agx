#!/usr/bin/env python3
"""SenseVoice-Small 端侧推理脚本 (NVIDIA Orin AGX)."""

import argparse
import os
import statistics
import sys
import time

# cuSPARSELt 库路径（Jetson PyTorch 依赖）
_PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
_LIB_DIR = os.path.join(_PROJECT_ROOT, "lib")
if os.path.isdir(_LIB_DIR):
    os.environ["LD_LIBRARY_PATH"] = _LIB_DIR + (
        ":" + os.environ.get("LD_LIBRARY_PATH", "")
        if os.environ.get("LD_LIBRARY_PATH")
        else ""
    )

DEFAULT_MODEL_DIR = "/home/admin/stephen/02-weight/SenseVoiceSmall"
DEFAULT_AUDIO = os.path.join(DEFAULT_MODEL_DIR, "example", "zh.mp3")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="SenseVoice-Small ASR inference")
    parser.add_argument(
        "--model-dir",
        default=os.environ.get("SENSEVOICE_MODEL_DIR", DEFAULT_MODEL_DIR),
        help="本地模型目录（含 model.pt / config.yaml）",
    )
    parser.add_argument(
        "--audio",
        nargs="+",
        default=[DEFAULT_AUDIO],
        help="输入音频路径，可指定多个文件进行批量推理",
    )
    parser.add_argument(
        "--language",
        default="auto",
        choices=["auto", "zh", "en", "yue", "ja", "ko", "nospeech"],
        help="识别语言",
    )
    parser.add_argument(
        "--device",
        default=os.environ.get("SENSEVOICE_DEVICE"),
        help="推理设备，如 cuda:0 或 cpu；默认自动选择",
    )
    parser.add_argument(
        "--use-itn",
        action="store_true",
        default=True,
        help="输出标点与逆文本正则化结果",
    )
    parser.add_argument(
        "--no-itn",
        action="store_true",
        help="禁用 ITN",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=64,
        help="静态 batch size（仅 SenseVoice，不含 VAD）",
    )
    parser.add_argument(
        "--warmup",
        type=int,
        default=10,
        help="预热推理次数（不计入统计）",
    )
    parser.add_argument(
        "--runs",
        type=int,
        default=30,
        help="计时推理次数（用于统计）",
    )
    parser.add_argument(
        "--once",
        action="store_true",
        help="仅单次推理，跳过预热与批量统计",
    )
    return parser.parse_args()


def resolve_device(requested: str | None) -> str:
    import torch

    if requested:
        return requested
    return "cuda:0" if torch.cuda.is_available() else "cpu"


def sync_device(device: str) -> None:
    import torch

    if device.startswith("cuda") and torch.cuda.is_available():
        torch.cuda.synchronize()


def run_inference(model, audio_inputs, language: str, use_itn: bool, batch_size: int):
    return model.generate(
        input=audio_inputs,
        cache={},
        language=language,
        use_itn=use_itn,
        batch_size=batch_size,
    )


def timed_inference(
    model,
    audio_inputs,
    language: str,
    device: str,
    use_itn: bool,
    batch_size: int,
):
    sync_device(device)
    start = time.perf_counter()
    result = run_inference(model, audio_inputs, language, use_itn, batch_size)
    sync_device(device)
    elapsed = time.perf_counter() - start
    return elapsed, result


def print_results(result, use_rich: bool = True) -> None:
    from funasr.utils.postprocess_utils import rich_transcription_postprocess

    for i, item in enumerate(result):
        raw_text = item.get("text", "")
        text = rich_transcription_postprocess(raw_text) if use_rich else raw_text
        print(f"  [{i + 1}] {text}")


def print_benchmark_stats(
    times_ms: list[float],
    audio_paths: list[str],
    device: str,
    batch_size: int,
    warmup: int,
    runs: int,
) -> None:
    total_s = sum(times_ms) / 1000
    print("\n=== 推理时长统计 ===")
    print(f"音频数量: {len(audio_paths)}")
    for i, path in enumerate(audio_paths, 1):
        print(f"  [{i}] {path}")
    print(f"设备: {device}")
    print(f"batch_size: {batch_size}")
    print(f"预热次数: {warmup}")
    print(f"有效推理次数: {len(times_ms)}")
    print(
        f"单次推理 (ms): min={min(times_ms):.2f}, max={max(times_ms):.2f}, "
        f"avg={statistics.mean(times_ms):.2f}, p50={statistics.median(times_ms):.2f}"
    )
    if len(times_ms) >= 2:
        print(f"单次推理 (ms): stdev={statistics.stdev(times_ms):.2f}")
    print(f"总推理时长: {total_s:.3f} s")
    print(f"平均推理时长: {statistics.mean(times_ms):.2f} ms")
    print("\n各次推理时长 (ms):")
    for i, t in enumerate(times_ms, 1):
        print(f"  [{i:02d}] {t:.2f}")


def benchmark(
    model,
    audio_paths: list[str],
    language: str,
    device: str,
    use_itn: bool,
    batch_size: int,
    warmup: int,
    runs: int,
) -> None:
    audio_inputs = audio_paths[0] if len(audio_paths) == 1 else audio_paths

    print(f"预热推理 {warmup} 次...")
    for i in range(warmup):
        run_inference(model, audio_inputs, language, use_itn, batch_size)
        print(f"  warmup {i + 1}/{warmup} done")

    print(f"\n批量推理 {runs} 次（计时）...")
    times_ms: list[float] = []
    last_result = None
    for i in range(runs):
        elapsed, result = timed_inference(
            model, audio_inputs, language, device, use_itn, batch_size
        )
        times_ms.append(elapsed * 1000)
        last_result = result
        print(f"  run {i + 1}/{runs}: {elapsed * 1000:.2f} ms")

    print_benchmark_stats(times_ms, audio_paths, device, batch_size, warmup, runs)

    if last_result is not None:
        print("\n最后一次识别结果:")
        print_results(last_result)


def main() -> int:
    args = parse_args()

    if not os.path.isdir(args.model_dir):
        print(f"[ERROR] 模型目录不存在: {args.model_dir}", file=sys.stderr)
        return 1

    audio_paths = [os.path.abspath(p) for p in args.audio]
    for path in audio_paths:
        if not os.path.isfile(path):
            print(f"[ERROR] 音频文件不存在: {path}", file=sys.stderr)
            return 1

    from torchaudio_stub import install as install_torchaudio_stub

    install_torchaudio_stub()

    from funasr import AutoModel

    device = resolve_device(args.device)
    use_itn = args.use_itn and not args.no_itn

    print(f"模型目录 : {args.model_dir}")
    print(f"输入音频 : {len(audio_paths)} 个")
    for i, path in enumerate(audio_paths, 1):
        print(f"  [{i}] {path}")
    print(f"设备     : {device}")
    print(f"语言     : {args.language}")
    print(f"ITN      : {use_itn}")
    print(f"batch_size: {args.batch_size}")
    if not args.once:
        print(f"预热次数 : {args.warmup}")
        print(f"推理次数 : {args.runs}")
    print("-" * 50)

    load_start = time.perf_counter()
    model = AutoModel(
        model=args.model_dir,
        device=device,
        disable_update=True,
    )
    load_elapsed = time.perf_counter() - load_start
    print(f"模型加载耗时: {load_elapsed:.2f}s")

    if args.once:
        audio_inputs = audio_paths[0] if len(audio_paths) == 1 else audio_paths
        infer_start = time.perf_counter()
        result = run_inference(
            model, audio_inputs, args.language, use_itn, args.batch_size
        )
        infer_elapsed = time.perf_counter() - infer_start
        print(f"推理耗时  : {infer_elapsed:.3f}s")
        print("识别结果:")
        print_results(result)
        return 0

    benchmark(
        model,
        audio_paths,
        args.language,
        device,
        use_itn,
        args.batch_size,
        args.warmup,
        args.runs,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
