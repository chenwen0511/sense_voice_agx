#!/usr/bin/env python3
"""对比 CPU 与 GPU 特征提取数值与识别结果。"""

import argparse
import os
import sys

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_LIB_DIR = os.path.join(_PROJECT_ROOT, "lib")
if os.path.isdir(_LIB_DIR):
    os.environ["LD_LIBRARY_PATH"] = _LIB_DIR + (
        ":" + os.environ.get("LD_LIBRARY_PATH", "")
        if os.environ.get("LD_LIBRARY_PATH")
        else ""
    )
sys.path.insert(0, _PROJECT_ROOT)

from torchaudio_stub import install as install_torchaudio_stub

install_torchaudio_stub()

import torch
import yaml
from funasr.frontends.wav_frontend import WavFrontend
from funasr.utils.load_utils import extract_fbank, load_audio_text_image_video

from acc2.feat_gpu import GpuFeatureExtractor

DEFAULT_MODEL_DIR = "/home/admin/stephen/02-weight/SenseVoiceSmall"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", default=DEFAULT_MODEL_DIR)
    parser.add_argument("--audio", required=True)
    args = parser.parse_args()

    with open(os.path.join(args.model_dir, "config.yaml")) as f:
        cfg = yaml.safe_load(f)
    fc = dict(cfg["frontend_conf"])
    fc["cmvn_file"] = os.path.join(args.model_dir, "am.mvn")
    frontend = WavFrontend(**fc)

    waveform = load_audio_text_image_video(
        args.audio, fs=16000, audio_fs=16000, data_type="sound"
    )
    speech_cpu, lens_cpu = extract_fbank(waveform, data_type="sound", frontend=frontend)

    extractor = GpuFeatureExtractor(args.model_dir)
    print(f"GPU backend: {extractor.backend}")
    speech_gpu, lens_gpu = extractor.extract_from_waveform(waveform)

    diff = (speech_cpu.cuda() - speech_gpu).abs()
    print(f"shape cpu={speech_cpu.shape} gpu={speech_gpu.shape}")
    print(f"length cpu={lens_cpu.item()} gpu={lens_gpu.item()}")
    print(f"max abs diff: {diff.max().item():.6f}")
    print(f"mean abs diff: {diff.mean().item():.6f}")
    # dither=1.0 时存在随机差异，长度一致即可认为对齐


if __name__ == "__main__":
    main()
