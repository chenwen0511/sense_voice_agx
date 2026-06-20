"""GPU 特征提取：kaldifeat（可选）或 torch kaldi.fbank + LFR + CMVN。"""

from __future__ import annotations

import math
import os
from typing import Tuple

import numpy as np
import torch
import torch.nn as nn
import yaml

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


class LFR(nn.Module):
    """Batch LFR，对齐 FunASR Triton feature_extractor。"""

    def __init__(self, m: int = 7, n: int = 6) -> None:
        super().__init__()
        self.m = m
        self.n = n
        self.left_padding_nums = math.ceil((self.m - 1) // 2)

    def forward(
        self, input_tensor: torch.Tensor, input_lens: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        B, _, D = input_tensor.size()
        n_lfr = torch.ceil(input_lens / self.n)

        prepad_nums = input_lens + self.left_padding_nums
        right_padding_nums = torch.where(
            self.m >= (prepad_nums - self.n * (n_lfr - 1)),
            self.m - (prepad_nums - self.n * (n_lfr - 1)),
            0,
        )

        T_all = self.left_padding_nums + input_lens + right_padding_nums
        new_len = T_all // self.n
        T_all_max = T_all.max().int()

        tail_frames_index = (input_lens - 1).view(B, 1, 1).repeat(1, 1, D)
        tail_frames = torch.gather(input_tensor, 1, tail_frames_index)
        tail_frames = tail_frames.repeat(1, right_padding_nums.max().int(), 1)
        head_frames = input_tensor[:, 0:1, :].repeat(1, self.left_padding_nums, 1)

        input_tensor = torch.cat([head_frames, input_tensor, tail_frames], dim=1)

        index = (
            torch.arange(T_all_max, device=input_tensor.device, dtype=input_lens.dtype)
            .unsqueeze(0)
            .repeat(B, 1)
        )
        index_mask = index < (self.left_padding_nums + input_lens).unsqueeze(1)
        tail_index_mask = torch.logical_not(index >= (T_all.unsqueeze(1))) & index_mask
        tail = (
            torch.ones(T_all_max, dtype=input_lens.dtype, device=input_tensor.device)
            .unsqueeze(0)
            .repeat(B, 1)
            * (T_all_max - 1)
        )
        indices = torch.where(
            torch.logical_or(index_mask, tail_index_mask), index, tail
        )
        input_tensor = torch.gather(
            input_tensor, 1, indices.unsqueeze(2).repeat(1, 1, D)
        )

        input_tensor = input_tensor.unfold(1, self.m, step=self.n).transpose(2, 3)
        return input_tensor.reshape(B, -1, D * self.m), new_len


def _load_cmvn(cmvn_file: str) -> np.ndarray:
    means_list: list[str] = []
    vars_list: list[str] = []
    with open(cmvn_file, "r", encoding="utf-8") as f:
        lines = f.readlines()
    for i, line in enumerate(lines):
        parts = line.split()
        if parts[0] == "<AddShift>":
            next_parts = lines[i + 1].split()
            if next_parts[0] == "<LearnRateCoef>":
                means_list = next_parts[3 : len(next_parts) - 1]
        elif parts[0] == "<Rescale>":
            next_parts = lines[i + 1].split()
            if next_parts[0] == "<LearnRateCoef>":
                vars_list = next_parts[3 : len(next_parts) - 1]
    means = np.array(means_list).astype(np.float64)
    vars_ = np.array(vars_list).astype(np.float64)
    return np.array([means, vars_])


def _apply_cmvn_batch(inputs: torch.Tensor, cmvn: np.ndarray) -> torch.Tensor:
    batch, frame, dim = inputs.shape
    means = np.tile(cmvn[0:1, :dim], (frame, 1))
    vars_ = np.tile(cmvn[1:2, :dim], (frame, 1))
    means_t = torch.from_numpy(means).to(device=inputs.device, dtype=inputs.dtype)
    vars_t = torch.from_numpy(vars_).to(device=inputs.device, dtype=inputs.dtype)
    return (inputs + means_t) * vars_t


class GpuFeatureExtractor:
    """SenseVoice GPU 特征提取（Fbank → LFR → CMVN）。"""

    def __init__(self, model_dir: str, device: str = "cuda:0") -> None:
        self.model_dir = model_dir
        self.device = torch.device(device)

        config_path = os.path.join(model_dir, "config.yaml")
        with open(config_path, "r", encoding="utf-8") as f:
            config = yaml.safe_load(f)
        fc = config["frontend_conf"]

        self.fs = int(fc["fs"])
        self.n_mels = int(fc["n_mels"])
        self.frame_length = float(fc["frame_length"])
        self.frame_shift = float(fc["frame_shift"])
        self.window = fc["window"]
        self.lfr_m = int(fc["lfr_m"])
        self.lfr_n = int(fc["lfr_n"])
        self.dither = float(fc.get("dither", 1.0))
        self.upsacle_samples = True

        cmvn_path = os.path.join(model_dir, "am.mvn")
        self.cmvn = _load_cmvn(cmvn_path)
        self.lfr = LFR(self.lfr_m, self.lfr_n).to(self.device)

        self._use_kaldifeat = False
        self._fbank = None
        try:
            import kaldifeat

            opts = kaldifeat.FbankOptions()
            opts.frame_opts.samp_freq = self.fs
            opts.frame_opts.frame_length_ms = self.frame_length
            opts.frame_opts.frame_shift_ms = self.frame_shift
            opts.frame_opts.dither = self.dither
            opts.frame_opts.window_type = self.window
            opts.mel_opts.num_bins = self.n_mels
            opts.device = self.device
            self._fbank = kaldifeat.Fbank(opts)
            self._use_kaldifeat = True
        except ImportError:
            import sys

            if _PROJECT_ROOT not in sys.path:
                sys.path.insert(0, _PROJECT_ROOT)
            from torchaudio_stub import install as install_stub

            install_stub()
            from torchaudio_stub.compliance import kaldi

            self._kaldi = kaldi

        self._warmup()

    def _warmup(self, steps: int = 3) -> None:
        """预热 CUDA kernel，避免首次推理计时虚高。"""
        dummy = torch.zeros(self.fs, device=self.device)
        for _ in range(steps):
            self.extract_from_waveform(dummy)
        torch.cuda.synchronize()

    @property
    def backend(self) -> str:
        return "kaldifeat" if self._use_kaldifeat else "torch-kaldi-gpu"

    def _fbank_gpu(self, waveform: torch.Tensor) -> torch.Tensor:
        if self._use_kaldifeat:
            return self._fbank([waveform])[0]

        w = waveform.unsqueeze(0)
        frame_len = min(self.frame_length, waveform.shape[0] / self.fs * 1000)
        return self._kaldi.fbank(
            w,
            num_mel_bins=self.n_mels,
            frame_length=frame_len,
            frame_shift=self.frame_shift,
            dither=self.dither,
            energy_floor=0.0,
            window_type=self.window,
            sample_frequency=self.fs,
            snip_edges=True,
        )

    def extract_from_waveform(
        self, waveform: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        wav = waveform.float().to(self.device)
        if self.upsacle_samples:
            wav = wav * (1 << 15)

        features = self._fbank_gpu(wav)
        feat_len = torch.tensor([features.shape[0]], dtype=torch.int64, device=self.device)

        max_len = features.shape[0]
        speech = torch.zeros(
            (1, max_len, self.n_mels), dtype=torch.float32, device=self.device
        )
        speech[0, :max_len] = features

        feats, feats_len = self.lfr(speech, feat_len)
        feats_len = feats_len.type(torch.int32)
        feats = _apply_cmvn_batch(feats, self.cmvn)
        return feats, feats_len

    def extract_from_audio_file(
        self, audio_path: str, audio_loader
    ) -> Tuple[torch.Tensor, torch.Tensor, float, float]:
        import time

        t0 = time.perf_counter()
        waveform = audio_loader(
            audio_path,
            fs=self.fs,
            audio_fs=self.fs,
            data_type="sound",
        )
        t1 = time.perf_counter()
        feats, feats_len = self.extract_from_waveform(waveform)
        torch.cuda.synchronize()
        t2 = time.perf_counter()
        return feats, feats_len, (t1 - t0) * 1000, (t2 - t1) * 1000
