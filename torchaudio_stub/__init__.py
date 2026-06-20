"""Minimal torchaudio stub for Jetson PyTorch + FunASR."""

import types
import sys

import torch
import soundfile as sf


def load(path, *args, **kwargs):
    data, sr = sf.read(path, dtype="float32")
    tensor = torch.from_numpy(data)
    if tensor.ndim == 1:
        tensor = tensor.unsqueeze(0)
    elif tensor.ndim == 2:
        tensor = tensor.T
    return tensor, sr


class Resample:
    def __init__(self, orig_freq: int, new_freq: int):
        self.orig_freq = orig_freq
        self.new_freq = new_freq

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        if self.orig_freq == self.new_freq:
            return x
        if x.ndim == 1:
            x = x.unsqueeze(0)
        ratio = self.new_freq / self.orig_freq
        new_len = max(1, int(round(x.shape[-1] * ratio)))
        y = torch.nn.functional.interpolate(
            x.unsqueeze(0),
            size=new_len,
            mode="linear",
            align_corners=False,
        )
        return y.squeeze(0)


transforms = types.ModuleType("torchaudio.transforms")
transforms.Resample = Resample

compliance = types.ModuleType("torchaudio.compliance")
compliance.kaldi = None  # set after import


def install() -> None:
    if "torchaudio" in sys.modules and hasattr(sys.modules["torchaudio"], "_is_stub"):
        return

    from torchaudio_stub.compliance import kaldi

    compliance.kaldi = kaldi

    stub = types.ModuleType("torchaudio")
    stub._is_stub = True
    stub.load = load
    stub.transforms = transforms

    functional = types.ModuleType("torchaudio.functional")

    def create_dct(num_input, num_output, norm="ortho"):
        import math

        n = torch.arange(float(num_input))
        k = torch.arange(float(num_output))
        dct = torch.cos(math.pi / float(num_input) * (n + 0.5) * k.unsqueeze(1))
        if norm == "ortho":
            dct[0] *= math.sqrt(1.0 / float(num_input))
            dct[1:] *= math.sqrt(2.0 / float(num_input))
        return dct

    functional.create_dct = create_dct
    functional.istft = torch.functional.istft if hasattr(torch.functional, "istft") else None

    stub.functional = functional
    stub.compliance = compliance

    sys.modules["torchaudio"] = stub
    sys.modules["torchaudio.transforms"] = transforms
    sys.modules["torchaudio.compliance"] = compliance
    sys.modules["torchaudio.compliance.kaldi"] = kaldi
    sys.modules["torchaudio.functional"] = functional
