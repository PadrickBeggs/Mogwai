"""Mogwai's voice recognition: speech-to-text via faster-whisper.

faster-whisper is a CTranslate2 reimplementation of OpenAI's Whisper --
same accuracy, several times faster and lighter on CPU, which is what
matters running alongside everything else on a Pi 5. Models auto-download
from Hugging Face on first use and are cached under ~/.cache/huggingface.

CLI:
    python -m mogwai.stt transcribe clip.wav
    python -m mogwai.stt listen --seconds 5       # record from mic, then transcribe
"""

from __future__ import annotations

import argparse
import os
import sys
import wave

import numpy as np

from mogwai.ears import TARGET_RATE, dbfs, find_mic, record, resample

# "*.en" models are English-only and noticeably more accurate than the
# multilingual models at the same size, since they don't spend capacity on
# language identification. base.en is a reasonable default for a Pi 5 CPU;
# tiny.en trades accuracy for speed, small.en trades speed for accuracy.
DEFAULT_MODEL = os.environ.get("MOGWAI_STT_MODEL", "base.en")

# int8 quantization is 2-4x faster than float32 on CPU with minimal accuracy
# loss at these model sizes -- the difference that matters on ARM without a GPU.
DEFAULT_COMPUTE_TYPE = os.environ.get("MOGWAI_STT_COMPUTE", "int8")

SILENCE_DBFS = -70.0

_model_cache: dict[tuple, object] = {}


def load_model(model_size: str = DEFAULT_MODEL, compute_type: str = DEFAULT_COMPUTE_TYPE, device: str = "cpu"):
    """Load (and cache) a faster-whisper model. First call downloads it."""
    key = (model_size, compute_type, device)
    if key not in _model_cache:
        from faster_whisper import WhisperModel

        _model_cache[key] = WhisperModel(model_size, device=device, compute_type=compute_type)
    return _model_cache[key]


def read_wav(path: str) -> tuple[np.ndarray, int]:
    """Read a WAV file as mono float32 in [-1, 1], whatever its native format."""
    with wave.open(path, "rb") as wav:
        rate = wav.getframerate()
        channels = wav.getnchannels()
        raw = wav.readframes(wav.getnframes())
    pcm = np.frombuffer(raw, dtype="<i2").astype(np.float32) / 32768.0
    if channels > 1:
        pcm = pcm.reshape(-1, channels).mean(axis=1)
    return pcm, rate


def transcribe(audio: np.ndarray, rate: int = TARGET_RATE, model=None, **kwargs) -> str:
    """Transcribe mono float32 audio at any sample rate (resampled to 16 kHz internally).

    faster-whisper treats a raw ndarray as already-decoded audio at the rate
    you claim -- unlike a file path, it does not resample for you, so getting
    this wrong silently produces garbage transcriptions rather than an error.
    """
    model = model or load_model()
    if rate != TARGET_RATE:
        audio = resample(audio, rate, TARGET_RATE)
    segments, _info = model.transcribe(
        audio,
        language="en",
        vad_filter=True,  # skip non-speech segments instead of hallucinating text over them
        without_timestamps=True,
        **kwargs,
    )
    return "".join(seg.text for seg in segments).strip()


def transcribe_file(path: str, model=None, **kwargs) -> str:
    audio, rate = read_wav(path)
    return transcribe(audio, rate, model, **kwargs)


def listen_and_transcribe(
    seconds: float = 5.0,
    mic=None,
    gain: float = 1.0,
    model=None,
) -> str:
    """Record `seconds` from the mic and transcribe it. Blocking."""
    mic = mic or find_mic()
    model = model or load_model()
    audio, rate = record(seconds, mic, TARGET_RATE, gain)
    if dbfs(audio) < SILENCE_DBFS:
        print("warning: near-silence captured -- check the mic before trusting this result", file=sys.stderr)
    return transcribe(audio, rate, model)


def _main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="mogwai.stt", description="Mogwai speech-to-text")
    parser.add_argument("--model", default=DEFAULT_MODEL, help="whisper model size (tiny.en, base.en, small.en, ...)")
    parser.add_argument("--compute-type", default=DEFAULT_COMPUTE_TYPE)
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_file = sub.add_parser("transcribe", help="transcribe a WAV file")
    p_file.add_argument("path")

    p_listen = sub.add_parser("listen", help="record from the mic, then transcribe")
    p_listen.add_argument("--seconds", type=float, default=5.0)
    p_listen.add_argument("--mic", help="substring of the input device name")
    p_listen.add_argument("--gain", type=float, default=1.0)

    args = parser.parse_args(argv)
    print(f"Loading {args.model} ({args.compute_type})...", file=sys.stderr)
    model = load_model(args.model, args.compute_type)

    if args.cmd == "transcribe":
        print(transcribe_file(args.path, model))
        return 0

    if args.cmd == "listen":
        mic = find_mic(args.mic)
        print(f"Recording {args.seconds}s from {mic}...", file=sys.stderr)
        print(listen_and_transcribe(args.seconds, mic, args.gain, model))
        return 0

    return 2


if __name__ == "__main__":
    raise SystemExit(_main())
