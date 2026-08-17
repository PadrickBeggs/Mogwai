"""Mogwai's voice: text-to-speech via espeak-ng.

Piper (a neural TTS engine) was the first pass here, but it's built to sound
human -- not what a small robotic creature wants. espeak-ng is the classic
formant/klatt synthesizer: deliberately synthetic-sounding, tiny, instant on
CPU, and it ships a "klatt3" voice variant that already sounds mechanical
before any tuning. Install it with `brew install espeak-ng` (macOS) or
`sudo apt install espeak-ng` (Raspberry Pi OS) -- there's no Python
dependency, this module just shells out to the `espeak-ng` binary.

Tuning knobs, chosen by ear against this project's test phrase:
  VARIANT   which formant voice (see "Picking a different variant" below)
  PITCH     espeak-ng's own -p, 0-99; klatt3 was already deep at 0
  DEPTH     post-synthesis pitch-down, independent of PITCH (see below)
  WPM       perceived words-per-minute *after* the DEPTH shift is applied

DEPTH shifting: naively resampling audio to a lower pitch also slows it
down (the "slowed tape" effect) -- fine for a robot, but it drifts from a
natural speaking pace the more you push it. To keep DEPTH and pace
independent, synthesis speed is pre-compensated by 1/DEPTH (faster than WPM)
before the resample, so the two effects cancel out in duration and only the
pitch drop remains. See `_pitch_down()` and `synthesize()`.

CLI:
    python -m mogwai.voice say "hello there"
    python -m mogwai.voice say "hello there" --variant croak --pitch 20 --depth 1.0
    python -m mogwai.voice say "hello there" --out hello.wav
"""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
import wave
from io import BytesIO

import numpy as np
import sounddevice as sd

# Defaults settled on by ear: klatt3 is already a deep, mechanical formant
# voice; DEPTH=0.75 pushes it further into "raspy and lower" without
# becoming unintelligible.
DEFAULT_VARIANT = os.environ.get("MOGWAI_VOICE_VARIANT", "klatt3")
DEFAULT_LANG = os.environ.get("MOGWAI_VOICE_LANG", "en-us")
DEFAULT_PITCH = int(os.environ.get("MOGWAI_VOICE_PITCH", "0"))  # espeak-ng's -p, 0-99
DEFAULT_DEPTH = float(os.environ.get("MOGWAI_VOICE_DEPTH", "0.75"))  # 1.0 = no extra shift
DEFAULT_WPM = int(os.environ.get("MOGWAI_VOICE_WPM", "140"))  # perceived pace, after DEPTH is applied

# A few variants worth trying (`espeak-ng --voices=variant` lists all ~100).
# croak/Demonic/klatt1-6/robosoft1-8/UniRobot lean mechanical or raspy;
# m1-m8/f1-f5 are plain adult formant voices; grandpa/grandma are gravelly
# but more human. Pick with --variant.
KNOWN_VARIANTS = ["klatt3", "croak", "Demonic", "robosoft3", "m3", "grandpa"]


def _espeak_binary() -> str:
    path = shutil.which("espeak-ng") or shutil.which("espeak")
    if not path:
        raise RuntimeError(
            "espeak-ng not found on PATH. Install it with `brew install espeak-ng` "
            "(macOS) or `sudo apt install espeak-ng` (Raspberry Pi OS)."
        )
    return path


def _read_wav_bytes(data: bytes) -> tuple[np.ndarray, int]:
    with wave.open(BytesIO(data), "rb") as wav:
        rate = wav.getframerate()
        channels = wav.getnchannels()
        raw = wav.readframes(wav.getnframes())
    pcm = np.frombuffer(raw, dtype="<i2").astype(np.float32) / 32768.0
    if channels > 1:
        pcm = pcm.reshape(-1, channels).mean(axis=1)
    return pcm, rate


def _pitch_down(audio: np.ndarray, depth: float) -> np.ndarray:
    """Stretch the waveform to drop pitch by ~`depth` (1.0 = no change).

    This also slows playback proportionally -- callers that want to keep pace
    constant should pre-synthesize at wpm/depth (see `synthesize()`).
    """
    if depth == 1.0:
        return audio
    n_new = max(1, int(len(audio) / depth))
    old_index = np.arange(len(audio))
    new_index = np.linspace(0, len(audio) - 1, n_new)
    return np.interp(new_index, old_index, audio).astype(np.float32)


def synthesize(
    text: str,
    variant: str = DEFAULT_VARIANT,
    lang: str = DEFAULT_LANG,
    pitch: int = DEFAULT_PITCH,
    depth: float = DEFAULT_DEPTH,
    wpm: int = DEFAULT_WPM,
) -> tuple[np.ndarray, int]:
    """Text -> mono float32 audio in [-1, 1] plus its sample rate."""
    synth_wpm = int(round(wpm / depth)) if depth else wpm
    cmd = [
        _espeak_binary(),
        "-v", f"{lang}+{variant}",
        "-p", str(pitch),
        "-s", str(synth_wpm),
        "--stdout",
        text,
    ]
    result = subprocess.run(cmd, capture_output=True, check=True)
    audio, rate = _read_wav_bytes(result.stdout)
    return _pitch_down(audio, depth), rate


def say(
    text: str,
    volume: float = 1.0,
    block: bool = True,
    **synth_kwargs,
) -> None:
    """Synthesize and play text through the default output device."""
    audio, rate = synthesize(text, **synth_kwargs)
    if volume != 1.0:
        audio = np.clip(audio * volume, -1.0, 1.0)
    sd.play(audio, samplerate=rate)
    if block:
        sd.wait()


def write_wav(path: str, audio: np.ndarray, rate: int) -> None:
    pcm = (np.clip(audio, -1.0, 1.0) * 32767).astype("<i2")
    with wave.open(path, "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(rate)
        wav.writeframes(pcm.tobytes())


def _main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="mogwai.voice", description="Mogwai text-to-speech (espeak-ng)")
    sub = parser.add_subparsers(dest="cmd", required=True)

    sub.add_parser("variants", help="list a few known-good variant names (espeak-ng has ~100 in total)")

    p_say = sub.add_parser("say", help="synthesize and speak text")
    p_say.add_argument("text")
    p_say.add_argument("--variant", default=DEFAULT_VARIANT)
    p_say.add_argument("--lang", default=DEFAULT_LANG)
    p_say.add_argument("--pitch", type=int, default=DEFAULT_PITCH, help="espeak-ng's own pitch, 0-99")
    p_say.add_argument("--depth", type=float, default=DEFAULT_DEPTH, help="post-synthesis pitch-down; 1.0 = off")
    p_say.add_argument("--wpm", type=int, default=DEFAULT_WPM, help="perceived words/minute, after --depth")
    p_say.add_argument("--volume", type=float, default=1.0)
    p_say.add_argument("--out", help="write to a WAV file instead of (or in addition to) playing it")
    p_say.add_argument("--no-play", action="store_true", help="skip playback (only useful with --out)")

    args = parser.parse_args(argv)

    if args.cmd == "variants":
        for v in KNOWN_VARIANTS:
            print(v)
        print("(run `espeak-ng --voices=variant` for the full list of ~100)", file=sys.stderr)
        return 0

    if args.cmd == "say":
        audio, rate = synthesize(args.text, args.variant, args.lang, args.pitch, args.depth, args.wpm)
        if args.volume != 1.0:
            audio = np.clip(audio * args.volume, -1.0, 1.0)
        if args.out:
            write_wav(args.out, audio, rate)
            print(f"wrote {args.out}", file=sys.stderr)
        if not args.no_play:
            sd.play(audio, samplerate=rate)
            sd.wait()
        return 0

    return 2


if __name__ == "__main__":
    raise SystemExit(_main())
