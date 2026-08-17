"""Mogwai's ears: USB microphone capture.

Works the same on macOS (CoreAudio) and Raspberry Pi OS (ALSA) because
PortAudio hides the difference and the mic is picked by name, not index --
indices are not stable across machines or reboots.

CLI:
    python -m mogwai.ears list
    python -m mogwai.ears meter --seconds 10
    python -m mogwai.ears record out.wav --seconds 5
"""

from __future__ import annotations

import argparse
import os
import sys
import wave
from dataclasses import dataclass

import numpy as np
import sounddevice as sd

# Substrings we look for, in priority order. The cheap C-Media USB mics report
# "USB PnP Sound Device" on both macOS and Pi OS; the rest are common variants.
MIC_HINTS = ("USB PnP", "USB Audio", "USB Microphone", "USB")

# What downstream speech models want. Capture may happen at a higher native
# rate and get resampled -- see resample().
TARGET_RATE = 16000

SILENCE_DBFS = -70.0  # RMS below this is treated as "no signal at all"


@dataclass(frozen=True)
class Mic:
    index: int
    name: str
    channels: int
    native_rate: int

    def __str__(self) -> str:
        return f"[{self.index}] {self.name} ({self.channels}ch @ {self.native_rate} Hz)"


def list_inputs() -> list[Mic]:
    """Every device that can capture audio."""
    mics = []
    for i, dev in enumerate(sd.query_devices()):
        if dev["max_input_channels"] > 0:
            mics.append(
                Mic(
                    index=i,
                    name=dev["name"].strip(),
                    channels=dev["max_input_channels"],
                    native_rate=int(dev["default_samplerate"]),
                )
            )
    return mics


def find_mic(hint: str | None = None) -> Mic:
    """Pick the USB mic, or whatever `hint` (or $MOGWAI_MIC) names.

    Falls back to the system default input so a bare Pi with no USB mic
    plugged in still gives a useful error rather than an IndexError.
    """
    hint = hint or os.environ.get("MOGWAI_MIC")
    inputs = list_inputs()
    if not inputs:
        raise RuntimeError("No audio input devices found at all.")

    if hint:
        for mic in inputs:
            if hint.lower() in mic.name.lower():
                return mic
        raise RuntimeError(
            f"No input device matching {hint!r}. Available:\n"
            + "\n".join(f"  {m}" for m in inputs)
        )

    for needle in MIC_HINTS:
        for mic in inputs:
            if needle.lower() in mic.name.lower():
                return mic

    default_index = sd.default.device[0]
    for mic in inputs:
        if mic.index == default_index:
            return mic
    return inputs[0]


def dbfs(block: np.ndarray) -> float:
    """RMS level of a float32 block, in dBFS (0 = full scale)."""
    if block.size == 0:
        return -np.inf
    rms = float(np.sqrt(np.mean(np.square(block, dtype=np.float64))))
    if rms <= 0:
        return -np.inf
    return 20.0 * np.log10(rms)


def resample(audio: np.ndarray, src_rate: int, dst_rate: int) -> np.ndarray:
    """Rate-convert mono float32 audio.

    Integer ratios (the common 48k -> 16k) use a boxcar average, which both
    decimates and low-passes, so we do not alias. Everything else falls back
    to linear interpolation.
    """
    if src_rate == dst_rate:
        return audio
    if src_rate % dst_rate == 0:
        factor = src_rate // dst_rate
        usable = (audio.size // factor) * factor
        return audio[:usable].reshape(-1, factor).mean(axis=1).astype(np.float32)
    duration = audio.size / src_rate
    dst_len = int(duration * dst_rate)
    src_t = np.arange(audio.size) / src_rate
    dst_t = np.arange(dst_len) / dst_rate
    return np.interp(dst_t, src_t, audio).astype(np.float32)


def record(
    seconds: float,
    mic: Mic | None = None,
    rate: int = TARGET_RATE,
    gain: float = 1.0,
) -> tuple[np.ndarray, int]:
    """Capture `seconds` of mono audio, returned as float32 in [-1, 1].

    Captures at the mic's native rate and resamples, rather than asking
    PortAudio for 16 kHz -- USB mics routinely refuse rates they do not
    support natively, and the failure mode is a cryptic PortAudio error.
    """
    mic = mic or find_mic()
    frames = int(seconds * mic.native_rate)
    raw = sd.rec(
        frames,
        samplerate=mic.native_rate,
        channels=1,
        dtype="float32",
        device=mic.index,
    )
    sd.wait()
    audio = raw[:, 0]
    if gain != 1.0:
        audio = np.clip(audio * gain, -1.0, 1.0)
    return resample(audio, mic.native_rate, rate), rate


def write_wav(path: str, audio: np.ndarray, rate: int) -> None:
    """Write mono float32 audio as a 16-bit PCM WAV."""
    pcm = (np.clip(audio, -1.0, 1.0) * 32767).astype("<i2")
    with wave.open(path, "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(rate)
        wav.writeframes(pcm.tobytes())


def meter(seconds: float, mic: Mic | None = None, gain: float = 1.0) -> float:
    """Live level meter. Returns the peak dBFS seen, for scripted checks."""
    mic = mic or find_mic()
    block = int(mic.native_rate * 0.1)  # 100 ms refresh
    peak = -np.inf
    print(f"Listening on {mic}\nCtrl-C to stop.\n", file=sys.stderr)

    with sd.InputStream(
        samplerate=mic.native_rate,
        channels=1,
        dtype="float32",
        blocksize=block,
        device=mic.index,
    ) as stream:
        for _ in range(int(seconds / 0.1)):
            data, overflowed = stream.read(block)
            level = dbfs(data[:, 0] * gain)
            peak = max(peak, level)
            # Map -60..0 dBFS onto a 40-column bar.
            filled = 0 if level == -np.inf else int(np.clip((level + 60) / 60, 0, 1) * 40)
            bar = "#" * filled + "-" * (40 - filled)
            flag = " OVERFLOW" if overflowed else ""
            print(f"\r{bar} {level:6.1f} dBFS{flag}", end="", file=sys.stderr, flush=True)
    print(file=sys.stderr)
    return peak


def check(mic: Mic | None = None, timeout: float = 15.0, gain: float = 1.0) -> bool:
    """Measure the noise floor, then wait for the user to make a sound.

    The useful number is not the absolute level but the gap between speech
    and the floor: under ~15 dB of headroom, speech recognition struggles no
    matter how loud things sound to a human.
    """
    mic = mic or find_mic()
    block = int(mic.native_rate * 0.1)

    with sd.InputStream(
        samplerate=mic.native_rate,
        channels=1,
        dtype="float32",
        blocksize=block,
        device=mic.index,
    ) as stream:
        stream.read(block)  # discard the stream-open transient
        print(f"Using {mic}", file=sys.stderr)
        print("Measuring noise floor -- stay quiet for 2s...", file=sys.stderr)
        floor_blocks = [dbfs(stream.read(block)[0][:, 0] * gain) for _ in range(20)]
        floor = float(np.median(floor_blocks))
        print(f"Noise floor: {floor:.1f} dBFS", file=sys.stderr)

        if floor < SILENCE_DBFS:
            print("Digital silence -- the mic is not delivering samples.", file=sys.stderr)
            return False

        threshold = floor + 15.0
        print(f"\nNow SAY SOMETHING (need > {threshold:.0f} dBFS)...", file=sys.stderr)
        peak = -np.inf
        for _ in range(int(timeout / 0.1)):
            level = dbfs(stream.read(block)[0][:, 0] * gain)
            peak = max(peak, level)
            filled = 0 if level == -np.inf else int(np.clip((level + 60) / 60, 0, 1) * 40)
            mark = "#" * filled + "-" * (40 - filled)
            print(f"\r{mark} {level:6.1f} dBFS", end="", file=sys.stderr, flush=True)
            if level > threshold:
                snr = level - floor
                print(f"\n\nHeard you: {level:.1f} dBFS, {snr:.0f} dB above the floor.", file=sys.stderr)
                if snr < 25:
                    print("Usable, but quiet -- move closer or raise --gain.", file=sys.stderr)
                print("Microphone works.", file=sys.stderr)
                return True

    print(f"\n\nHeard nothing above {threshold:.0f} dBFS (peak was {peak:.1f}).", file=sys.stderr)
    return False


def _main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="mogwai.ears", description="Mogwai microphone tools")
    parser.add_argument("--mic", help="substring of the input device name")
    parser.add_argument("--gain", type=float, default=1.0, help="linear gain multiplier")
    sub = parser.add_subparsers(dest="cmd", required=True)

    sub.add_parser("list", help="show input devices")

    p_check = sub.add_parser("check", help="interactive works-or-not test")
    p_check.add_argument("--timeout", type=float, default=15.0)

    p_meter = sub.add_parser("meter", help="live input level meter")
    p_meter.add_argument("--seconds", type=float, default=10.0)

    p_rec = sub.add_parser("record", help="record to a WAV file")
    p_rec.add_argument("path")
    p_rec.add_argument("--seconds", type=float, default=5.0)
    p_rec.add_argument("--rate", type=int, default=TARGET_RATE)

    args = parser.parse_args(argv)

    if args.cmd == "list":
        chosen = find_mic(args.mic)
        for m in list_inputs():
            mark = "->" if m.index == chosen.index else "  "
            print(f"{mark} {m}")
        return 0

    mic = find_mic(args.mic)

    if args.cmd == "check":
        return 0 if check(mic, args.timeout, args.gain) else 1

    if args.cmd == "meter":
        peak = meter(args.seconds, mic, args.gain)
        print(f"peak: {peak:.1f} dBFS")
        return 0

    if args.cmd == "record":
        print(f"Recording {args.seconds}s from {mic}", file=sys.stderr)
        audio, rate = record(args.seconds, mic, args.rate, args.gain)
        write_wav(args.path, audio, rate)
        level = dbfs(audio)
        print(f"wrote {args.path}: {audio.size / rate:.2f}s @ {rate} Hz, RMS {level:.1f} dBFS")
        if level < SILENCE_DBFS:
            print(
                "warning: that is digital silence -- check the mic is selected "
                "and that this terminal has microphone permission.",
                file=sys.stderr,
            )
            return 1
        return 0

    return 2


if __name__ == "__main__":
    raise SystemExit(_main())
