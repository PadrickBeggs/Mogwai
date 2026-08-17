"""Mogwai's wake word: continuous listening via openWakeWord.

openWakeWord only detects a wake phrase ("hey jarvis", "alexa", ...) in a
live audio stream -- it does not transcribe speech. Full speech-to-text of
what is said *after* the wake word is a separate component (e.g. Vosk or
faster-whisper) layered on top of this one.

CLI:
    python -m mogwai.wake models                       # list available models
    python -m mogwai.wake listen --model hey_jarvis     # print scores live
    python -m mogwai.wake listen --model hey_jarvis --clip-dir clips/
"""

from __future__ import annotations

import argparse
import os
import sys
import time
import wave

import numpy as np
import sounddevice as sd

from mogwai.ears import Mic, TARGET_RATE, dbfs, find_mic, resample

FRAME_MS = 80  # openWakeWord's native frame size; longer frames cost less CPU
FRAME_SAMPLES = int(TARGET_RATE * FRAME_MS / 1000)  # 1280 @ 16 kHz

DEFAULT_THRESHOLD = 0.5

# Custom-trained models (e.g. a "mogwai" wake word from the openWakeWord
# training notebook) live here as <name>.tflite / <name>.onnx, alongside
# whatever ships with the library itself.
CUSTOM_MODELS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models")


def _bundled_model_dir() -> str:
    import openwakeword

    return os.path.join(os.path.dirname(openwakeword.__file__), "resources", "models")


def _bundled_models() -> list[str]:
    """Wake-word names bundled with openWakeWord (e.g. 'hey_jarvis', 'alexa')."""
    names = set()
    d = _bundled_model_dir()
    if not os.path.isdir(d):
        return []
    for f in os.listdir(d):
        if f.endswith((".onnx", ".tflite")) and f not in (
            "embedding_model.onnx",
            "embedding_model.tflite",
            "melspectrogram.onnx",
            "melspectrogram.tflite",
            "silero_vad.onnx",
        ):
            names.add(f.rsplit("_v", 1)[0])
    return sorted(names)


def _custom_models() -> dict[str, str]:
    """Custom model names -> file path, from CUSTOM_MODELS_DIR."""
    found = {}
    if not os.path.isdir(CUSTOM_MODELS_DIR):
        return found
    for f in sorted(os.listdir(CUSTOM_MODELS_DIR)):
        if f.endswith((".onnx", ".tflite")):
            found.setdefault(os.path.splitext(f)[0], os.path.join(CUSTOM_MODELS_DIR, f))
    return found


def available_models() -> list[str]:
    """All wake-word names usable with --model: bundled plus custom-trained."""
    return sorted(set(_bundled_models()) | set(_custom_models().keys()))


def resolve_model(name: str, inference_framework: str = "tflite") -> str:
    """Turn a wake-word name into whatever openWakeWord's Model() expects.

    Bundled names are passed through as-is -- openWakeWord resolves those
    itself. Custom names are resolved to a file path here, matching the
    active inference framework's extension (a mismatched extension raises
    inside openWakeWord with a confusing error otherwise).
    """
    if name in _bundled_models():
        return name

    custom = _custom_models()
    ext = ".tflite" if inference_framework == "tflite" else ".onnx"
    path = os.path.join(CUSTOM_MODELS_DIR, name + ext)
    if os.path.exists(path):
        return path
    if name in custom:
        # Only the other extension is present; still usable if the caller
        # sets inference_framework to match its actual format.
        return custom[name]

    known = ", ".join(available_models()) or "(none found)"
    raise ValueError(
        f"No wake-word model named {name!r}. Known: {known}. "
        f"Drop a trained model at {os.path.join(CUSTOM_MODELS_DIR, name + ext)} to add it."
    )


def load_model(
    model_names: list[str] | None = None,
    vad_threshold: float = 0.0,
    inference_framework: str = "tflite",
):
    """Load an openWakeWord Model for the given wake words (default: all bundled)."""
    from openwakeword.model import Model

    kwargs = {}
    if vad_threshold > 0:
        kwargs["vad_threshold"] = vad_threshold
    if model_names:
        kwargs["wakeword_models"] = [resolve_model(n, inference_framework) for n in model_names]
    return Model(**kwargs)


class RingClip:
    """Rolling pre-trigger buffer so a saved clip includes audio from just
    before the wake word crossed threshold, not only after."""

    def __init__(self, seconds: float, rate: int) -> None:
        self.buf = np.zeros(int(seconds * rate), dtype=np.float32)
        self.rate = rate

    def push(self, frame: np.ndarray) -> None:
        n = len(frame)
        if n >= len(self.buf):
            self.buf[:] = frame[-len(self.buf):]
        else:
            self.buf = np.concatenate([self.buf[n:], frame])

    def save(self, path: str) -> None:
        pcm = (np.clip(self.buf, -1.0, 1.0) * 32767).astype("<i2")
        with wave.open(path, "wb") as wav:
            wav.setnchannels(1)
            wav.setsampwidth(2)
            wav.setframerate(self.rate)
            wav.writeframes(pcm.tobytes())


def listen(
    model_names: list[str] | None = None,
    mic: Mic | None = None,
    threshold: float = DEFAULT_THRESHOLD,
    gain: float = 1.0,
    clip_dir: str | None = None,
    quiet: bool = False,
    max_seconds: float | None = None,
):
    """Stream the mic through openWakeWord until Ctrl-C (or max_seconds).

    Yields (name, score) every time a model's score crosses `threshold`, and
    (optionally) writes a WAV clip of the moment to `clip_dir`.
    """
    mic = mic or find_mic()
    model = load_model(model_names)
    native_frame = int(mic.native_rate * FRAME_MS / 1000)

    ring = RingClip(seconds=2.0, rate=TARGET_RATE) if clip_dir else None
    if clip_dir:
        os.makedirs(clip_dir, exist_ok=True)

    last_trigger: dict[str, float] = {}
    debounce_s = 1.5  # ignore repeat triggers of the same word for this long

    if not quiet:
        print(f"Listening on {mic} for: {', '.join(model.models.keys())}", file=sys.stderr)
        print(f"(threshold={threshold}, Ctrl-C to stop)\n", file=sys.stderr)

    start = time.monotonic()
    with sd.InputStream(
        samplerate=mic.native_rate,
        channels=1,
        dtype="float32",
        blocksize=native_frame,
        device=mic.index,
    ) as stream:
        while max_seconds is None or (time.monotonic() - start) < max_seconds:
            data, overflowed = stream.read(native_frame)
            frame = resample(data[:, 0] * gain, mic.native_rate, TARGET_RATE)
            if ring is not None:
                ring.push(frame)

            pcm16 = (np.clip(frame, -1.0, 1.0) * 32767).astype(np.int16)
            scores = model.predict(pcm16)

            now = time.monotonic()
            for name, score in scores.items():
                if not quiet:
                    bar = "#" * int(np.clip(score, 0, 1) * 20)
                    print(f"\r{name:>20}: {score:.2f} {bar:<20}", end="", file=sys.stderr)
                if score >= threshold and now - last_trigger.get(name, -999) > debounce_s:
                    last_trigger[name] = now
                    if not quiet:
                        print(f"\n>>> detected '{name}' ({score:.2f})", file=sys.stderr)
                    clip_path = None
                    if ring is not None:
                        clip_path = os.path.join(clip_dir, f"{name}_{int(now * 1000)}.wav")
                        ring.save(clip_path)
                    yield name, float(score), clip_path


def _main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="mogwai.wake", description="Mogwai wake-word detection")
    parser.add_argument("--mic", help="substring of the input device name")
    parser.add_argument("--gain", type=float, default=1.0)
    sub = parser.add_subparsers(dest="cmd", required=True)

    sub.add_parser("models", help="list bundled wake-word models")

    p_listen = sub.add_parser("listen", help="stream the mic and print detections")
    p_listen.add_argument(
        "--model", action="append", dest="models",
        help="wake word to load (repeatable); default: all bundled models",
    )
    p_listen.add_argument("--threshold", type=float, default=DEFAULT_THRESHOLD)
    p_listen.add_argument("--seconds", type=float, default=None, help="stop after N seconds")
    p_listen.add_argument("--clip-dir", help="save a 2s WAV clip on each detection")
    p_listen.add_argument("--quiet", action="store_true", help="only print on detection")

    args = parser.parse_args(argv)

    if args.cmd == "models":
        for name in available_models():
            print(name)
        return 0

    mic = find_mic(args.mic)
    try:
        count = 0
        for name, score, clip in listen(
            args.models, mic, args.threshold, args.gain,
            args.clip_dir, args.quiet, args.seconds,
        ):
            count += 1
            if args.quiet:
                extra = f" -> {clip}" if clip else ""
                print(f"{name} {score:.2f}{extra}")
        if not args.quiet:
            print(file=sys.stderr)
        return 0
    except KeyboardInterrupt:
        if not args.quiet:
            print("\nstopped.", file=sys.stderr)
        return 0


if __name__ == "__main__":
    raise SystemExit(_main())
