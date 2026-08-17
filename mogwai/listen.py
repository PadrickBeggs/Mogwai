"""Mogwai's full hearing pipeline: wait for the wake word, then a back-and-forth.

Ties mogwai.wake (openWakeWord) and mogwai.stt (faster-whisper) together into
the actual "hear" feature -- everything else so far was a building block.

Saying the wake word starts a conversation, not a single exchange: once
triggered, this keeps recording and transcribing turn after turn with no need
to repeat the wake word, until a farewell phrase ("thanks", "thank you") ends
it and control returns to waiting for the wake word again.

CLI:
    python -m mogwai.listen
    python -m mogwai.listen --wake-model mogwai --command-seconds 4
"""

from __future__ import annotations

import argparse
import os
import re
import sys
from collections.abc import Callable

import numpy as np
import sounddevice as sd

from mogwai.ears import TARGET_RATE, dbfs, find_mic, resample
from mogwai.stt import DEFAULT_MODEL as STT_DEFAULT_MODEL
from mogwai.stt import load_model as load_stt_model
from mogwai.stt import transcribe
from mogwai.wake import DEFAULT_THRESHOLD, FRAME_MS
from mogwai.wake import load_model as load_wake_model

# Matched as whole words/phrases (not a bare substring -- "thanksgiving"
# shouldn't end the conversation), tolerant of surrounding punctuation and
# filler ("Thanks!", "Okay, thank you.").
_FAREWELL_RE = re.compile(r"\b(thanks|thank you)\b", re.IGNORECASE)


def _is_farewell(text: str) -> bool:
    return _FAREWELL_RE.search(text) is not None


# Recording a turn used to mean "read a fixed N seconds" -- too short and you
# get cut off mid-sentence, too long and every short answer waits through
# dead air. Instead each turn reads in small blocks and ends on silence, so it
# adapts to how long the person actually talks; `command_seconds` becomes a
# safety-cap ceiling rather than the expected duration.
_VAD_BLOCK_SECONDS = 0.1
MIN_SPEECH_SECONDS = float(os.environ.get("MOGWAI_MIN_SPEECH_SECONDS", "0.6"))
SILENCE_HANG_SECONDS = float(os.environ.get("MOGWAI_SILENCE_HANG_SECONDS", "1.2"))
SILENCE_MARGIN_DB = float(os.environ.get("MOGWAI_SILENCE_MARGIN_DB", "12"))


def _record_turn(stream, mic, gain: float, max_seconds: float, floor_dbfs: float) -> np.ndarray:
    """Record one conversation turn, ending on trailing silence rather than a fixed duration."""
    block_frames = int(mic.native_rate * _VAD_BLOCK_SECONDS)
    max_blocks = max(1, int(max_seconds / _VAD_BLOCK_SECONDS))
    min_speech_blocks = max(1, int(MIN_SPEECH_SECONDS / _VAD_BLOCK_SECONDS))
    silence_hang_blocks = max(1, int(SILENCE_HANG_SECONDS / _VAD_BLOCK_SECONDS))
    threshold = floor_dbfs + SILENCE_MARGIN_DB

    chunks = []
    trailing_silence = 0
    for i in range(max_blocks):
        raw, _overflowed = stream.read(block_frames)
        block = raw[:, 0] * gain
        chunks.append(block)
        trailing_silence = 0 if dbfs(block) >= threshold else trailing_silence + 1
        if i + 1 >= min_speech_blocks and trailing_silence >= silence_hang_blocks:
            break

    audio = np.concatenate(chunks) if chunks else np.zeros(0, dtype=np.float32)
    return resample(audio, mic.native_rate, TARGET_RATE)


def _calibrate_floor(stream, mic, gain: float, seconds: float = 1.0) -> float:
    """Median ambient noise level, measured once so every turn's silence
    threshold is relative to the actual room rather than a fixed guess."""
    block_frames = int(mic.native_rate * _VAD_BLOCK_SECONDS)
    n_blocks = max(1, int(seconds / _VAD_BLOCK_SECONDS))
    levels = []
    for _ in range(n_blocks):
        raw, _overflowed = stream.read(block_frames)
        level = dbfs(raw[:, 0] * gain)
        if level != -np.inf:
            levels.append(level)
    return float(np.median(levels)) if levels else -60.0


def run(
    wake_models: list[str] | None = None,
    command_seconds: float = 10.0,
    mic=None,
    threshold: float = DEFAULT_THRESHOLD,
    gain: float = 1.0,
    stt_model_size: str = STT_DEFAULT_MODEL,
    once: bool = False,
    continuous: bool = True,
    on_wake: Callable[[str], None] | None = None,
):
    """Blocking loop: wake word -> record -> transcribe -> yield (word, text, is_farewell).

    With continuous=True (the default), once the wake word triggers, keeps
    recording and yielding further turns *without* requiring the wake word
    again, until a farewell phrase is heard -- that turn is yielded with
    is_farewell=True, then control returns to waiting for the wake word.

    Each turn records until the person stops talking (silence relative to the
    room's calibrated noise floor), not for a fixed duration -- `command_seconds`
    is a safety-cap ceiling in case silence detection never triggers, not the
    expected length of what someone says.

    Uses one continuous InputStream throughout, including for every
    conversation turn -- reopening a stream mid-conversation adds a
    device-dependent startup gap that can clip the start of what the user says.

    `on_wake`, if given, is called with the triggering word right before each
    command recording window opens (the initial one and every later turn of
    the same conversation) -- e.g. mogwai.display.Face.listening, so a display
    can show it's actively recording.
    """
    mic = mic or find_mic()
    wake_model = load_wake_model(wake_models or ["mogwai"], inference_framework="tflite")
    stt_model = load_stt_model(stt_model_size)
    native_frame = int(mic.native_rate * FRAME_MS / 1000)

    print(f"Listening on {mic} for: {', '.join(wake_model.models.keys())}", file=sys.stderr)
    print("(Ctrl-C to stop)\n", file=sys.stderr)

    with sd.InputStream(
        samplerate=mic.native_rate,
        channels=1,
        dtype="float32",
        blocksize=native_frame,
        device=mic.index,
    ) as stream:
        floor_dbfs = _calibrate_floor(stream, mic, gain)
        print(f"Calibrated noise floor: {floor_dbfs:.1f} dBFS\n", file=sys.stderr)

        while True:
            data, _overflowed = stream.read(native_frame)
            frame = resample(data[:, 0] * gain, mic.native_rate, TARGET_RATE)
            pcm16 = (np.clip(frame, -1.0, 1.0) * 32767).astype(np.int16)
            scores = wake_model.predict(pcm16)

            triggered = next((name for name, score in scores.items() if score >= threshold), None)
            if triggered is None:
                continue

            print(f">>> heard '{triggered}' -- go ahead...", file=sys.stderr)

            while True:
                if on_wake is not None:
                    on_wake(triggered)
                command_audio = _record_turn(stream, mic, gain, command_seconds, floor_dbfs)

                text = transcribe(command_audio, TARGET_RATE, stt_model)
                print(f"heard: {text!r}" if text else "heard: (nothing intelligible)", file=sys.stderr)

                farewell = bool(text) and _is_farewell(text)
                yield triggered, text, farewell

                if once:
                    return
                if farewell or not continuous:
                    break

            # Drop buffered predictions so trailing echo of the conversation
            # doesn't immediately retrigger the wake word next loop.
            wake_model.reset()


def _main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="mogwai.listen", description="Wake word -> speech-to-text, end to end")
    parser.add_argument("--wake-model", action="append", dest="wake_models", help="repeatable; default: mogwai")
    parser.add_argument("--stt-model", default=STT_DEFAULT_MODEL)
    parser.add_argument("--threshold", type=float, default=DEFAULT_THRESHOLD)
    parser.add_argument("--command-seconds", type=float, default=10.0, help="max seconds per turn (a safety cap; recording ends on silence)")
    parser.add_argument("--mic", help="substring of the input device name")
    parser.add_argument("--gain", type=float, default=1.0)
    parser.add_argument("--once", action="store_true", help="stop after the first turn")
    parser.add_argument(
        "--no-continuous", dest="continuous", action="store_false",
        help="require the wake word again before every turn, instead of a back-and-forth",
    )
    args = parser.parse_args(argv)

    mic = find_mic(args.mic)
    try:
        for _name, text, _farewell in run(
            args.wake_models, args.command_seconds, mic, args.threshold, args.gain,
            args.stt_model, args.once, args.continuous,
        ):
            print(text)
        return 0
    except KeyboardInterrupt:
        print("\nstopped.", file=sys.stderr)
        return 0


if __name__ == "__main__":
    raise SystemExit(_main())
