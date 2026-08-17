"""Mogwai talks back: wake word -> transcribe -> Claude -> speak.

The full loop -- everything before this was a building block. Composes
mogwai.listen (wake word + speech-to-text), mogwai.brain (Claude Haiku 4.5),
and mogwai.voice (espeak-ng) without duplicating any of their logic.

Say the wake word once; it's a back-and-forth from there, no need to repeat
it. Say "thanks" (or "thank you") to end it -- that turn skips the LLM
entirely and gets a short, fixed goodbye instead, so Mogwai never argues
about being done or tries to keep the conversation going.

Drives mogwai.display's Face through the loop, one state per thing this is
doing: listening while a command is being recorded, thinking while waiting on
Claude/Ollama, talking while a reply plays, idle otherwise. pygame's event
pump has to run on the main thread (macOS enforces this -- a hard crash, not
just a quirk, if violated), so with a display this flips the usual shape:
listen/respond/speak run on a background thread (see _turns()) while the main
thread's only job is ticking the Face at a steady frame rate, picking up
finished turns over a queue.

CLI:
    python -m mogwai.converse
    python -m mogwai.converse --once
    python -m mogwai.converse --no-display
"""

from __future__ import annotations

import argparse
import queue
import sys
import threading

from mogwai.brain import DEFAULT_MODEL as BRAIN_DEFAULT_MODEL
from mogwai.brain import respond
from mogwai.display import Face
from mogwai.ears import find_mic
from mogwai.listen import run as listen
from mogwai.stt import DEFAULT_MODEL as STT_DEFAULT_MODEL
from mogwai.wake import DEFAULT_THRESHOLD
from mogwai.voice import say as speak

# A fixed line, not an LLM call -- guarantees Mogwai just accepts a "thanks"
# and stops, rather than getting a chance to plead to keep talking.
GOODBYE = "Later."


def _turns(
    wake_models: list[str] | None,
    command_seconds: float,
    mic,
    threshold: float,
    gain: float,
    stt_model_size: str,
    brain_model: str,
    once: bool,
    face: Face | None,
):
    """The conversation loop itself -- see run(), which wraps this for display."""
    history: list[dict] = []
    for wake_word, text, farewell in listen(
        wake_models, command_seconds, mic, threshold, gain, stt_model_size, once,
        on_wake=(lambda _name: face.listening()) if face is not None else None,
    ):
        if not text:
            continue
        if farewell:
            print(f"mogwai: {GOODBYE}", file=sys.stderr)
            if face is not None:
                face.talk(GOODBYE)
            speak(GOODBYE)
            if face is not None:
                face.idle()
            yield wake_word, text, GOODBYE
            history = []
            continue
        if face is not None:
            face.thinking()
        try:
            reply, history = respond(text, brain_model, history=history)
        except RuntimeError as e:
            print(f"error: {e}", file=sys.stderr)
            if face is not None:
                face.idle()
            continue
        print(f"mogwai: {reply}", file=sys.stderr)
        if face is not None:
            face.talk(reply)
        speak(reply)
        if face is not None:
            face.idle()
        yield wake_word, text, reply


def run(
    wake_models: list[str] | None = None,
    command_seconds: float = 10.0,
    mic=None,
    threshold: float = DEFAULT_THRESHOLD,
    gain: float = 1.0,
    stt_model_size: str = STT_DEFAULT_MODEL,
    brain_model: str = BRAIN_DEFAULT_MODEL,
    once: bool = False,
    display: bool = True,
):
    """Blocking loop: for each heard command, get a reply from Claude and speak it.

    One wake word starts a whole back-and-forth (mogwai.listen keeps
    recording turn after turn on its own); conversation history is carried
    across those turns so later replies can refer back to earlier ones. A
    farewell phrase ends it here with a fixed goodbye instead of a model
    call, and clears the history -- the next wake word starts fresh.

    With display=True (the default), see the module docstring for why the
    actual work moves to a background thread while this one just ticks the
    Face; closing the display window (or Esc) ends the loop from that side.
    """
    args = (wake_models, command_seconds, mic, threshold, gain, stt_model_size, brain_model, once)

    if not display:
        yield from _turns(*args, None)
        return

    face = Face()
    out: queue.Queue = queue.Queue()

    def _worker() -> None:
        try:
            for item in _turns(*args, face):
                out.put(("item", item))
        except Exception as e:
            out.put(("error", e))
        out.put(("done", None))

    threading.Thread(target=_worker, daemon=True).start()

    try:
        while face.tick():
            try:
                kind, payload = out.get_nowait()
            except queue.Empty:
                continue
            if kind == "item":
                yield payload
            elif kind == "error":
                raise payload
            else:  # "done"
                return
    finally:
        face.close()


def _main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="mogwai.converse", description="Mogwai's full conversation loop")
    parser.add_argument("--wake-model", action="append", dest="wake_models", help="repeatable; default: mogwai")
    parser.add_argument("--stt-model", default=STT_DEFAULT_MODEL)
    parser.add_argument("--brain-model", default=BRAIN_DEFAULT_MODEL)
    parser.add_argument("--threshold", type=float, default=DEFAULT_THRESHOLD)
    parser.add_argument("--command-seconds", type=float, default=10.0, help="max seconds per turn (a safety cap; recording ends on silence)")
    parser.add_argument("--mic", help="substring of the input device name")
    parser.add_argument("--gain", type=float, default=1.0)
    parser.add_argument("--once", action="store_true", help="stop after the first exchange")
    parser.add_argument("--no-display", dest="display", action="store_false", help="skip the face display")
    args = parser.parse_args(argv)

    mic = find_mic(args.mic)
    try:
        for _wake_word, text, reply in run(
            args.wake_models, args.command_seconds, mic, args.threshold, args.gain,
            args.stt_model, args.brain_model, args.once, args.display,
        ):
            print(f"you: {text}")
            print(f"mogwai: {reply}")
        return 0
    except KeyboardInterrupt:
        print("\nstopped.", file=sys.stderr)
        return 0


if __name__ == "__main__":
    raise SystemExit(_main())
