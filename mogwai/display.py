"""Mogwai's face: a small animated display driven by pygame.

Draws two eyes and a mouth. On the Pi this runs headless -- no X11/Wayland --
so pygame is pointed at the "kmsdrm" video driver, which writes straight to
the framebuffer via DRM/KMS. macOS doesn't support kmsdrm, so it falls back to
a normal window there, which is how this gets developed and previewed before
it ever touches the Pi.

States map to what mogwai.converse is doing, one call each:
    idle()       -- between conversations: slow, random blinks, mouth still
    listening()  -- wake word heard, recording a command: eyes widen
    thinking()   -- waiting on Claude/Ollama: eyes narrow, mouth still
    talk(text)   -- speaking a reply: mouth cycles open/closed for roughly
                    as long as `text` takes to say, on a background thread,
                    so it can run alongside mogwai.voice.say()'s blocking
                    playback rather than waiting for it to finish first

pygame isn't in requirements.txt yet -- install it with `pip install pygame`.

CLI:
    python -m mogwai.display demo
"""

from __future__ import annotations

import argparse
import math
import os
import random
import sys
import threading
import time

# Must be set before pygame.display.init() below. Only Linux (the Pi) has a
# kmsdrm driver at all -- macOS ignores this and pygame opens a normal window.
if sys.platform.startswith("linux") and "SDL_VIDEODRIVER" not in os.environ:
    os.environ["SDL_VIDEODRIVER"] = "kmsdrm"
os.environ.setdefault("PYGAME_HIDE_SUPPORT_PROMPT", "1")

import pygame  # noqa: E402  (must follow the SDL_VIDEODRIVER/PYGAME_HIDE... env vars above)

BG_COLOR = (10, 10, 14)
EYE_COLOR = (90, 220, 255)
MOUTH_COLOR = (90, 220, 255)

BLINK_MIN_S, BLINK_MAX_S = 2.0, 6.0  # gap between idle blinks
BLINK_DURATION_S = 0.12

# Rough talk-cycle pace for the mouth-flap animation -- independent of
# mogwai.voice's actual WPM, since this only approximates speech, not
# amplitude-driven lip sync.
TALK_WPM = 150


class Face:
    """Owns the pygame window/surface and draws Mogwai's face each frame."""

    def __init__(self, size: tuple[int, int] = (480, 320)) -> None:
        pygame.init()
        pygame.mouse.set_visible(False)
        fullscreen = pygame.FULLSCREEN if os.environ.get("SDL_VIDEODRIVER") == "kmsdrm" else 0
        self.screen = pygame.display.set_mode(size, fullscreen)
        self.clock = pygame.time.Clock()
        self.w, self.h = self.screen.get_size()

        self._state = "idle"
        self._mouth_open = 0.0  # 0 (closed) - 1 (fully open)
        self._blink_start: float | None = None
        self._next_blink = time.monotonic() + random.uniform(BLINK_MIN_S, BLINK_MAX_S)
        self._talk_thread: threading.Thread | None = None
        self._stop_talk = threading.Event()

    # -- state changes, one per thing mogwai.converse is doing --

    def idle(self) -> None:
        self._stop_talking()
        self._state = "idle"

    def listening(self) -> None:
        self._stop_talking()
        self._state = "listening"

    def thinking(self) -> None:
        self._stop_talking()
        self._state = "thinking"

    def talk(self, text: str, wpm: float = TALK_WPM) -> None:
        """Cycle the mouth open/closed for roughly as long as `text` takes to say.

        Runs on a background thread so a caller driving mogwai.voice.say()
        (which blocks until playback finishes) can call this first and have
        the mouth animate for the same span, rather than one waiting on the
        other.
        """
        self._stop_talking()
        self._stop_talk.clear()
        self._state = "talking"

        words = max(1, len(text.split()))
        duration = words / (wpm / 60)

        def _cycle() -> None:
            start = time.monotonic()
            while not self._stop_talk.is_set() and time.monotonic() - start < duration:
                self._mouth_open = max(0.0, math.sin((time.monotonic() - start) * 14.0))
                time.sleep(1 / 30)
            self._mouth_open = 0.0

        self._talk_thread = threading.Thread(target=_cycle, daemon=True)
        self._talk_thread.start()

    def _stop_talking(self) -> None:
        self._stop_talk.set()
        if self._talk_thread is not None:
            self._talk_thread.join()
        self._talk_thread = None

    @property
    def talking(self) -> bool:
        return self._talk_thread is not None and self._talk_thread.is_alive()

    # -- per-frame draw --

    def tick(self) -> bool:
        """Advance one frame: handle blinking, draw, flip. False means quit was requested."""
        for event in pygame.event.get():
            if event.type == pygame.QUIT or (
                event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE
            ):
                return False

        self._draw(self._blink_amount(time.monotonic()))
        pygame.display.flip()
        self.clock.tick(30)
        return True

    def _blink_amount(self, now: float) -> float:
        """0 (open) - 1 (shut), triangular over BLINK_DURATION_S, on a random schedule."""
        if self._blink_start is None:
            if now < self._next_blink:
                return 0.0
            self._blink_start = now

        t = now - self._blink_start
        if t >= BLINK_DURATION_S:
            self._blink_start = None
            self._next_blink = now + random.uniform(BLINK_MIN_S, BLINK_MAX_S)
            return 0.0

        half = BLINK_DURATION_S / 2
        return t / half if t < half else (BLINK_DURATION_S - t) / half

    def _draw(self, blink: float) -> None:
        self.screen.fill(BG_COLOR)
        cx, cy = self.w // 2, self.h // 2
        eye_dx = self.w // 5
        eye_r = min(self.w, self.h) // 8

        widen = 1.15 if self._state == "listening" else 1.0
        narrow = 0.6 if self._state == "thinking" else 1.0

        for dx in (-eye_dx, eye_dx):
            eye_h = max(2, int(eye_r * (1 - blink) * widen * narrow))
            rect = pygame.Rect(0, 0, eye_r, eye_h)
            rect.center = (cx + dx, cy - eye_r // 2)
            pygame.draw.ellipse(self.screen, EYE_COLOR, rect)

        mouth_h = max(4, int(eye_r * self._mouth_open))
        rect = pygame.Rect(0, 0, eye_dx * 2, mouth_h)
        rect.center = (cx, cy + eye_r * 2)
        pygame.draw.ellipse(self.screen, MOUTH_COLOR, rect)

    def close(self) -> None:
        self._stop_talking()
        pygame.quit()


def _demo() -> int:
    face = Face()
    try:
        face.listening()
        for _ in range(60):
            if not face.tick():
                return 0

        face.thinking()
        for _ in range(60):
            if not face.tick():
                return 0

        face.talk("Hello, I am Mogwai, a small robotic assistant.")
        while face.talking:
            if not face.tick():
                return 0

        face.idle()
        while True:
            if not face.tick():
                return 0
    except KeyboardInterrupt:
        pass
    finally:
        face.close()
    return 0


def _main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="mogwai.display", description="Mogwai's face")
    sub = parser.add_subparsers(dest="cmd", required=True)
    sub.add_parser("demo", help="cycle through listening -> thinking -> talking -> idle")
    args = parser.parse_args(argv)

    if args.cmd == "demo":
        return _demo()
    return 2


if __name__ == "__main__":
    raise SystemExit(_main())
