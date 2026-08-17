# Mogwai

A Raspberry Pi 5 (8 GB) creature that will see (Pi AI Camera), hear (USB mic),
and speak.

Current state: **ears work, wake word works, speech-to-text works, voice works, Mogwai talks back.**

## Setup

```bash
git clone https://github.com/dscripka/openWakeWord.git
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
.venv/bin/python -c "import openwakeword; openwakeword.utils.download_models()"
```

The `sounddevice` package needs PortAudio underneath:

- macOS: `brew install portaudio`
- Raspberry Pi OS: `sudo apt install libportaudio2`

## Ears

```bash
.venv/bin/python -m mogwai.ears list                  # show inputs, mark the chosen one
.venv/bin/python -m mogwai.ears check                 # noise floor, then "say something"
.venv/bin/python -m mogwai.ears meter --seconds 10    # live level bar
.venv/bin/python -m mogwai.ears record out.wav --seconds 5
```

The mic is chosen by name, not index -- indices shuffle between machines and
reboots, so hardcoding one guarantees a break when this moves to the Pi.
Override with `--mic <substring>` or `MOGWAI_MIC=<substring>`.

Audio is captured at the device's native rate (48 kHz on the C-Media USB mic)
and resampled to 16 kHz mono float32, which is what speech models expect.
Asking PortAudio for 16 kHz directly tends to fail on USB mics that only do
48 kHz natively.

### Reading the levels

| Reading | Meaning |
|---|---|
| below -70 dBFS | digital silence: wrong device, or no mic permission |
| noise floor around -50 dBFS | normal for a quiet room with these mics |
| speech 15-25 dB above floor | usable, on the quiet side |
| speech 25+ dB above floor | good |
| 0 dBFS / OVERFLOW | clipping, lower `--gain` or back off the mic |

A 50/60 Hz peak in a recording is mains hum -- expected on unshielded USB
mics, and harmless once a high-pass filter goes in ahead of speech recognition.

## Wake word

Built on [openWakeWord](https://github.com/dscripka/openWakeWord) (vendored as
a git submodule-less clone at `openWakeWord/`, installed editable). It detects
a wake phrase in a live stream -- it does **not** transcribe speech. Turning
"what did you say after the wake word" into text is a separate component
(e.g. Vosk or faster-whisper) layered on top of the same 16 kHz stream.

```bash
.venv/bin/python -m mogwai.wake models                             # list bundled wake words
.venv/bin/python -m mogwai.wake listen --model hey_jarvis          # live scores, Ctrl-C to stop
.venv/bin/python -m mogwai.wake listen --model hey_jarvis --clip-dir clips/ --quiet
```

Bundled wake words: `alexa`, `hey_jarvis`, `hey_mycroft`, `hey_rhasspy`,
`timer`, `weather`. Pass `--model` more than once to listen for several at
once; omit it to load all of them.

Reuses `mogwai.ears` for mic selection and the 48kHz-to-16kHz resample, so
detection runs on the same mic-picking logic (by name, not index) as
recording. Audio is read in 80 ms frames -- openWakeWord's native frame size
-- and each frame is fed to `Model.predict()`, which internally re-buffers if
frames are not exact multiples of 80 ms.

`--clip-dir` saves a 2-second WAV around each detection (1s before + after,
via a rolling buffer) for tuning thresholds or building a false-positive set.

Default detection threshold is 0.5, per the model authors' recommendation.
`--threshold` overrides it; lower it if wake-word misses are the problem,
raise it if false triggers are.

### Training a custom "mogwai" wake word

openWakeWord ships 6 fixed phrases; "mogwai" isn't one of them. Training a new
one needs synthetic TTS data (via a Linux-only fork of Piper) and a training
run against a ~17GB precomputed negative-audio dataset -- both awkward on a
Mac and slow without a GPU. The path that actually works well: the project's
own [Colab notebook](https://colab.research.google.com/drive/1q1oe2zOyZp7UsB3jJiQ1IFn8z5YfjwEb?usp=sharing)
(free GPU, ~30-60 min):

1. Open the notebook, run the setup cells as-is.
2. In the config cell, set `config["target_phrase"] = ["mogwai"]` and
   `config["model_name"] = "mogwai"`.
3. Run clip generation, augmentation, and training.
4. Download `my_custom_model/mogwai.onnx` **and** `mogwai.tflite` from the
   Colab file browser.

Then drop both files into `mogwai/models/` in this repo (that directory is
scanned automatically):

```
mogwai/models/mogwai.onnx
mogwai/models/mogwai.tflite
```

```bash
.venv/bin/python -m mogwai.wake models              # now also lists "mogwai"
.venv/bin/python -m mogwai.wake listen --model mogwai
```

Custom models are resolved by file path, matched to whichever inference
framework is active (`.tflite` by default on both this Mac and the Pi, since
`ai-edge-litert` is installed on both) -- see `resolve_model()` in
`mogwai/wake.py`. A custom "mogwai" model was trained on far less data than
the bundled ones, so expect to spend time tuning `--threshold` (and consider
`--clip-dir` to collect false positives for review).

## Speech-to-text

Built on [faster-whisper](https://github.com/SYSTRAN/faster-whisper), a
CTranslate2 reimplementation of OpenAI's Whisper -- several times faster and
lighter than stock Whisper on CPU, which is what matters with no GPU on the
Pi. Models auto-download from Hugging Face on first use and are cached under
`~/.cache/huggingface`.

```bash
.venv/bin/python -m mogwai.stt transcribe clip.wav       # transcribe a WAV file
.venv/bin/python -m mogwai.stt listen --seconds 5        # record from mic, then transcribe
```

Default model is `base.en` at `int8` quantization -- a reasonable
speed/accuracy balance for a Pi 5 CPU. Override with `--model tiny.en` for
lower latency or `--model small.en` for better accuracy, or set
`MOGWAI_STT_MODEL` / `MOGWAI_STT_COMPUTE` env vars.

**The full pipeline** -- wake word, then a back-and-forth, then transcribe
each turn -- is `mogwai.listen`:

```bash
.venv/bin/python -m mogwai.listen                                   # say "mogwai", then talk
.venv/bin/python -m mogwai.listen --command-seconds 6 --no-continuous --once
```

Saying "mogwai" starts a conversation, not a single exchange: it keeps
recording and transcribing turn after turn with no need to repeat the wake
word, until a farewell phrase ("thanks" / "thank you", matched as a whole
word so "thanksgiving" doesn't trigger it) ends that turn and hands control
back to waiting for the wake word. `--no-continuous` restores the
one-wake-word-per-turn behavior.

It keeps one continuous mic stream open the whole time (reopening a stream
mid-conversation adds a device-dependent startup gap that can clip the start
of what you say), runs openWakeWord on it until "mogwai" crosses threshold,
then records each turn and hands it to faster-whisper.

Each turn ends on **silence**, not a fixed duration: on stream open, it
measures the room's ambient noise level once (`Calibrated noise floor: ...`),
then records in 100ms blocks until about 1.2s of quiet follows at least
0.6s of speech (`MOGWAI_SILENCE_HANG_SECONDS` / `MOGWAI_MIN_SPEECH_SECONDS`
/ `MOGWAI_SILENCE_MARGIN_DB` env vars). `--command-seconds` (default 10) is
just a safety-cap ceiling in case silence detection never triggers -- not
the expected length of what you say, so it doesn't cut a longer sentence off
partway through, and it doesn't make a quick answer wait through dead air
either.

Note: transcription accuracy on the literal word "mogwai" doesn't matter here
-- detecting it is openWakeWord's job, and `mogwai.listen` only ever feeds
Whisper the audio *after* the wake word. (For reference, asking Whisper to
transcribe "mogwai" itself in isolation -- not something the real pipeline
does -- produced "Maguai": expected, since it's not a real English word and
base.en has no reason to know it.)

## Voice

Built on [espeak-ng](https://github.com/espeak-ng/espeak-ng), the classic
formant/Klatt speech synthesizer. A neural TTS engine ([Piper](https://github.com/OHF-voice/piper1-gpl))
was the first pass here, but it's built to sound human -- not what a small
robotic creature wants. espeak-ng is deliberately synthetic-sounding, tiny,
and instant on CPU with no model download. Install it once:

```bash
brew install espeak-ng          # macOS
sudo apt install espeak-ng      # Raspberry Pi OS
```

`mogwai/voice.py` shells out to the `espeak-ng` binary directly -- no Python
package needed for it.

```bash
.venv/bin/python -m mogwai.voice say "hello there"                          # synthesize and play
.venv/bin/python -m mogwai.voice say "hello there" --out hi.wav             # also save to disk
.venv/bin/python -m mogwai.voice say "hello there" --variant croak --pitch 20 --depth 1.0
.venv/bin/python -m mogwai.voice variants                                    # a few known-good starting points
```

Default is the `klatt3` formant voice, pitch 0 (espeak-ng's own, 0-99 range),
further deepened and roughened by a `--depth 0.75` post-processing pitch-down
-- chosen by ear against this project's test phrase. Four independent knobs:

| Flag | What it does |
|---|---|
| `--variant` | which formant voice (`espeak-ng --voices=variant` lists ~100; `croak`, `Demonic`, `robosoft3` also lean mechanical/raspy) |
| `--pitch` | espeak-ng's own pitch, 0-99 |
| `--depth` | post-synthesis pitch-down; `1.0` = off, lower = deeper. Naively resampling to a lower pitch also slows playback (the "slowed tape" effect) |
| `--wpm` | perceived words-per-minute *after* `--depth` is applied -- synthesis speed is pre-compensated by `wpm / depth` so pace stays put while only pitch drops |

Env var overrides: `MOGWAI_VOICE_VARIANT`, `MOGWAI_VOICE_PITCH`,
`MOGWAI_VOICE_DEPTH`, `MOGWAI_VOICE_WPM`. Playback goes through
`sounddevice`, same as the rest of Mogwai, so it uses whatever the system
default output device is -- on the Pi, that'll need to be pointed at a
speaker (see below).

`synthesize()` in `mogwai/voice.py` returns mono float32 audio in [-1, 1] at
espeak-ng's native rate (22050 Hz) -- the same convention `mogwai.ears` and
`mogwai.stt` use, so it composes with the rest of the pipeline without any
format juggling.

## Talking back

`mogwai/brain.py` sends what you said to Claude and returns a short reply,
using Claude Haiku 4.5 -- the fastest, cheapest current Claude model
($1/$5 per million tokens) -- since a voice assistant needs the reply back
quickly and the reply itself is only ever a sentence or two. Its system
prompt establishes Mogwai's persona and asks for spoken-friendly output (no
markdown, no lists).

Needs the `anthropic` package (already in `requirements.txt`) and an API key:

```bash
export ANTHROPIC_API_KEY=sk-ant-...   # create one at console.anthropic.com/settings/keys
.venv/bin/python -m mogwai.brain ask "what time is it"
```

### Local fallback (works with no internet)

`brain.respond()` tries Claude first, and on *any* cloud failure -- no API
key, no internet, a network hiccup -- falls back automatically to a small
model running locally via [Ollama](https://ollama.com), so Mogwai still
answers with no connection at all. Ollama was picked over the alternative
(`llama-cpp-python`) because it ships prebuilt binaries -- no C++ compiling on
first install, which matters more on a Pi than on a dev machine.

```bash
brew install ollama && brew services start ollama    # macOS
curl -fsSL https://ollama.com/install.sh | sh         # Raspberry Pi OS / Linux
ollama pull llama3.2:1b                               # ~1.3GB, CPU-friendly

.venv/bin/python -m mogwai.brain ask "what are you"           # cloud, falls back if it fails
.venv/bin/python -m mogwai.brain --local ask "what are you"   # skip the cloud attempt entirely
```

Default local model is `llama3.2:1b` -- small and fast enough for a Pi 5 CPU,
but noticeably less coherent and less instruction-following than Claude.
Override with `--local-model` or `MOGWAI_BRAIN_LOCAL_MODEL` -- `llama3.2:3b`
is a reasonable step up in quality if the Pi has headroom to spare.

The system prompt is explicit that "Mogwai" is just a name, not the movie
creature (small local models especially will otherwise latch onto the
association and describe themselves as the *Gremlins* character if asked
"what are you") -- worth keeping in mind if the persona ever drifts after
further prompt edits.

**`mogwai.converse`** is the full loop -- wake word, back-and-forth, ask
Claude (or the local fallback) each turn, speak the reply -- composing
`mogwai.listen`, `mogwai.brain`, and `mogwai.voice` without duplicating any
of their logic:

```bash
.venv/bin/python -m mogwai.converse                 # say "mogwai", then talk -- keep going, no need to repeat it
.venv/bin/python -m mogwai.converse --once
```

Say "thanks" (or "thank you") to end the conversation -- that turn never
reaches Claude or the local model at all; it gets a short, fixed reply
(`GOODBYE` in `mogwai/converse.py`) instead, so there's no chance of Mogwai
arguing about being done or trying to keep the conversation going.

Override the cloud model with `--brain-model` or `MOGWAI_BRAIN_MODEL` (e.g.
`claude-sonnet-5` for better answers at higher latency and cost).

## Moving to the Pi

The mic and wake-word code is portable as written. On the Pi:

```bash
sudo apt install libportaudio2
arecord -l                    # confirm the kernel sees it as an ALSA card
python -m mogwai.ears check
```

No macOS-style permission prompt exists there; if `arecord -l` lists the card,
capture works. If ALSA fights over the device, the fix is usually to make the
USB mic the default card in `/etc/asound.conf`.

`faster-whisper`'s dependencies (`ctranslate2`, `av`) both ship `manylinux
aarch64` wheels, so `pip install -r requirements.txt` should work unmodified
on Pi OS (64-bit). CPU transcription speed on a Pi 5 hasn't been benchmarked
yet here -- if `base.en` feels sluggish for a few seconds of command audio,
try `tiny.en` first.

`sudo apt install espeak-ng` covers voice output -- it's a prebuilt Debian
package on Pi OS, nothing to compile. The Pi has no built-in speaker, though,
so voice output needs either a USB/3.5mm speaker or HDMI audio, and may need
`raspi-config` or `/etc/asound.conf` to pick the right output device the way
the mic setup needed for input.

`mogwai.brain` calls out over the network to the Claude API, so `anthropic`
(pure Python + `httpx`, no compiled extensions) installs the same way on the
Pi. **If the Pi has working WiFi and can reach the internet on its own, the
cloud path just works** -- nothing here is macOS-specific, it's an ordinary
HTTPS call. If the Pi is offline (no WiFi in range, captive portal, etc.),
`mogwai.brain` falls back automatically to the local Ollama model instead of
failing -- see the Local fallback section above for the Pi install command
(`curl -fsSL https://ollama.com/install.sh | sh`, official ARM64 binaries, no
compiling). Every other piece of Mogwai (ears, wake word, STT, voice) already
runs fully offline regardless.

**Getting the API key onto the Pi:** `pi-zshrc-snippet.sh` at the repo root
has the `~/.zshrc` block to paste on the Pi (only there -- it's deliberately
not applied to this Mac's `~/.zshrc`) so every new terminal on the Pi loads
`ANTHROPIC_API_KEY` automatically. The snippet itself holds no secret; it
just sources `~/.mogwai_api_key.sh`, which needs a real key placed on the Pi
separately (a fresh key generated for the Pi specifically, not copied from
the Mac, is the cleaner option -- see the snippet's comments).

## Planned

- Eyes: Pi AI Camera (IMX500), on-sensor inference
