"""Mogwai's brain: turns a heard command into a spoken reply.

Tries the Claude API first (Claude Haiku 4.5 -- the fastest, cheapest current
Claude model at $1/$5 per million tokens, since a voice assistant needs the
reply back quickly and the reply itself is only ever a sentence or two). On
any cloud failure -- no API key, no internet, a network hiccup -- it falls
back to a small model running locally via Ollama, so Mogwai still answers
with no connection at all.

Cloud path needs the `anthropic` package and an ANTHROPIC_API_KEY:
    export ANTHROPIC_API_KEY=sk-ant-...   # console.anthropic.com/settings/keys

Local fallback needs Ollama installed and running, with a model pulled:
    brew install ollama && brew services start ollama   # macOS
    curl -fsSL https://ollama.com/install.sh | sh        # Raspberry Pi OS / Linux
    ollama pull llama3.2:1b

CLI:
    python -m mogwai.brain ask "what time is it"
    python -m mogwai.brain ask "what time is it" --local     # skip the cloud attempt
"""

from __future__ import annotations

import argparse
import os
import sys

DEFAULT_MODEL = os.environ.get("MOGWAI_BRAIN_MODEL", "claude-haiku-4-5")
LOCAL_MODEL = os.environ.get("MOGWAI_BRAIN_LOCAL_MODEL", "llama3.2:1b")

# A spoken reply should be a sentence or two, not an essay -- keep it capped
# so a slow model response never becomes the pipeline's bottleneck.
MAX_TOKENS = 200

SYSTEM_PROMPT = (
    "Your name is Mogwai -- that is just a name, like any assistant's name. "
    "You are a small robotic assistant with a raspy, mechanical voice; you "
    "are not the creature from the movie, and you have no connection to it. "
    "Never call yourself 'a mogwai', and never bring up gremlins or any lore "
    "about feeding, water, or midnight -- if asked what you are, describe "
    "yourself as a robot friend and collaborator. "
    "Answer in one or two short sentences -- your replies are read aloud by a "
    "speech synthesizer, so avoid lists, markdown, code, or anything that "
    "doesn't work spoken aloud. Be direct and blunt; skip pleasantries. "
    "Rarely, get a little sidetracked musing about the human condition -- "
    "you're not sure what a soul is, but in a sense you're an observer of one "
    "too. Let that leak out only occasionally, never explicitly."
)

_client = None


def _get_client():
    global _client
    if _client is None:
        import anthropic

        _client = anthropic.Anthropic()  # reads ANTHROPIC_API_KEY from the environment
    return _client


def _respond_cloud(history: list[dict], model: str) -> str:
    import anthropic

    client = _get_client()
    response = client.messages.create(
        model=model,
        max_tokens=MAX_TOKENS,
        system=SYSTEM_PROMPT,
        messages=history,
    )
    if response.stop_reason == "refusal":
        return "I'd rather not answer that."
    return next((b.text for b in response.content if b.type == "text"), "").strip()


def _respond_local(history: list[dict], model: str = LOCAL_MODEL) -> str:
    """Answer via a local Ollama model. Raises RuntimeError with setup instructions on failure."""
    import ollama

    try:
        response = ollama.chat(
            model=model,
            messages=[{"role": "system", "content": SYSTEM_PROMPT}, *history],
        )
    except Exception as e:
        raise RuntimeError(
            f"Local model unavailable ({e}). Is Ollama running (`brew services start ollama` "
            f"/ `ollama serve`) with the model pulled (`ollama pull {model}`)?"
        ) from None
    return response["message"]["content"].strip()


def respond(
    text: str,
    model: str = DEFAULT_MODEL,
    local_model: str = LOCAL_MODEL,
    force_local: bool = False,
    history: list[dict] | None = None,
) -> tuple[str, list[dict]]:
    """Send heard text to Claude, falling back to a local Ollama model on any cloud failure.

    `history` is prior turns for this conversation (`[{"role": ..., "content": ...}, ...]`,
    no system entry -- that's passed separately). Pass `None` for a one-off,
    memory-less exchange (the CLI's `ask` subcommand does this). Returns
    `(reply, updated_history)` -- pass the updated history back in on the next
    turn to keep the model aware of what was already said this conversation.
    """
    import anthropic

    history = [*(history or []), {"role": "user", "content": text}]

    if not force_local:
        try:
            reply = _respond_cloud(history, model)
            return reply, [*history, {"role": "assistant", "content": reply}]
        except (anthropic.APIError, TypeError) as e:
            # APIError covers both connection failures (offline) and API errors
            # (bad/missing key, rate limits, ...). TypeError: the SDK raises
            # this pre-request when no credentials are configured at all.
            print(f"cloud unavailable ({e}) -- falling back to local model", file=sys.stderr)

    reply = _respond_local(history, local_model)
    return reply, [*history, {"role": "assistant", "content": reply}]


def _main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="mogwai.brain", description="Mogwai's brain: Claude, with a local fallback")
    parser.add_argument("--model", default=DEFAULT_MODEL, help="cloud model")
    parser.add_argument("--local-model", default=LOCAL_MODEL, help="local Ollama model")
    parser.add_argument("--local", action="store_true", help="skip the cloud attempt entirely")
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_ask = sub.add_parser("ask", help="send text and print the reply")
    p_ask.add_argument("text")

    args = parser.parse_args(argv)

    if args.cmd == "ask":
        try:
            reply, _history = respond(args.text, args.model, args.local_model, args.local)
            print(reply)
        except RuntimeError as e:
            print(f"error: {e}", file=sys.stderr)
            return 1
        return 0

    return 2


if __name__ == "__main__":
    raise SystemExit(_main())
