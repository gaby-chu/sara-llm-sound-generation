#!/usr/bin/env python3
"""Generate sound clips from an OpenAI audio-output model (Chat Completions or Realtime), then run them through tools.py prep.

Same prompts, reference recordings, take naming and transcripts.log as gen_gemini.py (OPENING, REF_INTRO, load_ref,
crop_to_onset are imported from it), so triage_takes.py and takes_csv.py work on the output unchanged.

gpt-audio-1.5 (alloy, 2026-10-20) returned near-silent takes for about half the prompts (raw peak under 0.05, vs about
0.45 for Gemini). tools.py prep normalises them to 0.8 peak, which turns the model's noise floor into static. So takes
whose raw peak is under --min-peak are kept in raw/ but not prepped, and logged as TOO QUIET.

--realtime uses the Realtime API (WebSocket) for gpt-realtime* models, as gen_gemini.py uses the Live API.

Setup (once):
  uv pip install --python .venv/bin/python openai
  put OPENAI_API_KEY=sk-... in .env (gitignored), or export it

Run:
  python gen_openai.py --model gpt-realtime-2.1 --realtime --only f_correct_01.wav
  python gen_openai.py --model gpt-realtime-2.1 --realtime --voice coral --takes 3 --out runs/openai_gpt-realtime-2.1_coral_ref

The opening goes in as the system message and the reference recording + REF_INTRO + the row's prompt as the user
turn, like gen_gemini.py. Voices: alloy, ash, ballad, coral, echo, fable, nova, onyx, sage, shimmer,
verse, marin, cedar (the model rejects ones it does not support).
"""

from __future__ import annotations

import argparse
import base64
import csv
import io
import os
import subprocess
import sys
import wave
from pathlib import Path

import numpy as np

from gen_gemini import OPENING, crop_to_onset, load_ref, save_pcm_as_wav


def load_env(path: Path) -> None:
    """KEY=value lines from .env into the environment, without overriding variables already set."""
    if path.exists():
        for line in path.read_text().splitlines():
            k, sep, v = line.partition("=")
            if sep and not line.startswith("#"):
                os.environ.setdefault(k.strip(), v.strip())


def pcm_to_wav_bytes(pcm: bytes, rate: int) -> bytes:
    buf = io.BytesIO()
    with wave.open(buf, "wb") as f:
        f.setnchannels(1)
        f.setsampwidth(2)
        f.setframerate(rate)
        f.writeframes(pcm)
    return buf.getvalue()


def generate(client, model: str, voice: str, ref: tuple[bytes, str]) -> tuple[bytes, int, str]:
    """Chat Completions: the user turn is the reference recording followed by its text (pcm, text from load_ref)."""
    pcm_ref, text = ref
    wav_b64 = base64.b64encode(pcm_to_wav_bytes(pcm_ref, 16000)).decode()
    content = [{"type": "input_audio", "input_audio": {"data": wav_b64, "format": "wav"}}, {"type": "text", "text": text}]
    resp = client.chat.completions.create(
        model=model,
        modalities=["text", "audio"],
        audio={"voice": voice, "format": "wav"},
        messages=[{"role": "system", "content": OPENING}, {"role": "user", "content": content}],
    )
    audio = resp.choices[0].message.audio
    if not audio or not audio.data:
        return b"", 24000, ""
    with wave.open(io.BytesIO(base64.b64decode(audio.data)), "rb") as f:
        assert f.getnchannels() == 1 and f.getsampwidth() == 2, "expected 16-bit mono wav"
        return f.readframes(f.getnframes()), f.getframerate(), audio.transcript or ""


def generate_realtime(client, model: str, voice: str, ref: tuple[bytes, str]) -> tuple[bytes, int, str]:
    """Realtime API: the opening goes in as the session instructions, and one user item holds the reference recording
    followed by its text (pcm, text from load_ref). Realtime audio in and out is 24 kHz 16-bit mono PCM, so the
    16 kHz reference is resampled first.
    """
    pcm_ref, text = ref
    pcm24 = subprocess.run(
        ["ffmpeg", "-v", "error", "-f", "s16le", "-ar", "16000", "-ac", "1", "-i", "-", "-ar", "24000", "-f", "s16le", "-"],
        input=pcm_ref, capture_output=True, check=True,
    ).stdout
    content = [{"type": "input_audio", "audio": base64.b64encode(pcm24).decode()}, {"type": "input_text", "text": text}]
    import ssl, certifi  # the WebSocket uses Python's own SSL, which has no CA certs on this Mac

    chunks, said = [], ""
    ctx = ssl.create_default_context(cafile=certifi.where())
    with client.realtime.connect(model=model, websocket_connection_options={"ssl": ctx}) as conn:
        conn.session.update(session={
            "type": "realtime", "instructions": OPENING, "output_modalities": ["audio"],
            "audio": {"input": {"turn_detection": None},
                      "output": {"voice": voice, "format": {"type": "audio/pcm", "rate": 24000}}},
        })
        conn.conversation.item.create(item={"type": "message", "role": "user", "content": content})
        conn.response.create()
        for event in conn:
            if event.type == "response.output_audio.delta":
                chunks.append(base64.b64decode(event.delta))
            elif event.type == "response.output_audio_transcript.delta":
                said += event.delta
            elif event.type == "error":
                raise RuntimeError(event.error.message)
            elif event.type == "response.done":
                break
    return b"".join(chunks), 24000, said


def raw_peak(pcm: bytes) -> float:
    x = np.frombuffer(pcm, dtype=np.int16)
    return float(np.abs(x).max()) / 32768 if len(x) else 0.0


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default="gpt-audio-1.5")
    ap.add_argument("--voice", default="alloy", help="prebuilt voice name; try a few and log which")
    ap.add_argument("--realtime", action="store_true", help="use the Realtime API (needed for gpt-realtime* models)")
    ap.add_argument("--min-peak", type=float, default=0.05,
                    help="raw takes quieter than this peak (0-1) are not prepped and are logged as TOO QUIET")
    ap.add_argument("--prompts", default="gen_prompts_v4.csv")
    ap.add_argument("--only", help="generate just this file name from the prompts csv")
    ap.add_argument("--skip-done", action="store_true", help="skip rows whose clips/<stem>.wav already exists")
    ap.add_argument("--takes", type=int, default=1)
    ap.add_argument("--take", type=int, help="make only this take number (to refill a failed take without redoing the others)")
    ap.add_argument("--ref-dir", default="sound_inputs", help="folder of reference recordings <target>.wav sent with each prompt")
    ap.add_argument("--out", help="folder for raw/ and clips/ (default: next to this script)")
    args = ap.parse_args()

    from openai import OpenAI

    load_env(Path(__file__).parent / ".env")
    client = OpenAI()  # reads OPENAI_API_KEY
    repo = Path(__file__).parent
    here = Path(args.out) if args.out else repo

    with open(repo / args.prompts, newline="", encoding="utf-8") as f:
        rows = [r for r in csv.DictReader(f) if not args.only or r["file"] == args.only]
    if not rows:
        sys.exit("no matching rows in the prompts csv")

    for row in rows:
        stem = Path(row["file"]).stem
        if args.skip_done and (here / "clips" / f"{stem}.wav").exists():
            continue
        ref = load_ref(Path(args.ref_dir), row)
        if not ref:
            print(f"FAILED {stem}: no reference recording in {args.ref_dir}", file=sys.stderr)
            continue
        for take in [args.take] if args.take else range(1, args.takes + 1):
            raw = here / "raw" / f"{stem}_take{take}.wav"
            try:
                gen = generate_realtime if args.realtime else generate
                pcm, rate, text = gen(client, args.model, args.voice, ref)
            except Exception as e:  # keep going: failures are results for the memo
                print(f"FAILED {stem} take{take}: {e}", file=sys.stderr)
                continue
            out = here / "clips" / f"{stem}_take{take}.wav"
            # a new take replaces the old one: drop its files so a quiet or empty take can't leave a stale clip behind
            for stale in (raw, here / "raw" / f"{stem}_take{take}_crop.wav", out):
                stale.unlink(missing_ok=True)
            if len(pcm) < 2 * rate // 20:  # under 50 ms is effectively empty
                print(f"FAILED {stem} take{take}: model returned no audio", file=sys.stderr)
                (here / "raw").mkdir(parents=True, exist_ok=True)
                with open(here / "raw" / "transcripts.log", "a", encoding="utf-8") as log:
                    log.write(f"{stem}\ttake{take}\t{args.model}\t{args.voice}\t0.0s\tNO AUDIO\n")
                continue
            save_pcm_as_wav(raw, pcm, rate)
            secs = len(pcm) / 2 / rate
            peak = raw_peak(pcm)
            if peak < args.min_peak:  # prep would boost the noise floor into static; keep raw/ for listening
                print(f"FAILED {stem} take{take}: too quiet (peak {peak:.3f}), not prepped", file=sys.stderr)
                with open(here / "raw" / "transcripts.log", "a", encoding="utf-8") as log:
                    log.write(f"{stem}\ttake{take}\t{args.model}\t{args.voice}\t{secs:.1f}s\tTOO QUIET {text!r}\n")
                continue
            print(f"{stem} take{take}: {secs:.1f}s, peak {peak:.2f}, model said {text!r}")
            with open(here / "raw" / "transcripts.log", "a", encoding="utf-8") as log:
                log.write(f"{stem}\ttake{take}\t{args.model}\t{args.voice}\t{secs:.1f}s\t{text!r}\n")
            if secs > 3.0:  # held sounds run long: keep the full raw file, crop a copy for prep
                print(f"  longer than 3 s, cropping to the first 1.8 s after onset")
                raw = here / "raw" / f"{stem}_take{take}_crop.wav"
                save_pcm_as_wav(raw, crop_to_onset(pcm, rate), rate)
            try:
                subprocess.run([sys.executable, str(repo / "tools.py"), "prep", str(raw), str(out)], check=True)
            except subprocess.CalledProcessError:  # keep going, like generation failures
                print(f"FAILED {stem} take{take}: prep failed on {raw}", file=sys.stderr)


if __name__ == "__main__":
    main()
