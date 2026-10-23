#!/usr/bin/env python3
"""Generate sound clips from a Gemini native-audio model on Vertex AI (Live API), then run them through tools.py prep.

Each prompt is sent as the reference recording for its sound (--ref-dir, sound_inputs/<ref>.wav) followed by
REF_INTRO and the row's prompt from the prompts csv.

Setup (once):
  pip install google-genai numpy
  gcloud auth application-default login
  # use the Columbia GCP project that holds your credits

Run:
  python gen_gemini.py --project YOUR_PROJECT --model gemini-3.8-live --only r_correct_01.wav
  python gen_gemini.py --project YOUR_PROJECT --model gemini-3.8-live          # all rows in gen_prompts_v4.csv

Raw takes go to raw/, prepped 16 kHz mono clips go to clips/, and the take number is added to the name so
repeat runs do not overwrite earlier ones. takes_csv.py logs the model, voice and prompt of every take.
"""

from __future__ import annotations

import argparse
import csv
import subprocess
import sys
import wave

import numpy as np
from pathlib import Path

OPENING = (
    "You are helping me build a speech therapy teaching tool. I need very short, isolated speech sounds, "
    "not words. Never say the letter name and never add a vowel afterward unless asked. Keep the clip under "
    "2 seconds. Use a clear, natural voice at a relaxed pace, as if recorded close to a good microphone in a quiet "
    "room: no background noise, music, effects or breathing sounds. The clips will be used to train an audio model, "
    "so make the sound clean and easy to hear. Reply with only the sound."
)

# Reference recordings (--ref-dir): <ref>.wav, where ref is the prompts csv's ref column (gen_prompts_v4.csv)
# or, without that column, the file-name prefix (r_, s_, b_, g_)
REF_IPA = {"r": "ɹ", "s": "s", "b": "b", "g": "ɡ"}

# The recording, then REF_INTRO, then the row's prompt. The voice comes from --voice, so the recording is only a
# reference for which sound to make, not for voice or recording quality.
REF_INTRO = (
    "The audio is a recording of a person making the correct {name} sound, IPA {ipa}. Use it only as a reference "
    "for which sound to make. Use your own voice, and make the sound as clean and natural as possible. Don't copy "
    "the recording's voice, recording quality, length, background noise, or any vowel before or after the sound. "
    "Then do the following."
)


def load_ref(ref_dir: Path, row: dict) -> tuple[bytes, str] | None:
    """Reference recording for this prompt's target sound as 16 kHz mono PCM (the Live API rejects audio/wav parts),
    and the text sent after it (the whole user message)."""
    target = row.get("ref") or row["file"].split("_", 1)[0]
    ipa = row.get("ref_ipa") or REF_IPA.get(target, target)
    path = ref_dir / f"{target}.wav"
    if not path.exists():
        return None
    pcm = subprocess.run(
        ["ffmpeg", "-v", "error", "-i", str(path), "-ac", "1", "-ar", "16000", "-f", "s16le", "-"],
        capture_output=True, check=True,
    ).stdout
    intro = REF_INTRO.format(name=target.replace("_", " ").upper(), ipa=ipa)
    return pcm, f"{intro}\n\n{row['prompt']}"


def save_pcm_as_wav(path: Path, pcm: bytes, rate: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(path), "wb") as f:
        f.setnchannels(1)
        f.setsampwidth(2)
        f.setframerate(rate)
        f.writeframes(pcm)


def crop_to_onset(pcm: bytes, rate: int, keep_s: float = 1.8) -> bytes:
    """Native-audio models often hold a sound for 30+ s. Keep the first keep_s seconds after onset, with a fade-out."""
    x = np.frombuffer(pcm, dtype=np.int16).astype(np.float32)
    if len(x) == 0:
        return pcm
    frame = rate // 100
    n = len(x) // frame
    rms = np.sqrt((x[: n * frame].reshape(n, frame) ** 2).mean(axis=1) + 1e-9)
    loud = np.where(rms > rms.max() * 0.01)[0]
    start = max((loud[0] if len(loud) else 0) * frame - int(0.05 * rate), 0)
    y = x[start : start + int(keep_s * rate)].copy()
    fade = min(int(0.15 * rate), len(y))
    y[len(y) - fade :] *= np.linspace(1, 0, fade)
    return y.astype(np.int16).tobytes()


def generate_live(client, types, model: str, voice: str, ref: tuple[bytes, str]) -> tuple[bytes, int, str]:
    """Live API (native audio models): the opening goes in as the system instruction, and the turn is the reference
    recording followed by its text (pcm, text from load_ref)."""
    import asyncio

    pcm_ref, text = ref
    parts = [types.Part.from_bytes(data=pcm_ref, mime_type="audio/pcm;rate=16000"), types.Part(text=text)]

    async def run() -> tuple[bytes, str]:
        config = types.LiveConnectConfig(
            response_modalities=["AUDIO"],
            system_instruction=OPENING,
            output_audio_transcription=types.AudioTranscriptionConfig(),
            speech_config=types.SpeechConfig(
                voice_config=types.VoiceConfig(prebuilt_voice_config=types.PrebuiltVoiceConfig(voice_name=voice))
            ),
        )
        chunks, text = [], ""
        async with client.aio.live.connect(model=model, config=config) as session:
            await session.send_client_content(
                turns=types.Content(role="user", parts=parts), turn_complete=True
            )
            async for msg in session.receive():
                sc = msg.server_content
                if sc and sc.output_transcription:
                    text += sc.output_transcription.text or ""
                if sc and sc.model_turn:
                    for part in sc.model_turn.parts:
                        if part.inline_data:
                            chunks.append(part.inline_data.data)
                if sc and sc.turn_complete:
                    break
        return b"".join(chunks), text

    pcm, text = asyncio.run(run())
    return pcm, 24000, text  # Live API output is 24 kHz 16-bit mono PCM


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--project", required=True)
    ap.add_argument("--location", default="us-central1")
    ap.add_argument("--model", required=True)
    ap.add_argument("--voice", default="Kore", help="prebuilt voice name; try a few and log which")
    ap.add_argument("--prompts", default="gen_prompts_v4.csv")
    ap.add_argument("--only", help="generate just this file name from the prompts csv")
    ap.add_argument("--skip-done", action="store_true", help="skip rows whose clips/<stem>.wav already exists")
    ap.add_argument("--takes", type=int, default=1)
    ap.add_argument("--take", type=int, help="make only this take number (to refill a failed take without redoing the others)")
    ap.add_argument("--ref-dir", default="sound_inputs", help="folder of reference recordings <target>.wav sent with each prompt")
    ap.add_argument("--out", help="folder for raw/ and clips/ (default: next to this script)")
    args = ap.parse_args()

    from google import genai
    from google.genai import types

    client = genai.Client(vertexai=True, project=args.project, location=args.location)
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
                pcm, rate, text = generate_live(client, types, args.model, args.voice, ref)
            except Exception as e:  # keep going: failures are results for the memo
                print(f"FAILED {stem} take{take}: {e}", file=sys.stderr)
                continue
            if len(pcm) < 2 * rate // 20:  # under 50 ms is effectively empty (seen: 0.3 ms takes that break prep)
                print(f"FAILED {stem} take{take}: model returned no audio", file=sys.stderr)
                with open(here / "raw" / "transcripts.log", "a", encoding="utf-8") as log:
                    log.write(f"{stem}\ttake{take}\t{args.model}\t{args.voice}\t0.0s\tNO AUDIO\n")
                continue
            save_pcm_as_wav(raw, pcm, rate)
            secs = len(pcm) / 2 / rate
            print(f"{stem} take{take}: {secs:.1f}s, model said {text!r}")
            with open(here / "raw" / "transcripts.log", "a", encoding="utf-8") as log:
                log.write(f"{stem}\ttake{take}\t{args.model}\t{args.voice}\t{secs:.1f}s\t{text!r}\n")
            if secs > 3.0:  # held sounds run long: keep the full raw file, crop a copy for prep
                print(f"  longer than 3 s, cropping to the first 1.8 s after onset")
                raw = here / "raw" / f"{stem}_take{take}_crop.wav"
                save_pcm_as_wav(raw, crop_to_onset(pcm, rate), rate)
            out = here / "clips" / f"{stem}_take{take}.wav"
            try:
                subprocess.run([sys.executable, str(repo / "tools.py"), "prep", str(raw), str(out)], check=True)
            except subprocess.CalledProcessError:  # keep going, like generation failures
                print(f"FAILED {stem} take{take}: prep failed on {raw}", file=sys.stderr)


if __name__ == "__main__":
    main()
