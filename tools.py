#!/usr/bin/env python3
"""Helpers for the multimodal-LLM sound generation method.

Two subcommands:

  prep   Convert any recorded audio into the shared clip format:
         16 kHz mono WAV, 2-3 s, the sound sitting in light noise.

           python tools.py prep raw/r_w_try1.m4a clips/r_w_substitution_01.wav

  eval   Score a folder of clips against labels.csv with two phone recognizers
         and write the verdicts back into the CSV, plus accuracy per sound.

           python tools.py eval --folder clips --csv labels.csv
           python tools.py eval --folder clips --csv labels.csv --stub   # no models, tests the plumbing

Dependencies: numpy and ffmpeg for prep. eval additionally needs torch,
transformers and (for the second scorer) whatever you wire into zipa_recognize.
"""

from __future__ import annotations

import argparse
import csv
import subprocess
import sys
import wave
from collections import defaultdict
from pathlib import Path

import numpy as np

SR = 16000
MIN_LEN_S = 2.0
MAX_LEN_S = 3.0
TARGET_LEN_S = 2.5
NOISE_FLOOR_DBFS = -60.0

W2V2_MODEL = "facebook/wav2vec2-lv-60-espeak-cv-ft"


# ----------------------------------------------------------------------------
# audio io
# ----------------------------------------------------------------------------
def load_audio(path: Path) -> np.ndarray:
    """Decode any file to 16 kHz mono float32 using ffmpeg."""
    cmd = ["ffmpeg", "-v", "error", "-i", str(path), "-ac", "1", "-ar", str(SR), "-f", "f32le", "-"]
    out = subprocess.run(cmd, capture_output=True, check=True).stdout
    return np.frombuffer(out, dtype=np.float32).copy()


def write_wav(path: Path, audio: np.ndarray) -> None:
    pcm = (np.clip(audio, -1.0, 1.0) * 32767).astype(np.int16)
    path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(path), "wb") as f:
        f.setnchannels(1)
        f.setsampwidth(2)
        f.setframerate(SR)
        f.writeframes(pcm.tobytes())


def read_wav(path: Path) -> np.ndarray:
    with wave.open(str(path), "rb") as f:
        assert f.getframerate() == SR and f.getnchannels() == 1, f"{path} is not 16 kHz mono"
        raw = f.readframes(f.getnframes())
    return np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32767.0


# ----------------------------------------------------------------------------
# prep
# ----------------------------------------------------------------------------
def trim_silence(audio: np.ndarray, thresh_db: float = -40.0, margin_s: float = 0.08) -> np.ndarray:
    """Trim leading/trailing silence using 10 ms frame energy relative to the peak."""
    frame = SR // 100
    n = len(audio) // frame
    if n == 0 or np.max(np.abs(audio)) == 0:
        return audio
    frames = audio[: n * frame].reshape(n, frame)
    rms = np.sqrt((frames**2).mean(axis=1) + 1e-12)
    db = 20 * np.log10(rms / (rms.max() + 1e-12) + 1e-12)
    active = np.where(db > thresh_db)[0]
    if len(active) == 0:
        return audio
    margin = int(margin_s * SR)
    start = max(active[0] * frame - margin, 0)
    end = min((active[-1] + 1) * frame + margin, len(audio))
    return audio[start:end]


def place_in_clip(sound: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Center the sound in 2.5 s of very light noise (natural silence, not digital zero)."""
    total = int(TARGET_LEN_S * SR)
    if len(sound) > int(MAX_LEN_S * SR):
        print(f"warning: sound is {len(sound) / SR:.2f}s, longer than {MAX_LEN_S}s; keeping it uncropped", file=sys.stderr)
        total = len(sound)
    total = max(total, len(sound))  # sounds between 2.5 s and 3 s must still fit
    amp = 10 ** (NOISE_FLOOR_DBFS / 20)
    clip = rng.normal(0.0, amp, size=total).astype(np.float32)
    start = (total - len(sound)) // 2
    clip[start : start + len(sound)] += sound
    return clip


def cmd_prep(args: argparse.Namespace) -> None:
    rng = np.random.default_rng(args.seed)
    audio = load_audio(Path(args.src))
    sound = trim_silence(audio)
    peak = np.max(np.abs(sound)) or 1.0
    sound = sound / peak * 0.8  # normalise to -2 dBFS peak
    clip = place_in_clip(sound, rng)
    write_wav(Path(args.dst), clip)
    print(f"{args.dst}: sound {len(sound) / SR:.2f}s inside {len(clip) / SR:.2f}s clip, 16 kHz mono")


# ----------------------------------------------------------------------------
# recognizers
# ----------------------------------------------------------------------------
_w2v2 = {}


def w2v2_recognize(audio: np.ndarray) -> str:
    """facebook/wav2vec2-lv-60-espeak-cv-ft: returns space separated espeak phones."""
    import torch
    from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor

    if not _w2v2:
        _w2v2["proc"] = Wav2Vec2Processor.from_pretrained(W2V2_MODEL)
        _w2v2["model"] = Wav2Vec2ForCTC.from_pretrained(W2V2_MODEL).eval()
    inputs = _w2v2["proc"](audio, sampling_rate=SR, return_tensors="pt")
    with torch.no_grad():
        logits = _w2v2["model"](inputs.input_values).logits
    ids = torch.argmax(logits, dim=-1)
    return _w2v2["proc"].batch_decode(ids)[0]


def zipa_recognize(audio: np.ndarray) -> str:
    """Second, independent phone recognizer (ZIPA).

    NOT WIRED UP YET. Fill this in from the ZIPA repo or model card once you have it
    installed: load the model once (cache it like _w2v2 above), run it on `audio`
    (16 kHz mono float32), and return the predicted phones as one string.
    Until then eval reports "n/a" for this column and still scores with wav2vec2.
    """
    raise NotImplementedError("wire ZIPA here")


def stub_recognize(audio: np.ndarray) -> str:
    return "ɹ"


# ----------------------------------------------------------------------------
# scoring
# ----------------------------------------------------------------------------
# Map recognizer spellings onto the label IPA so ɹ/r, tʃ/ʧ etc. compare equal.
EQUIV = {
    "r": "ɹ",
    "ɾ": "ɹ",
    "ɝ": "ɹ",
    "ɚ": "ɹ",
    "tʃ": "ʧ",
    "dʒ": "ʤ",
    "ɡ": "g",
    "y": "j",
}


def normalise(phones: str) -> list[str]:
    s = phones.replace("ː", "").replace("ˈ", "").replace("ˌ", "")
    s = s.replace("t ʃ", "tʃ").replace("d ʒ", "dʒ")
    toks = s.split()
    if len(toks) <= 1:  # some decoders return a run-together string
        toks = list(s.replace(" ", ""))
    return [EQUIV.get(t, t) for t in toks]


def verdict(predicted: str, produced_ipa: str, error_type: str) -> str:
    """'match' when the recognizer heard the label; 'miss' otherwise.

    Distortions have no IPA label, so they are always 'review' (human ears decide).
    """
    if not produced_ipa:
        return "review"
    toks = normalise(predicted)
    want = EQUIV.get(produced_ipa, produced_ipa)
    return "match" if want in toks else "miss"


def cmd_eval(args: argparse.Namespace) -> None:
    folder = Path(args.folder)
    with open(args.csv, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    fields = list(rows[0].keys())

    recognizers = {"scorer_w2v2": stub_recognize if args.stub else w2v2_recognize}
    recognizers["scorer_zipa"] = stub_recognize if args.stub else zipa_recognize

    stats = defaultdict(lambda: defaultdict(lambda: [0, 0]))  # sound -> scorer -> [match, scored]
    for row in rows:
        path = folder / row["file"]
        if not path.exists():
            continue
        audio = read_wav(path)
        for col, fn in recognizers.items():
            try:
                pred = fn(audio)
            except NotImplementedError:
                row[col] = "n/a"
                continue
            v = verdict(pred, row["produced_IPA"], row["error_type"])
            row[col] = f"{v}: heard '{pred}'"
            if v != "review":
                stats[row["target_sound"]][col][1] += 1
                stats[row["target_sound"]][col][0] += v == "match"

    with open(args.csv, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)

    print("\nAccuracy per target sound (match / scored):")
    for sound, per in sorted(stats.items()):
        parts = [f"{col}: {m}/{n}" for col, (m, n) in per.items()]
        print(f"  {sound:>4}  " + "   ".join(parts))
    print(
        "\nReminder: calibrate both recognizers on REAL labeled kid clips first and report "
        "accuracy per sound before trusting any verdict on synthetic clips."
    )


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)

    pp = sub.add_parser("prep", help="convert a recording into the shared clip format")
    pp.add_argument("src")
    pp.add_argument("dst")
    pp.add_argument("--seed", type=int, default=0)
    pp.set_defaults(fn=cmd_prep)

    pe = sub.add_parser("eval", help="score clips with two phone recognizers")
    pe.add_argument("--folder", required=True)
    pe.add_argument("--csv", required=True)
    pe.add_argument("--stub", action="store_true", help="fake recognizers, to test the plumbing")
    pe.set_defaults(fn=cmd_eval)

    args = p.parse_args()
    args.fn(args)


if __name__ == "__main__":
    main()
