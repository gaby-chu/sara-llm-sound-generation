#!/usr/bin/env python3
"""One row per take for a run folder: what was sent, what came back, what the triage listener heard.

  python takes_csv.py RUN_DIR --prompts gen_prompts_v4.csv --labels labels_v2.csv [--ref-dir sound_inputs]

Pass the same --prompts and --ref-dir as the gen_gemini.py run. Writes RUN_DIR/takes.csv and prints a summary.
The prompt text is rebuilt from the current files, so rerun it before editing prompts or OPENING.
"""
import argparse, csv, json, re
from pathlib import Path

import gen_gemini as g

EQ = {"r": "ɹ", "ɚ": "ɹ", "ɝ": "ɹ", "ɡ": "g", "y": "j", "ʧ": "tʃ", "ʤ": "dʒ", "t͡ʃ": "tʃ", "d͡ʒ": "dʒ"}


def norm(x: str) -> str:
    x = re.sub(r"[/\[\]ːˑ̞̩ ]", "", x or "").replace("ɡ", "g")
    return EQ.get(x, x)


ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
ap.add_argument("run")
ap.add_argument("--prompts", default="gen_prompts_v4.csv")
ap.add_argument("--labels", default="labels_v2.csv")
ap.add_argument("--ref-dir")
ap.add_argument("--ref-mode", choices=["ref", "mimic"], default="ref")
args = ap.parse_args()

here = Path(__file__).parent
root = Path(args.run)
prompts = {Path(r["file"]).stem: r for r in csv.DictReader(open(here / args.prompts, encoding="utf-8"))}
labels = {Path(r["file"]).stem: r for r in csv.DictReader(open(here / args.labels, encoding="utf-8"))}
tri_path = root / "triage_takes.json"
tri = json.load(open(tri_path)) if tri_path.exists() else {}

rows = []
latest = {}  # a refilled take (gen_gemini.py --take) is logged again; keep its last attempt
for line in open(root / "raw" / "transcripts.log", encoding="utf-8"):
    f = line.rstrip("\n").split("\t")
    latest[(f[0], f[1])] = f
for stem, take, model, voice, secs, said in latest.values():
    name = f"{stem}_{take}.wav"
    p, lab, t = prompts.get(stem, {}), labels.get(stem, {}), tri.get(name, {})
    ref = g.load_ref(Path(args.ref_dir), p, args.ref_mode) if args.ref_dir and p else None
    ref_file = Path(args.ref_dir) / f"{p.get('ref') or stem.split('_', 1)[0]}.wav" if ref else ""
    want = lab.get("produced_IPA", "")
    clean = bool(t) and bool(want) and norm(t.get("heard_ipa")) == norm(want) and not t.get("vowel_after") and not t.get("letter_name_or_word")
    rows.append({
        "clip": f"clips/{name}" if (root / "clips" / name).exists() else "",
        "prompt_id": stem, "take": take[4:], "model": model, "voice": voice,
        "target_sound": lab.get("target_sound", ""), "intended_ipa": want, "error_type": lab.get("error_type", ""),
        "system_instruction": g.OPENING, "mode": args.ref_mode if ref else "text", "reference_audio": str(ref_file),
        "user_prompt": ref[1] if ref else p.get("prompt", ""),
        "raw_seconds": secs[:-1], "cropped_to_1.8s": float(secs[:-1]) > 3.0,
        "model_transcript": "" if said == "NO AUDIO" else said.removeprefix("TOO QUIET ").strip("'"),
        "status": "no audio" if said == "NO AUDIO" else "too quiet" if said.startswith("TOO QUIET") else "ok",
        "listener_heard_ipa": t.get("heard_ipa", ""), "listener_vowel_after": t.get("vowel_after", ""),
        "listener_letter_or_word": t.get("letter_name_or_word", ""), "listener_notes": t.get("notes", ""),
        "listener_clean_match": clean,
    })
with open(root / "takes.csv", "w", newline="", encoding="utf-8") as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)

ok = [r for r in rows if r["status"] == "ok"]
hits = [r for r in rows if r["listener_clean_match"]]
sounds = {r["prompt_id"] for r in rows}
print(f"{root}/takes.csv: {len(rows)} takes, {len(rows) - len(ok)} empty, "
      f"{sum(r['cropped_to_1.8s'] for r in ok)} cropped, clean matches {len(hits)}/{len(ok)}, "
      f"prompts with a clean take {len({r['prompt_id'] for r in hits})}/{len(sounds)}")
