#!/usr/bin/env python3
"""One row per take for a run folder: what was sent and what came back.

  python takes_csv.py RUN_DIR --prompts gen_prompts_v4.csv --labels labels_v2.csv [--ref-dir sound_inputs]

Pass the same --prompts and --ref-dir as the gen_gemini.py run. Writes RUN_DIR/takes.csv and prints a summary.
The prompt text is rebuilt from the current files, so rerun it before editing prompts or OPENING.
"""
import argparse, csv
from pathlib import Path

import gen_gemini as g

ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
ap.add_argument("run")
ap.add_argument("--prompts", default="gen_prompts_v4.csv")
ap.add_argument("--labels", default="labels_v2.csv")
ap.add_argument("--ref-dir", default="sound_inputs")
args = ap.parse_args()

here = Path(__file__).parent
root = Path(args.run)
prompts = {Path(r["file"]).stem: r for r in csv.DictReader(open(here / args.prompts, encoding="utf-8"))}
labels = {Path(r["file"]).stem: r for r in csv.DictReader(open(here / args.labels, encoding="utf-8"))}

rows = []
latest = {}  # a refilled take (gen_gemini.py --take) is logged again; keep its last attempt
for line in open(root / "raw" / "transcripts.log", encoding="utf-8"):
    f = line.rstrip("\n").split("\t")
    latest[(f[0], f[1])] = f
for stem, take, model, voice, secs, said in latest.values():
    name = f"{stem}_{take}.wav"
    p, lab = prompts.get(stem, {}), labels.get(stem, {})
    ref = g.load_ref(Path(args.ref_dir), p) if p else None
    ref_file = Path(args.ref_dir) / f"{p.get('ref') or stem.split('_', 1)[0]}.wav" if ref else ""
    rows.append({
        "clip": f"clips/{name}" if (root / "clips" / name).exists() else "",
        "prompt_id": stem, "take": take[4:], "model": model, "voice": voice,
        "target_sound": lab.get("target_sound", ""), "intended_ipa": lab.get("produced_IPA", ""), "error_type": lab.get("error_type", ""),
        "system_instruction": g.OPENING, "mode": "ref", "reference_audio": str(ref_file),
        "user_prompt": ref[1] if ref else p.get("prompt", ""),
        "raw_seconds": secs[:-1], "cropped_to_1.8s": float(secs[:-1]) > 3.0,
        "model_transcript": "" if said == "NO AUDIO" else said.removeprefix("TOO QUIET ").strip("'"),
        "status": "no audio" if said == "NO AUDIO" else "too quiet" if said.startswith("TOO QUIET") else "ok",
    })
with open(root / "takes.csv", "w", newline="", encoding="utf-8") as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)

ok = [r for r in rows if r["status"] == "ok"]
sounds = {r["prompt_id"] for r in rows}
print(f"{root}/takes.csv: {len(rows)} takes, {len(rows) - len(ok)} empty, "
      f"{sum(r['cropped_to_1.8s'] for r in ok)} cropped, "
      f"prompts with audio {len({r['prompt_id'] for r in ok})}/{len(sounds)}")
