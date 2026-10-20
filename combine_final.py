#!/usr/bin/env python3
"""Build full sound sets: each voice/mode's latest full run, with the rewritten sounds (V5) replaced by their v5 takes.

  python combine_final.py VOICE [VOICE ...]      # modes text and ref

Base run is runs/v4_<VOICE>_<MODE>, or runs/v3_gemini-3.8-live_<VOICE>_<MODE> when there is no v4 one (text runs
for Zephyr, Puck, Fenrir). Output: runs/final_<VOICE>_<MODE>/ with clips/, raw/ (including the full, uncropped takes),
and takes.csv with a source_run column saying where each take came from. Copies only; the source runs are untouched.
"""
import csv, shutil, sys
from pathlib import Path

V5 = {f"{s}_correct_01" for s in ["n", "ng", "l", "f", "th_voiceless", "th_voiced", "y"]}
runs = Path(__file__).parent / "runs"

for v in sys.argv[1:]:
    for m in ["text", "ref"]:
        base = runs / f"v4_{v}_{m}"
        if not base.exists():
            base = runs / f"v3_gemini-3.8-live_{v}_{m}"
        new = runs / f"v5_{v}_{m}"
        out = runs / f"final_{v}_{m}"
        # update in place: deleting and recreating the folder makes iCloud Drive (Desktop sync) restore the old
        # files as "<name> 2.wav" conflict copies
        (out / "clips").mkdir(parents=True, exist_ok=True); (out / "raw").mkdir(exist_ok=True)
        before = {p for p in out.rglob("*.wav")}
        fields = ["source_run"] + next(csv.reader(open(new / "takes.csv", encoding="utf-8")))
        rows, log = [], []
        for src, keep in [(base, lambda p: p not in V5), (new, lambda p: p in V5)]:
            for r in csv.DictReader(open(src / "takes.csv", encoding="utf-8")):
                if not keep(r["prompt_id"]):
                    continue
                r.setdefault("mode", m)  # v3 csvs have no mode column
                rows.append({"source_run": src.name, **r})
                if r["clip"]:
                    shutil.copy2(src / r["clip"], out / r["clip"])
                for f in (src / "raw").glob(f"{r['prompt_id']}_take{r['take']}*.wav"):
                    shutil.copy2(f, out / "raw" / f.name)
            log += [l for l in open(src / "raw" / "transcripts.log", encoding="utf-8") if keep(l.split("\t")[0])]
        copied = {out / r["clip"] for r in rows if r["clip"]}
        copied |= {out / "raw" / f.name for src in (base, new) for f in (src / "raw").glob("*.wav")
                   if (f.name.split("_take")[0] in V5) == (src == new)}
        for stale in before - copied:  # takes no longer in the set
            stale.unlink()
        rows.sort(key=lambda r: (r["prompt_id"], r["take"]))
        with open(out / "takes.csv", "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=fields); w.writeheader(); w.writerows(rows)
        open(out / "raw" / "transcripts.log", "w", encoding="utf-8").writelines(log)
        ok = [r for r in rows if r["status"] == "ok"]
        crop = [r for r in ok if r["cropped_to_1.8s"] == "True"]
        print(f"{out.name}: {len(ok)}/{len(rows)} takes with audio, {len({r['prompt_id'] for r in rows})} prompts "
              f"({base.name} + {new.name}), {len(crop)} cropped:")
        print("   " + ", ".join(f"{r['prompt_id'].replace('_correct_01', '')} t{r['take']} ({r['raw_seconds']}s)" for r in crop))
