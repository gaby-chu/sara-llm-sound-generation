#!/usr/bin/env python3
"""The final dataset: per model, every take with audio in one folder plus one CSV.

  python build_dataset.py        # after takes_csv.py on every source run

Sources as in build_scores.py: runs/final_<VOICE>_ref/ (gemini-3.8-live) and runs/openai_gpt-realtime-2.1_<VOICE>_ref/
(gpt-realtime-2.1). Copies each clip to runs/dataset/<model>/<id>.wav (the source runs are untouched) and writes
runs/dataset/<model>/<model>.csv with id, target_sound, intended_ipa, error_type, voice.
id is <VOICE>_<prompt_id>_take<N>, e.g. Kore_b_correct_01_take1. Takes with no audio (empty or too quiet) are left out.
"""
import csv, shutil
from pathlib import Path

repo = Path(__file__).parent
runs = repo / "runs"
SOURCES = {
    "gemini-3.8-live": sorted(runs.glob("final_*_ref")),
    "gpt-realtime-2.1": sorted(runs.glob("openai_gpt-realtime-2.1_*_ref")),
}

for model, dirs in SOURCES.items():
    dest = runs / "dataset" / model
    dest.mkdir(parents=True, exist_ok=True)
    before = set(dest.glob("*.wav"))  # update in place: recreating the folder makes iCloud restore conflict copies
    rows = []
    for d in dirs:
        voice = d.name.split("_")[-2]
        for r in csv.DictReader(open(d / "takes.csv", encoding="utf-8")):
            if not r["clip"]:
                continue
            id_ = f"{voice}_{r['prompt_id']}_take{r['take']}"
            out = dest / f"{id_}.wav"
            shutil.copy2(d / r["clip"], out)
            before.discard(out)
            rows.append({"id": id_, "target_sound": r["target_sound"], "intended_ipa": r["intended_ipa"],
                         "error_type": r["error_type"], "voice": voice})
    for stale in before:  # clips no longer in any source run
        stale.unlink()
    assert len({r["id"] for r in rows}) == len(rows), f"duplicate ids in {model}"
    with open(dest / f"{model}.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    print(f"runs/dataset/{model}/: {len(rows)} clips from {len(dirs)} voices")
