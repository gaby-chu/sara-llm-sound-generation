#!/usr/bin/env python3
"""The final dataset: per model, every take with audio in one folder plus one CSV. Column definitions: SCHEMA.md.

  python build_dataset.py        # after takes_csv.py (and optionally score_local.py) on every source run

Sources as in build_scores.py: runs/final_<VOICE>_ref/ (gemini-3.8-live) and runs/openai_gpt-realtime-2.1_<VOICE>_ref/
(gpt-realtime-2.1). Copies each clip to runs/dataset/<model>/<id>.wav (the source runs are untouched) and writes
runs/dataset/<model>/<model>.csv.
id is <VOICE>_<prompt_id>_take<N>, e.g. Kore_b_correct_01_take1. Takes with no audio (empty or too quiet) are left out.

Substitutions (error_types.csv, the substitutions in Sara's phoneme-endpoint error_type.py): an isolated substituted
sound is the same audio as a correct production of the sound said instead. So every correct take of the produced sound
also gets one row per substitution it stands for, with the same file and source_id pointing at the correct row. For
example Kore_th_voiceless_correct_01_take1 (θ) also appears as id Kore_s_th_substitution_from_th_voiceless_correct_01_take1,
target_sound s, intended_ipa θ, error_type th_substitution. No audio is copied for these rows.
"""
import csv, json, shutil
from pathlib import Path

repo = Path(__file__).parent
runs = repo / "runs"
SOURCES = {
    "gemini-3.8-live": sorted(runs.glob("final_*_ref")),
    "gpt-realtime-2.1": sorted(runs.glob("openai_gpt-realtime-2.1_*_ref")),
}
SUBS = list(csv.DictReader(open(repo / "error_types.csv", encoding="utf-8")))
SCORERS = ["zipa", "w2v2", "w2v2_espeak"]

for model, dirs in SOURCES.items():
    if not dirs:
        continue
    dest = runs / "dataset" / model
    dest.mkdir(parents=True, exist_ok=True)
    before = set(dest.glob("*.wav"))  # update in place: recreating the folder makes iCloud restore conflict copies
    rows = []
    for d in dirs:
        voice = d.name.split("_")[-2]
        scores = json.load(open(d / "scores_local.json", encoding="utf-8")) if (d / "scores_local.json").exists() else {}
        for r in csv.DictReader(open(d / "takes.csv", encoding="utf-8")):
            if not r["clip"]:
                continue
            id_ = f"{voice}_{r['prompt_id']}_take{r['take']}"
            out = dest / f"{id_}.wav"
            shutil.copy2(d / r["clip"], out)
            before.discard(out)
            heard = scores.get(Path(r["clip"]).name, {})
            row = {"id": id_, "file": out.name, "target_sound": r["target_sound"], "intended_ipa": r["intended_ipa"],
                   "error_type": r["error_type"], "model": model, "voice": voice, "source_id": "",
                   **{f"{k}_ipa": heard.get(k, "").replace("▁", "").strip() for k in SCORERS}, "human_review": ""}
            rows.append(row)
            if r["error_type"] != "correct":
                continue
            for s in SUBS:
                if s["produced_sound"] != r["target_sound"]:
                    continue
                rows.append({**row, "id": f"{voice}_{s['target_sound']}_{s['error_type']}_from_{r['prompt_id']}_take{r['take']}",
                             "target_sound": s["target_sound"], "error_type": s["error_type"], "source_id": id_})
    for stale in before:  # clips no longer in any source run
        stale.unlink()
    assert len({r["id"] for r in rows}) == len(rows), f"duplicate ids in {model}"
    with open(dest / f"{model}.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    n_sub = sum(1 for r in rows if r["source_id"])
    print(f"runs/dataset/{model}/: {len(rows) - n_sub} clips from {len(dirs)} voices, plus {n_sub} substitution rows")
