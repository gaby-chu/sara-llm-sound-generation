#!/usr/bin/env python3
"""One score table for the Gemini (gemini-3.8-live) and OpenAI (gpt-realtime-2.1) clips, with each model's clips
gathered in its own folder.

  python build_scores.py        # after triage (takes.csv) and score_local.py on every source run

Sources: runs/final_<VOICE>_<MODE>/ (Gemini, from combine_final.py) and runs/openai_gpt-realtime-2.1_<VOICE>_<MODE>/.
Copies every clip to runs/<model>/<VOICE>_<MODE>_<clip>.wav (the source runs are untouched) and writes
runs/<model>/<model>_scores.csv per model and runs/combined_scores.csv with both.

Columns as in runs/scorer_report_clips.csv (see scorer_report.md for the rules), plus model and source_clip:
- gemini / zipa / w2v2: what each scorer heard (Gemini 2.5 Flash listener from takes.csv, ZIPA and wav2vec2 from
  scores_local.json).
- <scorer>_strict: the whole output is the intended IPA (Gemini: also no vowel after, no letter name or word).
- <scorer>_cons: the consonants are the intended ones, after dropping vowels and glottal stops and collapsing repeats
  (so 'bə' counts as /b/, 'p a' does not).
"""
import csv, json, re, shutil
from pathlib import Path

repo = Path(__file__).parent
runs = repo / "runs"
labels = {Path(r["file"]).stem: r for r in csv.DictReader(open(repo / "labels_v2.csv", encoding="utf-8"))}
SOURCES = {
    "gemini-3.8-live": sorted(runs.glob("final_*_*")),
    "gpt-realtime-2.1": sorted(runs.glob("openai_gpt-realtime-2.1_*_*")),
}

EQ = {"r": "ɹ", "ɚ": "ɹ", "ɝ": "ɹ", "y": "j", "ʧ": "tʃ", "ʤ": "dʒ"}  # build_metadata.py's strict spellings
CONS_EQ = {"ʧ": "tʃ", "ʤ": "dʒ", "ɡ": "g", "ɫ": "l", "r": "ɹ", "ɚ": "ɹ", "ɝ": "ɹ", "y": "j"}
VOWELS = set("aeiouæɑɒɐəɛɜɪʊʌɔøœɯɤʏɨʉɘɵɞɶʔ")  # and the glottal stop, also dropped for the consonant match


def norm(x: str) -> str:
    """build_metadata.norm: drop word marks, spaces, slashes, length, syllabic and tie marks; unify a few spellings."""
    x = re.sub(r"\sʰ", "", x or "")  # ZIPA outputs aspiration as its own token; the report drops it (not Gemini's pʰ)
    x = re.sub(r"[▁/\[\]ːˑ\u0329\u032f\u0361ˈˌ\s]", "", x).replace("ɡ", "g")
    return EQ.get(x, x)


def strict(heard: str, want: str) -> bool:
    return bool(norm(heard)) and norm(heard) == norm(want)


def consonants(x: str) -> str:
    """Characters left after unifying spellings, dropping length/aspiration/syllabic marks, vowels and ʔ, and
    collapsing repeats (so 'ʧʃ' and 't ʃ' are both tʃ, while 'd̪' keeps its dental mark)."""
    x = re.sub(r"[▁/\[\]ːˑ\u0329\u032f\u0361ˈˌʰ\s]", "", x or "")
    x = "".join(CONS_EQ.get(c, c) for c in x)
    out = ""
    for c in x:
        if c not in VOWELS and not out.endswith(c):
            out += c
    return out


def cons(heard: str, want: str) -> bool:
    return bool(consonants(heard)) and consonants(heard) == consonants(want)


def truthy(v) -> bool:
    return str(v).strip().lower() in ("true", "1", "yes")


def error_type(lab: dict) -> str:
    e = lab.get("error_type", "")
    return "distortion" if "distortion" in e else e


all_rows = []
for model, dirs in SOURCES.items():
    dest = runs / model
    dest.mkdir(exist_ok=True)
    before = set(dest.glob("*.wav"))  # update in place: recreating the folder makes iCloud restore conflict copies
    rows = []
    for d in dirs:
        voice, mode = d.name.split("_")[-2:]
        local = json.load(open(d / "scores_local.json", encoding="utf-8"))
        for r in csv.DictReader(open(d / "takes.csv", encoding="utf-8")):
            if not r["clip"]:
                continue
            lab = labels.get(r["prompt_id"], {})
            want = lab.get("produced_IPA") or r["intended_ipa"]
            src = d / r["clip"]
            out = dest / f"{voice}_{mode}_{src.name}"
            shutil.copy2(src, out)
            before.discard(out)
            s = local.get(src.name, {})
            g, z, w = r["listener_heard_ipa"], s.get("zipa", "").replace("▁", "").strip(), s.get("w2v2", "")
            vowel, letter = truthy(r["listener_vowel_after"]), truthy(r["listener_letter_or_word"])
            rows.append({
                "file": str(out.relative_to(repo)), "model": model, "sound": r["prompt_id"].replace("_correct_01", ""),
                "error_type": error_type(lab), "voice": voice, "mode": mode, "want": want,
                "gemini": g, "zipa": z, "w2v2": w,
                "gemini_vowel_after": vowel, "gemini_letter_or_word": letter,
                "gemini_strict": strict(g, want) and not vowel and not letter,
                "zipa_strict": strict(z, want), "w2v2_strict": strict(w, want),
                "gemini_cons": cons(g, want), "zipa_cons": cons(z, want), "w2v2_cons": cons(w, want),
                "source_clip": str(src.relative_to(repo)),
            })
    for stale in before:  # clips no longer in any source run
        stale.unlink()
    with open(dest / f"{model}_scores.csv", "w", newline="", encoding="utf-8") as f:
        wr = csv.DictWriter(f, fieldnames=list(rows[0])); wr.writeheader(); wr.writerows(rows)
    all_rows += rows
    n = len(rows)
    rate = lambda k: f"{sum(r[k] for r in rows) / n:.0%}"
    print(f"runs/{model}/: {n} clips from {len(dirs)} runs; strict / consonant match: "
          + ", ".join(f"{s} {rate(s + '_strict')} / {rate(s + '_cons')}" for s in ("gemini", "zipa", "w2v2")))

with open(runs / "combined_scores.csv", "w", newline="", encoding="utf-8") as f:
    wr = csv.DictWriter(f, fieldnames=list(all_rows[0])); wr.writeheader(); wr.writerows(all_rows)
print(f"runs/combined_scores.csv: {len(all_rows)} rows")
