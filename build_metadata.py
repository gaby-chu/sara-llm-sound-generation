#!/usr/bin/env python3
"""Metadata for the Gemini sets (runs/final_<VOICE>_ref), plus a generation log and a scorer comparison.

  python build_metadata.py        # after run_gemini.sh (generation + triage in takes.csv) and score_local.py

Writes, in runs/:
- final_metadata.csv: one row per clip. target_sound, produced_IPA (the intended IPA), error_type
  (correct), method, voice, source_clip (the input recording sent to the model, sound_inputs/<sound>.wav),
  scorer results, human_review (blank).
  Scorer results: <scorer>_ipa is what it heard, <scorer>_match is True when that equals produced_IPA. Gemini's
  match also needs no vowel after and no letter name or word. ZIPA and wav2vec2 output phone sequences, so they
  match only when the whole sequence is the target (an added vowel counts as no match).
- final_generation.csv: how each take was made (prompt file, system instruction, reference audio, prompt sent,
  raw file, length, crop, model transcript), including takes that came back with no audio. Joins on file.
- scorer_comparison.md: match rates, agreement and per-sound results for Gemini vs ZIPA vs wav2vec2.
"""
import csv, json, re
from collections import Counter, defaultdict
from pathlib import Path

repo = Path(__file__).parent
runs = repo / "runs"
labels = {Path(r["file"]).stem: r for r in csv.DictReader(open(repo / "labels_v2.csv", encoding="utf-8"))}

EQ = {"r": "ɹ", "ɚ": "ɹ", "ɝ": "ɹ", "y": "j", "ʧ": "tʃ", "ʤ": "dʒ"}


def norm(x: str) -> str:
    """Comparable IPA: drop word marks, spaces, slashes, length, syllabic and tie marks; unify a few spellings."""
    x = re.sub(r"[▁/\[\]ːˑ̩̯͡ˈˌ\s]", "", x or "").replace("ɡ", "g")
    return EQ.get(x, x)


def error_type(lab: dict) -> str:
    e = lab.get("error_type", "")
    if e == "correct":
        return "correct"
    if "distortion" in e:
        return "distortion"
    return f"sub:{lab.get('produced_IPA', '')}" if "substitution" in e else e


def truthy(v) -> bool:
    return str(v).strip().lower() in ("true", "1", "yes")


meta, gen = [], []
for d in sorted(runs.glob("final_*_ref")):
    _, voice, mode = d.name.split("_")
    local = json.load(open(d / "scores_local.json", encoding="utf-8")) if (d / "scores_local.json").exists() else {}
    for r in csv.DictReader(open(d / "takes.csv", encoding="utf-8")):
        lab = labels.get(r["prompt_id"], {})
        want = lab.get("produced_IPA", r["intended_ipa"])
        name = Path(r["clip"]).name
        file = f"runs/{d.name}/{r['clip']}" if r["clip"] else ""
        stem = f"{r['prompt_id']}_take{r['take']}"
        raw = d / "raw" / f"{stem}.wav"
        gen.append({
            "file": file, "prompt_id": r["prompt_id"], "take": r["take"], "model": r["model"], "voice": voice,
            "mode": mode, "prompt_file": "gen_prompts_v4.csv",
            "system_instruction": r["system_instruction"], "reference_audio": r["reference_audio"],
            "user_prompt": r["user_prompt"], "raw_file": f"runs/{d.name}/raw/{raw.name}" if raw.exists() else "",
            "raw_seconds": r["raw_seconds"], "cropped_to_1.8s": r["cropped_to_1.8s"],
            "model_transcript": r["model_transcript"], "status": r["status"], "source_run": r.get("source_run", d.name),
        })
        if not file:
            continue
        g_ipa = r["listener_heard_ipa"]
        z = local.get(name, {})
        z_ipa, w_ipa = z.get("zipa", "").replace("▁", "").strip(), z.get("w2v2", "")
        meta.append({
            "file": file, "target_sound": lab.get("target_sound", r["target_sound"]), "produced_IPA": want,
            "error_type": error_type(lab), "method": f"{r['model']} text prompt + reference recording",
            "voice": voice, "source_clip": r["reference_audio"],
            "gemini_ipa": g_ipa, "gemini_vowel_after": r["listener_vowel_after"],
            "gemini_letter_or_word": r["listener_letter_or_word"],
            "gemini_match": bool(g_ipa) and norm(g_ipa) == norm(want) and not truthy(r["listener_vowel_after"])
                            and not truthy(r["listener_letter_or_word"]),
            "zipa_ipa": z_ipa, "zipa_match": bool(z) and norm(z_ipa) == norm(want),
            "w2v2_ipa": w_ipa, "w2v2_match": bool(z) and norm(w_ipa) == norm(want),
            "human_review": "",
        })

for name, rows in [("final_metadata.csv", meta), ("final_generation.csv", gen)]:
    with open(runs / name, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    print(f"runs/{name}: {len(rows)} rows")


# ---- scorer comparison ----
S = [("gemini", "Gemini 2.5 Flash listener"), ("zipa", "ZIPA large CR-CTC"), ("w2v2", "wav2vec2 TIMIT phoneme")]


def contains(heard: str, want: str) -> bool:
    return bool(norm(want)) and norm(want) in norm(heard)


def kappa(a, b) -> float:
    n = len(a); po = sum(x == y for x, y in zip(a, b)) / n
    pa, pb = sum(a) / n, sum(b) / n
    pe = pa * pb + (1 - pa) * (1 - pb)
    return (po - pe) / (1 - pe) if pe < 1 else 1.0


scored = [m for m in meta if m["zipa_ipa"] or m["w2v2_ipa"] or m["file"]]
correct = [m for m in scored if m["error_type"] == "correct"]
lines = ["# Scorer comparison: Gemini listener vs ZIPA vs wav2vec2", "",
         f"{len(scored)} generated clips in runs/final_*_ref, all intended-correct sounds. No human labels yet, so these are agreement numbers, not accuracy.",
         "", "Match = the scorer heard exactly the intended IPA (Gemini: also no vowel after, no letter name/word). "
         "Contains = the intended sound appears somewhere in what the local model heard (it may have added a vowel).", ""]

human = repo / "runs" / "human_refs_scores.json"
if human.exists():
    h = json.load(open(human, encoding="utf-8"))
    lines += ["## Baseline: your 23 reference recordings (known correct)", "",
              "| Scorer | Match | Contains |", "|---|---|---|"]
    for k, title in S:
        m = sum(norm(v[k]) == norm(v["want"]) for v in h.values())
        c = sum(contains(v[k], v["want"]) for v in h.values())
        lines.append(f"| {title} | {m}/{len(h)} | {c}/{len(h)} |")
    lines += ["", "| Sound | Intended | Gemini | ZIPA | wav2vec2 |", "|---|---|---|---|---|"]
    for s, v in sorted(h.items()):
        lines.append(f"| {s} | {v['want']} | {v['gemini']} | {v['zipa'].replace('▁', '').strip()} | {v['w2v2']} |")
    lines.append("")

lines += ["## Generated clips: match rates (intended-correct sounds)", "",
          "| Scorer | Match | Contains |", "|---|---|---|"]
for k, title in S:
    m = sum(r[f"{k}_match"] for r in correct)
    c = sum(contains(r[f"{k}_ipa"], r["produced_IPA"]) for r in correct)
    lines.append(f"| {title} | {m}/{len(correct)} ({m / len(correct):.0%}) | {c}/{len(correct)} ({c / len(correct):.0%}) |")

lines += ["", "## Agreement on match / no match (intended-correct sounds)", "",
          "| Pair | Same decision | Cohen's kappa | Both match | Only first | Only second |", "|---|---|---|---|---|---|"]
for i in range(3):
    for j in range(i + 1, 3):
        a = [r[f"{S[i][0]}_match"] for r in correct]; b = [r[f"{S[j][0]}_match"] for r in correct]
        same = sum(x == y for x, y in zip(a, b))
        lines.append(f"| {S[i][0]} vs {S[j][0]} | {same}/{len(a)} ({same / len(a):.0%}) | {kappa(a, b):.2f} | "
                     f"{sum(x and y for x, y in zip(a, b))} | {sum(x and not y for x, y in zip(a, b))} | "
                     f"{sum(y and not x for x, y in zip(a, b))} |")

lines += ["", "## Per sound: clips matched (of all voices) and what each scorer heard most", "",
          "| Sound | IPA | n | Gemini | ZIPA | wav2vec2 | Gemini heard | ZIPA heard | wav2vec2 heard |",
          "|---|---|---|---|---|---|---|---|---|"]
by = defaultdict(list)
for r in scored:
    by[r["file"].split("/")[-1].rsplit("_take", 1)[0]].append(r)
for s, rs in sorted(by.items()):
    top = lambda k: ", ".join(f"{x or '–'} ×{n}" for x, n in Counter(r[f"{k}_ipa"] for r in rs).most_common(2))
    lines.append(f"| {s.replace('_correct_01', '')} | {rs[0]['produced_IPA']} | {len(rs)} | "
                 + " | ".join(str(sum(r[f'{k}_match'] for r in rs)) for k, _ in S)
                 + f" | {top('gemini')} | {top('zipa')} | {top('w2v2')} |")

lines += ["", "## By voice (Gemini / ZIPA / wav2vec2 matches)", "", "| Voice | Clips | Matches |", "|---|---|---|"]
for v in sorted({r["voice"] for r in correct}):
    rs = [r for r in correct if r["voice"] == v]
    lines.append(f"| {v} | {len(rs)} | {' / '.join(str(sum(r[f'{k}_match'] for r in rs)) for k, _ in S)} |")

open(runs / "scorer_comparison.md", "w", encoding="utf-8").write("\n".join(lines) + "\n")
print("runs/scorer_comparison.md")
