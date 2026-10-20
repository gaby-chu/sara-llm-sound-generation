# Gemini native-audio sound generation

Generates isolated speech sounds with Gemini's live native-audio model on Vertex AI, labels each take with a
second Gemini model, scores them locally with ZIPA and wav2vec2, and builds per-clip metadata. Split out of
[mihir-agarwal0211/sara-multimodal-llm-sounds](https://github.com/mihir-agarwal0211/sara-multimodal-llm-sounds),
keeping only the current pipeline (the first pilot's `finalize.py`, `triage.py`, `triage_rw.py` and their data were left behind).

## Setup

```bash
uv venv --python 3.13 .venv && uv pip install --python .venv/bin/python -r requirements.txt
gcloud auth application-default login     # only for generating and labeling (Vertex AI)
```

Needs `ffmpeg` on the PATH. The GCP project `capstone-510119` is hard-coded in `run_v4.sh` and `triage_takes.py`.
ZIPA and wav2vec2 models download to `~/.cache/huggingface` on first use; after that, prefix with `HF_HUB_OFFLINE=1`.

## Pipeline

1. `bash run_v4.sh VOICE text ref` (optional `RUN=v5 PROMPTS=gen_prompts_v4.csv ONLY="n_correct_01.wav ..."`):
   `gen_gemini.py` makes 3 takes per prompt (using `tools.py` for the clip prep), then `triage_takes.py` asks
   `gemini-2.5-flash` what it hears, then `takes_csv.py` writes `runs/<RUN>_<VOICE>_<MODE>/takes.csv`.
2. `python combine_final.py Fenrir Kore Leda Puck Zephyr`: builds `runs/final_<VOICE>_<MODE>/` from the latest v4 run
   (v3 where there is none) plus the v5 takes of the rewritten sounds.
3. `python score_local.py runs/final_*`: ZIPA and wav2vec2 transcriptions, `scores_local.json` per folder.
4. `python build_metadata.py`: `runs/final_metadata.csv`, `runs/final_generation.csv`, `runs/scorer_comparison.md`.

## Inputs and results

- `gen_prompts_v3.csv` (v3/v4 runs), `gen_prompts_v4.csv` (v5 runs, the default), `labels_v2.csv`: prompts and labels.
- `sound_inputs/`: the 23 reference recordings sent with "ref" prompts.
- `scorer_report.md`: Gemini listener vs ZIPA vs wav2vec2.
- `runs/`: all audio and run data. Like every `.wav`, it is gitignored, so it lives only on disk.
