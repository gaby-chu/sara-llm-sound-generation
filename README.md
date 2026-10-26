# Gemini native-audio sound generation

Generates isolated speech sounds with Gemini's live native-audio model on Vertex AI and OpenAI's realtime model,
scores them locally with ZIPA and wav2vec2, and builds per-clip metadata. 

## Setup

```bash
uv venv --python 3.13 .venv && uv pip install --python .venv/bin/python -r requirements.txt
gcloud auth application-default login     # only for generating with Gemini (Vertex AI)
```

Needs `ffmpeg` on the PATH. The GCP project `capstone-510119` is hard-coded in `run_gemini.sh`.
ZIPA and wav2vec2 models download to `~/.cache/huggingface` on first use; after that, prefix with `HF_HUB_OFFLINE=1`.

## Pipeline

Every clip is made from a reference recording (`sound_inputs/<sound>.wav`) plus that sound's prompt in
`gen_prompts_v4.csv` ("ref" mode), for the 23 correct sounds.

1. `bash run_gemini.sh VOICE` (optional `PROMPTS=... ONLY="n_correct_01.wav ..."`): `gen_gemini.py` makes 3 takes
   per prompt with `gemini-3.8-live` (using `tools.py` for the clip prep), then `takes_csv.py` writes
   `runs/final_<VOICE>_ref/takes.csv`.
   Done for all 8 voices: Aoede, Charon, Fenrir, Kore, Leda, Orus, Puck, Zephyr.
2. `bash run_openai.sh VOICE`: the same with `gen_openai.py` and OpenAI's `gpt-realtime-2.1` (Realtime API), into
   `runs/openai_gpt-realtime-2.1_<VOICE>_ref/`. Needs `OPENAI_API_KEY` (in `.env`). Done for all 10 voices the
   model accepts: alloy, ash, ballad, cedar, coral, echo, marin, sage, shimmer, verse. Takes with a raw peak under 0.05 are kept in `raw/` but not
   prepped. Quiet f/θ often fall under that, so coral f and sage θ (no take above it in 8 retries) use a take
   prepped by hand: take 2, a real fricative 24–31 dB above the noise floor.
3. `python score_local.py runs/final_*_ref runs/openai_*_ref`: ZIPA and wav2vec2 transcriptions, `scores_local.json`
   per folder.
4. `python build_metadata.py`: `runs/final_metadata.csv`, `runs/final_generation.csv`, `runs/scorer_comparison.md`
   (Gemini only). `python build_scores.py`: copies every clip into `runs/gemini-3.8-live/` and
   `runs/gpt-realtime-2.1/` as `<VOICE>_ref_<clip>.wav`, with per-model score tables and `runs/combined_scores.csv`.
5. `python build_dataset.py`: the final dataset, `runs/dataset/<model>/` with every take that has audio as
   `<id>.wav` and `<model>.csv` (id, target_sound, intended_ipa, error_type, voice).

## Inputs and results

- `gen_prompts_v4.csv`, `labels_v2.csv`: prompts and labels for the 23 correct sounds.
- `sound_inputs/`: the 23 reference recordings sent with each prompt.
- `scorer_report.md`: Gemini listener vs ZIPA vs wav2vec2 (written for the earlier text + ref runs; the Gemini
  listener, `triage_takes.py`, has since been removed).
- `runs/`: all audio and run data. Like every `.wav`, it is gitignored, so it lives only on disk.
- `runs_archive/`: earlier runs that aren't part of the set (text-prompt runs, the v3/v4/v5 runs the first five
  Gemini sets were combined from, `gpt-audio-1.5` runs, R distortion takes, old score tables). Also gitignored.
