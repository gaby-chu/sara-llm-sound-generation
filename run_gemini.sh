#!/bin/bash
# One voice: [RUN=final] [PROMPTS=gen_prompts_v4.csv] bash run_gemini.sh VOICE
# Optional ONLY="n_correct_01.wav l_correct_01.wav" limits the prompts.
# 3 takes per prompt (reference recording + prompt) -> runs/<RUN>_<VOICE>_ref/, then takes.csv.
cd "$(dirname "$0")"
RUN=${RUN:-final}; PROMPTS=${PROMPTS:-gen_prompts_v4.csv}
V=$1
OUT=runs/${RUN}_${V}_ref
for f in ${ONLY:-ALL}; do
  SEL=(); [ $f != ALL ] && SEL=(--only $f)
  .venv/bin/python -W ignore gen_gemini.py --project capstone-510119 --model gemini-3.8-live --voice $V \
    --takes 3 --prompts $PROMPTS --out $OUT ${SEL[@]+"${SEL[@]}"} 2>&1 | grep -E "FAILED|Traceback"
done
.venv/bin/python -W ignore takes_csv.py $OUT --prompts $PROMPTS --labels labels_v2.csv
echo "DONE $V"
