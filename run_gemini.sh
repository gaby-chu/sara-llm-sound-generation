#!/bin/bash
# One voice: [RUN=final] [PROMPTS=gen_prompts_v4.csv] bash run_gemini.sh VOICE [MODE ...]   MODE = ref (default) | text | mimic
# Optional ONLY="n_correct_01.wav l_correct_01.wav" limits the prompts.
# Each mode: 3 takes -> runs/<RUN>_<VOICE>_<MODE>/, then triage and takes.csv.
cd "$(dirname "$0")"
RUN=${RUN:-final}; PROMPTS=${PROMPTS:-gen_prompts_v4.csv}
V=$1; shift; [ $# -eq 0 ] && set -- ref
for mode in "$@"; do
  OUT=runs/${RUN}_${V}_${mode}
  REF=(); [ $mode != text ] && REF=(--ref-dir sound_inputs --ref-mode $mode)
  for f in ${ONLY:-ALL}; do
    SEL=(); [ $f != ALL ] && SEL=(--only $f)
    .venv/bin/python -W ignore gen_gemini.py --project capstone-510119 --model gemini-3.8-live --live --voice $V \
      --takes 3 --prompts $PROMPTS --out $OUT ${REF[@]+"${REF[@]}"} ${SEL[@]+"${SEL[@]}"} 2>&1 | grep -E "FAILED|Traceback"
  done
  .venv/bin/python -W ignore triage_takes.py $OUT labels_v2.csv > /dev/null 2>&1
  .venv/bin/python -W ignore takes_csv.py $OUT --prompts $PROMPTS --labels labels_v2.csv ${REF[@]+"${REF[@]}"}
done
echo "DONE $V"
