#!/bin/bash
# One OpenAI voice: [MODEL=gpt-audio-1.5] [PROMPTS=gen_prompts_v4.csv] bash run_openai.sh VOICE MODE [MODE ...]   MODE = text | ref | mimic
# Optional ONLY="n_correct_01.wav l_correct_01.wav" limits the prompts. Needs OPENAI_API_KEY (generation) and
# gcloud credentials (triage still uses the Gemini listener, so results compare with the Gemini runs).
# Each mode: 3 takes -> runs/openai_<MODEL>_<VOICE>_<MODE>/, then triage and takes.csv.
cd "$(dirname "$0")"
MODEL=${MODEL:-gpt-audio-1.5}; PROMPTS=${PROMPTS:-gen_prompts_v4.csv}
RT=(); [[ $MODEL == gpt-realtime* ]] && RT=(--realtime)
V=$1; shift
for mode in "$@"; do
  OUT=runs/openai_${MODEL}_${V}_${mode}
  REF=(); [ $mode != text ] && REF=(--ref-dir sound_inputs --ref-mode $mode)
  for f in ${ONLY:-ALL}; do
    SEL=(); [ $f != ALL ] && SEL=(--only $f)
    .venv/bin/python -W ignore gen_openai.py --model $MODEL ${RT[@]+"${RT[@]}"} --voice $V \
      --takes 3 --prompts $PROMPTS --out $OUT ${REF[@]+"${REF[@]}"} ${SEL[@]+"${SEL[@]}"} 2>&1 | grep -E "FAILED|Traceback"
  done
  .venv/bin/python -W ignore triage_takes.py $OUT labels_v2.csv > /dev/null 2>&1
  .venv/bin/python -W ignore takes_csv.py $OUT --prompts $PROMPTS --labels labels_v2.csv ${REF[@]+"${REF[@]}"}
done
echo "DONE $V"
