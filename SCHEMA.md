# Dataset schema (gold standard)

Every method (LLM voices, TTS, articulatory, Praat distortions) delivers clips in this format so the sets can be
combined and scored the same way. `build_dataset.py` writes it for the LLM sets; other methods can copy its columns.

## Audio

- 16 kHz, mono, 16-bit PCM WAV, 2 to 3 s (2.5 s here), the sound centred in light background noise (`tools.py prep`).
- Synthetic or adult voices only. No audio in git: clips live on Drive, the CSV and code live here.

## One CSV per set, one row per clip label

| Column | Required | Meaning | Example |
|---|---|---|---|
| `id` | yes | Unique row id within the set. | `Kore_s_th_substitution_from_th_voiceless_correct_01_take1` |
| `file` | yes | WAV file name, relative to the CSV. Several rows may share one file (see substitutions). | `Kore_th_voiceless_correct_01_take1.wav` |
| `target_sound` | yes | The sound the child was meant to say, one of the 23 values of `Sound` in Sara's `phoneme-endpoint` (`src/assets/types.py`). | `s` |
| `intended_ipa` | yes | IPA of the sound actually produced in the clip. | `θ` |
| `error_type` | yes | `correct`, a substitution from `error_types.csv`, or `<sound>_distortion`. Names follow `phoneme-endpoint`'s `error_type.py` without the `_initial`/`_medial`/`_final` suffix (isolated sounds have no word position). | `th_substitution` |
| `model` | yes | Generator, as specific as possible (model name and version, TTS engine, or method). | `gemini-3.8-live` |
| `voice` | yes | Voice or speaker name; blank if not applicable. | `Kore` |
| `source_id` | no | For a row that reuses or transforms another clip: that clip's `id`. Blank for original audio. | `Kore_th_voiceless_correct_01_take1` |
| `severity` | distortions | 1 (mild) to 5 (strong) for graded distortions. | `3` |
| `zipa_ipa`, `w2v2_ipa`, `w2v2_espeak_ipa` | no | What each local recognizer heard (`score_local.py`). Filled by the shared evaluation, not by hand. | `θ`, `f`, `θ` |
| `human_review` | no | Blank until an SLP or team listener reviews it: `ok`, `wrong:<what was heard>`, or `unsure`. | `ok` |

## Allowed values

- `target_sound`: r, s, l, z, th_voiced, th_voiceless, sh, ch, k, g, v, j, t, d, f, ng, y, w, h, n, m, p, b
  (`j` is dʒ as in "jam"; `y` is the glide j as in "yes").
- `intended_ipa`: one IPA sound: ɹ s l z ð θ ʃ tʃ k ɡ v dʒ t d f ŋ j w h n m p b. Use `tʃ`/`dʒ`, not `ʧ`/`ʤ`,
  and `ɡ` (U+0261), not `g`.
- Substitutions: the 66 rows of `error_types.csv` (`target_sound`, `error_type`, `produced_IPA`, `produced_sound`),
  taken from `EDIT_TYPE_MAPPING` in `error_type.py`. Omissions and `other_substitution` are out of scope.

## Substitution rows reuse correct audio

An isolated substituted sound is the same audio as a correct production of the sound said instead: "s said as θ" is
a θ. So the dataset does not generate substitutions separately. Each correct take of the produced sound gets one
extra row per substitution it stands for, pointing at the same `file`, with `source_id` set. A set therefore
covers 65 of the 66 substitution rows (`ts_substitution` has no source sound) as soon as it has the 23 correct sounds.
When a correct take is judged wrong in review, drop its substitution rows too (filter on `source_id`).

## Evaluation

`score_local.py DIR` fills the three `*_ipa` columns for any folder of WAVs, so other methods can run the same check
on their own clips before handing them in. None of the three recognizers is calibrated on isolated sounds
(`scorer_report.md`), so their outputs are a filter for review, not a label.
