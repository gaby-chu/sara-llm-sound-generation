# Gemini self-labeler vs ZIPA vs wav2vec2

Run 2026-10-19. ZIPA (`anyspeech/zipa-large-crctc-500k`, ONNX, greedy CTC) and wav2vec2 (`vitouphy/wav2vec2-xls-r-300m-timit-phoneme`) were re-run from scratch on CPU with `score_local.py`'s loaders: 742 generated clips in `runs/final_*` (684 intended-correct, 58 R distortions) plus the 23 reference recordings in `sound_inputs/`, ~4 min total. Gemini labels are the existing `gemini-2.5-flash` listener results in each run's `takes.csv` (not re-queried).

Files: this report; `runs/scorer_report_clips.csv` (one row per clip: what each scorer heard, strict and consonant
match); `runs/local_scores_rerun_2026-10-19.json` and `runs/local_scores_refs_prepped_2026-10-19.json` (raw fresh outputs).

## Summary

- **None of the three is a usable automatic scorer for isolated sounds yet.** On the 23 known-correct reference
  recordings, the best consonant-level accuracy is 48% (wav2vec2), then Gemini 43%, then ZIPA 22% (§1).
- **Gemini is the most lenient**: it accepts 61% of generated clips, against 44% for wav2vec2 and 27% for ZIPA (§2).
  The scorers agree only weakly (Cohen's κ, consonant match: Gemini–wav2vec2 0.37, Gemini–ZIPA 0.22,
  ZIPA–wav2vec2 0.30). The local models agree with each other no better than each agrees with Gemini (§3).
- **Gemini's "yes" is not specific for several sound groups.** It says "m" for 88 clips, but only 30 of those asked
  for /m/ (26 /n/, 20 /ŋ/). It says "s" for 59, of which 29 asked for /s/ (the rest were /f/ and /θ/), and "ʃ" for 52, of
  which 27 asked for /ʃ/ (13 were /tʃ/). Its 30/30 on /m/ and 29/30 on /s/ therefore say little (§4b). In 30% of the
  clips Gemini accepts, neither local model hears the target. When at least one local model does hear it,
  Gemini agrees 80% of the time (§3).
- **Gemini almost never reports an added vowel** (3% of clips), while ZIPA and wav2vec2 each transcribe a vowel in
  53–63% (κ ≈ 0). On the reference recordings, Gemini did write vowels (pɑ, gə, d͡ʒɑ), so it can hear them. Either
  the generated clips really are cleaner, or the local models hear a release burst as a vowel. Only listening can
  settle which (§4).
- **The local models fail differently.** ZIPA outputs nothing for isolated fricatives (/s/ 0/30, /ʃ/ 0/29, /f/ 0/30),
  though it handles /s/ inside a word. wav2vec2 is the best of the three on /tʃ/, /d/ and /h/. Their outputs also change
  with the background-noise seed of the clip prep (6–9 of 23 references), so a single clip's verdict is noisy.
- **Possible self-preference, weak evidence.** Gemini accepts generated clips more often (61%) than it is right on the
  human references (43%). wav2vec2 shows no such gap (44% vs 48%). With only 23 references, and the generated clips
  being studio-clean, this does not prove that Gemini favours Gemini audio.
- **Next step:** a human listening pass on a stratified sample. The most informative clips are where Gemini and both
  local models disagree (126 clips, mostly m, ð, r, s, k, ʃ) and the nasal and fricative groups. That would turn these
  agreement numbers into accuracy.

## Checks

- The fresh run reproduced every saved score in `runs/final_*/scores_local.json` exactly (742/742 clips, both models).
- The reference recordings reproduce `runs/human_refs_scores.json` exactly once they get the same prep as generated clips (trim, peak-normalise, centre in 2.5 s of -60 dBFS noise, seed 0). Scored raw, 22/23 outputs differ.
- The noise is not neutral: re-prepping the references with noise seeds 0/1/2 changed the output for 6/23 references under ZIPA and 9/23 under wav2vec2 (e.g. k: dɛ / də / kɛ). Single-clip outputs from the local models are fragile.
- Both local models work on ordinary words (macOS `say`): sister → ZIPA `s ɪ s t ə˞`, wav2vec2 `sɪs tɝ`; think → `s θ æ ŋ k` / `θɪŋ k`; go → `g o` / `gəm`. Their failures below come from isolated sounds being out of their training distribution, not from a broken pipeline. ZIPA in particular outputs nothing for most isolated fricatives (30/30 /s/ clips empty) though it transcribes /s/ fine inside "sister".
- The Gemini listener is blind: it gets only the audio and a generic prompt, never the intended sound. But it is asked for "the single main sound", while ZIPA/wav2vec2 transcribe every phone. So a strict whole-string match penalises the local models for every added vowel. This report adds a consonant match: drop vowels and glottal stops from the output, collapse repeats, and compare the remaining consonants to the target (so `bə` counts as /b/, `p a` does not). For Gemini the consonant match also ignores its own vowel-after flag.

Definitions: **strict** = whole output equals the target (Gemini: also vowel_after and letter_or_word both false; same rule as `build_metadata.py`). **consonant** = as above. Small spelling unifications for all scorers: ʧ=tʃ, ʤ=dʒ, ɡ=g, ɫ=l, r/ɚ/ɝ=ɹ, y=j, length and aspiration marks dropped.

## 1. Ground truth: the 23 reference recordings (known-correct productions)

This is the only part with real labels. Every clip is a correct production, so the rate is accuracy.

| Scorer | Strict | Consonant |
|---|---|---|
| Gemini 2.5 Flash (blind listener) | 7/23 (30%) | 10/23 (43%) |
| ZIPA large CR-CTC | 0/23 (0%) | 5/23 (22%) |
| wav2vec2 TIMIT phoneme | 2/23 (9%) | 11/23 (48%) |

| Sound | Target | Gemini | ZIPA | wav2vec2 |
|---|---|---|---|---|
| b | b | pɑ | p a | bɑ ✓ |
| ch | tʃ | ʃ | – | ʧ ✓ |
| d | d | ð | t a | ðə |
| f | f | p | – | təəʃə |
| g | ɡ | gə ✓ | k a | gə ✓ |
| h | h | h ✓ | a | hɑ ✓ |
| j | dʒ | d͡ʒɑ ✓ | d ɑ | ʤə ✓ |
| k | k | x | – | dɛ |
| l | l | ɫ ✓ | ə l a ✓ | ʊlə ✓ |
| m | m | m ✓ | ə n ə | m ✓ |
| n | n | m | ə | un ✓ |
| ng | ŋ | n | ʔ i ŋ ✓ | ɪun |
| p | p | p ✓ | p a ✓ | bə |
| r | ɹ | ɹ̩ ✓ | a r ✓ | ʊɝ ✓ |
| s | s | θ | – | – |
| sh | ʃ | ʃ ✓ | – | ʃʃ ✓ |
| t | t | p | – | ðə |
| th_voiced | ð | m | x a a x a a x a | θumʊv |
| th_voiceless | θ | ð | – | pʊʃ |
| v | v | ə | ə | muvə |
| w | w | wʌ ✓ | w ɔ ✓ | wə ✓ |
| y | j | iː | ə e | eɪ |
| z | z | z ✓ | s ɯ p a | zʊvuzəm |

✓ = consonant match.

## 2. Generated clips: how often each scorer says "correct" (684 intended-correct clips)

No human labels, so these are acceptance rates, not accuracy.

| Scorer | Strict | Consonant |
|---|---|---|
| Gemini 2.5 Flash (blind listener) | 380/684 (56%) | 414/684 (61%) |
| ZIPA large CR-CTC | 47/684 (7%) | 187/684 (27%) |
| wav2vec2 TIMIT phoneme | 83/684 (12%) | 303/684 (44%) |

## 3. Agreement between scorers (consonant match, 684 clips)

| Pair | Same decision | Cohen's κ | Both yes | Only first | Only second | Both no |
|---|---|---|---|---|---|---|
| gemini vs zipa | 391/684 (57%) | 0.22 | 154 | 260 | 33 | 237 |
| gemini vs w2v2 | 463/684 (68%) | 0.37 | 248 | 166 | 55 | 215 |
| zipa vs w2v2 | 458/684 (67%) | 0.30 | 132 | 55 | 171 | 326 |

Strict-match κ for comparison: gemini–zipa 0.08, gemini–w2v2 0.11, zipa–w2v2 0.17.

- Gemini accepts 414 clips; for 126/414 (30%) of them neither local model hears the target consonant. Most common sounds there: m ×25, th_voiced ×13, r ×13, s ×12, k ×10, sh ×10.
- When at least one local model hears the target (358 clips), Gemini agrees on 288/358 (80%).
- When both local models hear it (132 clips), Gemini agrees on 114.

## 4. Added vowel: Gemini's vowel_after flag vs a vowel in the local transcripts

564 correct-target clips (glides and liquids w/y/r/l left out).

| Scorer | Says a vowel is present | κ with Gemini's flag |
|---|---|---|
| Gemini vowel_after flag | 19/564 (3%) | – |
| Gemini heard_ipa has a vowel | 22/564 (4%) | 0.82 |
| zipa transcript has a vowel | 301/564 (53%) | 0.01 |
| w2v2 transcript has a vowel | 356/564 (63%) | 0.04 |

Both local models put a vowel in 242/564 (43%) of these clips; Gemini flags 11 of those.

## 4b. Label bias: how specific is a "yes"?

A match for target X only means something if the scorer does not also say X for other targets. For each label, how many clips asked for that sound got it, versus clips that asked for a different sound (consonant match; intended-correct clips; each target has ~30 clips). Labels given to other targets ≥8 times:

| Scorer | Label | Given to its own target | Given to other targets (which) |
|---|---|---|---|
| gemini | m | 30 | 58 (n 26, ng 20, v 4, l 4) |
| gemini | ð | 15 | 55 (d 12, ng 7, v 7, f 5) |
| gemini | s | 29 | 30 (f 12, th_voiceless 12, h 2, sh 2) |
| gemini | ʃ | 27 | 25 (ch 13, f 5, th_voiceless 5, v 1) |
| gemini | z | 26 | 12 (th_voiced 6, v 5, th_voiceless 1) |
| gemini | p | 26 | 8 (b 5, f 2, h 1) |
| zipa | n | 12 | 51 (ng 14, m 13, z 8, r 5) |
| zipa | t | 27 | 29 (ch 18, d 4, th_voiced 3, b 1) |
| zipa | p | 25 | 19 (b 15, th_voiceless 1, l 1, z 1) |
| zipa | k | 14 | 15 (g 11, th_voiceless 2, r 1, t 1) |
| w2v2 | d | 26 | 17 (k 5, th_voiced 5, v 2, th_voiceless 2) |
| w2v2 | n | 7 | 17 (ng 4, l 3, m 3, th_voiced 3) |
| w2v2 | h | 19 | 14 (k 3, th_voiceless 2, f 2, l 2) |
| w2v2 | z | 18 | 11 (v 4, th_voiced 3, f 1, s 1) |
| w2v2 | g | 22 | 9 (k 9) |
| w2v2 | s | 17 | 9 (th_voiceless 5, f 2, h 1, z 1) |
| w2v2 | m | 5 | 9 (ng 3, v 2, b 1, l 1) |
| w2v2 | p | 17 | 8 (f 3, b 2, h 1, th_voiceless 1) |

## 5. Per sound (consonant match; all voices and modes)

| Sound | Target | n | Gemini | ZIPA | wav2vec2 | Gemini heard most | ZIPA heard most | wav2vec2 heard most |
|---|---|---|---|---|---|---|---|---|
| b | b | 30 | 23 | 12 | 23 | b ×21, p ×5 | p a ×11, b a ×9 | bə ×19, ðə ×2 |
| ch | tʃ | 30 | 11 | 0 | 22 | ʃ ×13, tʃ ×11 | t ×18, – ×5 | ʧ ×20, ʃ ×3 |
| d | d | 30 | 17 | 22 | 26 | d ×16, ð ×10 | d a ×9, d i ×4 | də ×19, dɪ ×4 |
| f | f | 30 | 2 | 0 | 0 | s ×11, ð ×5 | – ×24, a ×2 | – ×15, p ×2 |
| g | ɡ | 29 | 24 | 15 | 22 | ɡ ×15, g ×8 | g a ×7, k ɤ ×3 | gə ×11, gʊ ×11 |
| h | h | 29 | 16 | 0 | 19 | h ×15, ɑ ×4 | – ×24, a ×5 | hɑ ×12, ɑ ×2 |
| j | dʒ | 30 | 27 | 19 | 17 | d͡ʒ ×13, dʒ ×11 | d ʒ a ×14, d ʒ ə ×2 | ʤə ×12, nʃə ×3 |
| k | k | 30 | 28 | 14 | 8 | k ×28, s ×1 | – ×14, k ×12 | kɪ ×5, dɪ ×3 |
| l | l | 30 | 7 | 4 | 3 | uː ×9, l ×5 | – ×5, ə ×4 | – ×7, ʊ ×3 |
| m | m | 30 | 30 | 0 | 5 | m ×28, m̩ ×1 | ə n ×7, – ×6 | – ×11, m ×4 |
| n | n | 30 | 3 | 12 | 7 | m ×25, n ×3 | ə n ×8, – ×6 | u ×11, – ×8 |
| ng | ŋ | 28 | 1 | 0 | 0 | m ×18, ð ×7 | ə n ×7, – ×5 | u ×7, – ×4 |
| p | p | 30 | 26 | 25 | 17 | p ×25, b ×3 | p a ×15, p ×5 | pə ×9, pɑ ×6 |
| r | ɹ | 30 | 24 | 2 | 10 | r ×17, ɹ ×5 | – ×5, ə ×3 | ɝɝ ×2, wə ×2 |
| r_distortion_labialized_01 | ɹʷ | 29 | 0 | 0 | 0 | r ×17, ɹ ×4 | ə ×5, a ×3 | ɝɝɝ ×2, ɪ ×1 |
| r_distortion_retroflex_labialized_01 | ɻʷ | 29 | 0 | 0 | 0 | r ×7, ɹ ×5 | ə ×3, ə n ×2 | ɝɝ ×3, ʊ ×2 |
| s | s | 30 | 29 | 0 | 17 | s ×29, r ×1 | – ×30 | s ×16, – ×11 |
| sh | ʃ | 29 | 27 | 0 | 17 | ʃ ×27, s ×2 | – ×26, ɯ ×1 | ʃ ×13, – ×5 |
| t | t | 29 | 25 | 27 | 27 | t ×25, ð ×3 | t ×23, t a ×3 | t ×17, tɪ ×4 |
| th_voiced | ð | 30 | 15 | 1 | 1 | ð ×14, z ×6 | – ×2, ə ×2 | – ×4, dʊ ×2 |
| th_voiceless | θ | 30 | 3 | 0 | 0 | s ×12, ʃ ×5 | – ×23, p ʰ a ×1 | – ×15, s ×2 |
| v | v | 30 | 12 | 1 | 3 | v ×12, ð ×7 | – ×8, ə ×4 | – ×6, ə ×2 |
| w | w | 30 | 25 | 20 | 25 | w ×19, wə ×2 | w ɔ ×7, w a ×5 | wə ×21, uwə ×2 |
| y | j | 30 | 13 | 12 | 16 | iː ×8, j ×8 | i ×5, i j a ×4 | jə ×6, ʤɪ ×3 |
| z | z | 30 | 26 | 1 | 18 | z ×26, ð ×3 | – ×7, ə ×4 | z ×5, zɪ ×2 |

## 6. By voice and mode (consonant match, Gemini / ZIPA / wav2vec2, of ~69 clips each)

| Voice | text prompt | prompt + reference |
|---|---|---|
| Fenrir | 39 / 19 / 27 | 49 / 23 / 37 |
| Kore | 45 / 15 / 27 | 43 / 19 / 31 |
| Leda | 38 / 17 / 27 | 41 / 21 / 31 |
| Puck | 41 / 18 / 30 | 43 / 20 / 33 |
| Zephyr | 35 / 18 / 29 | 40 / 17 / 31 |

## 7. R distortions (58 clips, targets ɹʷ and ɻʷ)

No scorer can confirm a distortion. Gemini (asked for one main sound) mostly says plain r/ɹ: r ×24, ɹ ×9, ɹ̩ ×4, v ×4. ZIPA: ə ×8, a ×5, ə n ×4, – ×3. wav2vec2: ɝɝ ×4, ɝɝɝ ×2, ʊʊ ×2, ʊ ×2.

