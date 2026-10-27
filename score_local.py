#!/usr/bin/env python3
"""Local phone recognition for clips: ZIPA (ONNX) and wav2vec2, no API calls.

  python score_local.py DIR [DIR ...]      # each DIR has clips/*.wav, or is a folder of .wav files

Writes DIR/scores_local.json: {clip file name: {"zipa": "t ʃ", "w2v2": "tʃ", "w2v2_espeak": "tʃ"}}, phones as each
model outputs them. Clips already scored by every model are skipped, so adding a model only runs that model.
Matching against the intended sound happens in build_metadata.py.

Models (downloaded to ~/.cache/huggingface on first use):
- ZIPA: anyspeech/zipa-large-crctc-500k, model.onnx (CC BY 4.0), 80-bin Kaldi fbank, greedy CTC; feature setup and
  decoding as in github.com/lingjzhu/zipa inference/utils.py.
- wav2vec2: vitouphy/wav2vec2-xls-r-300m-timit-phoneme, the model behind Sara's phoneme-endpoint.
- wav2vec2 espeak: facebook/wav2vec2-lv-60-espeak-cv-ft, a multilingual IPA phone recognizer trained on different data,
  so it is a check that does not share the TIMIT model's blind spots. The tokenizer is loaded with
  do_phonemize=False, so neither phonemizer nor espeak needs to be installed.
Needs onnxruntime<=1.21 on macOS 13.2 (newer wheels are built for 13.4+).
"""
import json, sys
from pathlib import Path

import numpy as np
import librosa
import torch

ZIPA_REPO = "anyspeech/zipa-large-crctc-500k"
W2V2_REPO = "vitouphy/wav2vec2-xls-r-300m-timit-phoneme"
ESPEAK_REPO = "facebook/wav2vec2-lv-60-espeak-cv-ft"


def load_zipa():
    import onnxruntime as ort
    from huggingface_hub import hf_hub_download
    from lhotse.features.kaldi.extractors import Fbank, FbankConfig
    session = ort.InferenceSession(hf_hub_download(ZIPA_REPO, "model.onnx"))
    vocab = {}
    for line in open(hf_hub_download(ZIPA_REPO, "tokens.txt"), encoding="utf-8"):
        parts = line.split()
        if parts:
            vocab[int(parts[1]) if len(parts) > 1 else len(vocab)] = parts[0]
    fbank = Fbank(FbankConfig(num_filters=80, dither=0.0, snip_edges=False))

    def run(audio: np.ndarray) -> str:
        feat = fbank.extract_batch([torch.from_numpy(audio).float().unsqueeze(0)], sampling_rate=16000)[0].unsqueeze(0)
        logp = session.run(None, {"x": feat.numpy(), "x_lens": np.array([feat.shape[1]], dtype=np.int64)})[0][0]
        out, prev = [], -1
        for i in logp.argmax(-1):  # greedy CTC, blank = 0
            if i != 0 and i != prev:
                out.append(vocab.get(int(i), ""))
            prev = i
        return " ".join(out)
    return run


def load_w2v2():
    from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor
    proc = Wav2Vec2Processor.from_pretrained(W2V2_REPO)
    model = Wav2Vec2ForCTC.from_pretrained(W2V2_REPO).eval()

    def run(audio: np.ndarray) -> str:
        with torch.no_grad():
            logits = model(proc(audio, sampling_rate=16000, return_tensors="pt").input_values).logits
        return proc.batch_decode(logits.argmax(-1))[0].strip()
    return run


def load_w2v2_espeak():
    from transformers import Wav2Vec2FeatureExtractor, Wav2Vec2ForCTC, Wav2Vec2PhonemeCTCTokenizer
    fe = Wav2Vec2FeatureExtractor.from_pretrained(ESPEAK_REPO)
    tok = Wav2Vec2PhonemeCTCTokenizer.from_pretrained(ESPEAK_REPO, do_phonemize=False)
    model = Wav2Vec2ForCTC.from_pretrained(ESPEAK_REPO).eval()

    def run(audio: np.ndarray) -> str:
        with torch.no_grad():
            logits = model(fe(audio, sampling_rate=16000, return_tensors="pt").input_values).logits
        return tok.batch_decode(logits.argmax(-1))[0].strip()
    return run


def main() -> None:
    models = {"zipa": load_zipa(), "w2v2": load_w2v2(), "w2v2_espeak": load_w2v2_espeak()}
    for d in map(Path, sys.argv[1:]):
        clips = d / "clips" if (d / "clips").is_dir() else d
        out = d / "scores_local.json"
        res = json.load(open(out, encoding="utf-8")) if out.exists() else {}
        for f in sorted(clips.glob("*.wav")):
            todo = [k for k in models if k not in res.get(f.name, {})]
            if not todo:
                continue
            audio, _ = librosa.load(f, sr=16000, mono=True)
            res.setdefault(f.name, {}).update({k: models[k](audio) for k in todo})
        json.dump(res, open(out, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
        print(f"{out}: {len(res)} clips")


if __name__ == "__main__":
    main()
