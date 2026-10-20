#!/usr/bin/env python3
"""Local phone recognition for clips: ZIPA (ONNX) and wav2vec2, no API calls.

  python score_local.py DIR [DIR ...]      # each DIR has clips/*.wav, or is a folder of .wav files

Writes DIR/scores_local.json: {clip file name: {"zipa": "t ʃ", "w2v2": "tʃ"}}, space-separated phones as each model
outputs them. Clips already scored are skipped. Matching against the intended sound happens in build_metadata.py.

Models (downloaded to ~/.cache/huggingface on first use):
- ZIPA: anyspeech/zipa-large-crctc-500k, model.onnx (CC BY 4.0), 80-bin Kaldi fbank, greedy CTC; feature setup and
  decoding as in github.com/lingjzhu/zipa inference/utils.py.
- wav2vec2: vitouphy/wav2vec2-xls-r-300m-timit-phoneme, the model behind Sara's phoneme-endpoint.
Needs onnxruntime<=1.21 on macOS 13.2 (newer wheels are built for 13.4+).
"""
import json, sys
from pathlib import Path

import numpy as np
import librosa
import torch

ZIPA_REPO = "anyspeech/zipa-large-crctc-500k"
W2V2_REPO = "vitouphy/wav2vec2-xls-r-300m-timit-phoneme"


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


def main() -> None:
    zipa, w2v2 = load_zipa(), load_w2v2()
    for d in map(Path, sys.argv[1:]):
        clips = d / "clips" if (d / "clips").is_dir() else d
        out = d / "scores_local.json"
        res = json.load(open(out, encoding="utf-8")) if out.exists() else {}
        for f in sorted(clips.glob("*.wav")):
            if f.name in res:
                continue
            audio, _ = librosa.load(f, sr=16000, mono=True)
            res[f.name] = {"zipa": zipa(audio), "w2v2": w2v2(audio)}
        json.dump(res, open(out, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
        print(f"{out}: {len(res)} clips")


if __name__ == "__main__":
    main()
