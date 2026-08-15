"""Speech emotion recognition via emotion2vec+ large (FunASR).

Torch, not MLX. The MLX conversion of emotion2vec ships weights and a config but no
Python loader — its consumer is a Swift package — so running it from Python would mean
porting Data2Vec 2.0 first. FunASR runs the reference implementation today, and its
output is the ground truth any future MLX port would be validated against.

Feed this the *vocals stem*, not a full mix: emotion2vec is trained on speech, and
instrumental energy degrades it badly.
"""

import json
import time

import numpy as np

import comfy.model_management
from comfy.utils import ProgressBar

_MODEL_ID = "iic/emotion2vec_plus_large"
_TARGET_RATE = 16000

# FunASR returns bilingual labels like "开心/happy"; keep the English half.
def _clean(label):
    return label.split("/")[-1].strip().strip("<>")


# AutoModel load is slow and the weights are large, so hold one instance per process.
_CACHE = {}


def _get_model():
    if _MODEL_ID not in _CACHE:
        from funasr import AutoModel

        start = time.perf_counter()
        _CACHE[_MODEL_ID] = AutoModel(model=_MODEL_ID, hub="hf", disable_update=True)
        print(f"[emotion] loaded {_MODEL_ID} in {time.perf_counter() - start:.1f}s", flush=True)
    return _CACHE[_MODEL_ID]


def _to_mono_16k(audio):
    import scipy.signal as ss

    waveform = audio["waveform"]
    rate = audio["sample_rate"]
    data = waveform[0].mean(axis=0).cpu().numpy().astype(np.float32)
    if rate != _TARGET_RATE:
        data = ss.resample_poly(data, _TARGET_RATE, rate).astype(np.float32)
    return data


class SpeechEmotionRecognition:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "audio": ("AUDIO",),
                "granularity": (
                    ["whole", "windowed"],
                    {"default": "windowed", "tooltip": "'whole' scores the clip once; 'windowed' scores each window and also reports a mean."},
                ),
                "window_seconds": ("FLOAT", {"default": 3.0, "min": 0.5, "max": 30.0, "step": 0.5}),
                "hop_seconds": ("FLOAT", {"default": 1.5, "min": 0.25, "max": 30.0, "step": 0.25}),
            }
        }

    RETURN_TYPES = ("STRING", "FLOAT", "STRING")
    RETURN_NAMES = ("emotion", "confidence", "report")
    FUNCTION = "analyze"
    CATEGORY = "MLX/Audio Analysis"
    OUTPUT_NODE = True

    def analyze(self, audio, granularity, window_seconds, hop_seconds):
        model = _get_model()
        data = _to_mono_16k(audio)
        duration = len(data) / _TARGET_RATE

        def score(chunk):
            res = model.generate(chunk, granularity="utterance", extract_embedding=False)[0]
            labels = [_clean(x) for x in res["labels"]]
            return labels, np.asarray(res["scores"], dtype=np.float32)

        if granularity == "whole":
            windows = [(0.0, data)]
        else:
            size = int(window_seconds * _TARGET_RATE)
            hop = int(hop_seconds * _TARGET_RATE)
            windows = [
                (i / _TARGET_RATE, data[i : i + size])
                for i in range(0, max(1, len(data) - size + 1), hop)
            ]
            # emotion2vec needs a reasonable amount of audio to be meaningful.
            windows = [w for w in windows if len(w[1]) >= _TARGET_RATE // 2] or [(0.0, data)]

        pbar = ProgressBar(len(windows))
        labels = None
        rows = []
        stacked = []
        for index, (offset, chunk) in enumerate(windows):
            comfy.model_management.throw_exception_if_processing_interrupted()
            labels, scores = score(chunk)
            stacked.append(scores)
            top = int(scores.argmax())
            rows.append({"t": round(offset, 2), "emotion": labels[top], "confidence": round(float(scores[top]), 3)})
            pbar.update_absolute(index + 1, len(windows))

        mean = np.mean(np.stack(stacked), axis=0)
        top = int(mean.argmax())
        emotion, confidence = labels[top], float(mean[top])

        report = {
            "duration_seconds": round(duration, 2),
            "windows": len(windows),
            "overall": {"emotion": emotion, "confidence": round(confidence, 3)},
            "mean_scores": {l: round(float(s), 3) for l, s in zip(labels, mean) if s >= 0.001},
            "timeline": rows if granularity == "windowed" else [],
        }
        text = json.dumps(report, indent=2)
        print(f"[emotion] {emotion} ({confidence:.2f}) over {len(windows)} window(s)", flush=True)
        return {"ui": {"text": [text]}, "result": (emotion, confidence, text)}


NODE_CLASS_MAPPINGS = {"SpeechEmotionRecognition": SpeechEmotionRecognition}
NODE_DISPLAY_NAME_MAPPINGS = {"SpeechEmotionRecognition": "Speech Emotion (emotion2vec+)"}
