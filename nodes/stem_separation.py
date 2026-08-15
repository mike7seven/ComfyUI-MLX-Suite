"""Open-Unmix stem separation (vocals / drums / bass / other).

Torch, not MLX — Open-Unmix has no MLX port, and the MLX instance already carries
torch. Kept in its own family module so it stays usable if the emotion nodes'
dependencies are absent.

Separating first is what makes downstream vocal analysis viable: speech models
score poorly on a full mix, and much better on an isolated vocal stem.
"""

import os

import torch

import comfy.model_management
import folder_paths

import openunmix

_TARGETS = ["vocals", "drums", "bass", "other"]

# Bundled Separator checkpoints (all four targets in one file), as shipped by the
# ComfyUI ecosystem. Keys are the dropdown labels.
_MODELS = {"umxl": "umxl.pth", "umxhq": "umxhq.pth"}

_SAMPLE_RATE = 44100


def _local_checkpoint(filename):
    """Return the first existing openunmix checkpoint across known model roots."""
    roots = [os.path.join(r, "openunmix") for r in folder_paths.get_folder_paths("checkpoints")]
    roots.append(os.path.join(folder_paths.models_dir, "openunmix"))
    roots.append(os.path.expanduser("~/Local/ComfyUI/models/openunmix"))
    for root in roots:
        path = os.path.join(root, filename)
        if os.path.isfile(path):
            return path
    return None


class OpenUnmixSeparator:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "audio": ("AUDIO",),
                "model": (list(_MODELS.keys()), {"default": "umxl"}),
                "device": (["mps", "cpu"], {"default": "mps"}),
            }
        }

    RETURN_TYPES = ("AUDIO", "AUDIO", "AUDIO", "AUDIO")
    RETURN_NAMES = tuple(_TARGETS)
    FUNCTION = "separate"
    CATEGORY = "MLX/Audio Analysis"

    def separate(self, audio, model, device):
        import torchaudio

        waveform = audio["waveform"]
        rate = audio["sample_rate"]

        # openunmix is trained at 44.1 kHz stereo.
        if waveform.dim() == 2:
            waveform = waveform.unsqueeze(0)
        if waveform.shape[1] == 1:
            waveform = waveform.repeat(1, 2, 1)
        if rate != _SAMPLE_RATE:
            waveform = torchaudio.functional.resample(waveform, rate, _SAMPLE_RATE)

        checkpoint = _local_checkpoint(_MODELS[model])
        factory = getattr(openunmix, model)
        separator = factory(pretrained=checkpoint is None, device="cpu", niter=1)
        if checkpoint is not None:
            separator.load_state_dict(
                torch.load(checkpoint, map_location="cpu", weights_only=False)
            )
            print(f"[stems] loaded local {os.path.basename(checkpoint)}", flush=True)
        separator.freeze()

        comfy.model_management.throw_exception_if_processing_interrupted()

        try:
            separator = separator.to(device)
            estimates = separator(waveform.to(device))
        except Exception as exc:
            # istft and complex ops are not uniformly supported on MPS across
            # torch versions; CPU is slower but always correct.
            if device == "cpu":
                raise
            print(f"[stems] {device} failed ({type(exc).__name__}), retrying on cpu", flush=True)
            separator = separator.to("cpu")
            estimates = separator(waveform.to("cpu"))

        # Separator returns [B, n_targets, C, S] ordered by separator.target_models.
        order = list(separator.target_models.keys())
        estimates = estimates.cpu()
        out = []
        for target in _TARGETS:
            stem = estimates[:, order.index(target)] if target in order else torch.zeros_like(waveform)
            out.append({"waveform": stem, "sample_rate": _SAMPLE_RATE})
        print(f"[stems] {model} -> {', '.join(_TARGETS)} @ {_SAMPLE_RATE} Hz", flush=True)
        return tuple(out)


NODE_CLASS_MAPPINGS = {"OpenUnmixSeparator": OpenUnmixSeparator}
NODE_DISPLAY_NAME_MAPPINGS = {"OpenUnmixSeparator": "Stem Separator (Open-Unmix)"}
