"""ComfyUI-MLX-Suite — Apple Silicon MLX model nodes for ComfyUI.

Each model family lives in its own module under nodes/ and is loaded independently:
one family failing to import (missing optional dependency, missing weights package)
disables only that family, never the whole pack.
"""

import importlib
import traceback

NODE_CLASS_MAPPINGS = {}
NODE_DISPLAY_NAME_MAPPINGS = {}

# Add a module name here to register a new model family.
_FAMILIES = [
    "minimax_music3",
    "mflux_image",
    "stem_separation",
    "emotion",
]

for _name in _FAMILIES:
    try:
        _mod = importlib.import_module(f".nodes.{_name}", __name__)
    except Exception:
        print(f"[ComfyUI-MLX-Suite] '{_name}' unavailable — skipping:")
        traceback.print_exc()
        continue
    NODE_CLASS_MAPPINGS.update(getattr(_mod, "NODE_CLASS_MAPPINGS", {}))
    NODE_DISPLAY_NAME_MAPPINGS.update(getattr(_mod, "NODE_DISPLAY_NAME_MAPPINGS", {}))

if NODE_CLASS_MAPPINGS:
    print(f"[ComfyUI-MLX-Suite] registered {len(NODE_CLASS_MAPPINGS)} nodes", flush=True)

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]
