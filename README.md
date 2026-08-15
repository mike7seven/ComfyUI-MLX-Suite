# ComfyUI MLX Suite

Apple Silicon MLX models as ComfyUI nodes. MLX runs on Metal through Apple's own array
framework rather than PyTorch/MPS, which is a different runtime — not a different file format.

**Shipping now:** MiniMax Music 3 (text-to-music).
**Planned:** mflux image models (Qwen-Image, Z-Image, FLUX).

This repo contains node wrappers only. Inference implementations stay upstream as
dependencies, and model weights are downloaded by you from Hugging Face — nothing is
vendored or redistributed here. See [NOTICE](NOTICE).

## Install

MLX packages conflict with a torch-oriented ComfyUI (`mlx-lm` requires `transformers` 5.x,
which breaks nodes pinned to 4.x). **Use a ComfyUI instance dedicated to MLX.** Comfy Desktop
supports multiple instances, and enabling shared models means weights are not duplicated.

```bash
cd <mlx-instance>/ComfyUI/custom_nodes
git clone https://github.com/mike7seven/ComfyUI-MLX-Suite
../.venv/bin/python -m pip install -r ComfyUI-MLX-Suite/requirements.txt
```

Restart ComfyUI — custom nodes are imported at boot, with no hot reload.

### Version pinning

`mlx` is pinned to **0.31.2**: `mflux` requires `<0.32.0` while `mlx-lm` requires `>=0.31.2`,
leaving exactly one version. Re-check that intersection before bumping either.

## MiniMax Music 3

| Node | Inputs | Output |
|---|---|---|
| MiniMax Music 3 MLX Loader | `weights` | `MINIMAX_MUSIC3_MLX` |
| MiniMax Music 3 MLX Generate | modules, prompt, lyrics, duration, steps, seed | `AUDIO` |

Workflow: **Loader → Generate → Save Audio (Advanced)**.
Ready-made graph in [`workflows/minimax-music3-mlx.json`](workflows/minimax-music3-mlx.json).

Split into two nodes so ComfyUI's execution cache holds the loaded modules — the loader
re-runs only when `weights` changes. Progress bar and Cancel both work during generation.

### Weights

The loader scans for any folder containing `mlx_config.json` in:

- `<ComfyUI models>/minimax_music3_mlx/`
- `~/Local/minimax-music3-mlx/weights/`

From the [MLX collection](https://huggingface.co/collections/mikoy92/minimax-music-3-mlx-6a7eb2a8e09d246e6d029fad)
— bf16 ~23.5 GB, 8-bit ~12 GB, 6-bit ~10 GB, 4-bit ~7 GB:

```bash
hf download mikoy92/MiniMax-Music3-MLX-8bit \
  --local-dir <ComfyUI models>/minimax_music3_mlx/mlx-8bit
```

### Upstream patch

The Generate node needs an optional `progress_callback` in the runner, without which
ComfyUI cannot draw a progress bar or honour Cancel during a multi-minute generation.

It is offered upstream as [mikolaj92/minimax-music3-mlx#1](https://github.com/mikolaj92/minimax-music3-mlx/pull/1).
Until that merges, `requirements.txt` installs from a fork branch carrying it. A standalone
copy is kept at [`patches/minimax-mlx-progress.patch`](patches/minimax-mlx-progress.patch).
Once merged, switch the requirement back to upstream and delete the patch.

### Behaviour worth knowing

- **`duration` is an upper bound, not a target.** The model routinely ends a song early —
  a 30s request returning ~20s is correct.
- **Output is not reproducible across `mlx` versions.** The same seed on 0.31.2 and 0.32.0
  produces different songs; RNG and kernels changed between them.
- Roughly **2–5.5x realtime** depending on `steps`; peak MLX memory ~16–17 GB at 8-bit.
- Weights load lazily (~0.03s) — the real cost is paging during the first generate, so
  caching the loader saves a few percent, not a load.

## Credits

MiniMax Music 3 MLX port by [mikolaj92](https://github.com/mikolaj92/minimax-music3-mlx);
MLX weight conversions by [mikoy92](https://huggingface.co/mikoy92); model by
[MiniMax](https://huggingface.co/MiniMaxAI/MiniMax-Music3).
