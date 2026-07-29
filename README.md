# mvgen

Reproducible music-video generation, fully local. A track goes in; a beat-aware
shot manifest is planned; Z-Image Turbo renders one keyframe still per scene;
LTX-2.3 animates each shot from its scene still; ffmpeg assembles the shots and
muxes the original track back over the top.

## Requirements

- A running ComfyUI at `127.0.0.1:8188` (`~/programs/comfyui/start.sh`) with:
  - `diffusion_models/LTX-2.3-22B-distilled-1.1-Q4_K_M.gguf` (+ ComfyUI-GGUF node)
  - `text_encoders/gemma_3_12B_it_fp8_scaled.safetensors`, `ltx-2.3_text_projection_bf16.safetensors`
  - `vae/LTX23_video_vae_bf16.safetensors`, `vae/LTX23_audio_vae_bf16.safetensors`
    (audio VAE also symlinked into `checkpoints/`)
  - `diffusion_models/z_image_turbo_bf16.safetensors`, `text_encoders/qwen_3_4b_fp8_mixed.safetensors`, `vae/z_image_ae.safetensors`
- `uv sync` for the analysis deps (librosa), ffmpeg on PATH.
- Stop any resident LLM first (`llm stop`) — the GPU can't be shared.

## Usage

```bash
V=.venv/bin/python
J=jobs/<job>          # contains scenes.json (see jobs/true-to-myself)

$V -m mvgen.analyze <track file> $J/analysis.json
$V -m mvgen.plan $J/analysis.json $J/scenes.json $J/manifest.json
$V -m mvgen.render $J/manifest.json $J/work [limit]   # ~2 min per shot
$V -m mvgen.assemble $J/manifest.json $J/work $J/out.mp4
```

`render` is resumable and idempotent (`work/state.json`): kill it any time and
re-run; to re-roll one shot, edit its prompt/seed in the manifest, delete its
entry from `state.json`, and re-run render + assemble.

## How the plan works

- Bars are grouped low/mid/high by RMS-energy tercile; contiguous runs become
  sections; sections map to scenes in order (cycling).
- Shot length in bars per energy level is set in `scenes.json`
  (`bars_per_shot`); each shot's prompt = scene `video_prompt` + the scene's
  `motion` text for that energy level.
- LTX frame counts must be 8n+1, so exact bar-boundary cuts are impossible;
  each shot is quantized against its *absolute* track-time boundary so error
  never accumulates (every cut lands within 4 frames of its beat).
- Everything that affects output lives in `scenes.json` + `manifest.json` —
  commit them and the video is reproducible bit-for-bit modulo GPU
  nondeterminism.
