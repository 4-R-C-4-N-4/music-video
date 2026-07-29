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

## Fully automatic

```bash
.venv/bin/python -m mvgen build <track file> jobs/<job>
```

That is the whole thing. With no `scenes.json` present, a local llama.cpp
model reads the track's structure and transcript and writes one — concept,
recurring figure, per-scene settings, per-shot blocking beats, motion
language, and a material treatment per scene chosen from the validated
palette in `mvgen/materials.py`. Then it plans, renders and assembles.

Stages are skipped when their outputs exist, so re-running resumes. Delete a
stage's output to redo it. The GPU is taken and released in turn: demucs,
then the LLM, then ComfyUI (started automatically if it isn't up).

Flags: `--no-lyrics` to skip transcription, `--model=<name>` to pick another
llama.cpp runner (default `gemma4-nothink`; reasoning models return
transcripts instead of JSON, so use a no-think runner).

The director is constrained, not trusted: it picks material *keys* from the
tested palette rather than writing its own treatment prompts, and its output
is rejected and regenerated if it echoes the transcript's wording.

## Workflow: hand-authoring a video for a new song

Write `scenes.json` yourself when you want control — it is always preferred
over regenerating, and the automatic director is a good starting point to
edit rather than a black box.


1. **Create a job dir** and write the creative spec:
   ```bash
   mkdir jobs/my-song
   cp jobs/true-to-myself/scenes.json jobs/my-song/scenes.json
   ```
   Edit `scenes.json`: this is the entire creative surface. Each scene has a
   `still_prompt` (the look — one keyframe image per scene), a `video_prompt`
   (what's always in motion), and three `motion` variants (`low`/`mid`/`high`)
   that get appended depending on the music's energy at that point. Scenes are
   assigned to the song's sections in listed order, cycling. A recurring
   character/palette across scenes ("figure in a long dark coat", one color
   grade) is what makes the result read as one video. 3–6 scenes works well.

2. **Preview cheaply before the long render.** Stills are ~10s each, so build
   with a shot limit, look at `work/still-*.png` and the first shot clips, and
   iterate on `scenes.json` (delete `work/state.json` entries — or the whole
   `work/` dir — for scenes you re-prompt):
   ```bash
   .venv/bin/python -m mvgen build <track file> jobs/my-song 2
   ```

3. **Full build** (same command, no limit — resumes past everything already
   rendered, then assembles `jobs/my-song/my-song.mp4`):
   ```bash
   .venv/bin/python -m mvgen build <track file> jobs/my-song
   ```
   Budget ~2 min of GPU per shot; a 3–4 min track is ~40–50 shots.

4. **Reroll bad shots**: edit that shot's `video_prompt`/`seed` in
   `manifest.json`, delete its entry from `work/state.json`, re-run step 3.
   Only that shot re-renders, then assembly re-runs.

5. **Commit** `scenes.json` + `manifest.json` (and `analysis.json`) — that's
   the reproducible record of the video.

Individual stages are also runnable on their own (`mvgen.analyze`,
`mvgen.plan`, `mvgen.render`, `mvgen.assemble`) — see each module's docstring
for arguments. `analysis.json`/`manifest.json` are only generated if missing;
delete them to force a re-plan after editing scenes.json.

## How the plan works

- Bars are grouped low/mid/high by RMS-energy tercile; contiguous runs become
  sections; sections map to scenes in order (cycling).
- Shot length in bars per energy level is set in `scenes.json`
  (`bars_per_shot`); each shot's prompt = scene `video_prompt` + the scene's
  `motion` text for that energy level.
- LTX frame counts must be 8n+1, so exact bar-boundary cuts are impossible;
  each shot is quantized against its *absolute* track-time boundary so error
  never accumulates (every cut lands within 4 frames of its beat).
- A scene may set its own `style` (and `lead`) to override the global style
  clause — LTX i2v preserves the still's material treatment, so a video can
  genuinely change medium between scenes (validated: needle-felted wool,
  cyanotype, copperplate etching and stained glass all survive the video
  stage intact). Prompt the *process*, not an abstract effect.
- Everything that affects output lives in `scenes.json` + `manifest.json` —
  commit them and the video is reproducible bit-for-bit modulo GPU
  nondeterminism.
