# Roadmap

Ordered by expected value, and grounded in failures this tool actually had
rather than in capability for its own sake.

The recurring failure mode is worth stating first, because it shapes the
ordering: **mvgen generates confidently and verifies narrowly.** Feathers on
ice, a three-scene loop, a video ending 29s before its own soundtrack — none of
these were caught by the pipeline. They were caught by a human watching or by a
measurement taken by hand. So the first two items close verification loops
rather than adding capability.

---

## 1. Judge the video, not just the still

**Why.** Every quality mechanism stops at the keyframe. We spend three
candidates and a global dedup check choosing a still, then render ~90s of video
from it and never look at the result. A shot that freezes, drifts into mush, or
abandons its material is invisible to the pipeline.

**Design.** New `judge_video()` in `mvgen/judge.py`, called from `render()`
after each shot completes. Sample 3 frames (first / middle / last) via ffmpeg,
then score:

| check | method | fails when |
|---|---|---|
| froze | mean frame-to-frame diff over sampled frames | motion below a floor — shot is a still |
| drifted to mush | last frame vs `judge_text` adherence | adherence collapses vs the still's score |
| lost material | last frame vs its own source still | similarity below a floor |

On failure, re-render once with a different seed; accept the better of the two.
Cap at one retry — a second failure usually means the prompt is wrong, not the
sample.

**Cost.** ~2s CPU per shot to score (siglip already runs on CPU, no GPU
contention). Retries only on failure; if the floors are set right that should be
a small minority of shots.

**Risk.** Thresholds set too tight would re-render constantly at ~90s a time.
Mitigation: run the scorer over already-rendered jobs first and check that it
flags exactly the shots that look bad by eye, before letting it trigger anything.

**Validation.** It must agree with judgements already made by hand — the six
identical galaxies, the ceramic pots. If it disagrees with what is visibly true,
the scoring is wrong, not the video.

---

## 2. Audit whether audio conditioning does anything — DONE: CUT

**Why.** This one is a correction, not a feature. Audio conditioning was proven
to *render*, and then described as making motion answer the music. That was
never demonstrated. It may be doing nothing.

**Design.** Standalone experiment, no pipeline changes:

1. Pick 6 shots spanning quiet and loud passages.
2. Render each twice — same still, same seed — with `MVGEN_AUDIO_COND` on and off.
3. Extract a per-frame motion series from each clip.
4. Extract the onset-strength envelope of the matching audio slice (librosa).
5. Correlate motion against onset strength for both conditions.

**Decision rule.** If the audio-on correlation is not meaningfully higher than
audio-off, **cut the feature** rather than carry it. It costs an ffmpeg slice and
an encode per shot, and a feature that does nothing is worse than absent because
it invites building on top of it.

**Cost.** 12 renders, ~25 min. No code to maintain if the answer is negative.

### Result (measured, 6 shots on `wedbecute`)

| shot | motion audio-on | motion audio-off |
|---|---|---|
| 0 | 0.51 | 0.52 |
| 10 | 1.46 | 1.46 |
| 20 | 5.52 | 5.51 |
| 30 | 1.71 | 1.71 |
| 40 | 1.53 | 1.53 |
| 50 | 0.35 | 0.36 |

Mean correlation against the onset envelope: **-0.176 on, -0.177 off, delta
+0.000.** Same still, same seed, real audio versus silent latent — the video is
identical either way.

The slices were checked before drawing the conclusion, in case the audit was
measuring silence: RMS 0.03-0.34, peaks to full scale. Real audio, no effect.

**Removed.** `slice_audio`, the `LTXVAudioVAEEncode` branch, `--audio` and
`MVGEN_AUDIO_COND` are gone. Every video built with the flag on was unaffected
by it.

**Why it probably failed.** Supplying a real audio latent in place of the empty
one was the wrong mechanism. LTX-2 lists audio-to-video as a *separate
pipeline*, not a parameter of image-to-video — so the video branch presumably
never attends to the audio stream in this configuration, and the AV pair is
simply carried alongside each other. Anyone revisiting this should start from
that pipeline rather than from the latent swap, and should re-run this audit as
the acceptance test rather than trusting that it renders.

**Lesson.** "It renders without error" was treated as "it works" for several
hours, and the benefit was described in summaries before it was ever measured.

---

## 3. 50fps via the temporal upscaler

**Why.** Lowest-effort visible improvement. At 25fps with 3s shots, fast
material reads as slightly strobed. The strength calibration bought motion
*amount*; this buys motion *smoothness*, which is a different axis.

**Design.** Download `ltx-2.3-temporal-upscaler-x2` (0.26GB) into
`models/latent_upscale_models/`. Confirm first whether it operates on the video
latent (before `LTXVSeparateAVLatent`) or on decoded frames — that determines
whether it slots into the graph or becomes a post-pass. Then double `fps` in
`CreateVideo` for the upscaled stream.

**Open question.** Interaction with the audio latent. The AV pair is
concatenated at a fixed frame rate, so temporal upscaling the video half may
desynchronise them — the same class of bug as the audio-slice length. Test on
one shot with audio conditioning on before wiring it in.

**Fallback.** `ffmpeg minterpolate` needs no download and no graph changes, but
interpolates in pixel space and will smear fast material. Only worth it if the
LTX upscaler proves incompatible with the AV latent.

---

## 4. Palette arc

**Why.** Materials are currently chosen per scene independently, so a video has
variety but no trajectory. Nothing makes the sequence move from cold to warm, or
flat to dimensional, across a song.

**Design.** Every family already carries a 7-axis affinity vector, so the data
exists. After the director picks its set, order the scenes along whichever axis
best matches the song's own progression (e.g. sort by `warmth` when the energy
curve rises toward the end).

**Subtlety.** The director also writes a `concept` describing an arc, and
reordering after the fact would fight it. Better to compute the intended
ordering first and *tell* the director the sequence, so the concept and the
material progression agree instead of competing.

**Cost.** Nearly free — no extra rendering, no new models.

---

## Explicitly not doing

- **More materials.** 63 families; count stopped being the constraint several
  fixes ago. Adding more would repeat the mistake of treating variety as a
  quantity problem.
- **Spatial upscaling.** Poor return for the render cost at these durations.
- **Camera-control LoRAs.** Motion prompts are doing this job adequately; the
  LoRA loading path adds failure surface for a marginal gain.

---

## Sequencing

1 and 2 first, together — they are both verification and neither needs a
download. 2 may *delete* code, which is worth knowing before building on it.
Then 3, which is self-contained. Then 4, which is cheap and benefits from the
video judge existing (a reordered palette is easier to trust when bad shots are
caught automatically).
