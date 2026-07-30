"""Shot planning: analysis + scene spec -> reproducible shot manifest.

Sections are assigned to scenes in order (cycling). Each section is split
into shots whose length in bars depends on the section's energy level.
Shot frame counts obey LTX's 8n+1 constraint; each shot's frame count is
chosen against the *absolute* track-time boundary so quantization error
never accumulates (every cut stays within 4 frames of its target beat).
"""
import json
import sys

from .materials import FAMILIES, jitter

# Fallback beats when a scene doesn't author its own: generic blocking that
# leads the prompt (the image model follows lead content, so composition MUST
# come first — a camera phrase appended after a long setting prompt is noise).
# Abstract fields have strong compositional attractors — "cosmic" will render
# the same spiral galaxy from any seed. Forcing a different framing per shot is
# what actually breaks the repetition; varying the seed alone does not.
VIZ_FRAMINGS = [
    "Extreme macro, so close the structure is barely identifiable, filling the frame.",
    "Wide field, the whole expanse visible, structure repeating to the edges.",
    "Off-centre close view, the mass pushed hard to one side, empty space opposite.",
    "Steep oblique angle across the material, receding into blur.",
    "Tight crop on a single feature, everything else out of frame.",
    "Symmetrical head-on view, the structure radiating from the centre.",
    "Shallow focus with the near material dissolved, one plane sharp deep in.",
    "Frame filled with the densest part of the mass, no edges or margins visible.",
]

FALLBACK_BEATS = [
    "Extreme wide establishing shot, the protagonist tiny in the landscape.",
    "Medium tracking shot from behind the protagonist as she moves deeper in.",
    "Close-up on the protagonist's face in profile, the scene's light playing across it.",
    "Overhead aerial shot, her figure a small dark point far below.",
    "Detail close-up of the scene's central element, the protagonist blurred beyond it.",
    "Reverse wide angle from deep within the scene, her small figure approaching.",
]


def quantize_frames(seconds: float, fps: int, lo: int = 33, hi: int = 241) -> int:
    n = round((seconds * fps - 1) / 8)
    return max(lo, min(hi, 8 * n + 1))


def phrase_cuts(lyrics: dict | None) -> list[float]:
    """End times of sung phrases — the cut points a human editor would use."""
    if not lyrics or not lyrics.get("has_sung_lyrics"):
        return []
    return [s["end"] for s in lyrics["segments"]]


def snap(t: float, cuts: list[float], window: float, used: set[float],
         floor: float, min_shot: float) -> tuple[float, bool]:
    """Pull a bar-derived boundary onto a nearby phrase end.

    Cutting mid-vocal-phrase is what reads as arbitrary, so within one bar
    the phrase end wins. Sung phrases rarely land on 4-bar multiples — a
    tighter window than a full bar catches almost nothing in practice.

    A phrase end is consumed once (`used`) so two boundaries can't collapse
    onto the same point, and a snap that would leave too short a shot is
    declined rather than clamped.
    """
    free = [c for c in cuts if c not in used and c - floor >= min_shot]
    if not free:
        return t, False
    best = min(free, key=lambda c: abs(c - t))
    if abs(best - t) > window:
        return t, False
    used.add(best)
    return best, True


def plan(analysis: dict, spec: dict, lyrics: dict | None = None) -> dict:
    fps = spec.get("fps", 25)
    # Visualizer mode cuts on the bar rather than every two-to-four bars, so
    # the edit sits on the rhythm instead of on the song's sections.
    visualizer = spec.get("mode") == "visualizer"
    default_bps = ({"low": 2, "mid": 1, "high": 1} if visualizer
                   else {"low": 4, "mid": 2, "high": 2})
    bars_per_shot = spec.get("bars_per_shot", default_bps)
    base_seed = spec.get("seed", 1000)
    scenes = spec["scenes"]
    bars = analysis["bars"]
    duration = analysis["duration"]

    style = spec.get("style", "")
    n_sections = len(analysis["sections"])
    cuts = [] if visualizer else phrase_cuts(lyrics)
    bar_secs = 240.0 / max(analysis["tempo"], 1)  # 4 beats
    window = spec.get("snap_window", bar_secs)
    min_shot = spec.get("min_shot_secs", 1.6)
    used_cuts: set[float] = set()

    shots = []
    cum = 0.0  # seconds of video planned so far
    for si, sec in enumerate(analysis["sections"]):
        # Narrative mode: linear allocation, one scene per section, a journey
        # rather than a cycle. Visualizer mode instead rotates material every
        # shot — dwelling on one abstract family produces near-identical
        # consecutive frames, because a strong attractor (a spiral galaxy, say)
        # overrides any per-shot framing instruction. Rotation is the only
        # thing that reliably makes successive cuts differ.
        scene = scenes[min(si * len(scenes) // n_sections, len(scenes) - 1)]
        level = sec["level"]
        step = bars_per_shot[level]
        b = sec["start_bar"]
        while b < sec["end_bar"]:
            b_end = min(b + step, sec["end_bar"])
            # absolute target end-time for this shot
            t1 = bars[b_end]["t0"] if b_end < len(bars) else duration
            t1, snapped = snap(t1, cuts, window, used_cuts, cum, min_shot)
            frames = quantize_frames(t1 - cum, fps)
            shot_scene = scenes[len(shots) % len(scenes)] if visualizer else scene
            shots.append({
                "idx": len(shots),
                "scene": shot_scene["id"],
                "level": level,
                "t0": round(cum, 3),
                "frames": frames,
                "seed": base_seed + len(shots),
                "video_prompt": scene["video_prompt"] + " " + scene["motion"][level],
                "still_seed": base_seed + 9000 + len(shots),
                "phrase_cut": snapped,
            })
            cum += frames / fps
            b = b_end
        if cum >= duration:
            break

    # Per-shot stills: shot k of n in a scene walks monotonically through the
    # scene's authored `beats` (concrete blocking — these LEAD the prompt so
    # they actually control composition), followed by the scene setting and
    # the global style clause.
    by_scene: dict[str, list[dict]] = {}
    for s in shots:
        by_scene.setdefault(s["scene"], []).append(s)
    scene_map = {s["id"]: s for s in scenes}
    for sid, group in by_scene.items():
        beats = scene_map[sid].get("beats", FALLBACK_BEATS)
        n = len(group)
        for k, shot in enumerate(group):
            beat = beats[min(k * len(beats) // max(n, 1), len(beats) - 1)]
            scene = scene_map[sid]
            fam = scene.get("material")
            if fam in FAMILIES:
                # Compose the treatment per shot: the scene keeps one family,
                # substrate and palette, but each shot re-rolls its flaws so
                # it reads as its own physical artefact rather than one image
                # with a filter applied.
                t = jitter(fam, scene.get("material_seed", base_seed), shot["seed"])
                lead, scene_style = t["lead"], t["style"]
            else:
                scene_style = scene.get("style", style)
                lead = scene.get("lead", "Cinematic film still:")
            framing = (VIZ_FRAMINGS[k % len(VIZ_FRAMINGS)]
                       if spec.get("mode") == "visualizer" else "")
            shot["still_prompt"] = " ".join(p for p in (
                lead + " " + beat,
                framing,
                "Setting: " + scene["still_prompt"],
                scene_style) if p)
            # What the shot is *meant* to be, compactly — the judge scores
            # against this, not the full prompt whose flaw and wear clauses
            # would crowd out the subject inside SigLIP's token limit.
            shot["judge_text"] = f"{fam or 'photograph'}. {beat} {scene['still_prompt'][:140]}"

    return {
        "track": analysis["track"],
        "fps": fps,
        "width": spec.get("width", 1280),
        "height": spec.get("height", 704),
        "video_duration": round(cum, 3),
        "track_duration": duration,
        "shots": shots,
    }


def main():
    import pathlib
    analysis_path, spec_path, out = sys.argv[1], sys.argv[2], sys.argv[3]
    lyr_path = pathlib.Path(analysis_path).parent / "lyrics.json"
    lyrics = json.load(open(lyr_path)) if lyr_path.exists() else None
    manifest = plan(json.load(open(analysis_path)), json.load(open(spec_path)), lyrics)
    json.dump(manifest, open(out, "w"), indent=1)
    n = len(manifest["shots"])
    snapped = sum(1 for s in manifest["shots"] if s.get("phrase_cut"))
    print(f"{n} shots (one still each), {snapped} cut on vocal phrase ends, "
          f"video {manifest['video_duration']:.1f}s vs track {manifest['track_duration']:.1f}s, "
          f"~{n * 2} min render")


if __name__ == "__main__":
    main()
