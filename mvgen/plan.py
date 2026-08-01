"""Shot planning: analysis + scene spec -> reproducible shot manifest.

Sections are assigned to scenes in order (cycling). Each section is split
into shots whose length in bars depends on the section's energy level.
Shot frame counts obey LTX's 8n+1 constraint; each shot's frame count is
chosen against the *absolute* track-time boundary so quantization error
never accumulates (every cut stays within 4 frames of its target beat).
"""
import json
import random
import sys

from .materials import ABSTRACT_FRAMING, FAMILIES, jitter

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

    # Compliance to the still, keyed to dynamics rather than applied flat: a
    # quiet passage should hold the composed frame, a loud one should be allowed
    # to tear away from it. Defaults are deliberately conservative pending
    # calibration — the usable floor is not yet measured.
    # Calibrated by sweep, not guessed: motion rises 63% from strength 1.0 to
    # 0.7 and then flattens (0.55 adds 4% more), while material fidelity to the
    # composed still only falls 0.969 -> 0.952 across the whole range. The
    # dissolution risk these were originally set against did not materialise,
    # so the floor is set by diminishing returns rather than by damage.
    # Widened and shifted down after side-by-side viewing: the 0.55/0.45/0.35
    # build read as more interesting than the 0.85/0.75/0.68 one. The spread
    # matters as much as the level — 0.17 between quiet and loud was too narrow
    # for the dynamics to show. Note the sweeps could not resolve an optimum at
    # one seed per setting, so this is set by eye, not by measurement.
    strength_by_level = spec.get("strength_by_level",
                                 {"low": 0.70, "mid": 0.50, "high": 0.32})
    compression_by_level = spec.get("compression_by_level",
                                    {"low": 34, "mid": 42, "high": 50})

    style = spec.get("style", "")
    n_sections = len(analysis["sections"])
    cuts = phrase_cuts(lyrics) if spec.get("snap_phrases", True) else []
    bar_secs = 240.0 / max(analysis["tempo"], 1)  # 4 beats
    # Visualizer shots are about one bar long, so a full-bar snap window could
    # halve or double a shot and break the grid the mode exists to sit on. Half
    # a bar still lands cuts on sung phrases without that damage.
    window = spec.get("snap_window", bar_secs / 2 if visualizer else bar_secs)
    min_shot = spec.get("min_shot_secs", 1.6)
    used_cuts: set[float] = set()

    shots = []
    cum = 0.0  # seconds of video planned so far
    # Both modes progress through the scene list once and never return. In
    # narrative mode a scene is a place, so it spans a whole section. In
    # visualizer mode a scene is a single phenomenon with no reason to be
    # dwelt in, so scenes are many and each gets a short run of shots —
    # progression, not rotation. Cycling a handful of phenomena reads as an
    # obvious loop; clustering many shots on one produces near-identical
    # frames. A short run of each, moving forward, avoids both.
    total_shots_est = max(1, sum(
        max(1, (sec["end_bar"] - sec["start_bar"]) // bars_per_shot[sec["level"]])
        for sec in analysis["sections"]))

    for si, sec in enumerate(analysis["sections"]):
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
            if visualizer:
                # position through the track -> position through the phenomena
                pos = len(shots) / max(total_shots_est, 1)
                shot_scene = scenes[min(int(pos * len(scenes)), len(scenes) - 1)]
            else:
                shot_scene = scene
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
                "strength": strength_by_level.get(level, 1.0),
                "img_compression": compression_by_level.get(level, 33),
                # Window of the track this shot covers, for audio conditioning.
                "audio_t0": round(cum, 3),
                "audio_dur": round(frames / fps, 3),
            })
            cum += frames / fps
            b = b_end
        if cum >= duration:
            break

    # Beat tracking can stop well before the audio does — an ambient outro or
    # a fade leaves no detectable beat, so the last bar may sit tens of seconds
    # from the end. A single shot is capped at 241 frames, so without this the
    # remainder is silently dropped and the assembled video ends early against
    # its own soundtrack.
    if scenes and cum < duration - 0.5:
        tail_scene = scenes[-1]
        level = analysis["sections"][-1]["level"] if analysis["sections"] else "low"
        while cum < duration - 0.5:
            frames = quantize_frames(min(duration - cum, 9.6), fps)
            shots.append({
                "idx": len(shots),
                "scene": tail_scene["id"],
                "level": level,
                "t0": round(cum, 3),
                "frames": frames,
                "seed": base_seed + len(shots),
                "video_prompt": tail_scene["video_prompt"] + " " + tail_scene["motion"][level],
                "still_seed": base_seed + 9000 + len(shots),
                "phrase_cut": False,
                "strength": strength_by_level.get(level, 1.0),
                "img_compression": compression_by_level.get(level, 33),
                "audio_t0": round(cum, 3),
                "audio_dur": round(frames / fps, 3),
                "tail": True,
            })
            cum += frames / fps

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
                if visualizer and fam in ABSTRACT_FRAMING:
                    scene_style += " " + ABSTRACT_FRAMING[fam]
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
