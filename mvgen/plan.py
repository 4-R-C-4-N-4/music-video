"""Shot planning: analysis + scene spec -> reproducible shot manifest.

Sections are assigned to scenes in order (cycling). Each section is split
into shots whose length in bars depends on the section's energy level.
Shot frame counts obey LTX's 8n+1 constraint; each shot's frame count is
chosen against the *absolute* track-time boundary so quantization error
never accumulates (every cut stays within 4 frames of its target beat).
"""
import json
import sys

# Fallback beats when a scene doesn't author its own: generic blocking that
# leads the prompt (the image model follows lead content, so composition MUST
# come first — a camera phrase appended after a long setting prompt is noise).
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


def plan(analysis: dict, spec: dict) -> dict:
    fps = spec.get("fps", 25)
    bars_per_shot = spec.get("bars_per_shot", {"low": 4, "mid": 2, "high": 2})
    base_seed = spec.get("seed", 1000)
    scenes = spec["scenes"]
    bars = analysis["bars"]
    duration = analysis["duration"]

    style = spec.get("style", "")
    n_sections = len(analysis["sections"])

    shots = []
    cum = 0.0  # seconds of video planned so far
    for si, sec in enumerate(analysis["sections"]):
        # Linear allocation: scenes play through once, in order, across the
        # whole song — a journey, not a cycle. No scene ever returns.
        scene = scenes[min(si * len(scenes) // n_sections, len(scenes) - 1)]
        level = sec["level"]
        step = bars_per_shot[level]
        b = sec["start_bar"]
        while b < sec["end_bar"]:
            b_end = min(b + step, sec["end_bar"])
            # absolute target end-time for this shot
            t1 = bars[b_end]["t0"] if b_end < len(bars) else duration
            frames = quantize_frames(t1 - cum, fps)
            shots.append({
                "idx": len(shots),
                "scene": scene["id"],
                "level": level,
                "t0": round(cum, 3),
                "frames": frames,
                "seed": base_seed + len(shots),
                "video_prompt": scene["video_prompt"] + " " + scene["motion"][level],
                "still_seed": base_seed + 9000 + len(shots),
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
            shot["still_prompt"] = " ".join(p for p in (
                "Cinematic film still: " + beat,
                "Setting: " + scene_map[sid]["still_prompt"],
                style) if p)

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
    analysis_path, spec_path, out = sys.argv[1], sys.argv[2], sys.argv[3]
    manifest = plan(json.load(open(analysis_path)), json.load(open(spec_path)))
    json.dump(manifest, open(out, "w"), indent=1)
    n = len(manifest["shots"])
    print(f"{n} shots (one still each), "
          f"video {manifest['video_duration']:.1f}s vs track {manifest['track_duration']:.1f}s, "
          f"~{n * 2} min render")


if __name__ == "__main__":
    main()
