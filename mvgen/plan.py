"""Shot planning: analysis + scene spec -> reproducible shot manifest.

Sections are assigned to scenes in order (cycling). Each section is split
into shots whose length in bars depends on the section's energy level.
Shot frame counts obey LTX's 8n+1 constraint; each shot's frame count is
chosen against the *absolute* track-time boundary so quantization error
never accumulates (every cut stays within 4 frames of its target beat).
"""
import json
import sys

# Per-shot camera variation within a scene: same world, different setup.
# Kills the "same keyframe looping" feel of naive one-still-per-scene.
ANGLES = [
    "Wide establishing shot.",
    "Medium shot, closer to the figure.",
    "Low angle looking up, foreground elements towering.",
    "Detail close-up of the scene's central element.",
    "Reverse angle, looking back the way we came.",
    "Slightly elevated shot, the figure small in the landscape.",
]

# Progression through a scene: shot k of n moves through these stages.
STAGES = [
    "The scene at its threshold, just arrived.",
    "Deeper in now, the scene fully surrounding.",
    "At the heart of the scene, its central element close and dominant.",
    "Passing through the far side, the scene beginning to open up.",
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

    # Per-shot stills: each shot in a scene gets its own camera angle and a
    # progression stage through the scene, plus the global style clause.
    by_scene: dict[str, list[dict]] = {}
    for s in shots:
        by_scene.setdefault(s["scene"], []).append(s)
    scene_map = {s["id"]: s for s in scenes}
    for sid, group in by_scene.items():
        n = len(group)
        for k, shot in enumerate(group):
            stage = STAGES[min(k * len(STAGES) // max(n, 1), len(STAGES) - 1)]
            angle = ANGLES[k % len(ANGLES)]
            shot["still_prompt"] = " ".join(
                p for p in (scene_map[sid]["still_prompt"], stage, angle, style) if p)

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
