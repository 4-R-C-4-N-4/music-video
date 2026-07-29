"""Shot planning: analysis + scene spec -> reproducible shot manifest.

Sections are assigned to scenes in order (cycling). Each section is split
into shots whose length in bars depends on the section's energy level.
Shot frame counts obey LTX's 8n+1 constraint; each shot's frame count is
chosen against the *absolute* track-time boundary so quantization error
never accumulates (every cut stays within 4 frames of its target beat).
"""
import json
import sys


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

    shots = []
    cum = 0.0  # seconds of video planned so far
    for si, sec in enumerate(analysis["sections"]):
        scene = scenes[si % len(scenes)]
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
            })
            cum += frames / fps
            b = b_end
        if cum >= duration:
            break

    return {
        "track": analysis["track"],
        "fps": fps,
        "width": spec.get("width", 1280),
        "height": spec.get("height", 704),
        "video_duration": round(cum, 3),
        "track_duration": duration,
        "stills": {s["id"]: {"prompt": s["still_prompt"], "seed": base_seed + 9000 + i}
                   for i, s in enumerate(scenes)},
        "shots": shots,
    }


def main():
    analysis_path, spec_path, out = sys.argv[1], sys.argv[2], sys.argv[3]
    manifest = plan(json.load(open(analysis_path)), json.load(open(spec_path)))
    json.dump(manifest, open(out, "w"), indent=1)
    n = len(manifest["shots"])
    est = n * 2
    print(f"{n} shots, {len(manifest['stills'])} stills, "
          f"video {manifest['video_duration']:.1f}s vs track {manifest['track_duration']:.1f}s, "
          f"~{est} min render")


if __name__ == "__main__":
    main()
