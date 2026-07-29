"""Track analysis: tempo, beat grid, bars, per-bar energy, energy-labelled sections."""
import json
import sys

import librosa
import numpy as np


def analyze(track_path: str) -> dict:
    y, sr = librosa.load(track_path, sr=22050, mono=True)
    duration = float(len(y) / sr)

    tempo, beat_frames = librosa.beat.beat_track(y=y, sr=sr, units="frames")
    beat_times = librosa.frames_to_time(beat_frames, sr=sr).tolist()
    tempo = float(np.atleast_1d(tempo)[0])

    # Assume 4/4; bars start on every 4th beat. No downbeat tracker in v0.1 —
    # phase 0 is close enough for cut placement on electronic tracks.
    bar_times = beat_times[::4]

    rms = librosa.feature.rms(y=y)[0]
    rms_times = librosa.times_like(rms, sr=sr)

    bars = []
    for i, t0 in enumerate(bar_times):
        t1 = bar_times[i + 1] if i + 1 < len(bar_times) else duration
        mask = (rms_times >= t0) & (rms_times < t1)
        energy = float(rms[mask].mean()) if mask.any() else 0.0
        bars.append({"i": i, "t0": float(t0), "t1": float(t1), "energy": energy})

    # Label bars low/mid/high by energy tercile. Raw per-bar RMS flips labels
    # constantly on dynamically-compressed tracks, so smooth over a 4-bar
    # window first; then merge contiguous runs (min 4 bars) into sections.
    energies = np.array([b["energy"] for b in bars])
    kernel = np.ones(4) / 4
    smoothed = np.convolve(energies, kernel, mode="same")
    lo, hi = np.quantile(smoothed, [0.33, 0.75])
    labels = ["low" if e <= lo else "high" if e >= hi else "mid" for e in smoothed]

    sections = []
    for i, lab in enumerate(labels):
        if sections and sections[-1]["level"] == lab:
            sections[-1]["end_bar"] = i + 1
        else:
            sections.append({"start_bar": i, "end_bar": i + 1, "level": lab})
    merged = []
    for s in sections:
        if merged and s["end_bar"] - s["start_bar"] < 4:
            merged[-1]["end_bar"] = s["end_bar"]
        else:
            merged.append(s)

    return {
        "track": track_path,
        "duration": duration,
        "tempo": tempo,
        "beat_times": [float(t) for t in beat_times],
        "bars": bars,
        "sections": merged,
    }


def main():
    track, out = sys.argv[1], sys.argv[2]
    result = analyze(track)
    json.dump(result, open(out, "w"), indent=1)
    print(f"{result['duration']:.1f}s @ {result['tempo']:.1f} BPM, "
          f"{len(result['bars'])} bars, {len(result['sections'])} sections")
    for s in result["sections"]:
        print(f"  bars {s['start_bar']:3d}-{s['end_bar']:3d}  {s['level']}")


if __name__ == "__main__":
    main()
