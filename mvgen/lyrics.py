"""Lyric extraction: Demucs vocal isolation -> Whisper word-level timestamps.

Whisper on a full mix does badly on sung vocals — instrumentation smears the
acoustics it expects — so we split the vocal stem out first and transcribe
that alone. Output feeds the planner: phrase boundaries make better cut points
than arbitrary bar multiples, and vocal-present vs instrumental is often a
truer section signal than RMS energy alone.

Usage: python -m mvgen.lyrics <track> <out.json> [--model large-v3]

Writes lyrics.json; prints only timing metadata, never transcript text.
"""
import json
import pathlib
import subprocess
import sys
import tempfile


def isolate_vocals(track: str, workdir: pathlib.Path) -> pathlib.Path:
    """Run Demucs two-stem separation; return path to the vocal stem."""
    subprocess.run([sys.executable, "-m", "demucs", "--two-stems=vocals",
                    "-n", "htdemucs", "-o", str(workdir), track],
                   check=True, capture_output=True)
    hits = list(workdir.glob("htdemucs/*/vocals.wav"))
    if not hits:
        raise RuntimeError("demucs produced no vocal stem")
    return hits[0]


def transcribe(vocal_path: pathlib.Path, model_size: str = "large-v3") -> list[dict]:
    import os

    from faster_whisper import WhisperModel

    # CPU by default, and deliberately not "try CUDA then fall back": a failed
    # CUDA init still allocates a context (~3GB measured) that is never
    # released for the life of the process, which later starves ComfyUI of
    # exactly the headroom the video model needs. Opt in only where CUDA is
    # known to work — CTranslate2 links cuBLAS 12, so a cu13 torch in the same
    # venv makes it unloadable regardless.
    device = os.environ.get("MVGEN_WHISPER_DEVICE", "cpu")
    compute = "float16" if device == "cuda" else "int8"
    model = WhisperModel(model_size, device=device, compute_type=compute)
    segments, _info = model.transcribe(str(vocal_path), word_timestamps=True,
                                       vad_filter=True, beam_size=5)
    # faster-whisper hands back numpy scalars; coerce everything to plain
    # Python types or json.dump chokes downstream.
    out = []
    for seg in segments:
        out.append({
            "start": round(float(seg.start), 3),
            "end": round(float(seg.end), 3),
            "text": seg.text.strip(),
            "words": [{"w": w.word.strip(), "start": round(float(w.start), 3),
                       "end": round(float(w.end), 3),
                       "prob": round(float(w.probability), 3)}
                      for w in (seg.words or [])],
        })
    return out


def vocal_spans(segments: list[dict], gap: float = 2.0) -> list[dict]:
    """Merge segments separated by < `gap` seconds into contiguous sung spans.

    The complement of these spans is instrumental — a structural signal the
    planner can use independently of energy.
    """
    spans = []
    for seg in segments:
        if spans and seg["start"] - spans[-1]["end"] <= gap:
            spans[-1]["end"] = seg["end"]
            spans[-1]["n"] += 1
        else:
            spans.append({"start": seg["start"], "end": seg["end"], "n": 1})
    return spans


def energy_spans(vocal_path: pathlib.Path, thresh_ratio: float = 0.35,
                 min_len: float = 1.5, gap: float = 2.0) -> list[dict]:
    """Vocal-presence spans from stem energy alone.

    Works on instrumental tracks with vocal texture (chops, pads, samples)
    where transcription yields nothing — the planner can still use "vocal
    present here" as a structural signal.
    """
    import numpy as np
    import soundfile as sf

    y, sr = sf.read(str(vocal_path))
    y = y.mean(axis=1) if y.ndim > 1 else y
    win = int(sr * 0.5)
    rms = np.array([np.sqrt((y[i:i + win] ** 2).mean())
                    for i in range(0, max(len(y) - win, 1), win)])
    if not len(rms) or rms.max() <= 0:
        return []
    hot = rms >= rms.max() * thresh_ratio

    spans, start = [], None
    for i, on in enumerate(hot):
        t = i * 0.5
        if on and start is None:
            start = t
        elif not on and start is not None:
            spans.append({"start": round(start, 2), "end": round(t, 2)})
            start = None
    if start is not None:
        spans.append({"start": round(start, 2), "end": round(len(y) / sr, 2)})

    merged: list[dict] = []
    for s in spans:
        if merged and s["start"] - merged[-1]["end"] <= gap:
            merged[-1]["end"] = s["end"]
        else:
            merged.append(s)
    return [s for s in merged if s["end"] - s["start"] >= min_len]


def extract(track: str, out_path: str, model_size: str = "large-v3") -> dict:
    with tempfile.TemporaryDirectory() as td:
        workdir = pathlib.Path(td)
        print("separating vocal stem...", flush=True)
        vocal = isolate_vocals(track, workdir)
        print(f"transcribing with whisper {model_size}...", flush=True)
        segments = transcribe(vocal, model_size)
        energy = energy_spans(vocal)

    words = [w for s in segments for w in s["words"]]
    mean_conf = float(sum(w["prob"] for w in words) / len(words)) if words else 0.0
    # Sparse, low-confidence output on a track with real vocal energy means
    # Whisper is hallucinating speech onto non-speech texture. Say so rather
    # than handing the planner garbage phrases.
    sung = bool(len(words) >= 20 and mean_conf >= 0.6)

    result = {
        "track": track,
        "has_sung_lyrics": sung,
        "mean_confidence": round(mean_conf, 3),
        "segments": segments if sung else [],
        "rejected_segments": [] if sung else segments,
        "vocal_spans": vocal_spans(segments) if sung else energy,
        "vocal_spans_source": "transcript" if sung else "stem_energy",
    }
    json.dump(result, open(out_path, "w"), indent=1)
    return result


def main():
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    model = next((a.split("=", 1)[1] for a in sys.argv if a.startswith("--model=")),
                 "large-v3")
    result = extract(args[0], args[1], model)
    spans = result["vocal_spans"]
    if result["has_sung_lyrics"]:
        segs = result["segments"]
        words = sum(len(s["words"]) for s in segs)
        print(f"sung lyrics: {len(segs)} phrases, {words} words, "
              f"mean confidence {result['mean_confidence']:.2f}")
    else:
        n = len(result["rejected_segments"])
        print(f"no sung lyrics detected (only {n} low-confidence fragments, "
              f"mean {result['mean_confidence']:.2f}) — treating as instrumental")
    print(f"{len(spans)} vocal spans (from {result['vocal_spans_source']}):")
    for s in spans:
        print(f"  {s['start']:6.1f}s - {s['end']:6.1f}s")


if __name__ == "__main__":
    main()
