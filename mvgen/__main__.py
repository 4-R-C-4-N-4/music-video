"""One-command build: analyze -> lyrics -> direct -> render -> assemble.

    python -m mvgen build <track> jobs/<job> [limit] [--no-lyrics] [--model=X]

Every stage is skipped if its output already exists, so re-running resumes.
Delete a stage's output to redo it (analysis.json, lyrics.json, scenes.json,
manifest.json, or an entry in work/state.json for a single shot).

Nothing here needs a human in the loop: if scenes.json is absent, a local
model writes it from the track's structure and transcript. Hand-author or
edit scenes.json when you *do* want control — it is always preferred over
regenerating.

GPU is exclusive: the local LLM, then ComfyUI. Each stage takes the card and
releases it before the next.
"""
import json
import pathlib
import subprocess
import sys
import time
import urllib.request

from . import analyze as _analyze
from . import assemble as _assemble
from . import direct as _direct
from . import plan as _plan
from . import render as _render

COMFY_HEALTH = "http://127.0.0.1:8188/system_stats"


def comfy_up() -> bool:
    try:
        urllib.request.urlopen(COMFY_HEALTH, timeout=3)
        return True
    except Exception:
        return False


def ensure_comfy(timeout: int = 120) -> None:
    if comfy_up():
        return
    start = pathlib.Path.home() / "programs/comfyui/start.sh"
    if not start.exists():
        sys.exit(f"ComfyUI not running and {start} not found")
    print("starting comfyui...", flush=True)
    subprocess.Popen(["nohup", str(start)],
                     stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                     start_new_session=True)
    deadline = time.time() + timeout
    while time.time() < deadline:
        if comfy_up():
            print("comfyui up", flush=True)
            return
        time.sleep(3)
    sys.exit("comfyui did not come up")


def build(track: str, jobdir: str, limit: int | None = None,
          do_lyrics: bool = True, model: str = "gemma4-nothink"):
    job = pathlib.Path(jobdir)
    job.mkdir(parents=True, exist_ok=True)

    analysis_path = job / "analysis.json"
    if not analysis_path.exists():
        result = _analyze.analyze(track)
        json.dump(result, open(analysis_path, "w"), indent=1)
        print(f"analyzed: {result['duration']:.1f}s @ {result['tempo']:.1f} BPM, "
              f"{len(result['sections'])} sections", flush=True)

    lyrics_path = job / "lyrics.json"
    if do_lyrics and not lyrics_path.exists():
        try:
            from . import lyrics as _lyrics
            res = _lyrics.extract(track, str(lyrics_path))
            print("lyrics: "
                  + (f"{len(res['segments'])} phrases" if res["has_sung_lyrics"]
                     else "instrumental")
                  + f", {len(res['vocal_spans'])} vocal spans", flush=True)
        except Exception as e:  # transcription is optional enrichment
            print(f"lyrics step skipped ({e})", flush=True)

    spec_path = job / "scenes.json"
    if not spec_path.exists():
        print("no scenes.json — directing automatically", flush=True)
        spec = _direct.direct(str(job), model)
        print(f"concept: {spec['concept']}", flush=True)
        for s in spec["scenes"]:
            print(f"  {s['id']} [{s['material']}]", flush=True)

    manifest_path = job / "manifest.json"
    if not manifest_path.exists():
        lyrics = json.load(open(lyrics_path)) if lyrics_path.exists() else None
        manifest = _plan.plan(json.load(open(analysis_path)),
                              json.load(open(spec_path)), lyrics)
        json.dump(manifest, open(manifest_path, "w"), indent=1)
        snapped = sum(1 for s in manifest["shots"] if s.get("phrase_cut"))
        print(f"planned: {len(manifest['shots'])} shots, {snapped} on phrase ends, "
              f"~{2*len(manifest['shots'])} min render", flush=True)

    ensure_comfy()
    _render.render(str(manifest_path), str(job / "work"), limit)

    if limit is None:
        out = job / (job.name + ".mp4")
        _assemble.assemble(str(manifest_path), str(job / "work"), str(out))


def main():
    flags = [a for a in sys.argv[1:] if a.startswith("--")]
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    if len(args) < 3 or args[0] != "build":
        sys.exit(__doc__)
    model = next((f.split("=", 1)[1] for f in flags if f.startswith("--model=")),
                 "gemma4-nothink")
    build(args[1], args[2],
          int(args[3]) if len(args) > 3 else None,
          do_lyrics="--no-lyrics" not in flags,
          model=model)


if __name__ == "__main__":
    main()
