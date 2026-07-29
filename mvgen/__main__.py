"""One-command build: analyze -> plan -> render -> assemble.

Usage: python -m mvgen build <track file> jobs/<job> [render limit]

The job dir must contain scenes.json. Analysis/manifest are only regenerated
if missing (delete them to force re-analysis); render resumes from state.json.
"""
import json
import pathlib
import subprocess
import sys

from . import analyze as _analyze
from . import assemble as _assemble
from . import plan as _plan
from . import render as _render


def build(track: str, jobdir: str, limit: int | None = None):
    job = pathlib.Path(jobdir)
    spec_path = job / "scenes.json"
    if not spec_path.exists():
        sys.exit(f"no scenes.json in {job} — copy one from an existing job and edit")

    analysis_path = job / "analysis.json"
    if not analysis_path.exists():
        result = _analyze.analyze(track)
        json.dump(result, open(analysis_path, "w"), indent=1)
        print(f"analyzed: {result['duration']:.1f}s @ {result['tempo']:.1f} BPM, "
              f"{len(result['sections'])} sections")

    manifest_path = job / "manifest.json"
    if not manifest_path.exists():
        manifest = _plan.plan(json.load(open(analysis_path)), json.load(open(spec_path)))
        json.dump(manifest, open(manifest_path, "w"), indent=1)
        print(f"planned: {len(manifest['shots'])} shots, ~{2*len(manifest['shots'])} min render")

    _render.render(str(manifest_path), str(job / "work"), limit)

    if limit is None:
        out = job / (job.name + ".mp4")
        _assemble.assemble(str(manifest_path), str(job / "work"), str(out))


def main():
    if len(sys.argv) < 4 or sys.argv[1] != "build":
        sys.exit(__doc__)
    build(sys.argv[2], sys.argv[3], int(sys.argv[4]) if len(sys.argv) > 4 else None)


if __name__ == "__main__":
    main()
