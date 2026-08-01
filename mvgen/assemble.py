"""Assembly: concat rendered shots (dropping their generated audio), mux the real track."""
import json
import pathlib
import subprocess
import sys


def clip_fps(path, default: int) -> float:
    """Frame rate of a rendered shot, not of the plan.

    --fps50 doubles the rate of every shot, but the manifest still records the
    planned 25. Forcing the manifest value here decimated 50fps clips back to
    25 at the final step, so the feature rendered correctly and was then thrown
    away during assembly.
    """
    r = subprocess.run(["ffprobe", "-v", "error", "-select_streams", "v:0",
                        "-show_entries", "stream=avg_frame_rate",
                        "-of", "csv=p=0", str(path)],
                       capture_output=True, text=True)
    try:
        num, den = r.stdout.strip().split("/")
        return float(num) / float(den)
    except (ValueError, ZeroDivisionError):
        return float(default)


def assemble(manifest_path: str, workdir: str, out_path: str):
    manifest = json.load(open(manifest_path))
    workdir = pathlib.Path(workdir)
    state = json.load(open(workdir / "state.json"))

    missing = [s["idx"] for s in manifest["shots"] if str(s["idx"]) not in state["shots"]]
    if missing:
        sys.exit(f"missing shots: {missing}")

    concat = workdir / "concat.txt"
    concat.write_text("".join(
        f"file '{(workdir / state['shots'][str(s['idx'])]).resolve()}'\n"
        for s in manifest["shots"]))

    first = workdir / state["shots"][str(manifest["shots"][0]["idx"])]
    fps = clip_fps(first, manifest["fps"])
    if abs(fps - manifest["fps"]) > 0.1:
        print(f"clips are {fps:g}fps (plan says {manifest['fps']}) — keeping {fps:g}",
              flush=True)

    silent = workdir / "concat-silent.mp4"
    subprocess.run([
        "ffmpeg", "-y", "-v", "error", "-f", "concat", "-safe", "0", "-i", str(concat),
        "-an", "-c:v", "libx264", "-crf", "18", "-preset", "medium",
        "-pix_fmt", "yuv420p", "-r", f"{fps:g}", str(silent)], check=True)

    subprocess.run([
        "ffmpeg", "-y", "-v", "error", "-i", str(silent), "-i", manifest["track"],
        "-map", "0:v", "-map", "1:a", "-c:v", "copy", "-c:a", "aac", "-b:a", "192k",
        "-movflags", "+faststart", "-shortest", out_path], check=True)
    print(f"assembled: {out_path}")


if __name__ == "__main__":
    assemble(sys.argv[1], sys.argv[2], sys.argv[3])
