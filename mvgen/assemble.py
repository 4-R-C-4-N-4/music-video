"""Assembly: concat rendered shots (dropping their generated audio), mux the real track."""
import json
import pathlib
import subprocess
import sys


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

    silent = workdir / "concat-silent.mp4"
    subprocess.run([
        "ffmpeg", "-y", "-v", "error", "-f", "concat", "-safe", "0", "-i", str(concat),
        "-an", "-c:v", "libx264", "-crf", "18", "-preset", "medium",
        "-pix_fmt", "yuv420p", "-r", str(manifest["fps"]), str(silent)], check=True)

    subprocess.run([
        "ffmpeg", "-y", "-v", "error", "-i", str(silent), "-i", manifest["track"],
        "-map", "0:v", "-map", "1:a", "-c:v", "copy", "-c:a", "aac", "-b:a", "192k",
        "-movflags", "+faststart", "-shortest", out_path], check=True)
    print(f"assembled: {out_path}")


if __name__ == "__main__":
    assemble(sys.argv[1], sys.argv[2], sys.argv[3])
