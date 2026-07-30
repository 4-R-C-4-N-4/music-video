"""Render a shot manifest against a local ComfyUI: Z-Image stills, LTX-2.3 i2v shots.

Idempotent: finished outputs are recorded in <workdir>/state.json and skipped
on re-run, so a killed render resumes and editing one shot's prompt/seed in
the manifest re-renders only that shot (delete its state entry or use --redo).
"""
import json
import os
import pathlib
import shutil
import subprocess
import sys
import time
import urllib.error
import urllib.request

COMFY = "http://127.0.0.1:8188"
COMFY_OUT = pathlib.Path.home() / "programs/comfyui/output"
COMFY_IN = pathlib.Path.home() / "programs/comfyui/input"
SIGMAS = "1., 0.99375, 0.9875, 0.98125, 0.975, 0.909375, 0.725, 0.421875, 0.0"

MODELS = {
    "ltx": "LTX-2.3-22B-distilled-1.1-Q4_K_M.gguf",
    "gemma": "gemma_3_12B_it_fp8_scaled.safetensors",
    "proj": "ltx-2.3_text_projection_bf16.safetensors",
    "vvae": "LTX23_video_vae_bf16.safetensors",
    "avae": "LTX23_audio_vae_bf16.safetensors",
    "zimage": "z_image_turbo_bf16.safetensors",
    "qwen": "qwen_3_4b_fp8_mixed.safetensors",
    "zvae": "z_image_ae.safetensors",
}


def api(path, data=None, timeout=30):
    req = urllib.request.Request(COMFY + path,
                                 data=json.dumps(data).encode() if data else None,
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read())


def comfy_up() -> bool:
    try:
        api("/system_stats", timeout=3)
        return True
    except Exception:
        return False


def ensure_comfy(timeout: int = 180) -> bool:
    """Start ComfyUI if it isn't answering. Returns True once it is.

    The server dies unpredictably on long unattended runs, so every stage
    that talks to it must be able to bring it back rather than failing the
    whole build.
    """
    if comfy_up():
        return True
    start = pathlib.Path.home() / "programs/comfyui/start.sh"
    if not start.exists():
        return False
    subprocess.run(["pkill", "-9", "-f", "main.py --listen 127.0.0.1 --port 8188"],
                   capture_output=True)
    time.sleep(2)
    print("  (re)starting comfyui...", flush=True)
    # Keep the server's own output — when it dies mid-render this file is the
    # only record of why, and discarding it makes the failure undiagnosable.
    log = pathlib.Path.home() / ".cache/mvgen-comfyui.log"
    log.parent.mkdir(parents=True, exist_ok=True)
    fh = open(log, "a")
    subprocess.Popen([str(start)], stdout=fh, stderr=subprocess.STDOUT,
                     start_new_session=True)
    deadline = time.time() + timeout
    while time.time() < deadline:
        if comfy_up():
            return True
        time.sleep(3)
    return False


def _run_once(graph, label, timeout=1800):
    pid = api("/prompt", {"prompt": graph})["prompt_id"]
    t0 = time.time()
    while time.time() - t0 < timeout:
        time.sleep(4)
        h = api(f"/history/{pid}")
        if pid in h:
            st = h[pid]["status"]
            if st.get("status_str") == "error":
                for m in st.get("messages", []):
                    if m[0] == "execution_error":
                        raise RuntimeError(f"{label}: {m[1].get('exception_message')}")
                raise RuntimeError(f"{label}: unknown comfy error")
            print(f"  {label} done in {time.time()-t0:.0f}s", flush=True)
            return h[pid]["outputs"]
    raise TimeoutError(label)


def run_graph(graph, label, timeout=1800, attempts=4):
    """Run a graph, surviving ComfyUI dying underneath us.

    A dead server is a transport failure (connection refused / timeout), not
    a RuntimeError — those come from the graph itself and are not retried,
    since re-running a broken graph just fails again more slowly.
    """
    for attempt in range(attempts):
        try:
            return _run_once(graph, label, timeout)
        except (urllib.error.URLError, TimeoutError, ConnectionError, OSError) as e:
            if attempt == attempts - 1:
                raise
            print(f"  {label}: comfy unreachable ({e}); recovering "
                  f"[{attempt+1}/{attempts-1}]", flush=True)
            if not ensure_comfy():
                raise RuntimeError("could not bring comfyui back up") from e


def still_graph(prompt, seed, w, h, prefix):
    return {
        "1": {"class_type": "UNETLoader", "inputs": {"unet_name": MODELS["zimage"], "weight_dtype": "default"}},
        "2": {"class_type": "CLIPLoader", "inputs": {"clip_name": MODELS["qwen"], "type": "lumina2", "device": "default"}},
        "3": {"class_type": "CLIPTextEncode", "inputs": {"text": prompt, "clip": ["2", 0]}},
        "4": {"class_type": "ConditioningZeroOut", "inputs": {"conditioning": ["3", 0]}},
        "5": {"class_type": "VAELoader", "inputs": {"vae_name": MODELS["zvae"]}},
        "6": {"class_type": "EmptySD3LatentImage", "inputs": {"width": w, "height": h, "batch_size": 1}},
        "7": {"class_type": "ModelSamplingAuraFlow", "inputs": {"model": ["1", 0], "shift": 3}},
        "8": {"class_type": "KSampler", "inputs": {
            "model": ["7", 0], "seed": seed, "steps": 8, "cfg": 1.0,
            "sampler_name": "res_multistep", "scheduler": "simple", "denoise": 1.0,
            "positive": ["3", 0], "negative": ["4", 0], "latent_image": ["6", 0]}},
        "9": {"class_type": "VAEDecode", "inputs": {"samples": ["8", 0], "vae": ["5", 0]}},
        "10": {"class_type": "SaveImage", "inputs": {"images": ["9", 0], "filename_prefix": prefix}},
    }


def i2v_graph(image_name, prompt, seed, w, h, frames, fps, prefix,
              audio_name=None, strength=1.0, img_compression=33):
    """Image-to-video graph.

    `audio_name` is a wav in ComfyUI's input dir covering exactly this shot's
    window. When given, the real audio is encoded and conditioned on instead of
    a silent latent, so motion answers the music rather than merely being cut
    to it. Its duration must equal frames/fps or the audio and video latents
    disagree in length.

    `strength` is how hard the video is pinned to the still. 1.0 leaves almost
    no licence to depart from the keyframe, which reads as a slow drift;
    lowering it buys motion at the cost of fidelity to the composed frame.
    """
    return {
        "1": {"class_type": "UnetLoaderGGUF", "inputs": {"unet_name": MODELS["ltx"]}},
        "2": {"class_type": "DualCLIPLoader", "inputs": {
            "clip_name1": MODELS["gemma"], "clip_name2": MODELS["proj"],
            "type": "ltxv", "device": "default"}},
        "3": {"class_type": "CLIPTextEncode", "inputs": {"text": prompt, "clip": ["2", 0]}},
        "4": {"class_type": "CLIPTextEncode", "inputs": {"text": "", "clip": ["2", 0]}},
        "5": {"class_type": "VAELoader", "inputs": {"vae_name": MODELS["vvae"]}},
        "6": {"class_type": "LTXVAudioVAELoader", "inputs": {"ckpt_name": MODELS["avae"]}},
        "7": {"class_type": "EmptyLTXVLatentVideo", "inputs": {"width": w, "height": h, "length": frames, "batch_size": 1}},
        "8": ({"class_type": "LTXVAudioVAEEncode",
               "inputs": {"audio": ["24", 0], "audio_vae": ["6", 0]}}
              if audio_name else
              {"class_type": "LTXVEmptyLatentAudio",
               "inputs": {"frames_number": frames, "frame_rate": fps,
                          "batch_size": 1, "audio_vae": ["6", 0]}}),
        "21": {"class_type": "LoadImage", "inputs": {"image": image_name}},
        "22": {"class_type": "LTXVPreprocess", "inputs": {"image": ["21", 0], "img_compression": img_compression}},
        "23": {"class_type": "LTXVImgToVideoInplace", "inputs": {
            "vae": ["5", 0], "image": ["22", 0], "latent": ["7", 0],
            "strength": strength, "bypass": False}},
        "9": {"class_type": "LTXVConcatAVLatent", "inputs": {"video_latent": ["23", 0], "audio_latent": ["8", 0]}},
        "10": {"class_type": "LTXVConditioning", "inputs": {"positive": ["3", 0], "negative": ["4", 0], "frame_rate": fps}},
        "11": {"class_type": "KSamplerSelect", "inputs": {"sampler_name": "euler_ancestral"}},
        "12": {"class_type": "ManualSigmas", "inputs": {"sigmas": SIGMAS}},
        "13": {"class_type": "RandomNoise", "inputs": {"noise_seed": seed}},
        "14": {"class_type": "CFGGuider", "inputs": {"model": ["1", 0], "positive": ["10", 0], "negative": ["10", 1], "cfg": 1.0}},
        "15": {"class_type": "SamplerCustomAdvanced", "inputs": {
            "noise": ["13", 0], "guider": ["14", 0], "sampler": ["11", 0],
            "sigmas": ["12", 0], "latent_image": ["9", 0]}},
        "16": {"class_type": "LTXVSeparateAVLatent", "inputs": {"av_latent": ["15", 0]}},
        "17": {"class_type": "VAEDecodeTiled", "inputs": {
            "samples": ["16", 0], "vae": ["5", 0],
            "tile_size": 512, "overlap": 64, "temporal_size": 4096, "temporal_overlap": 8}},
        "18": {"class_type": "LTXVAudioVAEDecode", "inputs": {"samples": ["16", 1], "audio_vae": ["6", 0]}},
        "19": {"class_type": "CreateVideo", "inputs": {"images": ["17", 0], "fps": fps, "audio": ["18", 0]}},
        "20": {"class_type": "SaveVideo", "inputs": {"video": ["19", 0], "filename_prefix": "video/" + prefix, "format": "auto", "codec": "auto"}},
        **({"24": {"class_type": "LoadAudio", "inputs": {"audio": audio_name}}}
           if audio_name else {}),
    }


def grab_output(outputs, node, key, workdir, dest_name):
    info = outputs[node][key][0]
    sub = info.get("subfolder", "")
    src = COMFY_OUT / sub / info["filename"]
    dest = workdir / dest_name
    shutil.copy(src, dest)
    return dest


def slice_audio(track: str, t0: float, dur: float, dest_name: str) -> str | None:
    """Cut the shot's own window out of the track, into ComfyUI's input dir.

    The slice must be exactly frames/fps long: the audio VAE derives its latent
    length from duration, and a mismatch puts the audio and video latents out of
    step when they are concatenated. Padded with silence if the track ends mid
    shot.
    """
    out = COMFY_IN / dest_name
    r = subprocess.run(
        ["ffmpeg", "-y", "-v", "error", "-ss", f"{t0:.3f}", "-i", track,
         "-t", f"{dur:.3f}", "-ac", "2", "-ar", "44100",
         "-af", f"apad=whole_dur={dur:.3f}", str(out)],
        capture_output=True)
    return dest_name if r.returncode == 0 and out.exists() else None


def pick_still(shot, i, total, w, h, job, workdir, picked,
               candidates: int, sim_max: float):
    """Render N candidate stills, score them, keep the best non-duplicate.

    Stills are ~8s against ~90s for the video they seed, so spending 3x here
    is the cheapest possible place to buy quality — the shot inherits whatever
    the still got wrong.

    Appends the winner's embedding to `picked` so later shots are checked
    against every frame already chosen, not just the previous one.
    """
    from .judge import rank_with_dedup, stack

    print(f"still {i+1}/{total} (scene {shot['scene']}, "
          f"best of {candidates})", flush=True)
    paths = []
    for c in range(candidates):
        seed = shot["still_seed"] + c * 7919
        out = run_graph(still_graph(shot["still_prompt"], seed, w, h,
                                    f"mv-{job}-s{i:03d}c{c}"), f"still:{i}.{c}")
        paths.append(str(grab_output(out, "10", "images", workdir,
                                     f"cand-{i:03d}-{c}.png")))

    if candidates == 1:
        winner = pathlib.Path(paths[0])
        final = winner.with_name(f"still-{i:03d}.png")
        winner.rename(final)
        return final

    desc = shot.get("judge_text") or shot["still_prompt"][:300]
    ranked = rank_with_dedup(paths, desc, stack(picked), sim_max=sim_max)
    best = ranked[0]
    dupes = sum(1 for r in ranked if r["duplicate"])
    note = f", {dupes} dup" if dupes else ""
    if best["duplicate"]:
        note += " (ALL duplicates — kept least similar)"
    print(f"  picked score {best['score']:+.3f} "
          f"(adh {best['adherence']:.3f}, dup {best['dup_sim']:.3f}{note})",
          flush=True)

    picked.append(best["embedding"])
    final = pathlib.Path(workdir) / f"still-{i:03d}.png"
    pathlib.Path(best["path"]).rename(final)
    for r in ranked[1:]:  # discard the losers, keep the directory readable
        pathlib.Path(r["path"]).unlink(missing_ok=True)
    return final


def render(manifest_path: str, workdir: str, limit: int | None = None,
           stills_only: bool = False, candidates: int | None = None,
           sim_max: float = 0.95, use_audio: bool | None = None):
    manifest = json.load(open(manifest_path))
    workdir = pathlib.Path(workdir)
    workdir.mkdir(parents=True, exist_ok=True)
    state_path = workdir / "state.json"
    state = json.load(open(state_path)) if state_path.exists() else {"stills": {}, "shots": {}}

    def save_state():
        json.dump(state, open(state_path, "w"), indent=1)

    w, h, fps = manifest["width"], manifest["height"], manifest["fps"]
    job = pathlib.Path(manifest_path).parent.name
    if candidates is None:
        candidates = int(os.environ.get("MVGEN_CANDIDATES", "3"))
    if use_audio is None:
        use_audio = os.environ.get("MVGEN_AUDIO_COND", "0") == "1"
    track = manifest.get("track")
    if use_audio:
        print(f"audio conditioning ON (slices cut per shot from {track})", flush=True)

    # Dedup is against every still already chosen, so on resume the existing
    # winners have to be re-embedded — cheap on CPU and it keeps a resumed
    # build held to the same uniqueness bar as a fresh one.
    picked: list = []
    if candidates > 1 and state["stills"]:
        from .judge import embed_images
        existing = [str(workdir / n) for n in state["stills"].values()
                    if (workdir / n).exists()]
        if existing:
            print(f"re-embedding {len(existing)} already-picked stills for dedup",
                  flush=True)
            emb = embed_images(existing)
            picked = [emb[j:j + 1] for j in range(len(existing))]

    todo = [s for s in manifest["shots"] if str(s["idx"]) not in state["shots"]]
    if limit:
        todo = todo[:limit]
    for shot in todo:
        i = shot["idx"]
        key = str(i)
        if key not in state["stills"]:
            dest = pick_still(shot, i, len(manifest["shots"]), w, h, job,
                              workdir, picked, candidates, sim_max)
            shutil.copy(dest, COMFY_IN / dest.name)
            state["stills"][key] = dest.name
            save_state()
        if stills_only:
            continue
        print(f"shot {i+1}/{len(manifest['shots'])} "
              f"(scene {shot['scene']}, {shot['frames']}f, {shot['level']})", flush=True)
        audio_name = None
        if use_audio and track and shot.get("audio_dur"):
            audio_name = slice_audio(track, shot["audio_t0"], shot["audio_dur"],
                                     f"mv-{job}-a{i:03d}.wav")
        g = i2v_graph(state["stills"][key], shot["video_prompt"], shot["seed"],
                      w, h, shot["frames"], fps, f"mv-{job}-{i:03d}",
                      audio_name=audio_name,
                      strength=shot.get("strength", 1.0),
                      img_compression=shot.get("img_compression", 33))
        out = run_graph(g, f"shot:{i}")
        dest = grab_output(out, "20", "images", workdir, f"shot-{i:03d}.mp4")
        state["shots"][str(i)] = dest.name
        save_state()
    done = len(state["shots"])
    print(f"rendered: {done}/{len(manifest['shots'])} shots complete", flush=True)


def main():
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    manifest_path, workdir = args[0], args[1]
    limit = int(args[2]) if len(args) > 2 else None
    cands = next((int(a.split("=", 1)[1]) for a in sys.argv
                  if a.startswith("--candidates=")), None)
    render(manifest_path, workdir, limit,
           stills_only="--stills-only" in sys.argv, candidates=cands)


if __name__ == "__main__":
    main()


def comfy_down() -> None:
    """Stop ComfyUI and wait for the GPU to actually be released.

    The local LLM needs ~14GB and ComfyUI holds ~18GB resident, so the two
    cannot overlap even briefly. Any stage that wants the card must take it
    explicitly rather than assuming the previous stage tidied up.
    """
    subprocess.run(["pkill", "-9", "-f", "main.py --listen 127.0.0.1 --port 8188"],
                   capture_output=True)
    for _ in range(20):
        time.sleep(1)
        r = subprocess.run(["nvidia-smi", "--query-gpu=memory.used",
                            "--format=csv,noheader,nounits"],
                           capture_output=True, text=True)
        try:
            if int(r.stdout.strip().splitlines()[0]) < 4000:
                return
        except (ValueError, IndexError):
            return
