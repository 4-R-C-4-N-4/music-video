"""Render a shot manifest against a local ComfyUI: Z-Image stills, LTX-2.3 i2v shots.

Idempotent: finished outputs are recorded in <workdir>/state.json and skipped
on re-run, so a killed render resumes and editing one shot's prompt/seed in
the manifest re-renders only that shot (delete its state entry or use --redo).
"""
import json
import pathlib
import shutil
import sys
import time
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


def run_graph(graph, label, timeout=1800):
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


def i2v_graph(image_name, prompt, seed, w, h, frames, fps, prefix):
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
        "8": {"class_type": "LTXVEmptyLatentAudio", "inputs": {"frames_number": frames, "frame_rate": fps, "batch_size": 1, "audio_vae": ["6", 0]}},
        "21": {"class_type": "LoadImage", "inputs": {"image": image_name}},
        "22": {"class_type": "LTXVPreprocess", "inputs": {"image": ["21", 0], "img_compression": 33}},
        "23": {"class_type": "LTXVImgToVideoInplace", "inputs": {
            "vae": ["5", 0], "image": ["22", 0], "latent": ["7", 0], "strength": 1.0, "bypass": False}},
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
    }


def grab_output(outputs, node, key, workdir, dest_name):
    info = outputs[node][key][0]
    sub = info.get("subfolder", "")
    src = COMFY_OUT / sub / info["filename"]
    dest = workdir / dest_name
    shutil.copy(src, dest)
    return dest


def render(manifest_path: str, workdir: str, limit: int | None = None,
           stills_only: bool = False):
    manifest = json.load(open(manifest_path))
    workdir = pathlib.Path(workdir)
    workdir.mkdir(parents=True, exist_ok=True)
    state_path = workdir / "state.json"
    state = json.load(open(state_path)) if state_path.exists() else {"stills": {}, "shots": {}}

    def save_state():
        json.dump(state, open(state_path, "w"), indent=1)

    w, h, fps = manifest["width"], manifest["height"], manifest["fps"]
    job = pathlib.Path(manifest_path).parent.name

    todo = [s for s in manifest["shots"] if str(s["idx"]) not in state["shots"]]
    if limit:
        todo = todo[:limit]
    for shot in todo:
        i = shot["idx"]
        key = str(i)
        if key not in state["stills"]:
            print(f"still {i+1}/{len(manifest['shots'])} (scene {shot['scene']})", flush=True)
            out = run_graph(still_graph(shot["still_prompt"], shot["still_seed"], w, h,
                                        f"mv-{job}-s{i:03d}"), f"still:{i}")
            dest = grab_output(out, "10", "images", workdir, f"still-{i:03d}.png")
            shutil.copy(dest, COMFY_IN / dest.name)
            state["stills"][key] = dest.name
            save_state()
        if stills_only:
            continue
        print(f"shot {i+1}/{len(manifest['shots'])} "
              f"(scene {shot['scene']}, {shot['frames']}f, {shot['level']})", flush=True)
        g = i2v_graph(state["stills"][key], shot["video_prompt"], shot["seed"],
                      w, h, shot["frames"], fps, f"mv-{job}-{i:03d}")
        out = run_graph(g, f"shot:{i}")
        dest = grab_output(out, "20", "images", workdir, f"shot-{i:03d}.mp4")
        state["shots"][str(i)] = dest.name
        save_state()
    done = len(state["shots"])
    print(f"rendered: {done}/{len(manifest['shots'])} shots complete", flush=True)


def main():
    args = [a for a in sys.argv[1:] if a != "--stills-only"]
    manifest_path, workdir = args[0], args[1]
    limit = int(args[2]) if len(args) > 2 else None
    render(manifest_path, workdir, limit, stills_only="--stills-only" in sys.argv)


if __name__ == "__main__":
    main()
