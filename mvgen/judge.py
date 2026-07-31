"""Judge: score candidate stills so the pipeline can pick its own best frame.

Runs SigLIP on CPU, in-process. That is deliberate: everything painful in this
pipeline has come from two processes wanting the GPU at once, and an 800MB
vision tower scoring ~40 images does not need the card. Nothing here can
starve the renderer.

Three signals, each aimed at a failure actually observed rather than at a
general notion of quality:

  adherence  image vs the shot's own description. Catches a still that ignored
             its material — the SEM frame that came back as a tinted photo.
  margin     adherence minus similarity to explicit anti-patterns. Catches the
             ceramic bowls that appeared when an abstract sequence used a
             family whose substrate implies an object, and photoreal drift.
  novelty    distance from the previous shot's embedding. Catches the six
             near-identical spiral galaxies that no amount of prompt wording
             would fix — the one defect prompting could not reach.

A still is scored, not judged pass/fail: best-of-N picks a winner rather than
rejecting everything and stalling.
"""
import pathlib
from functools import lru_cache

MODEL = "google/siglip-base-patch16-384"

ANTI_PATTERNS = [
    "an ordinary photorealistic colour photograph",
    "a single object photographed on a plain studio background",
    "a ceramic bowl or vase on a table",
    "a blurry featureless grey image",
]


def _norm(feats):
    """L2-normalise, tolerating either a tensor or a model-output wrapper.

    transformers 5.x returns BaseModelOutputWithPooling from get_*_features
    where 4.x returned a bare tensor.
    """
    for attr in ("pooler_output", "last_hidden_state"):
        if hasattr(feats, attr):
            feats = getattr(feats, attr)
            break
    if feats.ndim == 3:  # unpooled sequence output — mean over tokens
        feats = feats.mean(dim=1)
    return feats / feats.norm(dim=-1, keepdim=True)


@lru_cache(maxsize=1)
def _model():
    import torch
    from transformers import AutoModel, AutoProcessor

    torch.set_num_threads(max(1, (torch.get_num_threads() or 4) // 2))
    model = AutoModel.from_pretrained(MODEL).eval()
    proc = AutoProcessor.from_pretrained(MODEL)
    return model, proc


def embed_images(paths: list[str]):
    import torch
    from PIL import Image

    model, proc = _model()
    imgs = [Image.open(p).convert("RGB") for p in paths]
    with torch.no_grad():
        inputs = proc(images=imgs, return_tensors="pt")
        feats = model.get_image_features(**inputs)
    return _norm(feats)


def embed_texts(texts: list[str]):
    import torch

    model, proc = _model()
    with torch.no_grad():
        inputs = proc(text=texts, padding="max_length", truncation=True,
                      max_length=64, return_tensors="pt")
        feats = model.get_text_features(**inputs)
    return _norm(feats)


def score_candidates(paths: list[str], description: str,
                     previous=None, w_margin: float = 1.0,
                     w_novelty: float = 0.5) -> list[dict]:
    """Score each candidate; higher total is better.

    `description` should be the shot's intent (material + subject), not the
    whole prompt — SigLIP truncates, and the lead clauses carry the signal.
    `previous` is the winning embedding of the preceding shot, or None.
    """
    img = embed_images(paths)
    txt = embed_texts([description] + ANTI_PATTERNS)
    want, avoid = txt[0:1], txt[1:]

    adherence = (img @ want.T).squeeze(-1)
    anti = (img @ avoid.T).max(dim=-1).values
    margin = adherence - anti

    out = []
    for i, p in enumerate(paths):
        novelty = 1.0
        if previous is not None:
            novelty = float(1.0 - (img[i:i + 1] @ previous.T).max())
        total = (w_margin * float(margin[i]) + w_novelty * novelty)
        out.append({
            "path": p,
            "adherence": round(float(adherence[i]), 4),
            "anti": round(float(anti[i]), 4),
            "margin": round(float(margin[i]), 4),
            "novelty": round(novelty, 4),
            "score": round(total, 4),
            "_embedding": img[i:i + 1],
        })
    out.sort(key=lambda d: -d["score"])
    return out


def rank_with_dedup(paths: list[str], description: str, picked=None,
                    sim_max: float = 0.95, w_margin: float = 1.0,
                    w_novelty: float = 0.5) -> list[dict]:
    """Rank candidates and flag any that duplicate an already-picked still.

    `picked` is a stacked tensor of every winning embedding so far, not just
    the previous shot's: the same image can resurface many shots later from a
    different seed, and a previous-shot-only check misses that entirely.

    Duplicates are sorted last rather than dropped, so a shot where every
    candidate is a near-duplicate still yields something (the least similar)
    instead of stalling the build.
    """
    import torch

    img = embed_images(paths)
    txt = embed_texts([description] + ANTI_PATTERNS)
    want, avoid = txt[0:1], txt[1:]

    adherence = (img @ want.T).squeeze(-1)
    anti = (img @ avoid.T).max(dim=-1).values
    margin = adherence - anti

    out = []
    for i, p in enumerate(paths):
        dup = 0.0
        if picked is not None and len(picked):
            dup = float((img[i:i + 1] @ picked.T).max())
        novelty = 1.0 - dup
        out.append({
            "path": p,
            "adherence": round(float(adherence[i]), 4),
            "margin": round(float(margin[i]), 4),
            "dup_sim": round(dup, 4),
            "duplicate": dup > sim_max,
            "score": round(w_margin * float(margin[i]) + w_novelty * novelty, 4),
            "embedding": img[i:i + 1],
        })
    # non-duplicates first, then by score within each group
    out.sort(key=lambda d: (d["duplicate"], -d["score"]))
    return out


def stack(embeddings: list):
    """Stack a list of (1, D) embeddings into one (N, D) tensor, or None."""
    import torch

    return torch.cat(embeddings, dim=0) if embeddings else None


def similarity(path_a: str, path_b: str) -> float:
    """Cosine similarity between two stills — the repetition detector."""
    e = embed_images([path_a, path_b])
    return round(float(e[0:1] @ e[1:2].T), 4)


# --- video judging (roadmap item 1) ---------------------------------------
# Every quality mechanism above stops at the still. These score the rendered
# clip, so a shot that freezes, dissolves, or abandons its material can be
# caught instead of silently shipping.

def _sample_frames(video: str, out_dir) -> list[str]:
    """Pull first / middle / last frames out of a clip.

    Seeks by time rather than frame-selecting: ffmpeg's select filter has no
    total-frame variable, so expressions like eq(n,n-1) never match and yield
    nothing. -sseof is the reliable way to reach the final frame.
    """
    import subprocess
    out_dir = pathlib.Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = pathlib.Path(video).stem

    r = subprocess.run(["ffprobe", "-v", "error", "-show_entries",
                        "format=duration", "-of", "csv=p=0", str(video)],
                       capture_output=True, text=True)
    try:
        dur = float(r.stdout.strip())
    except ValueError:
        return []

    seeks = [("a", ["-ss", "0"]), ("b", ["-ss", f"{dur/2:.3f}"]),
             ("c", ["-sseof", "-0.2"])]
    paths = []
    for tag, seek in seeks:
        pth = out_dir / f"{stem}-{tag}.png"
        subprocess.run(["ffmpeg", "-y", "-v", "error", *seek, "-i", str(video),
                        "-frames:v", "1", str(pth)], capture_output=True)
        if pth.exists() and pth.stat().st_size > 0:
            paths.append(str(pth))
    return paths


def motion_of(video: str) -> float:
    """Mean frame-to-frame difference — how much actually moves."""
    import subprocess

    import numpy as np
    r = subprocess.run(["ffmpeg", "-v", "error", "-i", str(video),
                        "-vf", "select=not(mod(n\\,4)),scale=160:88,format=gray",
                        "-f", "rawvideo", "-"], capture_output=True)
    a = np.frombuffer(r.stdout, dtype=np.uint8).astype(np.float32)
    n = len(a) // (160 * 88)
    if n < 2:
        return 0.0
    a = a[:n * 160 * 88].reshape(n, -1)
    return float(np.abs(np.diff(a, axis=0)).mean())


def judge_video(video: str, still: str, description: str, tmp_dir,
                end_ref: str | None = None,
                motion_floor: float = 0.12, material_floor: float = 0.70) -> dict:
    """Score a rendered shot. Returns reasons it failed, empty list if fine.

    - froze:    almost nothing moved; the clip is effectively a still.
    - lost:     the final frame no longer resembles where it should have
                ended. `end_ref` is that target: normally the shot's own
                still, but under tweening the NEXT shot's still, because a
                tweened shot is *supposed* to land on the following frame.
                Measured on real output: tweened shots score ~0.6 against
                their own still and ~0.98 against the next one, so checking
                the wrong reference flags healthy shots as failures.
    There is deliberately no separate "dissolved into mush" check. The obvious
    one — final-frame adherence relative to the still's — divides two raw
    SigLIP similarities, which are small and sometimes negative, so the ratio
    goes meaningless (measured values of -1.01 and -0.15 on healthy shots).
    Dissolution is already caught by the landing check: mush does not resemble
    the target frame either.

    Thresholds are deliberately loose. A false reroll costs ~90s of GPU, so
    this should only fire on genuine failure, not on mild drift.
    """
    target = end_ref or still
    frames = _sample_frames(video, tmp_dir)
    if len(frames) < 3:
        # Do not report a pass we did not actually establish — a broken judge
        # that silently approves everything is worse than no judge at all.
        return {"ok": True, "unjudged": True,
                "reasons": [], "note": "could not sample frames"}

    motion = motion_of(video)
    emb = embed_images([still, target] + frames)
    src, tgt, f_last = emb[0:1], emb[1:2], emb[4:5]
    material = float(f_last @ tgt.T)

    reasons = []
    if motion < motion_floor:
        reasons.append(f"froze (motion {motion:.3f} < {motion_floor})")
    if material < material_floor:
        reasons.append(f"lost material ({material:.3f} < {material_floor})")
    return {"ok": not reasons, "reasons": reasons, "motion": round(motion, 4),
            "material": round(material, 4)}


def main():
    import sys
    args = sys.argv[1:]
    if len(args) >= 2 and args[0] == "sim":
        for a, b in zip(args[1:-1], args[2:]):
            print(f"{similarity(a, b):.4f}  {pathlib.Path(a).name} vs {pathlib.Path(b).name}")
    elif len(args) >= 2:
        desc, paths = args[0], args[1:]
        for r in score_candidates(paths, desc):
            print(f"{r['score']:+.4f}  adh {r['adherence']:.3f}  anti {r['anti']:.3f}  "
                  f"{pathlib.Path(r['path']).name}")
    else:
        sys.exit("usage: judge.py <description> <image>...   |   judge.py sim <image>...")


if __name__ == "__main__":
    main()
