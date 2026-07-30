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


def similarity(path_a: str, path_b: str) -> float:
    """Cosine similarity between two stills — the repetition detector."""
    e = embed_images([path_a, path_b])
    return round(float(e[0:1] @ e[1:2].T), 4)


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
