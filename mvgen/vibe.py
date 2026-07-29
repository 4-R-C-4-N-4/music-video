"""Vibe analysis: what does this song feel like, and which materials suit it?

Two independent readings, deliberately kept separate:

  acoustic   librosa features that correlate with mood regardless of words —
             mode (major/minor) for valence, tempo and loudness for energy,
             spectral centroid for warmth. Works on instrumentals.
  semantic   a local model reads the title and transcript and rates the same
             axes, plus the ones audio can't give: how antique the song
             feels, how tactile it wants to look, how alive it is.

They are blended (audio wins on what audio measures) into a seven-axis
vector — valence, energy, warmth, age, tactility, scale, organic — and
material families are ranked by distance to it. So a bleak, slow, cold song
surfaces daguerreotype and cyanotype; a vast one surfaces cosmic or lithic;
a teeming one surfaces mycology or microbial — rather than picking blind.

The transcript is only ever sent to the local model. Nothing from it is
returned or stored here: the output is adjectives and numbers.
"""
import json
import re
import sys

AXES = ("valence", "energy", "warmth", "age", "tactility", "scale", "organic")

# Krumhansl-Kessler key profiles, used only to decide major vs minor.
MAJOR = [6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52, 5.19, 2.39, 3.66, 2.29, 2.88]
MINOR = [6.33, 2.68, 3.52, 5.38, 2.60, 3.53, 2.54, 4.75, 3.98, 2.69, 3.34, 3.17]

SEMANTIC_SYSTEM = """You rate the emotional character of songs. Reply with strict \
JSON only — no markdown, no commentary. Never quote or reproduce lyrics."""

SEMANTIC_USER = """Rate this song on seven axes from 0.0 to 1.0.

title: {title}
{lyrics}

Axes:
  valence    0 = bleak, desolate, grieving; 1 = joyful, radiant, elated
  energy     0 = still, sparse, suspended; 1 = frantic, pounding, overloaded
  warmth     0 = cold, metallic, clinical; 1 = warm, soft, human
  age        0 = contemporary, synthetic, digital; 1 = antique, weathered, historical
  tactility  0 = flat, graphic, screen-like; 1 = dimensional, handmade, physical
  scale      0 = intimate, close, small; 1 = vast, cosmic, immense
  organic    0 = inert, geometric, mineral, synthetic; 1 = living, growing, biological

Return:
{{"valence":0.0,"energy":0.0,"warmth":0.0,"age":0.0,"tactility":0.0,"scale":0.0,"organic":0.0,
 "adjectives":["<6 mood words, your own, not from the lyrics>"],
 "imagery":["<4 concrete physical nouns the song evokes, your own words>"]}}"""


def acoustic(track: str) -> dict:
    """Mood axes measurable from the audio alone."""
    import librosa
    import numpy as np

    y, sr = librosa.load(track, sr=22050, mono=True, duration=180)

    chroma = librosa.feature.chroma_cqt(y=y, sr=sr).mean(axis=1)
    best_maj = max(np.corrcoef(np.roll(MAJOR, i), chroma)[0, 1] for i in range(12))
    best_min = max(np.corrcoef(np.roll(MINOR, i), chroma)[0, 1] for i in range(12))
    majorness = float(np.clip(0.5 + (best_maj - best_min) * 2.0, 0, 1))

    tempo = float(np.atleast_1d(librosa.beat.beat_track(y=y, sr=sr)[0])[0])
    rms = float(librosa.feature.rms(y=y).mean())
    onsets = librosa.onset.onset_detect(y=y, sr=sr, units="time")
    density = len(onsets) / max(len(y) / sr, 1)

    tempo_n = float(np.clip((tempo - 60) / 120, 0, 1))
    loud_n = float(np.clip(rms / 0.15, 0, 1))
    dens_n = float(np.clip(density / 5.0, 0, 1))
    energy = float(np.clip(0.45 * tempo_n + 0.3 * loud_n + 0.25 * dens_n, 0, 1))

    centroid = float(librosa.feature.spectral_centroid(y=y, sr=sr).mean())
    warmth = float(np.clip(1.0 - (centroid - 800) / 3200, 0, 1))

    # Spaciousness proxy for scale: a wide spectrum with sparse events and
    # slow amplitude decay reads as vast; dense, narrow and dry reads close.
    bandwidth = float(librosa.feature.spectral_bandwidth(y=y, sr=sr).mean())
    env = librosa.onset.onset_strength(y=y, sr=sr)
    sustain = float(np.mean(env < np.percentile(env, 60)))  # fraction of quiet frames
    scale_n = float(np.clip(0.5 * (bandwidth / 3000) + 0.3 * sustain
                            + 0.2 * (1 - dens_n), 0, 1))

    # Brightness nudges valence, but mode dominates.
    valence = float(np.clip(0.7 * majorness + 0.3 * (1 - warmth) * 0.6 + 0.1, 0, 1))

    return {"valence": round(valence, 3), "energy": round(energy, 3),
            "warmth": round(warmth, 3), "scale": round(scale_n, 3),
            "tempo": round(tempo, 1), "majorness": round(majorness, 3)}


def semantic(title: str, lyrics: dict | None, model: str = "gemma4-nothink") -> dict:
    """Ask the local model how the song feels. Returns ratings, never text."""
    from .direct import ask, llm_down, llm_up
    from .render import comfy_down

    if lyrics and lyrics.get("has_sung_lyrics"):
        text = " ".join(s["text"] for s in lyrics["segments"])
        block = ("transcript (for judgement only, never reproduce it):\n"
                 f'"""\n{text}\n"""')
    else:
        block = "This track is instrumental — judge from the title alone."

    prompt = SEMANTIC_USER.format(title=title, lyrics=block)
    comfy_down()
    llm_up(model)
    try:
        raw = ask(prompt, system=SEMANTIC_SYSTEM)
    finally:
        llm_down()

    out = {a: float(raw.get(a, 0.5)) for a in AXES}
    out["adjectives"] = [str(w) for w in raw.get("adjectives", [])][:6]
    out["imagery"] = [str(w) for w in raw.get("imagery", [])][:4]
    return out


def blend(ac: dict, sem: dict) -> dict:
    """Audio wins where audio can measure; the model supplies the rest."""
    v = {
        "valence": 0.6 * ac["valence"] + 0.4 * sem["valence"],
        "energy": 0.7 * ac["energy"] + 0.3 * sem["energy"],
        "warmth": 0.6 * ac["warmth"] + 0.4 * sem["warmth"],
        "age": sem["age"],
        "tactility": sem["tactility"],
        # Audio measures spaciousness; the model judges aliveness.
        "scale": 0.6 * ac["scale"] + 0.4 * sem["scale"],
        "organic": sem["organic"],
    }
    return {k: round(float(x), 3) for k, x in v.items()}


def tension(ac: dict, sem: dict) -> float:
    """How far the music's mood sits from the words' mood.

    Bright music carrying bleak words (or the reverse) is a real and common
    songwriting move, and averaging the two readings erases exactly the thing
    that makes such a song interesting. Kept as its own signal so the
    director can be told to stage the contradiction rather than resolve it.
    """
    return round(float(abs(ac["valence"] - sem["valence"])), 3)


def rank_families(vec: dict, top: int = 8) -> list[tuple[str, float]]:
    """Rank material families by distance to the song's vibe vector."""
    from .materials import FAMILIES

    scored = []
    for key, fam in FAMILIES.items():
        aff = fam["affinity"]
        d = sum((aff[a] - vec[a]) ** 2 for a in AXES) ** 0.5
        scored.append((key, round(d, 3)))
    scored.sort(key=lambda t: t[1])
    return scored[:top]


def analyse(track: str, title: str, lyrics: dict | None,
            model: str = "gemma4-nothink") -> dict:
    ac = acoustic(track)
    try:
        sem = semantic(title, lyrics, model)
    except Exception as e:
        print(f"  semantic read failed ({e}); using audio only", flush=True)
        sem = {a: 0.5 for a in AXES} | {"adjectives": [], "imagery": []}
    vec = blend(ac, sem)
    return {"acoustic": ac, "semantic": sem, "vector": vec,
            "tension": tension(ac, sem),
            "families": rank_families(vec)}


def main():
    import pathlib
    job = pathlib.Path(sys.argv[1])
    track = sys.argv[2]
    lyr = job / "lyrics.json"
    lyrics = json.load(open(lyr)) if lyr.exists() else None
    title = re.sub(r"[-_]+", " ", job.name).title()
    res = analyse(track, title, lyrics)
    json.dump(res, open(job / "vibe.json", "w"), indent=1)
    v = res["vector"]
    print(f"tempo {res['acoustic']['tempo']:.0f}, "
          f"{'major' if res['acoustic']['majorness'] > 0.5 else 'minor'}-leaning")
    print("vector: " + "  ".join(f"{a}={v[a]:.2f}" for a in AXES))
    if res["semantic"]["adjectives"]:
        print("mood:   " + ", ".join(res["semantic"]["adjectives"]))
    t = res["tension"]
    print(f"tension: {t:.2f}" + ("  (music and words disagree — bittersweet)"
                                 if t >= 0.35 else ""))
    print("materials by fit: " + ", ".join(f"{k}({d})" for k, d in res["families"]))


if __name__ == "__main__":
    main()
