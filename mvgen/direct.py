"""Automatic art direction: track analysis (+ lyrics) -> scenes.json.

Removes the human from the creative loop. A local llama.cpp model reads the
song's structure and its transcript, then returns scene settings, per-shot
blocking beats, and motion language. Materials are chosen by key from the
validated palette in materials.py, so generated specs can only use looks
that are known to render and survive i2v.

GPU note: the LLM and ComfyUI cannot share the card, so this step starts the
model, generates, and stops it again before rendering begins.

Usage: python -m mvgen.direct jobs/<job> [--model gemma4-nothink]
"""
import json
import pathlib
import re
import subprocess
import sys
import time
import urllib.error
import urllib.request

from .materials import BY_LEVEL, MATERIALS

ENDPOINT = "http://127.0.0.1:8080/v1/chat/completions"

SYSTEM = """You are an art director for music videos. You answer only with strict JSON \
matching the requested schema. No markdown fences, no commentary.

Hard rules:
- Never quote, paraphrase, or transcribe any lyric line. The transcript is \
context for mood and theme only. Your output describes places and actions.
- Describe concrete, filmable places. No abstractions ("a sense of longing"), \
no text or writing in frame, no logos.
- Never name a real person, artist, band, or trademark.
- Each scene must be a genuinely different location, not a variation."""

USER = """Design the visual plan for a music video.

TRACK STRUCTURE
duration: {duration:.0f}s, tempo: {tempo:.0f} BPM
sections (in order, one scene each):
{sections}

MOOD CONTEXT
{mood}

MATERIAL PALETTE (choose one key per scene; each scene must differ)
Each entry gives the material's feel, then what it can plausibly depict.
{palette}

COHERENCE RULE — this matters more than any other choice:
The setting and the material must belong together. The material is not a
filter laid over an arbitrary place; it either IS the place, or it is the
instrument the place is seen through. A microscope cannot image a harbour;
a nebula is not a room. If you pick a material with a scale restriction,
the setting must live at that scale. If you cannot write a setting that
honestly suits the material, choose a different material.

Return JSON:
{{
 "concept": "<one sentence describing the through-line, no lyrics>",
 "figure": "<one recurring figure appearing in every scene: age, clothing, no name>",
 "scenes": [
   {{
    "id": "<short-kebab-slug>",
    "material": "<key from the palette>",
    "setting": "<one sentence: a concrete filmable location with specific physical detail, no camera directions, no figure>",
    "beats": ["<5 or 6 distinct shots in this location: each names a camera setup AND what the figure does; they must progress through the location>"],
    "motion": {{"low": "<slow camera movement sentence>", "mid": "<walking-pace camera movement sentence>", "high": "<fast camera movement sentence>"}},
    "ambient": "<one sentence: what physically moves in this location, continuously>"
   }}
 ]
}}

Exactly {n} scenes, in order, matching the sections listed above."""


def llm_up(model: str, timeout: int = 240) -> None:
    subprocess.run(["llm", model], check=True, capture_output=True, timeout=timeout)


def llm_down() -> None:
    subprocess.run(["llm", "stop"], check=False, capture_output=True)


def ask(prompt: str, system: str = SYSTEM, retries: int = 3) -> dict:
    body = {
        "messages": [{"role": "system", "content": system},
                     {"role": "user", "content": prompt}],
        "temperature": 0.8,
        "max_tokens": 4096,
    }
    last = None
    for attempt in range(retries):
        req = urllib.request.Request(
            ENDPOINT, data=json.dumps(body).encode(),
            headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=600) as r:
            text = json.loads(r.read())["choices"][0]["message"]["content"]
        raw = re.sub(r"^```(?:json)?|```$", "", text.strip(), flags=re.M).strip()
        try:
            return json.loads(raw)
        except json.JSONDecodeError as e:
            last = e
            m = re.search(r"\{.*\}", raw, re.S)
            if m:
                try:
                    return json.loads(m.group(0))
                except json.JSONDecodeError:
                    pass
            print(f"  bad json (attempt {attempt+1}/{retries}), retrying", flush=True)
    raise RuntimeError(f"model never returned valid json: {last}")


def lyric_ngrams(lyrics: dict | None, n: int = 5) -> set[str]:
    """Word n-grams from the transcript, for leak detection."""
    if not lyrics:
        return set()
    segs = lyrics.get("segments") or lyrics.get("rejected_segments") or []
    words = re.findall(r"[a-z']+", " ".join(s["text"] for s in segs).lower())
    return {" ".join(words[i:i + n]) for i in range(len(words) - n + 1)}


def leaks(text: str, grams: set[str], n: int = 5) -> bool:
    words = re.findall(r"[a-z']+", text.lower())
    return any(" ".join(words[i:i + n]) in grams
               for i in range(len(words) - n + 1))


def mood_context(lyrics: dict | None, vibe: dict | None = None) -> str:
    """Mood context for the director: the vibe read, plus the transcript."""
    parts = []
    if vibe:
        v = vibe["vector"]
        parts.append("Measured character (0-1): " + ", ".join(
            f"{a} {v[a]:.2f}" for a in ("valence", "energy", "warmth", "age", "tactility")))
        sem = vibe.get("semantic", {})
        if sem.get("adjectives"):
            parts.append("Mood: " + ", ".join(sem["adjectives"]))
        if sem.get("imagery"):
            parts.append("Evoked objects: " + ", ".join(sem["imagery"]))
        if vibe.get("tension", 0) >= 0.35:
            parts.append(
                "IMPORTANT: the music and the words disagree — the sound is "
                "brighter than the sentiment (or the reverse). Stage that "
                "contradiction rather than resolving it: settings that are "
                "beautiful and wrong at once, warmth in a bleak place.")
    if not lyrics or not lyrics.get("has_sung_lyrics"):
        parts.append("This track is instrumental — there are no lyrics.")
    else:
        text = " ".join(s["text"] for s in lyrics["segments"])
        parts.append("Transcript of the sung vocal, for mood and theme only — "
                     f"do not reuse its wording:\n\"\"\"\n{text}\n\"\"\"")
    return "\n".join(parts)


def build_spec(plan_json: dict, analysis: dict, lyrics: dict | None,
               title: str, seed: int = 3000) -> dict:
    levels = [s["level"] for s in analysis["sections"]]
    scenes = []
    for i, sc in enumerate(plan_json["scenes"]):
        level = levels[i] if i < len(levels) else "mid"
        key = sc.get("material")
        if key not in MATERIALS:
            key = BY_LEVEL[level][i % len(BY_LEVEL[level])]
        scenes.append({
            "id": sc.get("id") or f"scene-{i+1}",
            "material": key,
            # The treatment is composed at plan time from this family plus
            # the seed, so two videos using the same family still differ.
            "material_seed": seed + 4400 + i,
            "still_prompt": sc["setting"],
            "beats": sc["beats"],
            "video_prompt": sc.get("ambient", "") + " The material texture of the image stays constant and visible throughout.",
            "motion": sc["motion"],
        })
    return {
        "title": title,
        "concept": plan_json.get("concept", ""),
        "fps": 25,
        "width": 1280,
        "height": 704,
        "seed": seed,
        "bars_per_shot": {"low": 4, "mid": 2, "high": 2},
        "style": f"A recurring figure appears throughout: {plan_json.get('figure','a lone figure')}.",
        "scenes": scenes,
    }


def direct(jobdir: str, model: str = "gemma4-nothink", keep_llm: bool = False) -> dict:
    job = pathlib.Path(jobdir)
    analysis = json.load(open(job / "analysis.json"))
    lyr_path = job / "lyrics.json"
    lyrics = json.load(open(lyr_path)) if lyr_path.exists() else None

    secs = analysis["sections"]
    bars = analysis["bars"]
    lines = []
    for i, s in enumerate(secs):
        t0 = bars[s["start_bar"]]["t0"]
        t1 = bars[s["end_bar"]]["t0"] if s["end_bar"] < len(bars) else analysis["duration"]
        lines.append(f"  {i+1}. {t0:.0f}s-{t1:.0f}s ({t1-t0:.0f}s), energy: {s['level']}")

    # Constrain the palette to families that actually fit this song's vibe,
    # rather than offering all 20 blind. The ranking is the whole point of
    # the vibe read: material choice becomes rooted in the track.
    vibe_path = job / "vibe.json"
    vibe = json.load(open(vibe_path)) if vibe_path.exists() else None
    if vibe:
        keys = [k for k, _ in vibe["families"]]
        print("  vibe-matched palette: " + ", ".join(keys), flush=True)
    else:
        keys = list(MATERIALS)
    palette = "\n".join(
        f"  {k}: {MATERIALS[k]['mood']}\n      suits: {MATERIALS[k]['suits']}"
        for k in keys)

    prompt = USER.format(duration=analysis["duration"], tempo=analysis["tempo"],
                         sections="\n".join(lines),
                         mood=mood_context(lyrics, vibe),
                         palette=palette, n=len(secs))

    print(f"starting {model}...", flush=True)
    llm_up(model)
    try:
        grams = lyric_ngrams(lyrics)
        for attempt in range(3):
            plan_json = ask(prompt)
            spec = build_spec(plan_json, analysis, lyrics,
                              title=job.name.replace("-", " ").title())
            blob = json.dumps(spec)
            if not leaks(blob, grams):
                break
            print("  output echoed the transcript; regenerating", flush=True)
        else:
            raise RuntimeError("director kept reusing lyric wording")
    finally:
        if not keep_llm:
            llm_down()

    out = job / "scenes.json"
    json.dump(spec, open(out, "w"), indent=2)
    return spec


def main():
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    model = next((a.split("=", 1)[1] for a in sys.argv if a.startswith("--model=")),
                 "gemma4-nothink")
    spec = direct(args[0], model)
    print(f"\nconcept: {spec['concept']}")
    print(f"figure:  {spec['style']}")
    for s in spec["scenes"]:
        print(f"  {s['id']:20s} [{s['material']}] {len(s['beats'])} beats")
        print(f"      {s['still_prompt'][:100]}")


if __name__ == "__main__":
    main()
