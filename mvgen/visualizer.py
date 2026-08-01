"""Visualizer mode: abstract phenomena cut to the rhythm, no figure, no place.

The narrative mode gives each song a character walking through locations. That
is the wrong shape for a lot of electronic music, where there is no story to
tell and a one-bar cut lands harder than a scene transition. Here each "scene"
is a *phenomenon* rather than a place — a field of matter doing something —
and the plan cuts on the bar.

Two consequences worth knowing:
  - phrase-end snapping is switched off, because pulling cuts onto vocal
    phrases fights the rhythmic grid this mode exists to sit on;
  - only families from materials.ABSTRACT are offered, since an abstract cut
    sequence has no room to establish a figure or a location.
"""
import json
import pathlib
import re
import sys

from .materials import ABSTRACT, MATERIALS

SYSTEM = """You design abstract music visualisers. You answer only with strict \
JSON matching the requested schema. No markdown, no commentary.

Hard rules:
- NO people, figures, characters or animals. NO buildings, rooms, landscapes or
  any recognisable place. NO text or writing.
- Every scene is a PHENOMENON: matter doing something, filling the frame edge
  to edge. It is NOT an object photographed against a background — no vessels,
  bowls, panels, plates, sheets or any thing with an outline in the frame. The
  camera is inside the material, not looking at it.
- Never quote or reproduce lyrics. Transcript context is for mood only.
- Each scene must be a genuinely different phenomenon, not a variation.
- Use a DIFFERENT material for every scene. Do not group several scenes under
  one material — varying the phenomenon while repeating the material still
  reads as dwelling on one look."""

USER = """Design an abstract visualiser for a track.

TRACK
duration: {duration:.0f}s, tempo: {tempo:.0f} BPM
sections (in order, one scene each):
{sections}

MOOD CONTEXT
{mood}

MATERIAL PALETTE (choose one key per scene; each scene must differ)
Each entry gives the material's feel, then what it can plausibly be.
{palette}

COHERENCE RULE: several of these are instruments with a fixed scale — an
electron micrograph resolves microns, a spectrogram shows sound, a schlieren
image shows air. The phenomenon must be something that instrument could
actually be pointed at. If you cannot write one honestly, pick another
material.

Return JSON:
{{
 "concept": "<one sentence describing the visual through-line, no lyrics>",
 "scenes": [
   {{
    "id": "<short-kebab-slug>",
    "material": "<key from the palette>",
    "phenomenon": "<one sentence: matter filling the frame, doing something. No figure, no place, no horizon. e.g. 'a dense field of black spikes rising and collapsing in ranks'>",
    "beats": ["<5 or 6 framings of this same phenomenon: each names how close the camera is and what the matter is doing. They must escalate.>"],
    "motion": {{"low": "<slow movement sentence>", "mid": "<moderate movement sentence>", "high": "<violent movement sentence>"}},
    "ambient": "<one sentence: what moves continuously, with no cause shown>"
   }}
 ]
}}

Exactly {n} scenes. There may be more scenes than sections — that is intended,
the extra phenomena keep an abstract sequence from visibly cycling."""


def build_spec(plan_json: dict, analysis: dict, title: str, seed: int = 5000,
               ranked: list[str] | None = None) -> dict:
    levels = [s["level"] for s in analysis["sections"]]
    # Draw the palette down rather than letting it be resampled. Measured over
    # every job since the palette expanded: given a free choice the model picks
    # ferrofluid 2.75 times per offer and sumi 0.14, and it groups scenes under
    # one material (one build came back as 18 phenomena across only 6 materials,
    # in blocks of three). Enforcing one material per scene guarantees the
    # spread instead of asking the model to override its own taste.
    pool = [k for k in (ranked or ABSTRACT) if k in ABSTRACT] or list(ABSTRACT)
    used: set[str] = set()

    def take(preferred: str | None, i: int) -> str:
        if preferred in ABSTRACT and preferred in MATERIALS and preferred not in used:
            return preferred
        for k in pool:                      # next best fit not yet spent
            if k not in used:
                return k
        return (preferred if preferred in MATERIALS
                else ABSTRACT[i % len(ABSTRACT)])   # pool exhausted

    scenes = []
    for i, sc in enumerate(plan_json["scenes"]):
        level = levels[i] if i < len(levels) else levels[i % len(levels)]
        key = take(sc.get("material"), i)
        used.add(key)
        scenes.append({
            "id": sc.get("id") or f"scene-{i+1}",
            "material": key,
            "material_seed": seed + 4400 + i,
            "still_prompt": sc.get("phenomenon", ""),
            "beats": sc["beats"],
            "video_prompt": sc.get("ambient", "")
                            + " The material texture stays constant and visible throughout.",
            "motion": sc["motion"],
        })
    return {
        "title": title,
        "mode": "visualizer",
        "concept": plan_json.get("concept", ""),
        "fps": 25,
        "width": 1280,
        "height": 704,
        "seed": seed,
        # No figure clause at all — that is the point of this mode.
        "style": ("No people, no figures, no buildings, no landscape, no horizon. "
                  "No object with a visible outline — no vessel, bowl, panel or plate. "
                  "The matter fills the frame edge to edge and continues past it."),
        "scenes": scenes,
    }


def direct(jobdir: str, model: str = "gemma4-nothink") -> dict:
    from .direct import ask, llm_down, llm_up, mood_context
    from .render import comfy_down

    job = pathlib.Path(jobdir)
    analysis = json.load(open(job / "analysis.json"))
    lyr = job / "lyrics.json"
    lyrics = json.load(open(lyr)) if lyr.exists() else None
    vib = job / "vibe.json"
    vibe = json.load(open(vib)) if vib.exists() else None

    secs, bars = analysis["sections"], analysis["bars"]
    lines = []
    for i, s in enumerate(secs):
        t0 = bars[s["start_bar"]]["t0"]
        t1 = (bars[s["end_bar"]]["t0"] if s["end_bar"] < len(bars)
              else analysis["duration"])
        lines.append(f"  {i+1}. {t0:.0f}s-{t1:.0f}s ({t1-t0:.0f}s), energy: {s['level']}")

    # Rank abstract families by vibe fit when we have a reading.
    keys = ABSTRACT
    if vibe:
        ranked = [k for k, _ in vibe["families"] if k in ABSTRACT]
        keys = ranked + [k for k in ABSTRACT if k not in ranked]
    bar_secs_est = 240.0 / max(analysis["tempo"], 1)
    est_shots_pre = max(1, int(analysis["duration"] / (bar_secs_est * 1.2)))
    keys = keys[:max(12, min(est_shots_pre // 2, 20))]
    palette = "\n".join(
        f"  {k}: {MATERIALS[k]['mood']}\n      suits: {MATERIALS[k]['suits']}"
        for k in keys)

    # A visualiser rotates material every shot, so the number of phenomena
    # governs how long before the eye sees a cycle — not the song's section
    # count. Ask for enough that the rotation never visibly repeats: roughly
    # one phenomenon per three shots, bounded by the palette on offer.
    bar_secs = 240.0 / max(analysis["tempo"], 1)
    est_shots = max(1, int(analysis["duration"] / (bar_secs * 1.2)))
    # One phenomenon per ~2 shots: enough that each gets a short run and the
    # sequence never revisits one, few enough that the model reliably returns
    # them all in a single well-formed response.
    n_scenes = max(len(secs), min(max(4, est_shots // 2), len(keys), 20))
    print(f"  {n_scenes} phenomena for ~{est_shots} shots "
          f"({len(secs)} sections)", flush=True)

    prompt = USER.format(duration=analysis["duration"], tempo=analysis["tempo"],
                         sections="\n".join(lines),
                         mood=mood_context(lyrics, vibe),
                         palette=palette, n=n_scenes)
    comfy_down()
    llm_up(model)
    try:
        plan_json = ask(prompt, system=SYSTEM)
    finally:
        llm_down()

    title = re.sub(r"[-_]+", " ", job.name).title()
    spec = build_spec(plan_json, analysis, title, ranked=keys)
    json.dump(spec, open(job / "scenes.json", "w"), indent=2)
    return spec


def main():
    spec = direct(sys.argv[1],
                  next((a.split("=", 1)[1] for a in sys.argv if a.startswith("--model=")),
                       "gemma4-nothink"))
    print("concept:", spec["concept"])
    for s in spec["scenes"]:
        print(f"  {s['id']:22s} [{s['material']}] {s['still_prompt'][:70]}")


if __name__ == "__main__":
    main()
