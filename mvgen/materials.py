"""Validated material treatments.

Each entry is a tested (lead, style) pair: Z-Image renders it convincingly
AND it survives LTX i2v with its material identity intact. The art director
picks keys from this table rather than inventing treatments, so generated
specs can only use looks that are known to work.

Rule learned the hard way: name the *physical process*. Asking for an
abstract optical effect ("formed from thin-film interference") makes the
model fall back to photorealism.
"""

MATERIALS = {
    "cyanotype": {
        "lead": "Cyanotype sun print on rough cotton rag paper:",
        "style": "Deep Prussian blue and raw paper white only, brush-stroked emulsion edges, fibrous paper grain visible throughout, high contrast with no midtones, hand-coated unevenness.",
        "mood": "cold, still, elegiac, archival",
    },
    "felted": {
        "lead": "Needle-felted wool diorama photographed in soft daylight:",
        "style": "Everything built from dyed carded wool fibers — fuzzy matted surfaces, visible needle-poke marks, loose fiber wisps catching the light, handmade craft object on a tabletop, shallow depth of field.",
        "mood": "warm, tender, handmade, childlike",
    },
    "etching": {
        "lead": "Copperplate etching, intaglio print on laid paper:",
        "style": "Dense cross-hatched line work, drypoint burr, plate tone and ink held in bitten lines, warm sepia-black ink on off-white laid paper, visible plate edge.",
        "mood": "austere, antique, melancholy, precise",
    },
    "collodion": {
        "lead": "Wet-plate collodion tintype photograph:",
        "style": "Silver halide grain, bromide drips and swirl artifacts at the plate edges, shallow tonal range, tarnished silver highlights, glass-plate imperfections, monochrome warm grey.",
        "mood": "haunted, historical, decayed, solemn",
    },
    "risograph": {
        "lead": "Risograph print, two ink layers only (fluorescent pink and teal):",
        "style": "Visible halftone dot screens, slight misregistration between layers, ink smudge, matte uncoated paper grain, flat poster shapes.",
        "mood": "bright, punchy, graphic, youthful",
    },
    "stainedglass": {
        "lead": "Stained glass window panel, backlit:",
        "style": "Leaded came dividing saturated colored glass, luminous jewel tones, glass texture and trapped bubbles, heavy dark lead lines, medieval craft.",
        "mood": "transcendent, devotional, radiant",
    },
    "cutpaper": {
        "lead": "Layered cut-paper diorama in a shadow box:",
        "style": "Six stacked planes of hand-cut textured cardstock, visible paper edges and drop shadows between layers, matte colored stock, craft-object lighting.",
        "mood": "delicate, storybook, quiet, constructed",
    },
    "verdigris": {
        "lead": "Oxidized copper relief panel in raking light:",
        "style": "Hammered copper sheet with deep verdigris green patina, mineral crust, corroded pitting, raised repoussé forms catching the light, aged metal.",
        "mood": "ancient, corroded, heavy, elemental",
    },
    "ukiyoe": {
        "lead": "Japanese woodblock ukiyo-e print:",
        "style": "Flat carved color planes, visible woodgrain in the ink, bold black keyline, mineral pigment, aged mulberry paper.",
        "mood": "stylized, flowing, poised",
    },
    "raku": {
        "lead": "Raku-fired ceramic glaze surface:",
        "style": "Crackled glaze craquelure, iridescent copper-lustre sheen, carbon-blackened unglazed clay, thick glossy pooling, kiln-scarred.",
        "mood": "volatile, lustrous, earthy",
    },
}

# Reasonable default arcs by energy level, used when the director declines
# to choose. Cold/flat materials for low energy, warm/tactile for peaks.
BY_LEVEL = {
    "low": ["cyanotype", "collodion", "etching", "cutpaper"],
    "mid": ["etching", "ukiyoe", "verdigris", "cutpaper"],
    "high": ["felted", "risograph", "stainedglass", "raku"],
}
