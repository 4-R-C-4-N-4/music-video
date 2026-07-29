"""Material grammar: compose treatments instead of storing them.

The first version of this file held ten finished treatment strings, so every
cyanotype scene in every video got the identical prompt and the look read as
a filter preset. Here a treatment is *composed* at plan time from a physical
family plus sampled modifiers, so two cyanotypes are two different objects —
different sheet, different exposure, different damage, different lighting.

Structure:
    family    a physically-real substrate+process pair (the part that must
              stay coherent — "cyanotype on rag paper", "intaglio etching")
    palette   a colour constraint the family actually admits
    emphasis  which physical quality to foreground (fibre, tooth, burr...)
    artifacts 1-3 process defects, the main source of felt uniqueness
    capture   how the object is lit and photographed
    wear      its age and condition

Sampling is seeded, so a manifest reproduces exactly. Per-shot jitter
re-rolls artifacts only, which makes each shot read as a separate physical
artefact rather than one image with a filter over it.

Rule that still governs everything: name the *physical process*. Abstract
optical asks ("formed from thin-film interference") collapse to photoreal.
"""
import random

FAMILIES = {
    "cyanotype": {
        "mood": "cold, still, elegiac, archival",
        "leads": [
            "Cyanotype sun print on {sub}:",
            "Hand-coated cyanotype contact print on {sub}:",
            "Blueprint-process sun exposure on {sub}:",
        ],
        "substrates": ["rough cotton rag paper", "heavy watercolour paper",
                       "thin unsized mulberry paper", "coarse linen cloth",
                       "buff-toned printmaking stock"],
        "core": "Prussian blue where the light struck and bare substrate where it did not, no true midtones",
        "palettes": ["deep Prussian blue and raw paper white only",
                     "faded slate-blue washed almost to grey",
                     "blue-black shadows with warm cream highlights",
                     "toned brown-violet from a tannin bath"],
        "emphasis": ["visible paper fibre in the highlights",
                     "the brush-stroked edge of the hand-coated emulsion",
                     "the deckle edge of the sheet",
                     "uneven coating leaving lighter streaks"],
        "artifacts": ["streaks where the emulsion was brushed unevenly",
                      "pale fingerprints at the sheet margin",
                      "a bleached halo where the sheet lifted during exposure",
                      "hard-edged contact shadows from the negative's tape",
                      "foxing spots blooming through the blue",
                      "a crease running through the print",
                      "uneven wash-out leaving a tidemark"],
        "captures": ["photographed flat under diffuse north light",
                     "scanned on a flatbed, dead flat and slightly over-lit",
                     "held up against a window, light coming through the sheet",
                     "lying on a table with the sheet curling at one corner"],
        "wear": ["freshly washed and still faintly damp", "decades old and sun-faded",
                 "handled often, soft at the corners", "archival and pristine"],
    },
    "etching": {
        "mood": "austere, antique, melancholy, precise",
        "leads": ["Copperplate etching, intaglio print on {sub}:",
                  "Hand-pulled intaglio etching on {sub}:",
                  "Drypoint and etching impression on {sub}:"],
        "substrates": ["off-white laid paper", "heavy cream printmaking paper",
                       "thin India paper", "toned tan stock"],
        "core": "the image built entirely from bitten lines and cross-hatching, ink sitting in the grooves",
        "palettes": ["warm sepia-black ink on off-white",
                     "cold blue-black ink", "sanguine red-brown ink",
                     "dense carbon black with grey plate tone"],
        "emphasis": ["the velvety burr of drypoint lines",
                     "the embossed plate mark pressed into the paper",
                     "dense cross-hatching modelling the shadows",
                     "open unworked paper carrying the highlights"],
        "artifacts": ["plate tone smeared unevenly across the background",
                      "a scratch running across the plate",
                      "over-bitten passages gone blotchy and dark",
                      "the plate edge printing as a dark rectangle",
                      "ink squeezed into a ridge at one margin",
                      "foul-biting speckle in the open areas"],
        "captures": ["photographed flat in raking light so the embossing shows",
                     "scanned flat and sharp", "lying under glass with a slight reflection"],
        "wear": ["a crisp early impression", "a late worn impression, lines going grey",
                 "foxed and yellowed with age"],
    },
    "felted": {
        "mood": "warm, tender, handmade, childlike",
        "leads": ["Needle-felted wool diorama, {cap}:",
                  "Handmade felted wool model, {cap}:",
                  "Wet-felted and needle-sculpted wool scene, {cap}:"],
        "substrates": ["dyed carded wool", "undyed natural fleece with dyed accents",
                       "wool over a wire armature", "mixed wool and silk fibre"],
        "core": "every surface built from matted wool fibre, soft and slightly fuzzy, sitting on a tabletop",
        "palettes": ["heathered naturals with one saturated accent colour",
                     "soft dusty pastels", "deep saturated dyed jewel tones",
                     "undyed creams and greys"],
        "emphasis": ["loose fibre wisps catching the light",
                     "visible needle-poke pockmarks over every surface",
                     "the fuzzy halo at every silhouette edge",
                     "compacted denser felt where the piece was worked hardest"],
        "artifacts": ["a felting needle left stuck in the piece",
                      "loose fibres scattered on the table around it",
                      "a seam where two felted parts were joined",
                      "the wire armature showing through at one point",
                      "slight sagging where the felt is under-worked",
                      "stray coloured fibres migrated into the wrong area"],
        "captures": ["photographed in soft daylight with shallow depth of field",
                     "lit from one side with a warm lamp, long soft shadows",
                     "overhead flat-lay on a plain linen cloth",
                     "macro close focus, the background falling away"],
        "wear": ["freshly made, fibres still crisp", "pilled and dusty from handling",
                 "slightly compressed as if stored in a box"],
    },
    "risograph": {
        "mood": "bright, punchy, graphic, youthful",
        "leads": ["Risograph print, {pal}, on {sub}:",
                  "Two-colour riso duplicator print on {sub}:",
                  "Mimeograph-style stencil duplicator print on {sub}:"],
        "substrates": ["matte uncoated newsprint", "cheap grey pulp paper",
                       "bright white copy stock", "recycled speckled stock"],
        "core": "flat poster shapes built from overlapping ink layers, no continuous tone",
        "palettes": ["fluorescent pink and teal", "federal blue and warm red",
                     "bright yellow and black", "green and fluorescent orange"],
        "emphasis": ["visible halftone dot screens in the mid-tones",
                     "the moiré where two dot screens overlap",
                     "flat blocks of unbroken ink",
                     "the paper showing through the thin ink"],
        "artifacts": ["misregistration offsetting one ink layer by a few millimetres",
                      "roller streaks running down the sheet",
                      "ink smudged by handling at one corner",
                      "a paper-feed crease across the print",
                      "patchy coverage where the drum ran dry",
                      "show-through from the reverse side"],
        "captures": ["scanned flat", "photographed on a desk under flat light",
                     "pinned to a wall, edges curling"],
        "wear": ["fresh off the drum, ink still offsetting",
                 "sun-bleached, the fluorescent gone pale", "creased from being folded"],
    },
    "collodion": {
        "mood": "haunted, historical, decayed, solemn",
        "leads": ["Wet-plate collodion tintype, {cap}:",
                  "Ambrotype on glass, {cap}:",
                  "Wet-collodion negative printed as a positive, {cap}:"],
        "substrates": ["blackened iron plate", "clear glass plate",
                       "aluminium trophy plate"],
        "core": "shallow tonal range with silver highlights and deep empty shadows, extremely shallow focus",
        "palettes": ["warm neutral grey with tarnished silver highlights",
                     "cold blue-grey", "sepia-toned brown-black",
                     "silver going iridescent where it has tarnished"],
        "emphasis": ["the sweep of collodion pour visible across the plate",
                     "the fall-off into total black at the edges",
                     "silver highlights that read almost metallic",
                     "one plane of razor focus and everything else dissolving"],
        "artifacts": ["comet-tail streaks from dust in the pour",
                      "bromide drips running from the plate edge",
                      "the collodion lifting and flaking at one corner",
                      "chemical staining blooming across one side",
                      "fingerprints of the plate holder at the margins",
                      "a scratch through the emulsion"],
        "captures": ["photographed on black velvet", "held in a hand at an angle so the silver flares",
                     "scanned flat against black"],
        "wear": ["freshly varnished and glossy", "flaking and half-lost",
                 "the varnish yellowed and crazed"],
    },
    "cutpaper": {
        "mood": "delicate, storybook, quiet, constructed",
        "leads": ["Layered cut-paper diorama in a shadow box, {cap}:",
                  "Hand-cut paper theatre with stacked planes, {cap}:",
                  "Papercut scene built in receding layers, {cap}:"],
        "substrates": ["textured coloured cardstock", "heavy watercolour paper",
                       "thin tracing paper and cardstock mixed", "corrugated board and tissue"],
        "core": "the scene built from separate stacked planes of cut paper with real air between them",
        "palettes": ["a narrow range of blues and greys deepening with distance",
                     "warm creams and ochres", "high-contrast black and white with one red",
                     "soft faded pastels"],
        "emphasis": ["the cut edge of every layer catching the light",
                     "drop shadows falling from each plane onto the next",
                     "the paper's tooth visible across the flat areas",
                     "translucent tissue glowing where light passes through"],
        "artifacts": ["a slightly ragged cut where the blade slipped",
                      "visible glue tabs holding a layer",
                      "one layer buckling away from flat",
                      "a pencil guideline left uncut",
                      "dust settled in the gaps between layers"],
        "captures": ["lit from one side to throw the layer shadows long",
                     "backlit so the tissue layers glow",
                     "photographed straight on in flat museum light"],
        "wear": ["crisply new", "sun-faded on one side", "dented from storage"],
    },
    "stainedglass": {
        "mood": "transcendent, devotional, radiant",
        "leads": ["Stained glass panel, {cap}:",
                  "Leaded glass window, {cap}:",
                  "Painted and fired glass panel in lead came, {cap}:"],
        "substrates": ["hand-blown cathedral glass", "opalescent glass",
                       "streaky antique glass", "thick slab glass set in concrete"],
        "core": "saturated coloured glass divided by heavy dark lead lines, lit from behind",
        "palettes": ["deep cobalt and ruby with gold", "cool greens and blue-greys",
                     "amber and bottle green", "predominantly white and silver-stain yellow"],
        "emphasis": ["bubbles and striations trapped in the hand-blown glass",
                     "the black lead came drawing every contour",
                     "painted grisaille shading fired onto the glass",
                     "colour spilling onto the surrounding stone"],
        "artifacts": ["a cracked pane held by a repair lead",
                      "one pane replaced in a mismatched colour",
                      "dirt and grime built up in the lower corners",
                      "the lead sagging and bowing with age",
                      "condensation blurring part of the panel"],
        "captures": ["backlit by grey daylight", "backlit by low warm sun, colour thrown across the frame",
                     "photographed at an angle so the glass surface reflects"],
        "wear": ["newly leaded and bright", "centuries old, buckled and grimy",
                 "partly restored, some panes clean and some dark"],
    },
    "verdigris": {
        "mood": "ancient, corroded, heavy, elemental",
        "leads": ["Oxidised copper relief panel, {cap}:",
                  "Hammered bronze plaque with patina, {cap}:",
                  "Repoussé copper sheet, corroded, {cap}:"],
        "substrates": ["hammered copper sheet", "cast bronze", "beaten brass plate"],
        "core": "raised metal forms worked in relief, the surface eaten by patina",
        "palettes": ["deep verdigris green over raw copper",
                     "blue-green patina with black recesses",
                     "brown-black bronze with green only in the hollows",
                     "bright polished metal against corroded areas"],
        "emphasis": ["hammer facets across the flat areas",
                     "mineral crust built up thick in the recesses",
                     "corrosion pitting eating into the raised forms",
                     "polished high points where hands have rubbed it"],
        "artifacts": ["a repair patch soldered across one area",
                      "drips of green running down from the recesses",
                      "a dent distorting the relief",
                      "rivet heads punching through the design",
                      "salt bloom crusting one edge"],
        "captures": ["in hard raking light so the relief throws shadows",
                     "under flat overcast light", "lit from below, shadows running upward"],
        "wear": ["freshly patinated", "centuries buried and only just cleaned",
                 "weathered outdoors for decades"],
    },
    "ukiyoe": {
        "mood": "stylized, flowing, poised",
        "leads": ["Japanese woodblock ukiyo-e print on {sub}:",
                  "Multi-block nishiki-e colour woodcut on {sub}:",
                  "Hand-printed woodblock impression on {sub}:"],
        "substrates": ["aged mulberry washi paper", "thin kozo paper",
                       "cream hosho paper"],
        "core": "flat carved colour planes bounded by a printed keyline, no shading",
        "palettes": ["indigo and safflower red with black keyline",
                     "muted mineral greens and ochres",
                     "predominantly indigo blues (aizuri-e)",
                     "faded pinks and greys, the fugitive pigments gone"],
        "emphasis": ["woodgrain printing through the flat colour areas",
                     "the bokashi gradient hand-wiped on the block",
                     "the embossed impression of the block in the paper",
                     "the crisp carved keyline"],
        "artifacts": ["one colour block slightly out of register",
                      "a wormhole through the sheet",
                      "the paper browned and brittle at the edges",
                      "an ink-heavy pull with the colour gone muddy",
                      "a horizontal fold from being bound in an album"],
        "captures": ["photographed flat in even light", "scanned flat",
                     "lit at a low angle so the paper texture and embossing read"],
        "wear": ["a fresh early impression", "faded by two centuries of light",
                 "water-stained along one edge"],
    },
    "raku": {
        "mood": "volatile, lustrous, earthy",
        "leads": ["Raku-fired ceramic surface, {cap}:",
                  "Glazed and reduction-fired stoneware, {cap}:",
                  "Pit-fired earthenware with lustre glaze, {cap}:"],
        "substrates": ["grogged stoneware clay", "smooth white earthenware",
                       "dark iron-rich clay body"],
        "core": "a thick glaze crazed into a fine crackle over a scorched clay body",
        "palettes": ["copper lustre iridescence over black",
                     "white crackle glaze with carbon-blackened crazing",
                     "turquoise glaze breaking to bronze at the edges",
                     "unglazed smoke-blackened clay with one glazed passage"],
        "emphasis": ["the crackle network filled with carbon",
                     "glaze pooling thick in the hollows and thin on the ridges",
                     "flashing where the flame licked the piece",
                     "raw grogged clay contrasting with glassy glaze"],
        "artifacts": ["a firing crack running through the piece",
                      "a bare patch where the glaze crawled",
                      "kiln shelf scars on the underside",
                      "pinholes bubbled through the glaze",
                      "an ash deposit fused to one side"],
        "captures": ["lit hard so the lustre flares", "in soft light showing the crackle",
                     "macro close on the glaze surface"],
        "wear": ["straight from the reduction bin", "handled for years, lustre worn thin",
                 "chipped at the rim"],
    },
}

# Cross-family palette-ish global look, sampled per video so different videos
# of the same family still differ.
GLOBAL_GRADE = [
    "the whole image slightly cool", "the whole image slightly warm",
    "low contrast and dusty", "high contrast and clean",
    "a touch overexposed", "a touch dark and dense",
]

BY_LEVEL = {
    "low": ["cyanotype", "collodion", "etching", "cutpaper", "ukiyoe"],
    "mid": ["etching", "ukiyoe", "verdigris", "cutpaper", "raku"],
    "high": ["felted", "risograph", "stainedglass", "raku", "verdigris"],
}

MATERIALS = FAMILIES  # back-compat for specs that reference family keys


def compose(family_key: str, seed: int, n_artifacts: int = 2) -> dict:
    """Sample one concrete treatment from a family.

    Same key + same seed always yields the same treatment, so a manifest
    reproduces. Different seeds yield materially different objects rather
    than the same string every time.
    """
    fam = FAMILIES[family_key]
    rng = random.Random(seed)

    sub = rng.choice(fam["substrates"])
    cap = rng.choice(fam["captures"])
    pal = rng.choice(fam["palettes"])
    lead = rng.choice(fam["leads"]).format(sub=sub, cap=cap, pal=pal)

    picks = rng.sample(fam["artifacts"], min(n_artifacts, len(fam["artifacts"])))
    style = ". ".join([
        fam["core"].capitalize(),
        f"Colour: {pal}",
        f"Surface: {rng.choice(fam['emphasis'])}",
        f"Condition: {rng.choice(fam['wear'])}",
        f"Flaws: {'; '.join(picks)}",
        rng.choice(GLOBAL_GRADE).capitalize(),
        "No photorealism — the material and the process that made it are the subject",
    ]) + "."
    return {"lead": lead, "style": style, "family": family_key,
            "substrate": sub, "palette": pal, "artifacts": picks}


def jitter(family_key: str, base_seed: int, shot_seed: int) -> dict:
    """Per-shot variation: same object family and palette, fresh flaws.

    Keeps a scene coherent while making each shot read as its own physical
    artefact — a different sheet, a different pull — instead of one image
    with a filter over it.
    """
    base = compose(family_key, base_seed)
    rng = random.Random(shot_seed)
    fam = FAMILIES[family_key]
    picks = rng.sample(fam["artifacts"], min(2, len(fam["artifacts"])))
    style = base["style"].replace(
        f"Flaws: {'; '.join(base['artifacts'])}", f"Flaws: {'; '.join(picks)}")
    return {"lead": base["lead"], "style": style, "family": family_key}
