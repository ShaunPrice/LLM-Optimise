"""Deterministic 1080p LLM-Optimise marketing compositor.

Public API: draw_scene(scene: dict, t: float, duration: float) -> RGB PIL.Image.
Times are scene-local seconds. Pixels y >= 940 are reserved for subtitles.
Run --proof for midpoint PNGs and a contact sheet. Missing screenshot placeholders
are permitted only in proof mode; normal rendering raises FileNotFoundError.
"""

from __future__ import annotations

import argparse
import json
import math
from functools import lru_cache
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[3]
SHOTS = ROOT / "marketing/video/shots"
WIDTH, HEIGHT = 1920, 1080
SAFE_BOTTOM = 940
NAVY = (11, 23, 40)
COBALT = (47, 98, 203)
TEAL = (60, 211, 197)
IVORY = (242, 247, 251)
MUTED = (161, 180, 202)
DIM = (75, 101, 133)
PANEL = (17, 35, 58)
PROOF_MODE = False
SCREENSHOTS = {
    2: "lab",
    3: "results",
    4: "workbench-result",
    6: "routing",
    7: "develop",
    8: "training",
    9: "chat",
}
# Crop genuine screenshots, excluding the sidebar and private workspace footer.
# Coordinates are fractions of the supplied 1185x782 captures, so HiDPI works too.
CROPS = {
    "lab": (180, 65, 1170, 720),
    "results": (475, 275, 1150, 740),
    "workbench": (198, 0, 1150, 728),
    "workbench-result": (198, 0, 1150, 728),
    "routing": (200, 18, 1150, 706),
    "develop": (180, 155, 1170, 720),
    "training": (180, 0, 1170, 700),
    "chat": (200, 219, 880, 721),
}


# Explicit privacy masks over paths in the reviewed source captures. Source files
# remain unchanged, and the composited footer discloses that paths were omitted.
REDACTIONS = {
    "develop": [(518, 493, 1132, 515)],
    "training": [(540, 65, 1132, 91), (542, 143, 1132, 188)],
}


def ease(value):
    value = max(0.0, min(1.0, value))
    return value * value * (3.0 - 2.0 * value)


def mix(a, b, fraction):
    return tuple(round(x + (y - x) * fraction) for x, y in zip(a, b, strict=True))


@lru_cache(maxsize=80)
def font(size, weight="regular"):
    candidates = [
        (
            Path("/System/Library/Fonts/Avenir Next.ttc"),
            {"regular": 7, "medium": 5, "bold": 2}[weight],
        ),
        (Path("/System/Library/Fonts/Helvetica.ttc"), 1 if weight == "bold" else 0),
        (
            Path(
                "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"
                if weight == "bold"
                else "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"
            ),
            0,
        ),
        (
            Path(
                "C:/Windows/Fonts/arialbd.ttf" if weight == "bold" else "C:/Windows/Fonts/arial.ttf"
            ),
            0,
        ),
    ]
    for path, index in candidates:
        if path.exists():
            return ImageFont.truetype(str(path), int(size), index=index)
    raise RuntimeError("Install Avenir Next, Helvetica, DejaVu Sans or Arial for video rendering")


def text(draw, xy, value, size=32, color=IVORY, weight="regular", anchor=None):
    draw.text(tuple(map(round, xy)), str(value), font=font(size, weight), fill=color, anchor=anchor)


def tracked(draw, xy, value, size=20, color=TEAL, tracking=3):
    x, y = xy
    f = font(size, "bold")
    for char in value:
        draw.text((round(x), round(y)), char, font=f, fill=color)
        x += draw.textlength(char, font=f) + tracking
    return x


def wrapped(draw, xy, value, width, size=70, leading=1.13, color=IVORY, weight="bold"):
    x, y = xy
    f = font(size, weight)
    for paragraph in value.split("\n"):
        line = ""
        for word in paragraph.split():
            proposed = (line + " " + word).strip()
            if line and draw.textlength(proposed, font=f) > width:
                text(draw, (x, y), line, size, color, weight)
                y += size * leading
                line = word
            else:
                line = proposed
        if line:
            text(draw, (x, y), line, size, color, weight)
            y += size * leading
    return y


@lru_cache(maxsize=1)
def background_base():
    y, x = np.mgrid[0:HEIGHT, 0:WIDTH]
    glow = np.exp(-(((x - 1450) / 980) ** 2 + ((y - 330) / 650) ** 2))
    second = np.exp(-(((x - 280) / 650) ** 2 + ((y - 820) / 600) ** 2))
    array = np.empty((HEIGHT, WIDTH, 3), dtype=np.uint8)
    for channel in range(3):
        array[:, :, channel] = np.clip(
            NAVY[channel] + glow * (3, 12, 25)[channel] + second * (1, 5, 8)[channel], 0, 255
        )
    image = Image.fromarray(array, "RGB")
    draw = ImageDraw.Draw(image)
    # Deliberately sparse paths, visible enough to move without competing with text.
    paths = [
        [(1070, 0), (1070, 85), (1370, 85), (1470, 185), (1920, 185)],
        [(1450, 0), (1450, 50), (1630, 50), (1740, 160), (1920, 160)],
        [(0, 825), (255, 825), (355, 725), (500, 725)],
        [(0, 850), (270, 850), (370, 750), (510, 750)],
        [(1660, 480), (1840, 480), (1920, 560)],
    ]
    for path in paths:
        draw.line(path, fill=(24, 52, 79), width=2, joint="curve")
    return image


def background(t, scene_id):
    image = background_base().copy()
    draw = ImageDraw.Draw(image)
    for index in range(28):
        x = (index * 197 + 601 + t * (4 + index % 3)) % WIDTH
        y = (index * 131 + 53 + math.sin(t * 0.28 + index) * 12) % 915
        brightness = 0.15 + 0.12 * (math.sin(t * 0.7 + index * 2.1) + 1) / 2
        color = mix(NAVY, TEAL, brightness)
        draw.ellipse((x, y, x + 2, y + 2), fill=color)
    # The caption reserve is consistently clean in every scene.
    draw.rectangle((0, SAFE_BOTTOM, WIDTH, HEIGHT), fill=NAVY)
    return image


@lru_cache(maxsize=12)
def icon(size):
    image = Image.open(ROOT / "docs/assets/app-icon.png").convert("RGBA")
    return image.resize((size, size), Image.Resampling.LANCZOS)


def brand(image, t=0):
    image.paste(icon(52), (92, 49), icon(52))
    draw = ImageDraw.Draw(image)
    text(draw, (162, 57), "LLM-OPTIMISE", 24, IVORY, "bold")
    draw.line((94, 122, 1826, 122), fill=(39, 60, 86), width=1)


def footer(image, scene, t, duration, label="PRODUCT WORKSPACE / ACTUAL APPLICATION"):
    draw = ImageDraw.Draw(image)
    text(draw, (96, 902), label, 17, MUTED, "medium")
    text(draw, (1824, 902), f"{int(scene['id']) + 1:02d} / 12", 17, MUTED, "medium", anchor="ra")
    x = 1370
    for i in range(12):
        color = TEAL if i < scene["id"] else DIM
        draw.rounded_rectangle((x, 916, x + 22, 919), radius=1, fill=color)
        if i == scene["id"]:
            draw.rounded_rectangle(
                (x, 916, x + 22 * min(1, max(0, t / duration)), 919), radius=1, fill=TEAL
            )
        x += 31


def pill(draw, xy, value, t, index=0):
    reveal = ease((t - 0.55 - index * 0.15) / 0.6)
    x, y = xy
    width = draw.textlength(value, font=font(23, "medium")) + 38
    y += 12 * (1 - reveal)
    draw.rounded_rectangle(
        (x, y, x + width, y + 46),
        radius=23,
        fill=mix(NAVY, PANEL, reveal),
        outline=mix(NAVY, (47, 82, 113), reveal),
        width=1,
    )
    text(draw, (x + 19, y + 7), value, 23, mix(NAVY, IVORY, reveal), "medium")


@lru_cache(maxsize=16)
def load_shot(name, stamp):
    del stamp  # mtime makes the cache refresh when root replaces a capture.
    path = SHOTS / f"{name}.png"
    source = Image.open(path).convert("RGB")
    crop = CROPS[name]
    factor_x, factor_y = source.width / 1185, source.height / 782
    draw = ImageDraw.Draw(source)
    for redaction in REDACTIONS.get(name, []):
        left, top, right, bottom = tuple(
            round(v * (factor_x if i % 2 == 0 else factor_y)) for i, v in enumerate(redaction)
        )
        draw.rectangle((left, top, right, bottom), fill=NAVY)
        text(
            draw,
            (left + 9 * factor_x, (top + bottom) / 2),
            "Local path omitted",
            max(10, round(12 * factor_y)),
            MUTED,
            "medium",
            "lm",
        )
    return source.crop(
        tuple(round(v * (factor_x if i % 2 == 0 else factor_y)) for i, v in enumerate(crop))
    )


def screenshot(image, name, box, t, duration, caption):
    x, y, width, height = map(int, box)
    reveal = ease(t / 0.8)
    y += round((1 - reveal) * 22)
    path = SHOTS / f"{name}.png"
    draw = ImageDraw.Draw(image)
    draw.rounded_rectangle(
        (x - 13, y - 43, x + width + 13, y + height + 13),
        radius=20,
        fill=(18, 34, 55),
        outline=(56, 86, 121),
        width=1,
    )
    for i in range(3):
        draw.ellipse((x + 8 + i * 16, y - 25, x + 14 + i * 16, y - 19), fill=(78, 111, 146))
    text(draw, (x + 76, y - 33), caption, 18, MUTED, "medium")
    if not path.exists():
        if not PROOF_MODE:
            raise FileNotFoundError(f"Production screenshot missing: {path}")
        draw.rectangle((x, y, x + width, y + height), fill=(21, 43, 69))
        text(draw, (x + width / 2, y + height / 2 - 22), "PROOF ONLY", 26, MUTED, "bold", "mm")
        text(
            draw,
            (x + width / 2, y + height / 2 + 22),
            f"{name.upper()} CAPTURE PENDING",
            24,
            MUTED,
            "medium",
            "mm",
        )
        return
    source = load_shot(name, path.stat().st_mtime_ns)
    # Use contain scaling: every pixel of the deliberately selected crop is kept.
    scale = min(width / source.width, height / source.height)
    target = (round(source.width * scale), round(source.height * scale))
    shot = source.resize(target, Image.Resampling.LANCZOS)
    panel = Image.new("RGB", (width, height), (234, 241, 248))
    panel.paste(shot, ((width - shot.width) // 2, (height - shot.height) // 2))
    # A 1.4% push; only the panel's safe peripheral padding is clipped.
    zoom = 1 + 0.014 * ease(t / duration)
    pushed = panel.resize((round(width * zoom), round(height * zoom)), Image.Resampling.BICUBIC)
    cx, cy = (pushed.width - width) // 2, (pushed.height - height) // 2
    panel = pushed.crop((cx, cy, cx + width, cy + height))
    mask = Image.new("L", (width, height), 0)
    ImageDraw.Draw(mask).rounded_rectangle(
        (0, 0, width - 1, height - 1), radius=7, fill=round(255 * reveal)
    )
    image.paste(panel, (x, y), mask)


def connector(draw, points, t, phase=0, color=TEAL, width=2):
    draw.line(points, fill=mix(NAVY, color, 0.32), width=width, joint="curve")
    lengths = [math.dist(a, b) for a, b in zip(points[:-1], points[1:], strict=True)]
    total = sum(lengths)
    if not total:
        return
    pos = (t * 100 + phase * 133) % total
    for a, b, length in zip(points[:-1], points[1:], lengths, strict=True):
        if pos <= length:
            fraction = pos / length if length else 0
            x, y = a[0] + (b[0] - a[0]) * fraction, a[1] + (b[1] - a[1]) * fraction
            draw.ellipse((x - 4, y - 4, x + 4, y + 4), fill=color)
            break
        pos -= length
    end, previous = points[-1], points[-2]
    angle = math.atan2(end[1] - previous[1], end[0] - previous[0])
    head = [
        (end[0] - 10 * math.cos(angle + a), end[1] - 10 * math.sin(angle + a)) for a in (-0.5, 0.5)
    ]
    draw.polygon([end, *head], fill=mix(NAVY, color, 0.65))


def node(draw, box, title, subtitle="", accent=False, title_size=30):
    x, y, width, height = box
    draw.rounded_rectangle(
        (x, y, x + width, y + height),
        radius=16,
        fill=(21, 44, 73) if accent else PANEL,
        outline=TEAL if accent else (57, 84, 115),
        width=2 if accent else 1,
    )
    text(draw, (x + 22, y + 19), title, title_size, IVORY, "bold")
    if subtitle:
        text(draw, (x + 22, y + 61), subtitle, 22, MUTED)


def title_block(image, scene, t, size=68):
    draw = ImageDraw.Draw(image)
    reveal = ease(t / 0.65)
    y = 194 + (1 - reveal) * 24
    tracked(draw, (96, y), scene["eyebrow"], 18, mix(NAVY, TEAL, reveal), 2)
    return wrapped(
        draw, (91, y + 66), scene["heading"], 520, size=size, color=mix(NAVY, IVORY, reveal)
    )


def wide_title(image, scene, t):
    draw = ImageDraw.Draw(image)
    tracked(draw, (96, 158), scene["eyebrow"], 19, TEAL, 2)
    wrapped(
        draw, (91, 200 + 14 * (1 - ease(t / 0.6))), scene["heading"].replace("\n", " "), 1750, 67
    )


def intro(image, scene, t, duration, outro=False):
    draw = ImageDraw.Draw(image)
    p = ease(t / 1.0)
    logo_size = 180 if outro else 230
    x = (WIDTH - logo_size) // 2
    y = 166 if outro else 147
    # Existing transparent icon, without repainting its identity.
    image.paste(icon(logo_size), (x, round(y + 18 * (1 - p))), icon(logo_size))
    label = "LLM-OPTIMISE"
    label_width = draw.textlength(label, font=font(27, "bold")) + (len(label) - 1) * 5
    tracked(draw, ((WIDTH - label_width) / 2, y + logo_size + 33), label, 27, TEAL, 5)
    headline_y = 466 if not outro else 438
    lines = scene["heading"].split("\n")
    size = 90 if not outro else 116
    for i, line in enumerate(lines):
        text(
            draw,
            (960, headline_y + i * size * 1.08 + 20 * (1 - p)),
            line,
            size,
            mix(NAVY, IVORY, p),
            "bold",
            "ma",
        )
    if outro:
        text(draw, (960, 729), "Open source  /  MIT licensed", 29, MUTED, "medium", "ma")
        width = 950
        draw.rounded_rectangle(
            (960 - width / 2, 797, 960 + width / 2, 861),
            radius=32,
            fill=(22, 53, 84),
            outline=(54, 108, 148),
            width=1,
        )
        text(draw, (960, 808), "github.com/ShaunPrice/LLM-Optimise", 35, IVORY, "medium", "ma")
    else:
        text(
            draw, (960, 716), "A laboratory for the hardware you have.", 32, MUTED, "regular", "ma"
        )
        for i, label in enumerate(["SPEED", "MEMORY", "TASK QUALITY"]):
            tracked(draw, (570 + i * 258, 810), label, 18, MUTED, 3)
    # Slow orbiting accent that continues after the title reveal.
    radius = 160 if outro else 185
    center = (960, y + logo_size / 2)
    angle = t * 0.18
    for offset in (0, math.pi):
        px, py = (
            center[0] + math.cos(angle + offset) * radius,
            center[1] + math.sin(angle + offset) * radius * 0.55,
        )
        draw.ellipse((px - 4, py - 4, px + 4, py + 4), fill=TEAL)
    footer(image, scene, t, duration, "LLM-OPTIMISE / SMALL HARDWARE. SPECIALISED INTELLIGENCE.")


def purpose(image, scene, t, duration):
    end = title_block(image, scene, t, 72)
    draw = ImageDraw.Draw(image)
    wrapped(
        draw,
        (96, end + 34),
        "Explore the limits.\nKeep quality in view.",
        475,
        31,
        1.4,
        MUTED,
        "regular",
    )
    center = (1265, 509)
    for radius in (174, 252):
        draw.ellipse(
            (center[0] - radius, center[1] - radius, center[0] + radius, center[1] + radius),
            outline=(30, 62, 91),
            width=1,
        )
    boxes = [
        (836, 238, "SPEED", "Response time"),
        (1362, 238, "MEMORY", "RAM + GPU"),
        (836, 651, "COST", "Your budget"),
        (1362, 651, "QUALITY", "Task correctness"),
    ]
    for i, (x, y, heading, body) in enumerate(boxes):
        target = (x + 190, y + 75)
        connector(draw, [center, target], t, i)
        node(draw, (x, y, 366, 132), heading, body)
    draw.rounded_rectangle(
        (1087, 422, 1443, 584), radius=30, fill=(26, 57, 89), outline=TEAL, width=2
    )
    text(draw, (1265, 445), "YOUR", 33, TEAL, "medium", "ma")
    text(draw, (1265, 492), "SPECIALISED TASK", 32, IVORY, "bold", "ma")
    footer(image, scene, t, duration, "OPTIMISATION TARGETS / CONCEPTUAL ILLUSTRATION")


def ui_scene(image, scene, t, duration):
    identifier = scene["id"]
    end = title_block(image, scene, t)
    draw = ImageDraw.Draw(image)
    copy = {
        2: ("Representative tasks.\nExplicit resource limits.", ["Baseline", "Quality gate"]),
        3: (
            "Latency. Memory. Correctness.\nRead the evidence together.",
            ["Actual measurements", "Exportable results"],
        ),
        4: (
            "Find viable settings through\nbounded experiment searches.",
            ["Context + KV cache", "Accelerators"],
        ),
        6: (
            "Choose placement, budget\nand performance priorities.",
            ["Measured routing", "Exact reuse"],
        ),
        7: ("Review proposed changes.\nTest a disposable copy.", ["Chosen model", "Docker tests"]),
        8: (
            "Train an adapter. Reload it.\nEvaluate unseen task examples.",
            ["Soup + MLX / Transformers", "Held-out evaluation"],
        ),
    }
    body, labels = copy[identifier]
    end = wrapped(draw, (96, end + 35), body, 500, 29, 1.38, MUTED, "regular")
    for i, label in enumerate(labels):
        pill(draw, (96, end + 39 + i * 62), label, t, i)
    name = SCREENSHOTS[identifier]
    screenshot(
        image,
        name,
        (692, 204, 1116, 638),
        t,
        duration,
        {
            2: "EXPERIMENT LAB",
            3: "MEASURED RESULTS",
            4: "EXPERIMENT WORKBENCH",
            6: "MODEL ROUTER",
            7: "DEVELOP WORKSPACE",
            8: "SOUP / RECIPE PREPARATION",
        }[identifier],
    )
    footer(
        image,
        scene,
        t,
        duration,
        "ACTUAL APPLICATION / LOCAL PATHS OMITTED"
        if name in REDACTIONS
        else "PRODUCT WORKSPACE / ACTUAL APPLICATION",
    )


def architecture(image, scene, t, duration):
    wide_title(image, scene, t)
    draw = ImageDraw.Draw(image)
    for index, label in enumerate(["GUI workspace", "Command line", "MCP clients"]):
        y = 350 + index * 130
        connector(draw, [(366, y + 41), (410, y + 41), (410, 523), (482, 523)], t, index)
        node(draw, (96, y, 270, 83), label, title_size=26)
    connector(draw, [(850, 523), (928, 523), (928, 370), (1030, 370)], t, 2)
    connector(draw, [(1350, 370), (1460, 370)], t, 3)
    connector(draw, [(850, 523), (1030, 523)], t, 4)
    connector(draw, [(928, 523), (928, 675), (1030, 675)], t, 5)
    connector(draw, [(928, 675), (928, 772), (1445, 772), (1445, 675), (1460, 675)], t, 6)
    connector(draw, [(666, 600), (666, 807)], t, 7)
    node(draw, (482, 445, 368, 155), "Shared application", "Python workflow engine", True, 31)
    node(draw, (1030, 324, 320, 99), "Rust supervisor", "Optional process control", title_size=28)
    node(draw, (1460, 324, 348, 99), "llama.cpp", "Local model execution", title_size=32)
    node(
        draw,
        (1030, 475, 778, 104),
        "Soup adapters",
        "Separate MLX / Transformers training and evaluation",
        title_size=31,
    )
    node(draw, (1030, 628, 370, 102), "Cloud providers", "Selected model + budget", title_size=29)
    node(draw, (1460, 628, 348, 102), "Docker", "Disposable build + tests", title_size=32)
    draw.rounded_rectangle(
        (482, 807, 1808, 867), radius=13, fill=(16, 49, 64), outline=(43, 109, 118), width=1
    )
    text(
        draw,
        (1145, 819),
        "CONFIGURATIONS  /  QUALITY SCORES  /  RESOURCE LOGS  /  REPORTS",
        24,
        IVORY,
        "medium",
        "ma",
    )
    footer(image, scene, t, duration, "SYSTEM ARCHITECTURE / ILLUSTRATION")


def mcp_scene(image, scene, t, duration):
    wide_title(image, scene, t)
    draw = ImageDraw.Draw(image)
    screenshot(
        image, "chat", (1160, 343, 648, 476), t, duration, "BUILT-IN CHAT / ACTUAL APPLICATION"
    )
    connector(draw, [(424, 429), (614, 429)], t, 0)
    connector(draw, [(992, 427), (1065, 427), (1065, 715), (992, 715)], t, 1)
    connector(draw, [(424, 621), (500, 621), (500, 459), (614, 459)], t, 2)
    node(draw, (96, 368, 328, 113), "Your MCP client", "Compatible assistant", title_size=29)
    node(draw, (96, 563, 328, 113), "Companion skill", "Discover tools + evidence", title_size=28)
    node(draw, (614, 368, 378, 119), "MCP interface", "Shared application tools", True, 31)
    node(draw, (614, 660, 378, 111), "LLM-Optimise", "Run workflows. Read results.", title_size=30)
    text(draw, (598, 512), "Local stdio", 25, TEAL, "medium")
    text(draw, (598, 552), "Remote HTTPS + OAuth", 25, IVORY, "medium")
    text(draw, (598, 590), "Requires remote gateway setup", 20, MUTED)
    wrapped(
        draw,
        (96, 732),
        "Use the built-in chat for guidance,\nor connect a compatible assistant.",
        430,
        26,
        1.4,
        MUTED,
        "regular",
    )
    footer(image, scene, t, duration, "CONNECTION OPTIONS / ILLUSTRATION + ACTUAL CHAT CAPTURE")


def evidence(image, scene, t, duration):
    wide_title(image, scene, t)
    draw = ImageDraw.Draw(image)
    rows = [
        ("SOFTWARE CI", "Linux  /  macOS  /  Windows"),
        ("MODEL WORKFLOWS", "Mac  /  Omen with WSL + CUDA"),
        ("INTEGRATIONS", "Docker  /  OpenRouter"),
        ("REPRODUCIBILITY", "Configuration  /  logs  /  exported evidence"),
    ]
    for i, (label, detail) in enumerate(rows):
        y = 358 + i * 112
        reveal = ease((t - i * 0.18) / 0.7)
        draw.line((96, y + 90, 1808, y + 90), fill=(39, 60, 86), width=1)
        tracked(draw, (115, y + 27), label, 19, mix(NAVY, TEAL, reveal), 2)
        text(draw, (570 + 25 * (1 - reveal), y + 7), detail, 39, mix(NAVY, IVORY, reveal), "medium")
    text(
        draw,
        (96, 837),
        "Hardware evidence is specific to the model, task and configuration tested.",
        27,
        MUTED,
    )
    footer(
        image, scene, t, duration, "VALIDATION SCOPE / SOFTWARE CHECKS AND RECORDED LIVE WORKFLOWS"
    )


def draw_scene(scene: dict, t: float, duration: float) -> Image.Image:
    """Compose one silent 1920x1080 RGB frame; captions/audio are added by root."""
    if not isinstance(scene, dict) or not 0 <= int(scene["id"]) <= 11:
        raise ValueError("scene must have id 0..11 and narration-scene heading/eyebrow")
    if not math.isfinite(t) or not math.isfinite(duration) or duration <= 0:
        raise ValueError("scene timing must be finite and duration positive")
    t = max(0.0, min(t, duration))
    identifier = int(scene["id"])
    image = background(t, identifier)
    if identifier in (0, 11):
        intro(image, scene, t, duration, outro=identifier == 11)
    else:
        brand(image, t)
        if identifier == 1:
            purpose(image, scene, t, duration)
        elif identifier == 5:
            architecture(image, scene, t, duration)
        elif identifier == 9:
            mcp_scene(image, scene, t, duration)
        elif identifier == 10:
            evidence(image, scene, t, duration)
        else:
            ui_scene(image, scene, t, duration)
    return image


def main():
    global PROOF_MODE
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--proof", action="store_true", help="Render 12 midpoint frames and a contact sheet"
    )
    parser.add_argument("--scene", type=int, help="Only render one scene ID")
    args = parser.parse_args()
    if not args.proof:
        parser.error("Use --proof, or import draw_scene from your render assembly script")
    PROOF_MODE = True
    scenes = json.loads((ROOT / "marketing/video/narration-scenes.json").read_text())
    selected = scenes if args.scene is None else [s for s in scenes if s["id"] == args.scene]
    SHOTS.mkdir(parents=True, exist_ok=True)
    thumbs = []
    for scene in selected:
        duration = float(scene.get("duration", 10))
        frame = draw_scene(scene, duration / 2, duration)
        output = SHOTS / f"proof-{scene['id']:02d}-{scene['name']}.png"
        frame.save(output)
        thumbs.append((scene, frame.resize((640, 360), Image.Resampling.LANCZOS)))
        print(output.relative_to(ROOT))
    if len(thumbs) > 1:
        columns = 3
        rows = math.ceil(len(thumbs) / columns)
        sheet = Image.new("RGB", (columns * 640, rows * 397), NAVY)
        draw = ImageDraw.Draw(sheet)
        for i, (scene, thumb) in enumerate(thumbs):
            x, y = i % columns * 640, i // columns * 397
            sheet.paste(thumb, (x, y))
            text(draw, (x + 15, y + 365), f"{scene['id']:02d} / {scene['name'].upper()}", 18, MUTED)
        sheet.save(SHOTS / "proof-contact-sheet.png")
        print("marketing/video/shots/proof-contact-sheet.png")


if __name__ == "__main__":
    main()
