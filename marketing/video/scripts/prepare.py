"""Align scenes to actual narration words; assemble a lossless voice track and captions."""

import json
import re
import subprocess
import textwrap
import wave
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
FPS, RATE, TRANSITION = 30, 48000, 0.4


def normalise(text):
    return re.sub(r"[^a-z0-9]", "", text.lower())


def srt_time(seconds):
    ms = round(seconds * 1000)
    h, ms = divmod(ms, 3600000)
    m, ms = divmod(ms, 60000)
    s, ms = divmod(ms, 1000)
    return f"{h:02}:{m:02}:{s:02},{ms:03}"


def main():
    scenes = json.loads((ROOT / "narration-scenes.json").read_text())
    words = json.loads((ROOT / "audio/alignment.json").read_text())["words"]
    # Opening anchors are deliberately matched to real timestamps, never word ratios.
    anchors = [
        "Small hardware",
        "Explore what your",
        "Choose a model",
        "Run bounded experiments",
        "Explore context sizes",
        "One shared application",
        "Route agents using",
        "Describe a solution",
        "Use Soup to",
        "Ask the built",
        "Validated workflows span",
        "Open Source",
    ]
    starts, cursor = [], 0
    for anchor in anchors:
        parts = [normalise(s) for s in anchor.split()]
        hits = [
            i
            for i in range(cursor, len(words) - len(parts) + 1)
            if [normalise(w["word"]) for w in words[i : i + len(parts)]] == parts
        ]
        if len(hits) != 1:
            raise ValueError(f"Ambiguous/missing anchor: {anchor}: {hits}")
        starts.append(hits[0])
        cursor = hits[0] + len(parts)
    decoded = subprocess.check_output(
        [
            "ffmpeg",
            "-v",
            "error",
            "-i",
            str(ROOT / "audio/narration-full.mp3"),
            "-f",
            "f32le",
            "-ar",
            str(RATE),
            "-ac",
            "1",
            "-",
        ]
    )
    voice = np.frombuffer(decoded, dtype="<f4")
    cuts = (
        [0.0]
        + [(words[i - 1]["end"] + words[i]["start"]) / 2 for i in starts[1:]]
        + [len(voice) / RATE]
    )
    starts.append(len(words))
    timeline, captions, clock = [], [], 0.0
    for i, scene in enumerate(scenes):
        lead, tail = (1.2 if i == 0 else 0.35), (3.0 if i == len(scenes) - 1 else 0.55)
        duration = np.ceil((cuts[i + 1] - cuts[i] + lead + tail) * FPS) / FPS
        current = {
            **scene,
            "start": round(clock, 6),
            "duration": float(duration),
            "source_start": cuts[i],
            "source_end": cuts[i + 1],
            "voice_lead": lead,
            "captions": [],
        }
        group = []
        for j in range(starts[i], starts[i + 1]):
            w = dict(words[j])
            w["word"] = {
                "specialized": "specialised",
                "Optimize.": "Optimise.",
                "optimize.": "Optimise.",
                "quantize": "quantized",
                "rust": "Rust",
                "evaluation": "evaluations",
            }.get(w["word"], w["word"])
            group.append(w)
            line = " ".join(x["word"] for x in group).replace(" -", "-")
            if (
                len(group) >= 8
                or len(line) >= 60
                or w["word"].endswith(".")
                or j == starts[i + 1] - 1
            ):
                line = line.replace("LLM Optimise", "LLM-Optimise").replace("held out", "held-out")
                cue = {
                    "start": group[0]["start"] - cuts[i] + lead,
                    "end": min(duration - 0.2, group[-1]["end"] - cuts[i] + lead + 0.12),
                    "text": "\n".join(textwrap.wrap(line, width=48)),
                }
                current["captions"].append(cue)
                captions.append({**cue, "start": clock + cue["start"], "end": clock + cue["end"]})
                group = []
        timeline.append(current)
        clock += duration - (TRANSITION if i < len(scenes) - 1 else 0)
    output = np.zeros(round(clock * RATE), dtype=np.float32)
    for scene in timeline:
        piece = voice[round(scene["source_start"] * RATE) : round(scene["source_end"] * RATE)]
        offset = round((scene["start"] + scene["voice_lead"]) * RATE)
        output[offset : offset + len(piece)] += piece
    assert len(captions) > 35 and all(
        captions[i]["end"] <= captions[i + 1]["start"] + 0.15 for i in range(len(captions) - 1)
    )
    with wave.open(str(ROOT / "audio/narration-timeline.wav"), "wb") as f:
        f.setparams((1, 2, RATE, 0, "NONE", "not compressed"))
        f.writeframes((np.clip(output, -1, 1) * 32767).astype("<i2").tobytes())
    data = {
        "fps": FPS,
        "transition": TRANSITION,
        "duration": round(clock, 6),
        "word_count": len(words),
        "scenes": timeline,
    }
    (ROOT / "timeline.json").write_text(json.dumps(data, indent=2) + "\n")
    (ROOT / "output/captions.srt").write_text(
        "\n\n".join(
            f"{i + 1}\n{srt_time(c['start'])} --> {srt_time(c['end'])}\n{c['text']}"
            for i, c in enumerate(captions)
        )
        + "\n"
    )
    print(
        json.dumps(
            {
                "duration": clock,
                "words": len(words),
                "captions": len(captions),
                "scene_anchors": starts[:-1],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
