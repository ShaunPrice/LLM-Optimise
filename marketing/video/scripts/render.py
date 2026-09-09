"""Render the authored film using Pillow, then mix voice and generated music with FFmpeg."""

import argparse
import json
import subprocess
import time
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont
from visuals import draw_scene, font

ROOT = Path(__file__).resolve().parents[1]
FONT = (
    ImageFont.truetype("/System/Library/Fonts/Avenir Next.ttc", 34)
    if Path("/System/Library/Fonts/Avenir Next.ttc").is_file()
    else font(34, "medium")
)


def frame(scene, t):
    image = draw_scene(scene, max(0, t), scene["duration"]).convert("RGB")
    for cue in scene["captions"]:
        if cue["start"] <= t < cue["end"]:
            overlay = Image.new("RGBA", image.size)
            draw = ImageDraw.Draw(overlay)
            box = draw.multiline_textbbox((0, 0), cue["text"], font=FONT, spacing=6, align="center")
            width, height = box[2] - box[0], box[3] - box[1]
            x, y = (1920 - width) / 2, 969 if "\n" in cue["text"] else 990
            draw.rounded_rectangle(
                (x - 26, y - 12, x + width + 26, y + height + 18), radius=12, fill=(4, 12, 23, 225)
            )
            draw.multiline_text(
                (x, y - box[1]), cue["text"], font=FONT, fill="white", spacing=6, align="center"
            )
            image = Image.alpha_composite(image.convert("RGBA"), overlay).convert("RGB")
    return image


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--audio-only", action="store_true")
    parser.add_argument("--proof", action="store_true")
    args = parser.parse_args()
    data = json.loads((ROOT / "timeline.json").read_text())
    out, scenes, fps, duration = ROOT / "output", data["scenes"], data["fps"], data["duration"]
    if args.proof:
        for s in scenes:
            frame(s, s["duration"] * 0.5).save(out / f"frame-{s['id']:02}.jpg", quality=92)
        return
    if not args.audio_only:
        for name in (
            "lab",
            "results",
            "workbench",
            "workbench-result",
            "routing",
            "develop",
            "training",
            "chat",
        ):
            if not (ROOT / f"shots/{name}.png").is_file():
                raise FileNotFoundError(f"Real application capture required: {name}")
        cmd = [
            "ffmpeg",
            "-y",
            "-v",
            "warning",
            "-f",
            "rawvideo",
            "-pix_fmt",
            "rgb24",
            "-s",
            "1920x1080",
            "-r",
            str(fps),
            "-i",
            "-",
            "-an",
            "-c:v",
            "libx264",
            "-preset",
            "fast",
            "-crf",
            "19",
            "-threads",
            "4",
            "-pix_fmt",
            "yuv420p",
            "-movflags",
            "+faststart",
            str(out / "picture.mp4"),
        ]
        proc = subprocess.Popen(cmd, stdin=subprocess.PIPE)
        started = time.monotonic()
        try:
            for n in range(round(duration * fps)):
                t = n / fps
                active = [s for s in scenes if s["start"] <= t < s["start"] + s["duration"]]
                current = active[-1]
                im = frame(current, t - current["start"])
                if len(active) == 2:
                    previous = active[0]
                    alpha = min(1, (t - current["start"]) / data["transition"])
                    im = Image.blend(frame(previous, t - previous["start"]), im, alpha)
                proc.stdin.write(im.tobytes())
                if n % (fps * 10) == 0:
                    print(
                        f"Rendered {t:.0f}/{duration:.1f}s · elapsed {time.monotonic() - started:.1f}s",
                        flush=True,
                    )
        finally:
            proc.stdin.close()
        if proc.wait():
            raise RuntimeError("Picture encoding failed")
    # Six-second overlap avoids a hard seam when extending the original score.
    subprocess.run(
        [
            "ffmpeg",
            "-y",
            "-v",
            "error",
            "-i",
            str(ROOT / "audio/music-original.flac"),
            "-filter_complex",
            "[0:a]asplit=2[a][b];[a][b]acrossfade=d=6:c1=tri:c2=tri",
            "-t",
            str(duration),
            "-ar",
            "48000",
            str(ROOT / "audio/music-bed.wav"),
        ],
        check=True,
    )
    mix = (
        f"[1:a]aresample=48000,volume=1.0,asplit=2[vo][side];"
        f"[2:a]aresample=48000,volume=0.12,afade=t=in:d=1.5,afade=t=out:st={duration - 3}:d=3[music];"
        "[music][side]sidechaincompress=threshold=0.025:ratio=5:attack=25:release=450[ducked];"
        "[vo][ducked]amix=inputs=2:duration=first:normalize=0,loudnorm=I=-14:TP=-1.5:LRA=9:print_format=json[mix]"
    )
    cmd = [
        "ffmpeg",
        "-y",
        "-hide_banner",
        "-i",
        str(out / "picture.mp4"),
        "-i",
        str(ROOT / "audio/narration-timeline.wav"),
        "-i",
        str(ROOT / "audio/music-bed.wav"),
        "-filter_complex_threads",
        "1",
        "-filter_complex",
        mix,
        "-map",
        "0:v",
        "-map",
        "[mix]",
        "-t",
        str(duration),
        "-c:v",
        "copy",
        "-c:a",
        "aac",
        "-b:a",
        "256k",
        "-ar",
        "48000",
        "-ac",
        "2",
        "-movflags",
        "+faststart",
        "-metadata",
        "title=LLM-Optimise | Small hardware. Specialised intelligence.",
        str(out / "LLM-Optimise-product-film.mp4"),
    ]
    with (out / "mix.log").open("w") as log:
        subprocess.run(cmd, stderr=log, check=True)
    print(out / "LLM-Optimise-product-film.mp4", flush=True)


if __name__ == "__main__":
    main()
