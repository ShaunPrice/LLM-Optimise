"""Inspect the encoded deliverable, export review frames, and record media evidence."""

import hashlib
import json
import subprocess
from pathlib import Path

from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parents[1]


def main():
    output = ROOT / "output"
    movie = output / "LLM-Optimise-product-film.mp4"
    timeline = json.loads((ROOT / "timeline.json").read_text())
    probe = json.loads(
        subprocess.check_output(
            ["ffprobe", "-v", "error", "-show_streams", "-show_format", "-of", "json", str(movie)]
        )
    )
    video = next(s for s in probe["streams"] if s["codec_type"] == "video")
    audio = next(s for s in probe["streams"] if s["codec_type"] == "audio")
    assert (video["width"], video["height"], video["r_frame_rate"], video["codec_name"]) == (
        1920,
        1080,
        "30/1",
        "h264",
    )
    assert (
        audio["codec_name"] == "aac" and audio["sample_rate"] == "48000" and audio["channels"] == 2
    )
    assert abs(float(probe["format"]["duration"]) - timeline["duration"]) < 0.1
    frames = []
    for scene in timeline["scenes"]:
        cue = scene["captions"][0]
        at = scene["start"] + (cue["start"] + cue["end"]) / 2
        path = output / f"encoded-{scene['id']:02}.jpg"
        subprocess.run(
            [
                "ffmpeg",
                "-y",
                "-v",
                "error",
                "-ss",
                str(at),
                "-i",
                str(movie),
                "-frames:v",
                "1",
                "-q:v",
                "2",
                str(path),
            ],
            check=True,
        )
        frames.append(Image.open(path).convert("RGB"))
    sheet = Image.new("RGB", (1440, 4 * 296), "#0b1728")
    for i, frame in enumerate(frames):
        frame.thumbnail((480, 270))
        x, y = (i % 3) * 480, (i // 3) * 296
        sheet.paste(frame, (x, y))
        ImageDraw.Draw(sheet).text(
            (x + 10, y + 274), f"{i + 1:02} / {timeline['scenes'][i]['name']}", fill="white"
        )
    sheet.save(output / "encoded-contact-sheet.jpg", quality=94)
    # Poster comes from the encoded movie itself.
    subprocess.run(
        [
            "ffmpeg",
            "-y",
            "-v",
            "error",
            "-ss",
            "3",
            "-i",
            str(movie),
            "-frames:v",
            "1",
            "-q:v",
            "2",
            str(ROOT.parents[1] / "docs/assets/product-film-poster.jpg"),
        ],
        check=True,
    )
    measurement = subprocess.run(
        [
            "ffmpeg",
            "-hide_banner",
            "-i",
            str(movie),
            "-vn",
            "-af",
            "loudnorm=I=-14:TP=-1.5:LRA=9:print_format=json",
            "-f",
            "null",
            "-",
        ],
        capture_output=True,
        text=True,
        check=True,
    )
    loudness, _ = json.JSONDecoder().raw_decode(
        measurement.stderr[measurement.stderr.rfind("{\n") :]
    )
    assert -15 <= float(loudness["input_i"]) <= -13
    assert float(loudness["input_tp"]) <= -1.0
    report = {
        "film": movie.name,
        "sha256": hashlib.sha256(movie.read_bytes()).hexdigest(),
        "bytes": movie.stat().st_size,
        "duration_seconds": float(probe["format"]["duration"]),
        "video": {
            k: video[k] for k in ("codec_name", "width", "height", "r_frame_rate", "nb_frames")
        },
        "audio": {k: audio[k] for k in ("codec_name", "sample_rate", "channels")},
        "measured_lufs": float(loudness["input_i"]),
        "measured_true_peak_dbtp": float(loudness["input_tp"]),
        "caption_cues": sum(len(s["captions"]) for s in timeline["scenes"]),
        "actual_word_timestamps": timeline["word_count"],
        "scenes": len(timeline["scenes"]),
        "review_frames": "encoded-contact-sheet.jpg",
        "validation": "Encoded format, duration, captions in every scene, audio loudness and peak checks passed. Contact sheet requires visual review.",
    }
    (output / "quality-report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
