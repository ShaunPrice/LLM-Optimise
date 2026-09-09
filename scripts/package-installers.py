"""Package native Tauri outputs using consistent platform names and SHA-256 manifests."""

from __future__ import annotations

import argparse
import hashlib
import json
import plistlib
import shutil
import subprocess
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BUNDLES = ROOT / "desktop/src-tauri/target/release/bundle"


def sha256(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def mac_image(output, name):
    app = BUNDLES / "macos/LLM-Optimise.app"
    if not (app / "Contents/Resources/runtime/manifest.json").is_file():
        raise ValueError("The app is missing its bundled runtime")
    output.mkdir(parents=True, exist_ok=True)
    image = output / (name + ".dmg")
    with tempfile.TemporaryDirectory(prefix="dmg-stage-", dir=output.parent) as temp:
        stage = Path(temp)
        shutil.copytree(app, stage / app.name, symlinks=True)
        (stage / "Applications").symlink_to("/Applications", target_is_directory=True)
        subprocess.run(
            [
                "hdiutil",
                "create",
                "-volname",
                "LLM-Optimise",
                "-srcfolder",
                str(stage),
                "-format",
                "UDZO",
                "-ov",
                str(image),
            ],
            check=True,
        )
    subprocess.run(["hdiutil", "verify", str(image)], check=True)
    return image


def unique(pattern):
    paths = sorted(BUNDLES.glob(pattern))
    if len(paths) != 1:
        raise ValueError(
            f"Expected one {pattern} bundle, found {len(paths)}. Clean stale bundle outputs before building."
        )
    return paths[0]


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--platform", choices=("macos", "windows", "linux"), required=True)
    parser.add_argument("--arch", choices=("arm64", "x64"), required=True)
    parser.add_argument("--version", required=True)
    parser.add_argument("--output", type=Path, default=ROOT / "dist/installers")
    args = parser.parse_args(argv)
    configuration = json.loads((ROOT / "desktop/src-tauri/tauri.conf.json").read_text())
    if configuration["version"] != args.version:
        raise ValueError("Requested package version differs from the desktop app")
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    name = f"LLM-Optimise-{args.version}-{args.platform}-{args.arch}"
    if args.platform == "macos":
        with (BUNDLES / "macos/LLM-Optimise.app/Contents/Info.plist").open("rb") as stream:
            if plistlib.load(stream)["CFBundleShortVersionString"] != args.version:
                raise ValueError("Built macOS app version is stale")
        artifacts = [mac_image(output, name)]
    else:
        patterns = (
            [("nsis/*-setup.exe", "-setup.exe")]
            if args.platform == "windows"
            else [("deb/*.deb", ".deb"), ("appimage/*.AppImage", ".AppImage")]
        )
        artifacts = []
        for pattern, suffix in patterns:
            source = unique(pattern)
            target = output / (name + suffix)
            shutil.copy2(source, target)
            artifacts.append(target)
    records = [{"file": p.name, "bytes": p.stat().st_size, "sha256": sha256(p)} for p in artifacts]
    (output / f"SHA256SUMS-{args.platform}-{args.arch}.txt").write_text(
        "".join(f"{r['sha256']}  {r['file']}\n" for r in records)
    )
    report = {
        "version": args.version,
        "platform": args.platform,
        "architecture": args.arch,
        "artifacts": records,
        "python_bundled": True,
        "mcp_bundled": True,
        "distribution_signed": False,
        "notarized": False,
        "acceptance": "pending installed/native validation",
    }
    (output / f"package-{args.platform}-{args.arch}.json").write_text(
        json.dumps(report, indent=2) + "\n"
    )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
