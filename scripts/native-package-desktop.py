"""Package and inspect the built macOS app without Finder automation."""

import hashlib
import json
import shutil
import subprocess
from pathlib import Path

root = Path(__file__).resolve().parents[1]
artifact = root / "desktop/src-tauri/target/release/bundle/macos/LLM-Optimise.app"
output = root / "desktop/profile-local"
stage = output / "dmg-root"
stage.mkdir(parents=True, exist_ok=True)
shutil.copytree(artifact, stage / artifact.name, dirs_exist_ok=True, symlinks=True)
link = stage / "Applications"
if not link.exists():
    link.symlink_to("/Applications", target_is_directory=True)
image = output / "LLM-Optimise-macos-arm64.dmg"
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
mount = output / "dmg-inspection"
mount.mkdir(exist_ok=True)
attached = False
try:
    subprocess.run(
        ["hdiutil", "attach", "-readonly", "-nobrowse", "-mountpoint", str(mount), str(image)],
        check=True,
    )
    attached = True
    binary = mount / "LLM-Optimise.app/Contents/MacOS/llm-optimise-desktop"
    assert binary.is_file()
    assert (mount / "Applications").is_symlink()
    assert (mount / "LLM-Optimise.app/Contents/Resources/service.py").is_file()
finally:
    if attached:
        subprocess.run(["hdiutil", "detach", str(mount)], check=True)
result = {
    "artifact": image.name,
    "bytes": image.stat().st_size,
    "sha256": hashlib.sha256(image.read_bytes()).hexdigest(),
    "disk_image_checksum_verified": True,
    "mounted_layout_verified": True,
    "contains_app_bundle": True,
    "contains_applications_shortcut": True,
    "contains_python_service_bridge": True,
    "signed_for_distribution": False,
    "notarized": False,
}
(output / "package-validation.json").write_text(json.dumps(result, indent=2))
print(json.dumps(result, indent=2))
