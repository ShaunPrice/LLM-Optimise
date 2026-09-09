"""Fail closed on unpinned crypto builds and host-library/deployment leakage."""

import importlib.util
import json
import sys
from pathlib import Path

import pytest

pytestmark = pytest.mark.skipif(
    sys.version_info < (3, 12), reason="Installer builder uses Python3.12+"
)
ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location(
    "crypto_builder", ROOT / "scripts/build-bundled-runtime.py"
)
builder = importlib.util.module_from_spec(spec)
spec.loader.exec_module(builder)


def test_locally_built_wheel_replaces_only_its_matching_locked_dependency(tmp_path):
    wheel = tmp_path / "space here" / "cryptography-50.0.1-cp39-abi3-macosx_12_0_x86_64.whl"
    wheel.parent.mkdir()
    wheel.write_bytes(b"measured wheel fixture")
    common = (ROOT / "packaging/runtime-requirements.txt").read_text()
    patched = builder.replace_cryptography_requirement(common, wheel, "50.0.1")
    assert wheel.resolve().as_uri() in patched
    assert "--hash=sha256:" + builder.digest(wheel) in patched
    assert "cryptography==50.0.1" not in patched
    assert patched.split("cryptography @")[0] == common.split("cryptography==")[0]
    with pytest.raises(ValueError, match="version differs"):
        builder.replace_cryptography_requirement(common, wheel, "48.0.1")


def test_static_crypto_linkage_accepts_only_target_arch_and_macos12_system_libraries():
    libraries = "_rust.abi3.so:\n\t/usr/lib/libSystem.B.dylib (compatibility version 1.0.0)\n"
    commands = "Load command 1\n      cmd LC_BUILD_VERSION\n  cmdsize 32\n platform 1\n    minos 12.0\n      sdk 26.0\n"
    assert builder.macos_linkage(libraries, commands, "x86_64", "x86_64", "12.0")["minimum_os"] == [
        "12.0"
    ]
    identity = "@rpath/cryptography.hazmat.bindings._rust.abi3.so"
    own_identity = (
        "Load command 2\n      cmd LC_ID_DYLIB\n  cmdsize 80\n     name "
        + identity
        + " (offset 24)\n"
    )
    assert builder.macos_linkage(
        libraries + "\t" + identity + " (compatibility version 0.0.0)\n",
        commands + own_identity,
        "x86_64",
        "x86_64",
        "12.0",
    )["install_names"] == [identity]
    for dependency in (
        "/usr/local/opt/openssl/lib/libcrypto.3.dylib",
        "@rpath/libssl.4.dylib",
        "/usr/lib/libcrypto.dylib",
    ):
        with pytest.raises(ValueError, match="static"):
            builder.macos_linkage(
                libraries + "\t" + dependency + " (compatibility version 1.0.0)\n",
                commands,
                "x86_64",
                "x86_64",
                "12.0",
            )
    with pytest.raises(ValueError, match="architecture"):
        builder.macos_linkage(libraries, commands, "arm64", "x86_64", "12.0")
    with pytest.raises(ValueError, match="deployment target"):
        builder.macos_linkage(
            libraries, commands.replace("minos 12.0", "minos 15.0"), "x86_64", "x86_64", "12.0"
        )
    with pytest.raises(ValueError, match="deployment target"):
        builder.macos_linkage(libraries, "", "x86_64", "x86_64", "12.0")


def test_source_pins_keep_current_crypto_and_openssl_security_releases():
    pinned = json.loads((ROOT / "packaging/cryptography-source.json").read_text())
    assert pinned["cryptography_version"] == "50.0.1"
    assert pinned["openssl_version"] == "4.0.2"
    assert pinned["macos_deployment_target"] == "12.0"
    for component in ("cryptography", "openssl"):
        assert len(pinned[component]["sha256"]) == 64
        assert pinned[component]["bytes"] > 0
