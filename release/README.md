# Installer release procedure

The [installer workflow](../.github/workflows/installers.yml) is started manually from `main`. It builds and validates packages, then uploads accepted artifacts for review. It has read-only repository permissions and does not create a GitHub Release or publish an installer automatically.

## Target matrix

| Platform | Architecture | GitHub runner | Packages |
|---|---|---|---|
| macOS | ARM64 | `macos-15` | DMG |
| macOS | x64 | `macos-15-intel` | DMG |
| Windows | x64 | `windows-latest` | Per-user NSIS setup executable |
| Linux | x64 | `ubuntu-22.04` | Debian package and AppImage |
| Linux | ARM64 | `ubuntu-24.04-arm` | Debian package and AppImage |

These runner labels are listed in [GitHub's runner reference](https://docs.github.com/en/actions/reference/runners/github-hosted-runners). ARM AppImages are built on ARM runners, following [Tauri's native-build requirement](https://v2.tauri.app/distribute/appimage/#appimages-for-arm-based-devices). Linux ARM64 uses a newer system-library baseline than x64; disclose that in the release notes.

## Prepare and validate

1. Set matching application and desktop versions. Keep the packaging command and artifact name in the workflow aligned with that version.
2. Review the pinned Python archive references, dependency hashes and runtime manifest produced by `scripts/build-bundled-runtime.py`. Preserve the interpreter and dependency licenses in the complete runtime resource tree.
3. Run normal application CI. Start **Build and validate installers** on the intended `main` commit.
4. Require a successful package verification step for each platform you intend to publish. Inspect its JSON evidence and the exact package filenames/checksums. A successful compile alone is insufficient.
5. Download the accepted artifacts from the exact intended source commit. Require `cli_and_mcp_launchers_passed: true`: verification must assert meaningful help output from both the supplied launchers and their Python entry points. Confirm that the runtime relocates inside the installed package and the native smoke reports the bundled interpreter, local service checks and shutdown. Review Windows uninstall/workspace-preservation evidence and both Linux package routes.
6. Publish only reviewed packages to the matching experimental release, with their hashes and validation reports. Include the [installation guide](../docs/install.md), unsigned-package limitations and tested operating-system baselines.

Each accepted Actions artifact is named `llm-optimise-0.1.1-PLATFORM-ARCH` and contains `dist/installers/` output. A failed target can upload a separate `installer-failure-PLATFORM-ARCH` diagnostic artifact; that is not an accepted installer. Check all targeted jobs before representing a release as validated across platforms.

The installer verification script performs real OS installation actions on a disposable CI host. It copies a mounted Mac application to a fresh location, installs and uninstalls NSIS, or installs a `.deb` and extracts an AppImage. `desktop/validate_installed.py` observes the real webview and bundled interpreter using fresh preferences. Linux runs inside Xvfb and a temporary D-Bus session.

The [independent source review](installer-review.json) records the Opus 5 findings and their integration assessment. Installer acceptance still requires the reports from the actual packages.

The [0.1.1 release manifest](validation/v0.1.1/installer-release.json) records the accepted seven packages, exact build source, CI runs and local Mac acceptance. Its adjacent reports contain the individual checks. [Linux packaging diagnostics](linuxdeploy-review.json) explain the Tk pruning and RELR compatibility fixes; those earlier diagnostic candidates are separate from the final release assets.

The workflow does not sign with a Developer ID, notarize with Apple, sign Windows files with Authenticode, or provision an updater. Do not claim trusted-publisher status from an ad-hoc signature or checksum. Adding signing later requires explicit identity configuration and fresh validation of the final signed artifacts.

## Local reproduction

See [desktop build instructions](../desktop/README.md). Use the same target OS/architecture, Rust and Node versions, locked Python dependencies and bundle configuration as CI. macOS DMGs use the custom `hdiutil` packaging route; do not replace it with Finder automation.

Generated runtimes, local profiles, installers and extracted packages stay out of Git. Keep concise reviewed validation evidence in source if useful. Never commit a workspace, models, provider credentials or raw private prompts as packaging fixtures.
