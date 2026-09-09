# Local workspace security

The GUI is a single-user local tool. It binds to loopback by default, validates Host and Origin, uses a per-process CSRF token, and serves no remote scripts. Do not expose it as a public or multi-user service. `--host 0.0.0.0` is for container port publication bound to host loopback.

Model settings store environment variable **names** for credentials. Keys are read only by the provider adapter and are not returned by the API. Private prompts, responses, projects and runtime logs can be present in your workspace; keep `runs/`, `projects/`, `.llm-optimise/` and `.env` files private.

Code proposals require review and an explicit apply operation. Context changes during generation and file changes after preview invalidate a proposal. Portable relative paths, size limits, symlink rejection and atomic file replacement reduce accidental overwrites. This is a trusted local workspace, not a security boundary against another process modifying files concurrently.

Generated code runs only when you request a Docker build or test. The runner uses a disposable project copy, resource limits, an unprivileged user, a read-only container root filesystem, no Docker socket and no network by default. A container is not a complete sandbox against hostile code or kernel vulnerabilities. Do not grant it secrets or privileges. The application container intentionally cannot control the host Docker engine.

Local model endpoints must resolve to private or loopback addresses; HTTP redirects are refused. This relies on your trusted local DNS/network. Cloud requests require explicit registry entries, HTTPS and an environment credential. Routing has no automatic provider fallback. A configured cost budget is a preflight estimate, not a provider-enforced spending cap.

Report vulnerabilities privately to the repository owner. Include reproduction steps with credentials and private prompts removed.
