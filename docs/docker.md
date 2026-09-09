# Docker

Docker has two distinct roles: packaging the lab application and running a disposable copy of a development project under resource limits. The application image contains the lightweight GUI/CLI. It does not contain llama.cpp, model weights, accelerator drivers, Soup, the Docker CLI, or a mounted Docker socket.

## Run the application image

From the repository directory, with Docker Engine or Docker Desktop running:

```bash
docker compose up --build
```

Open [the local lab](http://127.0.0.1:8765). The process listens on `0.0.0.0` inside the container so Docker can reach it, while Compose publishes port 8765 only on host `127.0.0.1`. The Compose application has a 512 MiB memory limit, one CPU, 128-process limit, a read-only root filesystem and a temporary `/tmp`. Workspace data uses the named `lab-data` volume.

```bash
docker compose down
```

Stopping with this command retains the named volume. Deleting that volume removes its catalogue, projects and run artifacts; export anything you need before deliberately removing it.

For a one-off CLI check:

```bash
docker build -t llm-optimise .
docker run --rm llm-optimise --help
docker run --rm llm-optimise doctor
```

Hardware information from inside a container reflects its environment and available interfaces. It does not establish host GPU access or prove the native runtime's performance. Containerised macOS/Linux desktop workloads also include a VM boundary where Docker Desktop uses one.

## Connect the app to an inference server

An endpoint at `127.0.0.1` from inside the application container refers to that container. It does not refer to the host. Compose provides `host.docker.internal` through the host gateway mapping, but the host inference service must listen on an interface reachable through that gateway.

Configure the endpoint using its reachable private address and actual API prefix, for example `http://host.docker.internal:8080/v1` when your server is reachable there. Confirm the network path with your Docker/runtime configuration. A host server bound exclusively to loopback may not be reachable from a container; avoid exposing the lab publicly as a workaround.

For native Metal, CUDA or another host accelerator, the straightforward setup is the lab and model runtime on the host. The container app can call an existing endpoint, but does not manage a native host process or measure its process memory.

For cloud requests, pass the needed environment variable to the app container using your normal Docker secret/environment workflow. The committed Compose file does not forward API keys. Retain variable names in the model catalogue and avoid committing credential-bearing override files.

## Test or build a development project

Run this feature from the **host** CLI or host GUI, with Docker accessible. The small application container deliberately has no Docker socket. The host feature creates a copy of the project and mounts only that copy into its execution container.

```bash
llm-optimise container --project projects/temperature-tool --output runs/temperature-tests --runtime python --action test --memory-mib 512 --cpus 1 --timeout-s 60
```

Select `--pull` for an initial run if the chosen image is absent:

```bash
llm-optimise container --project projects/temperature-tool --output runs/temperature-tests-first --runtime python --action test --pull
```

An image pull is a host Docker operation and can access the registry even though the resulting test container has network disabled. `--network` independently permits network access from the executing project container. Neither option is enabled by default.

| Runtime | Default image | Test action | Build action |
| --- | --- | --- | --- |
| Python | `python:3.12-slim-bookworm` | `python -m unittest discover -s tests -v` | `python -m compileall -q .` |
| Node.js | `node:22-slim` | `node --test` | `npm run build` |

Python “build” checks compilation; it does not produce a wheel or run tests. Node build requires a suitable `build` script. Dependencies are not installed automatically. Projects requiring third-party packages need a prepared compatible image; the CLI accepts `--image` for this purpose. Enabling network alone does not run `pip install` or `npm install`.

```bash
llm-optimise container --project projects/web-tool --output runs/web-build --runtime node --action build --image my-prepared-node-image --memory-mib 1024 --cpus 2 --timeout-s 120
```

Replace that image reference with an image you have prepared. The recorded image ID identifies what was executed even when a mutable tag was used.

## Execution boundaries and artifacts

Containers run as a non-root user with all capabilities dropped, no new privileges, a read-only root filesystem and a bounded temporary filesystem. Default network is `none`; default memory/swap allowance is 512 MiB and CPU allowance is one. The memory and CPU limits are enforced by the Docker engine, unlike inference's sampled cooperative memory checks.

The output directory must be outside the source project and fresh for the run. Project copying skips hidden files/directories, symlinks, `node_modules`, `models`, `runs`, `__pycache__`, `dist` and `build`. Context is bounded to 256 MiB and 10,000 files. Check these exclusions when a project relies on hidden configuration or dependencies that were present on the host.

The writable `/workspace` mount is the disposable copy under the output directory. Changes made there do not automatically update the original project. The source project is not mounted, and no host Docker socket is supplied to generated code.

Artifacts include `stdout.log`, `stderr.log`, the copied `workspace/`, and execution metadata. CLI execution also writes `result.json`; GUI operation records live under `runs/jobs/`. Metadata records exit code, timeout/cancellation state, image and image ID, command, resource limits and elapsed time. Inspect logs for the number of tests and meaningful assertions: exit zero alone can accompany a test runner that found no tests.

Timeouts and cancellation remove the uniquely named execution container. Output is bounded, and excessive logs also stop execution. The timeout is for the project execution phase; an explicitly authorised image pull has its own timeout and can take additional time.

The feature supports 64–65,536 MiB, positive CPU values up to 64, and positive execution timeouts up to 3,600 seconds. These are application bounds, not recommendations to allocate all available resources.
