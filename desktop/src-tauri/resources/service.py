"""Managed desktop sidecar using the user's selected Python installation."""

import argparse
import json
import signal
import sys
import threading


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--workspace", required=True)
    args = parser.parse_args()
    try:
        from llm_optimise.server import make_server
    except ImportError as exc:
        print(
            json.dumps(
                {
                    "error": "This Python environment needs llm-optimise installed. Run: <selected-python> -m pip install /path/to/LLM-Optimise",
                    "detail": str(exc),
                }
            ),
            flush=True,
        )
        return 2
    stop = threading.Event()
    for sig in (signal.SIGINT, signal.SIGTERM):
        signal.signal(sig, lambda *_: stop.set())
    server = make_server(args.workspace, 0, "127.0.0.1")
    http = threading.Thread(target=server.serve_forever, daemon=True)
    http.start()

    def control():
        for line in sys.stdin:
            if line.strip() == "stop":
                break
        stop.set()

    threading.Thread(target=control, daemon=True).start()
    print(
        json.dumps(
            {
                "ready": True,
                "url": f"http://127.0.0.1:{server.server_address[1]}",
                "workspace": args.workspace,
            }
        ),
        flush=True,
    )
    stop.wait()
    for job in list(server.app.jobs.values()):
        if isinstance(job, dict) and job.get("cancel"):
            job["cancel"].set()
    # Support the existing core and the lifecycle-aware integration.
    for name in ("managed", "lifecycle", "model_manager", "local"):
        owned = getattr(server.app, name, None)
        if owned is not None and hasattr(owned, "close"):
            owned.close()
    server.shutdown()
    server.server_close()
    http.join(timeout=3)
    return 0


if __name__ == "__main__":
    sys.exit(main())
