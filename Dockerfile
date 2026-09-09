# Native llama.cpp accelerators are selected on the host; this image runs the GUI/CLI.
FROM python:3.12-slim-bookworm
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
WORKDIR /app
COPY pyproject.toml README.md ./
COPY src ./src
RUN pip install --no-cache-dir . && useradd --uid 10001 --create-home lab && mkdir /workspace && chown lab:lab /workspace
USER lab
WORKDIR /workspace
EXPOSE 8765
HEALTHCHECK --interval=30s --timeout=5s --start-period=10s CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8765/api/state',timeout=3)"
ENTRYPOINT ["llm-optimise"]
CMD ["ui", "--workspace", "/workspace", "--host", "0.0.0.0", "--port", "8765"]
