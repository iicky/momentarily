# python:3.13-slim (multi-arch index digest, pinned for reproducible builds).
FROM python@sha256:9d2e5553305c7c7b0097999bb17187c69b921ccd6bc9d40e4bb5ebe652c00285

# Minimal footprint: just httpx, no pip cache, no build tools needed for pure-Python deps.
WORKDIR /app

RUN pip install --no-cache-dir 'httpx==0.28.1'

COPY collector.py /app/collector.py

# Where appended data lives; bind-mounted in compose
ENV DATA_DIR=/data
VOLUME ["/data"]

# Run as non-root for safety
RUN useradd --create-home --uid 10001 collector \
    && mkdir -p /data \
    && chown -R collector:collector /data /app

USER collector

CMD ["python", "-u", "/app/collector.py"]
