FROM python:3.13-slim

# Minimal footprint: just httpx, no pip cache, no build tools needed for pure-Python deps.
WORKDIR /app

RUN pip install --no-cache-dir 'httpx>=0.27'

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
