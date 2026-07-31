FROM python:3.11-slim AS runtime

WORKDIR /app

ENV PIP_NO_CACHE_DIR=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

COPY pyproject.toml README.md ./
COPY src ./src
RUN pip install ".[inference]"

# Models are intentionally not baked into the image. Mount ./artifacts at runtime.
RUN mkdir -p /app/artifacts

ENV PYTHONUNBUFFERED=1
ENTRYPOINT ["sentiment-analyzer"]
CMD ["predict"]
