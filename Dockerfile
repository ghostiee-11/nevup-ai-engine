FROM python:3.12-slim

ENV PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1
WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    curl build-essential libpq-dev \
    && rm -rf /var/lib/apt/lists/*

COPY pyproject.toml ./
RUN pip install --no-cache-dir -e .

COPY app/ ./app/
COPY alembic/ ./alembic/
COPY alembic.ini ./
COPY scripts/ ./scripts/
COPY entrypoint.sh ./
RUN chmod +x entrypoint.sh

# Bake the seed dataset into the image so deploys (Render, Fly, etc.) don't need a volume mount.
RUN mkdir -p /data
COPY nevup_seed_dataset.json /data/nevup_seed_dataset.json

EXPOSE 8000
CMD ["./entrypoint.sh"]
