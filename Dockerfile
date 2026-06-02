FROM python:3.11-slim

ENV DEBIAN_FRONTEND=noninteractive
ENV TZ="Europe/Istanbul"

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends curl && rm -rf /var/lib/apt/lists/*

RUN --mount=type=cache,target=/root/.cache/pip \
    pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu

COPY requirements.txt .
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install -r requirements.txt

COPY app/ ./app/
COPY configs/ ./configs/

HEALTHCHECK --interval=30s --timeout=5s --start-period=60s \
    CMD curl --fail http://localhost:8000/health || exit 1

EXPOSE 8000

CMD ["python", "-m", "app.main"]
