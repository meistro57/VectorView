# syntax=docker/dockerfile:1

FROM golang:1.22-bookworm AS build

WORKDIR /src

COPY go.mod go.sum ./
RUN go mod download

COPY *.go ./
COPY pca_gpu.py ./
COPY static ./static

RUN CGO_ENABLED=0 go build -trimpath -ldflags="-s -w" -o /out/vectorview .

FROM python:3.12-slim AS runtime

ARG INSTALL_EXTRAS=""

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    VECTORVIEW_PORT=7433 \
    VECTORVIEW_CACHE_DIR=/app/.cache/projections \
    VECTORVIEW_PYTHON=/usr/local/bin/python3

COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt $INSTALL_EXTRAS

WORKDIR /app

COPY --from=build /out/vectorview /app/vectorview
COPY pca_gpu.py /app/pca_gpu.py

RUN useradd --system --create-home --uid 10001 vectorview \
    && mkdir -p /app/.cache/projections \
    && chown -R vectorview:vectorview /app

USER vectorview

EXPOSE 7433

HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD python3 -c "import urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://127.0.0.1:7433/healthz', timeout=3).status == 200 else 1)"

ENTRYPOINT ["/app/vectorview"]
