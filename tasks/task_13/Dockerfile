FROM python:3.11-slim

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

RUN apt-get update && apt-get install -y --no-install-recommends \
    bash \
    build-essential \
    ca-certificates \
    curl \
    git \
    jq \
    procps \
    ripgrep \
    tini \
    unzip \
    wget \
 && rm -rf /var/lib/apt/lists/*

RUN python -m pip install --no-cache-dir \
    duckdb \
    ipykernel \
    jupyterlab \
    matplotlib \
    numpy \
    openpyxl \
    pandas \
    pillow \
    pyarrow \
    scikit-learn \
    seaborn

WORKDIR /workspace

COPY data/ /workspace/data/

RUN mkdir -p /logs/artifacts

CMD ["bash", "-lc", "sleep infinity"]
