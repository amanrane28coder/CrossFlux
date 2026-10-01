# Reproducible research/demo image. Its default command is synthetic backtest
# mode and never submits exchange orders.
FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    MPLCONFIGDIR=/tmp/matplotlib \
    CROSSFLUX_LOG_FORMAT=json \
    CROSSFLUX_LOG_LEVEL=INFO

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential cmake libboost-all-dev libssl-dev nlohmann-json3-dev \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt pyproject.toml ./
RUN python -m pip install --no-cache-dir -r requirements.txt

COPY . .
RUN cmake -S . -B /tmp/crossflux-build \
        -DCROSSFLUX_OUTPUT_DIRECTORY=/opt/crossflux/artifacts \
        -Dpybind11_DIR="$(python -m pybind11 --cmakedir)" \
    && cmake --build /tmp/crossflux-build --parallel 2

ENV PYTHONPATH=/opt/crossflux/artifacts:/app

CMD ["python", "scripts/run_backtest.py", "--synthetic"]
