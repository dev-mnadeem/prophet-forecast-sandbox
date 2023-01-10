# syntax=docker/dockerfile:1
#
# Two stages, because building this package is expensive and the result is not.
#
# The vendored Prophet has no pure-Python install path: `pip install python/`
# downloads CmdStan (~1 GB of sources), compiles it, then compiles
# python/stan/prophet.stan into a native binary. That toolchain is needed once,
# to produce a wheel. The runtime stage installs that wheel and carries no
# compiler at all, which is both smaller and a smaller attack surface.
#
# Versions are pinned deliberately; see requirements-dev.txt for why.

# ---------------------------------------------------------------------------
# Stage 1 - build the Prophet wheel (compiles CmdStan and the Stan model)
# ---------------------------------------------------------------------------
FROM python:3.10-slim-bookworm AS builder

ENV PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PYTHONDONTWRITEBYTECODE=1

RUN apt-get update && apt-get install --no-install-recommends -y \
        build-essential \
        ca-certificates \
        curl \
        make \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /src

# Build requirements first, so the expensive layer caches independently of
# the source tree.
RUN python -m pip install --upgrade pip setuptools wheel \
    && python -m pip install "cmdstanpy>=1.0.4,<1.2" "numpy<2"

COPY python/ /src/python/

# Produces /wheels/prophet-*.whl containing prophet_model.bin and a pruned
# CmdStan tree, so the runtime stage never compiles anything.
RUN python -m pip wheel --no-deps --wheel-dir /wheels /src/python

# ---------------------------------------------------------------------------
# Stage 2 - runtime
# ---------------------------------------------------------------------------
FROM python:3.10-slim-bookworm AS runtime

ENV PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app

# Non-root from here on. Nothing this image does needs privilege.
RUN useradd --create-home --uid 10001 --shell /usr/sbin/nologin prophet

WORKDIR /app

COPY --from=builder /wheels /wheels
COPY requirements-dev.txt /app/requirements-dev.txt

RUN python -m pip install --upgrade pip \
    && python -m pip install /wheels/*.whl \
    && python -m pip install -r /app/requirements-dev.txt \
    && rm -rf /wheels

COPY --chown=prophet:prophet playground/ /app/playground/
COPY --chown=prophet:prophet examples/ /app/examples/
COPY --chown=prophet:prophet pyproject.toml /app/pyproject.toml
COPY --chown=prophet:prophet docker-healthcheck.py /app/docker-healthcheck.py

USER prophet

# A batch image, so "healthy" means the toolchain is intact: Prophet imports,
# the compiled Stan binary is present, and every pinned dataset still matches
# its recorded checksum.
HEALTHCHECK --interval=60s --timeout=30s --start-period=15s --retries=2 \
    CMD ["python", "/app/docker-healthcheck.py"]

CMD ["python", "-m", "playground", "backtest", "--dataset", "peyton", "--per-fold"]
