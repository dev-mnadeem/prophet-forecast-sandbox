# syntax=docker/dockerfile:1
#
# One stage, no compiler. The prophet wheel from PyPI already contains the
# compiled Stan model and a pruned CmdStan, so the image only installs pinned
# wheels. The vendored python/ tree is reference material and is not copied
# in (see .dockerignore); building it would mean downloading and compiling
# CmdStan, which is what the previous two-stage file did.

FROM python:3.12-slim

ENV PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app \
    MPLBACKEND=Agg \
    MPLCONFIGDIR=/tmp/matplotlib

# Non-root from here on. Nothing this image does needs privilege.
RUN useradd --create-home --uid 10001 --shell /usr/sbin/nologin playground

WORKDIR /app

# Dependencies first, so source edits do not invalidate the install layer.
# The dev pins (pytest, ruff) are included so the same image runs the suite.
COPY requirements.txt requirements-dev.txt /app/
RUN python -m pip install -r /app/requirements-dev.txt

COPY --chown=playground:playground pyproject.toml docker-healthcheck.py /app/
COPY --chown=playground:playground examples/ /app/examples/
COPY --chown=playground:playground playground/ /app/playground/

USER playground

# A batch image, so "healthy" means the toolchain is intact: Prophet imports,
# its compiled Stan model is present, and every pinned dataset still matches
# its recorded checksum.
HEALTHCHECK --interval=60s --timeout=30s --start-period=15s --retries=2 \
    CMD ["python", "/app/docker-healthcheck.py"]

CMD ["python", "-m", "playground", "backtest", "--dataset", "peyton", "--per-fold"]
