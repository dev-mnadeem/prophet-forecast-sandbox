"""Pinned loaders for the example series vendored in ``examples/``.

Every dataset is recorded with a SHA-256 digest.  Loading verifies the digest,
so a figure quoted in the README cannot silently drift because the underlying
CSV changed.  ``PLAYGROUND_SKIP_CHECKSUM=1`` relaxes the check for local
experiments; it is never relaxed in tests.
"""

from __future__ import annotations

import hashlib
import os
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent
EXAMPLES_DIR = REPO_ROOT / "examples"


class ChecksumError(RuntimeError):
    """Raised when a pinned dataset no longer matches its recorded digest."""


@dataclass(frozen=True)
class Dataset:
    """A pinned example series.

    Attributes
    ----------
    key:
        Short name used on the command line.
    filename:
        File under ``examples/``.
    sha256:
        Digest of the file as vendored, recorded with ``shasum -a 256``.
    freq:
        Pandas offset alias of the observation grid.
    seasonal_period:
        Number of observations in one dominant seasonal cycle, used by the
        seasonal-naive baseline.
    description:
        One line, for ``--list``.

    """

    key: str
    filename: str
    sha256: str
    freq: str
    seasonal_period: int
    description: str

    @property
    def path(self) -> Path:
        """Absolute path to the vendored CSV."""
        return EXAMPLES_DIR / self.filename


DATASETS: dict[str, Dataset] = {
    d.key: d
    for d in (
        Dataset(
            key="peyton",
            filename="example_wp_log_peyton_manning.csv",
            sha256="6b1383a6f458e317c5da0488abaf2cadccd7e80c84c7a543d084e1159d5bfa56",
            freq="D",
            seasonal_period=7,
            description="log daily Wikipedia page views for Peyton Manning, 2007-12-10..2016-01-20",
        ),
        Dataset(
            key="retail",
            filename="example_retail_sales.csv",
            sha256="6556f0683b78362c4bc4390b8feb66870beecd7189cc052e650ccac9b21cd65c",
            freq="MS",
            seasonal_period=12,
            description="US monthly retail sales, 1992-01..2016-05",
        ),
        Dataset(
            key="air",
            filename="example_air_passengers.csv",
            sha256="062d172e9b4f694dd798f4a3d41316a930c8c6bfd244bd6082193f6d6653a213",
            freq="MS",
            seasonal_period=12,
            description="Box & Jenkins monthly airline passengers, 1949-01..1960-12",
        ),
        Dataset(
            key="pedestrians",
            filename="example_pedestrians_covid.csv",
            sha256="e8971ec69aba730809d185ea2e0a24a56281f1a67b3a74ead01f0463c939ba20",
            freq="D",
            seasonal_period=7,
            description="Melbourne daily pedestrian counts spanning the 2020 COVID shock",
        ),
        Dataset(
            key="yosemite",
            filename="example_yosemite_temps.csv",
            sha256="c0ec9f2cb4bbf0bc53f7bfd2e39f88ae21e43b7b8912b2d1eb8185055f9510e2",
            freq="5min",
            seasonal_period=288,
            description="Yosemite Village 5-minute temperatures (sub-daily series)",
        ),
    )
}


def file_digest(path: Path) -> str:
    """SHA-256 of a file, streamed so large CSVs do not load into memory."""
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def load_dataset(key: str, *, verify: bool | None = None) -> pd.DataFrame:
    """Load a pinned dataset as a tidy, strictly increasing ``ds``/``y`` frame.

    The frame is sorted by ``ds`` and de-duplicated before it is returned.  That
    is not cosmetic: :func:`playground.splits.rolling_origin_splits` refuses
    unsorted input, because a shuffled time series is the cheapest way to
    introduce look-ahead bias.
    """
    try:
        spec = DATASETS[key]
    except KeyError:
        known = ", ".join(sorted(DATASETS))
        raise KeyError(f"unknown dataset {key!r}; known datasets: {known}") from None

    if not spec.path.exists():
        raise FileNotFoundError(f"missing vendored dataset: {spec.path}")

    if verify is None:
        verify = os.environ.get("PLAYGROUND_SKIP_CHECKSUM", "") not in ("1", "true", "True")
    if verify:
        actual = file_digest(spec.path)
        if actual != spec.sha256:
            raise ChecksumError(
                f"{spec.filename} has changed: expected sha256 {spec.sha256}, got {actual}. "
                "Any metric recorded against this dataset is now stale."
            )

    # example_air_passengers.csv uses classic-Mac CR line endings; the python
    # engine handles them where the default C engine does not.
    df = pd.read_csv(spec.path, engine="python")
    missing = {"ds", "y"} - set(df.columns)
    if missing:
        raise ValueError(f"{spec.filename} is missing column(s): {sorted(missing)}")

    df = df[["ds", "y"]].copy()
    df["ds"] = pd.to_datetime(df["ds"])
    df["y"] = pd.to_numeric(df["y"], errors="coerce")
    return (
        df.dropna(subset=["ds"])
        .drop_duplicates(subset=["ds"], keep="last")
        .sort_values("ds")
        .reset_index(drop=True)
    )


__all__ = ["DATASETS", "ChecksumError", "Dataset", "EXAMPLES_DIR", "file_digest", "load_dataset"]
