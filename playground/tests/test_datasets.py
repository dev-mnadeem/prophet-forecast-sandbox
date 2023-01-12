"""The pinned datasets: checksums, tidiness, and the loader's contract."""

from __future__ import annotations

import pandas as pd
import pytest

from playground.datasets import DATASETS, ChecksumError, file_digest, load_dataset


@pytest.mark.parametrize("key", sorted(DATASETS))
def test_vendored_file_matches_its_recorded_digest(key):
    """If this fails, every metric recorded against that dataset is stale."""
    spec = DATASETS[key]
    assert spec.path.exists(), f"missing {spec.path}"
    assert file_digest(spec.path) == spec.sha256


@pytest.mark.parametrize("key", sorted(DATASETS))
def test_loader_returns_a_tidy_strictly_increasing_series(key):
    df = load_dataset(key)
    assert list(df.columns) == ["ds", "y"]
    assert len(df) > 0
    assert pd.api.types.is_datetime64_any_dtype(df["ds"])
    assert pd.api.types.is_numeric_dtype(df["y"])
    assert df["ds"].is_monotonic_increasing
    assert not df["ds"].duplicated().any()


def test_air_passengers_parses_despite_cr_line_endings():
    """example_air_passengers.csv uses classic-Mac CR endings; regression guard."""
    df = load_dataset("air")
    assert len(df) == 144
    assert df["y"].iloc[0] == 112
    assert df["y"].iloc[-1] == 432


def test_unknown_dataset_lists_the_known_ones():
    with pytest.raises(KeyError, match="known datasets"):
        load_dataset("nasdaq")


def test_checksum_failure_is_loud(tmp_path, monkeypatch):
    """Corrupt the file on disk and confirm the loader refuses it."""
    spec = DATASETS["retail"]
    corrupted = tmp_path / spec.filename
    corrupted.write_text(spec.path.read_text() + "2099-01-01,999999\n")
    monkeypatch.setattr(type(spec), "path", property(lambda self: corrupted))
    with pytest.raises(ChecksumError, match="has changed"):
        load_dataset("retail")
