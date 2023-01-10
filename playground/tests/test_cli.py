"""The command line, driven in-process with the instant baselines only."""

from __future__ import annotations

import pytest

from playground.cli import main
from playground.forecasters import NaiveLastValue, SeasonalNaive
from playground.plotting import plot_last_fold

PNG_MAGIC = b"\x89PNG\r\n\x1a\n"


def test_datasets_verifies_every_checksum(capsys):
    assert main(["datasets"]) == 0
    out = capsys.readouterr().out
    assert out.count(" OK ") == 5
    assert "FAIL" not in out


def test_backtest_prints_one_summary_row_per_model(capsys):
    argv = "backtest --dataset air --models naive seasonal_naive --horizon 12 --initial 96"
    status = main([*argv.split(), "--max-folds", "3"])
    assert status == 0
    out = capsys.readouterr().out
    rows = [line for line in out.splitlines() if line.startswith(("naive ", "seasonal_naive "))]
    assert len(rows) == 2
    # Monthly airline traffic is dominated by its yearly cycle.
    assert "lowest mean MAE across identical folds: seasonal_naive" in out


def test_plot_writes_a_png(tmp_path, capsys):
    target = tmp_path / "nested" / "air.png"
    argv = "plot --dataset air --models naive seasonal_naive --horizon 12 --initial 96"
    status = main([*argv.split(), "--out", str(target)])
    assert status == 0
    assert target.read_bytes()[:8] == PNG_MAGIC
    assert str(target) in capsys.readouterr().out


def test_plot_draws_history_actuals_and_each_model(synthetic_daily):
    fig = plot_last_fold(
        synthetic_daily,
        {"naive": NaiveLastValue, "seasonal_naive": lambda: SeasonalNaive(period=7)},
        horizon=10,
        initial=60,
        context=20,
        title="synthetic",
    )
    (ax,) = fig.axes
    labels = [line.get_label() for line in ax.get_lines()]
    assert labels[:2] == ["history", "actual"]
    assert {"naive", "seasonal_naive"} <= set(labels)
    history = ax.get_lines()[0]
    assert len(history.get_xdata()) == 20


def test_plot_rejects_an_empty_model_list(synthetic_daily):
    with pytest.raises(ValueError, match="at least one model"):
        plot_last_fold(synthetic_daily, {}, horizon=10, initial=60, context=5, title="x")
