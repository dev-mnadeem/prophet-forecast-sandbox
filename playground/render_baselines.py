"""Render an observed holdout against two baselines without requiring Stan."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from playground.datasets import DATASETS, file_digest, load_dataset  # noqa: E402
from playground.forecasters import get_forecaster  # noqa: E402
from playground.metrics import mae  # noqa: E402


def main() -> None:
    """Save a reproducible last-30-day comparison on the pinned Peyton data."""
    key = "peyton"
    frame = load_dataset(key)
    horizon = 30
    train, actual = frame.iloc[:-horizon], frame.iloc[-horizon:]
    future = actual[["ds"]].copy()
    result = actual[["ds", "y"]].reset_index(drop=True)
    scores = {}
    fig, ax = plt.subplots(figsize=(11, 5), layout="constrained")
    history = train.iloc[-100:]
    ax.plot(history["ds"], history["y"], color="#64748b", label="Observed training data")
    ax.plot(actual["ds"], actual["y"], color="#0f172a", linewidth=2, label="Observed holdout")
    for name, color in [("naive", "#2563eb"), ("seasonal_naive", "#d97706")]:
        prediction = get_forecaster(name).fit(train).predict(future)
        result[name] = prediction["yhat"].to_numpy()
        scores[name] = float(mae(result["y"], result[name]))
        ax.plot(actual["ds"], prediction["yhat"], color=color, linestyle="--", label=name)
    cutoff = train["ds"].max()
    ax.axvline(cutoff, color="#94a3b8", linestyle=":", label="Training cutoff")
    ax.set(
        title="Peyton Manning page views: last 30 days held out",
        ylabel="Log page views",
        xlabel="Date",
    )
    ax.legend(loc="best")
    ax.grid(alpha=0.15)
    output = Path(__file__).resolve().parents[1] / "docs" / "captured"
    output.mkdir(parents=True, exist_ok=True)
    fig.savefig(output / "baseline-holdout.png", dpi=160)
    plt.close(fig)
    result.to_csv(output / "baseline-holdout.csv", index=False)
    provenance = {
        "dataset": key,
        "file": DATASETS[key].filename,
        "sha256": file_digest(DATASETS[key].path),
        "rows": len(frame),
        "training_cutoff": str(cutoff),
        "horizon": horizon,
        "holdout_mae": scores,
        "models": ["naive", "seasonal_naive"],
        "note": (
            "Actual pinned series; one terminal holdout, no Prophet fit or synthetic observations."
        ),
        "command": "python -m playground.render_baselines",
    }
    (output / "baseline-holdout.json").write_text(json.dumps(provenance, indent=2) + "\n")
    print(json.dumps(provenance, indent=2))


if __name__ == "__main__":
    main()
