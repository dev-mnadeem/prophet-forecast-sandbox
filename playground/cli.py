"""``python -m playground`` — reproduce every forecasting number in the README.

Two subcommands:

* ``datasets`` — list the pinned example series and verify their checksums.
* ``backtest`` — rolling-origin backtest of one or more registered models on a
  pinned dataset, printing per-fold and aggregate scores.

Output is plain text on stdout so it can be redirected into ``docs/captured/``
and quoted verbatim.
"""

from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence

import numpy as np
import pandas as pd

from playground.backtest import BacktestResult, backtest
from playground.datasets import DATASETS, ChecksumError, file_digest, load_dataset
from playground.forecasters import (
    available_forecasters,
    get_forecaster,
    silence_stan_logging,
)


def _fmt(value: float | None, places: int = 4) -> str:
    return "-" if value is None else f"{value:.{places}f}"


def _print_folds(result: BacktestResult) -> None:
    print(f"\n{result.model} / {result.dataset} / horizon={result.horizon}")
    print(
        f"{'fold':>4}  {'cutoff':<12} {'n_train':>7} {'mae':>8} {'rmse':>8} "
        f"{'smape':>8} {'cover':>7} {'secs':>7}"
    )
    for f in result.folds:
        print(
            f"{f.fold:>4}  {f.cutoff.date()!s:<12} {f.n_train:>7} {f.mae:>8.4f} "
            f"{f.rmse:>8.4f} {f.smape:>8.4f} {_fmt(f.coverage, 3):>7} {f.fit_seconds:>7.2f}"
        )


def _print_summary(results: Sequence[BacktestResult]) -> None:
    print(
        f"\n{'model':<16} {'folds':>5} {'mae':>8} {'rmse':>8} {'smape':>8} "
        f"{'cover':>7} {'total_s':>8}"
    )
    print("-" * 65)
    rows = [r.summary() for r in results]
    best = min(rows, key=lambda r: r["mae"])["model"] if rows else None
    for row in rows:
        marker = " *" if row["model"] == best else "  "
        print(
            f"{row['model']:<16} {row['folds']:>5} {row['mae']:>8.4f} {row['rmse']:>8.4f} "
            f"{row['smape']:>8.4f} {_fmt(row['coverage'], 3):>7} {row['fit_seconds']:>8.2f}{marker}"
        )
    if best:
        print(f"\n* lowest mean MAE across identical folds: {best}")


def cmd_datasets(_: argparse.Namespace) -> int:
    """List the pinned datasets and verify every recorded checksum."""
    print(f"{'key':<12} {'freq':<6} {'period':>6}  description")
    print("-" * 96)
    status = 0
    for key, spec in sorted(DATASETS.items()):
        print(f"{key:<12} {spec.freq:<6} {spec.seasonal_period:>6}  {spec.description}")
    print("\nchecksum verification:")
    for key, spec in sorted(DATASETS.items()):
        if not spec.path.exists():
            print(f"  {key:<12} MISSING {spec.path}")
            status = 1
            continue
        actual = file_digest(spec.path)
        ok = actual == spec.sha256
        status = status or (0 if ok else 1)
        print(f"  {key:<12} {'OK  ' if ok else 'FAIL'} sha256:{actual[:16]}...")
    return status


def cmd_determinism(args: argparse.Namespace) -> int:
    """Measure how far Prophet's output moves between identical runs.

    Prophet draws uncertainty samples from the *global* NumPy RNG and seeds it
    nowhere, so interval bounds wander while the MAP point forecast does not.
    This quantifies both, and shows that seeding fixes the intervals.
    """
    from prophet import Prophet

    silence_stan_logging()

    spec = DATASETS[args.dataset]
    df = load_dataset(args.dataset)[["ds", "y"]]
    future = pd.DataFrame({"ds": df["ds"].iloc[-args.horizon :].to_numpy()})
    train = df.iloc[: -args.horizon]

    print(f"dataset : {args.dataset} ({spec.filename}), {len(train)} train rows")
    print(f"horizon : {args.horizon}\n")

    def fit_predict(seed: int | None) -> pd.DataFrame:
        if seed is not None:
            np.random.seed(seed)
        model = Prophet()
        model.fit(train)
        if seed is not None:
            np.random.seed(seed)
        return model.predict(future)

    a, b = fit_predict(None), fit_predict(None)
    print("two independent UNSEEDED fits")
    print(f"  max |delta yhat|        = {np.abs(a.yhat - b.yhat).max():.3e}")
    print(f"  max |delta yhat_lower|  = {np.abs(a.yhat_lower - b.yhat_lower).max():.4f}")
    print(f"  max |delta yhat_upper|  = {np.abs(a.yhat_upper - b.yhat_upper).max():.4f}")
    print(f"  mean interval width     = {(a.yhat_upper - a.yhat_lower).mean():.4f}")

    c, d = fit_predict(args.seed), fit_predict(args.seed)
    print(f"\ntwo fits with np.random.seed({args.seed}) before each")
    print(f"  max |delta yhat|        = {np.abs(c.yhat - d.yhat).max():.3e}")
    print(f"  max |delta yhat_lower|  = {np.abs(c.yhat_lower - d.yhat_lower).max():.4f}")
    print(f"  max |delta yhat_upper|  = {np.abs(c.yhat_upper - d.yhat_upper).max():.4f}")

    print(
        "\nconclusion: point forecasts are deterministic (MAP fit); interval bounds "
        "are not, unless the global RNG is seeded. playground.ProphetForecaster seeds it."
    )
    return 0


def cmd_backtest(args: argparse.Namespace) -> int:
    """Run a rolling-origin backtest for each requested model."""
    spec = DATASETS[args.dataset]
    df = load_dataset(args.dataset)
    print(f"dataset   : {args.dataset} ({spec.filename})")
    print(f"rows      : {len(df)}  span {df['ds'].min().date()} .. {df['ds'].max().date()}")
    print(f"sha256    : {spec.sha256}")
    print(
        f"horizon   : {args.horizon}   initial: {args.initial}   "
        f"period: {args.period or args.horizon}   max_folds: {args.max_folds}"
    )

    results: list[BacktestResult] = []
    for name in args.models:
        kwargs = {"period": spec.seasonal_period} if name == "seasonal_naive" else {}
        if name == "prophet":
            kwargs = {"seed": args.seed}
        result = backtest(
            df,
            lambda n=name, k=kwargs: get_forecaster(n, **k),
            horizon=args.horizon,
            initial=args.initial,
            period=args.period,
            max_folds=args.max_folds,
            model_name=name,
            dataset_name=args.dataset,
        )
        results.append(result)
        if args.per_fold:
            _print_folds(result)

    _print_summary(results)
    return 0


def build_parser() -> argparse.ArgumentParser:
    """Assemble the argument parser for ``python -m playground``."""
    parser = argparse.ArgumentParser(
        prog="python -m playground",
        description="Reproducible, leakage-checked backtests over the vendored Prophet.",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("datasets", help="list pinned datasets and verify checksums").set_defaults(
        func=cmd_datasets
    )

    bt = sub.add_parser("backtest", help="run a rolling-origin backtest")
    bt.add_argument("--dataset", default="peyton", choices=sorted(DATASETS))
    bt.add_argument(
        "--models",
        nargs="+",
        default=["naive", "seasonal_naive", "prophet"],
        choices=available_forecasters(),
    )
    bt.add_argument("--horizon", type=int, default=30, help="observations forecast per fold")
    bt.add_argument(
        "--initial", type=int, default=730, help="observations in the first train window"
    )
    bt.add_argument("--period", type=int, default=None, help="observations between cutoffs")
    bt.add_argument("--max-folds", dest="max_folds", type=int, default=5)
    bt.add_argument("--seed", type=int, default=1234)
    bt.add_argument("--per-fold", action="store_true", help="print each fold, not just the summary")
    bt.set_defaults(func=cmd_backtest)

    det = sub.add_parser("determinism", help="measure run-to-run variation in Prophet's output")
    det.add_argument("--dataset", default="peyton", choices=sorted(DATASETS))
    det.add_argument("--horizon", type=int, default=30)
    det.add_argument("--seed", type=int, default=1234)
    det.set_defaults(func=cmd_determinism)

    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Entry point. Returns a process exit status."""
    silence_stan_logging()
    args = build_parser().parse_args(argv)
    try:
        return args.func(args)
    except ChecksumError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
