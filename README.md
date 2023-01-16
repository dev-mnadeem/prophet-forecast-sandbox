# prophet_playaround

**This repository is a fork — a vendored copy — of [facebook/prophet](https://github.com/facebook/prophet)
at v1.1.4.** Everything under `python/`, `python_shim/`, `notebooks/`, `examples/` and the
Jekyll site in `docs/` is Meta's code and documentation, unmodified. The upstream `R/`
package was deleted. **259 of the 296 tracked files are byte-identical to the upstream tag.**

The original work is one directory: **`playground/`** — 1,742 lines across 17 Python files
that turn the vendored library into something you can make a claim from. Pinned datasets,
rolling-origin backtests with the look-ahead invariant enforced per fold, a seeded Prophet
adapter, and two baselines replayed through the *same* folds. The Dockerfile, the CI
workflow and this README are first-party too. Nothing else here is.

## Whose code is whose, counted rather than asserted

| Against `facebook/prophet` v1.1.4 | files |
| --- | ---: |
| tracked in this repository | 296 |
| byte-identical to upstream | **259** |
| differ from upstream | 7 |
| not present upstream (all first-party) | 30 |

Reproduce that yourself:

```bash
curl -sL https://github.com/facebook/prophet/archive/refs/tags/v1.1.4.tar.gz | tar xz
diff -rq python/prophet prophet-1.1.4/python/prophet   # no output: identical
```

Of the seven that differ, five are first-party rewrites of files that exist upstream
(`README.md`, `Dockerfile`, `docker-compose.yml`, `Makefile`, `.gitignore`) and two are not
edits at all: `python/setup.py` and `python/pyproject.toml` are *newer* than the v1.1.4 tag,
carrying two upstream commits from just after the release, so the snapshot is upstream `main`
a little past the tag.

**No file under `python/prophet/`, `python/stan/`, `python_shim/`, `notebooks/` or
`examples/` has been modified. The upstream `LICENSE` and every per-file
`Copyright (c) Facebook, Inc.` header are exactly as they came.** The vendored tree is also
excluded from linting and formatting: reformatting it would destroy the ability to diff it
against upstream, which is this repository's most useful property.

## The finding: Prophet against two baselines on identical folds

Every model sees exactly the same five walk-forward folds of the Peyton Manning series.
The full transcript, with the per-fold rows this block elides, is in
[`docs/captured/backtest-peyton.txt`](docs/captured/backtest-peyton.txt); re-running it
reproduced every figure below, per fold and in aggregate, apart from wall-clock seconds.

```
$ python -m playground backtest --dataset peyton --horizon 30 --initial 730 --max-folds 5
dataset   : peyton (example_wp_log_peyton_manning.csv)
rows      : 2905  span 2007-12-10 .. 2016-01-20
sha256    : 6b1383a6f458e317c5da0488abaf2cadccd7e80c84c7a543d084e1159d5bfa56
horizon   : 30   initial: 730   period: 30   max_folds: 5

model            folds      mae     rmse    smape   cover  total_s
-----------------------------------------------------------------
naive                5   0.4633   0.6276   0.0281       -     0.00
seasonal_naive       5   0.6250   0.8753   0.0357       -     0.00
prophet              5   0.3149   0.4889   0.0188   0.893     2.15 *

* lowest mean MAE across identical folds: prophet
```

Three readings, the last two mattering more than the headline:

- Prophet earns its complexity here — **MAE 0.3149 against 0.4633** for
  last-value-carried-forward, a 32% reduction, on folds neither model could peek at.
- **Seasonal-naive is the worst of the three** (0.6250). Weekly seasonality alone is not
  what makes this series predictable; the fitted trend is.
- Nominal-80% intervals covered **89.3%** of actuals — conservative, which is a calibration
  cost, not a win.

Units are log page-views, so these are not percentages; `smape` is the comparable column.
This is five folds of one series at library defaults with no hyperparameter search. It is a
reproducible measurement, not a benchmark.

## Running it

```bash
python -m venv .venv && . .venv/bin/activate
pip install -r requirements-dev.txt

# Baselines only. No compiler, no Stan model, about a second.
python -m playground backtest --dataset peyton --models naive seasonal_naive

# Add Prophet. Compiles CmdStan and python/stan/prophet.stan the first time
# (tens of minutes), cached afterwards.
pip install -e python/
python -m playground backtest --dataset peyton --per-fold
```

`python -m playground {datasets,backtest,determinism} --help` lists the flags; `make help`
lists the same steps as targets. The pins in `requirements-dev.txt` are a correctness fix,
not hygiene — the vendored `python/pyproject.toml` declares `numpy>=1.15.4` unbounded, and a
fresh install today resolves NumPy 2.x, where `import prophet` dies on `np.float_`. They live
outside the vendored file deliberately.

On macOS arm64 there is no prophet 1.1.4 wheel and CmdStan 2.31 will not build with a
current clang, so the numbers above came from `python/` on `PYTHONPATH` against a prebuilt
`prophet_model.bin` — checked first to be a compilation of *this* repository's
`python/stan/prophet.stan`, which is byte-identical to the 1.1.5 wheel's.

## Why the folds are trustworthy

Leakage is prevented structurally first. The `Forecaster` protocol takes a training frame
in `fit` and **timestamps only** in `predict` — there is no parameter through which a future
`y` could reach a model. On top of that, `assert_no_look_ahead` runs when folds are
generated *and* again when they are materialised, and `playground/tests/test_leakage.py`
attacks the invariant from four directions: overlapping index sets, a training window past
its cutoff, unsorted or duplicated timestamps, and behaviourally — recording the widest
window a model was ever shown and asserting it ends before the test window begins.

The Prophet-specific one is the fitted scaler: `initialize_scales` derives `y_scale`,
`start` and `t_scale` from whatever frame it is handed, so fitting once on the full series
and slicing predictions afterwards is the textbook "scaler fitted on all data" leak.
`test_scaling_constants_come_from_training_data_only` plants a 500.0 spike inside the first
test window and asserts `m.y_scale` equals the *training* maximum.

A guard nobody has watched fail is decoration.
[`docs/captured/leakage-injection.txt`](docs/captured/leakage-injection.txt) is the
transcript of a run with the guard stubbed to a no-op and training windows deliberately
extended over the test window: **5 failed, 54 passed**, three of them in `test_leakage.py`.
Both injected leaks were reverted immediately; the committed tree is green.

| Suite | Command | Result, re-run for this README |
| --- | --- | --- |
| first-party, fast | `pytest playground -q` | 59 passed in 0.60s |
| first-party, real fits | `pytest playground -q -m slow` | 4 passed, 59 deselected |
| vendored upstream | `pytest python/prophet/tests python_shim -q` | 67 passed, 2 deselected in 21.7s |
| lint | `ruff check` / `ruff format --check` | clean, 19 files formatted |

The fast suite fits nothing — every fixture is a synthetic series of a few hundred seeded
points — so a broken invariant is reported in under a second and CI needs no compiler.
See [`playground/tests/README.md`](playground/tests/README.md) for what each file pins.

## Two things that only surfaced because something got measured

**Prophet's point forecast is reproducible and its intervals are not.**
[`docs/captured/determinism.txt`](docs/captured/determinism.txt) records two independent
unseeded fits: `max |delta yhat| = 0.000e+00`, the MAP estimate repeating exactly, while the
interval bounds moved by up to 0.0965 against a mean width of 1.2505 — about 8% of the
interval. Prophet draws those bounds from the *global* NumPy RNG and seeds it nowhere, so the
drift differs every time you look (a re-run for this README gave 0.0892 and 0.1168). Seeding
`np.random` before each fit collapses it to 0.0000, which is what
`playground.ProphetForecaster` does; there is no seed parameter to pass instead, because
v1.1.4 calls `np.random.normal` and friends directly.

**One vendored upstream test cannot pass, for any model.**
`python/prophet/tests/test_prophet.py:61` reads `assert 236 < res < 193` — unsatisfiable for
every real number. It sits behind `@pytest.mark.slow`, which upstream's `conftest.py` skips
unless `--test-slow` is passed, so nobody trips over it; the vendored run in the table above
is green with it deselected. Run deliberately, it gives `1 failed, 1 passed, 66 deselected`
and a measured **45.74**, nowhere near the adjacent comment claiming `~ 215.77`. Its sibling
has the same smell: `# this gives ~ 10.64` above `assert res == pytest.approx(23.44, 0.01)`.
This fork does not patch it — editing `python/` would forfeit the diff-against-upstream
property above, and the fix belongs upstream. It is recorded so the number is not mistaken
for a verified one.

## Out of scope, and not done

- **Docker was not built or booted.** `Dockerfile` and `docker-compose.yml` were written to
  a multi-stage, non-root, healthchecked standard and `docker compose config` parses them,
  but the daemon was off in the environment this work ran in. Build verified: *NOT RUN*.
  Boot verified: *NOT RUN*. Treat them as reviewed source, not as verified artefacts.
  `docker-healthcheck.py` was executed directly and exits 0.
- **CI has never run.** `.github/workflows/ci.yml` restores a pipeline the fork had deleted,
  written against the pinned requirements, with no run history to point at.
- **Only Python 3.10 was exercised** here; the CI matrix also lists 3.11.
- **The vendored library is frozen at v1.1.4 and is not maintained here.** Upstream has since
  shipped macOS arm64 wheels and NumPy 2 support. File library bugs with Meta, not here.
- **`docs/` is upstream's 8.6 MB Jekyll site**, documenting `facebook/prophet` including the
  R API this fork removed — kept for provenance, not as documentation of this repository.
- No hyperparameter search, no interval recalibration, no second dataset in the headline.

## License and attribution

The vendored Prophet code is MIT, **copyright Facebook, Inc. and its affiliates** — see
[LICENSE](LICENSE), which is upstream's, unaltered, as are all per-file copyright headers.
[facebook/prophet](https://github.com/facebook/prophet) and its own documentation are the
authoritative references for the library. The first-party `playground/` code is offered under
the same MIT terms.
