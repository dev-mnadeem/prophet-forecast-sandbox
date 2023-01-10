# playground tests

Fast by design: the default suite fits nothing, needs no network, no GPU and no
compiled Stan model. Every fixture is a synthetic series of at most a few
hundred points with a fixed seed.

| File | What it pins |
| --- | --- |
| `test_leakage.py` | The look-ahead invariant, from four directions. See below. |
| `test_splits.py` | Fold counts, expanding windows, cutoff spacing, argument validation. |
| `test_metrics.py` | Metric values worked out by hand, not recomputed from the implementation. |
| `test_forecasters.py` | The registry seam and the two baselines. |
| `test_backtest.py` | End-to-end harness behaviour, reproducibility, the `map_fn` seam. |
| `test_datasets.py` | Dataset checksums and loader tidiness guarantees. |
| `test_prophet_integration.py` | `slow`-marked; real Prophet fits. Run with `pytest -m slow`. |

## Proving the leakage tests can fail

They are only worth having if they break when leakage is reintroduced. To check:

```bash
# 1. Remove the guard in playground/splits.py: make assert_no_look_ahead a no-op.
# 2. Run the suite.
pytest playground -q
```

Three tests in `test_leakage.py` fail immediately
(`test_overlapping_train_and_test_is_rejected`,
`test_training_past_the_cutoff_is_rejected`, and the behavioural
`test_model_is_never_shown_a_row_at_or_after_its_cutoff` once `backtest`
also stops re-checking). A transcript of that run is in
`docs/captured/leakage-injection.txt`.
