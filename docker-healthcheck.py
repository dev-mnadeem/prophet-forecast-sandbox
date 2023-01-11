"""Container healthcheck for the runtime image.

A batch image has no port to probe, so "healthy" is defined as: the vendored
Prophet imports, its compiled Stan binary is actually present (the expensive
thing the builder stage produced), and every pinned dataset still matches its
recorded checksum.

Exits 0 when healthy, 1 otherwise.
"""

from __future__ import annotations

import sys


def main() -> int:
    """Return 0 if the image is usable, 1 with a reason on stderr otherwise."""
    try:
        import importlib_resources
        import prophet  # noqa: F401

        model = importlib_resources.files("prophet") / "stan_model" / "prophet_model.bin"
        if not model.is_file():
            print(f"compiled Stan model missing at {model}", file=sys.stderr)
            return 1

        from playground.cli import cmd_datasets

        class _NoArgs:
            pass

        return 1 if cmd_datasets(_NoArgs()) else 0
    except Exception as exc:  # noqa: BLE001 - a healthcheck must never raise
        print(f"healthcheck failed: {exc!r}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
