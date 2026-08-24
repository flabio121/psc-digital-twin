"""Measure how fast the surrogate is, and be honest about the comparison.

    python scripts/benchmark.py

The surrogate side of this benchmark is measured here, properly: warm up first,
then many repeats at randomly drawn conditions, reported as a median with an
interquartile range rather than a best case.

The COMSOL side is *not* measured here, because COMSOL is not installed in this
environment and inventing a runtime would make the headline speedup fiction.
The reference cost is a single clearly-named constant that must be replaced with
a timed run before the number is published. Every derived speedup in the output
is labelled with where its denominator came from.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from psc_twin.surrogate import predict as predict_mod  # noqa: E402

# --------------------------------------------------------------------------
# THE ONE NUMBER THAT IS NOT MEASURED HERE.
#
# Replace this with a timed COMSOL solve of one aging condition on the machine
# you intend to quote, then set COMSOL_REFERENCE_MEASURED = True. Until then the
# script prints the speedup as an estimate and says so on every line.
# --------------------------------------------------------------------------
COMSOL_SECONDS_PER_RUN = 240.0
COMSOL_REFERENCE_MEASURED = False
COMSOL_REFERENCE_NOTE = (
    "User-supplied estimate for one COMSOL drift-diffusion aging run "
    "(10 aging points with J-V sweeps). NOT measured on this machine. "
    "Replace COMSOL_SECONDS_PER_RUN in scripts/benchmark.py with a timed solve "
    "before quoting any speedup in a publication."
)

TABLE_DIR = ROOT / "outputs" / "tables"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repeats", type=int, default=100)
    args = parser.parse_args(argv)

    if not predict_mod.models_available():
        print("Trained models not found. Run scripts/train_models.py first.")
        return 1

    print("=" * 74)
    print("PSC Digital Twin - inference benchmark")
    print("=" * 74)
    print(f"\nMeasuring surrogate latency over {args.repeats} random conditions ...")

    stats = predict_mod.measure_latency(n_repeats=args.repeats)

    rows = []
    for key, label in (
        ("single_point", "Single condition (one aging time)"),
        ("trajectory", f"Full aging trajectory ({int(stats['n_points'])} points)"),
        ("jv_curve", "Full J-V curve (72 voltage points)"),
    ):
        rows.append(
            {
                "quantity": label,
                "median_ms": stats[f"{key}_median_ms"],
                "p25_ms": stats[f"{key}_p25_ms"],
                "p75_ms": stats[f"{key}_p75_ms"],
                "measured": True,
            }
        )

    table = pd.DataFrame(rows)
    print()
    print(table.round(3).to_string(index=False))

    traj_ms = float(stats["trajectory_median_ms"])
    speedup = COMSOL_SECONDS_PER_RUN * 1000.0 / traj_ms if traj_ms > 0 else float("nan")

    print("\n" + "-" * 74)
    print("Speedup against the physics solver")
    print("-" * 74)
    status = "MEASURED" if COMSOL_REFERENCE_MEASURED else "ESTIMATE, NOT MEASURED"
    print(f"  COMSOL reference : {COMSOL_SECONDS_PER_RUN:,.0f} s per run   [{status}]")
    print(f"  Surrogate        : {traj_ms:.2f} ms per equivalent trajectory   [MEASURED]")
    print(f"  Speedup          : {speedup:,.0f}x   [{status} denominator]")
    print()
    for line in (COMSOL_REFERENCE_NOTE[i : i + 70] for i in range(0, len(COMSOL_REFERENCE_NOTE), 70)):
        print(f"  {line}")

    TABLE_DIR.mkdir(parents=True, exist_ok=True)
    table.to_csv(TABLE_DIR / "benchmark.csv", index=False)

    payload = {
        "generated_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "repeats": int(args.repeats),
        "surrogate_latency_ms": {k: v for k, v in stats.items()},
        "comsol_reference_seconds": COMSOL_SECONDS_PER_RUN,
        "comsol_reference_measured": COMSOL_REFERENCE_MEASURED,
        "comsol_reference_note": COMSOL_REFERENCE_NOTE,
        "speedup_estimate": speedup,
        "speedup_is_estimate": not COMSOL_REFERENCE_MEASURED,
    }
    (TABLE_DIR / "benchmark.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")

    print(f"\nWrote {(TABLE_DIR / 'benchmark.csv').relative_to(ROOT)}")
    print("=" * 74)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
