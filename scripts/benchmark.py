"""Measure how fast the surrogate is, and be honest about the comparison.

    python scripts/benchmark.py

The surrogate side of this benchmark is measured here, properly: warm up first,
then many repeats at randomly drawn conditions, reported as a median with an
interquartile range rather than a best case.

The COMSOL side is *not* measured here. Timing it needs the model, a COMSOL
install, and a licence the machine running this script often cannot reach, and
inventing a runtime would make the headline speedup fiction.

So the denominator comes from ``outputs/tables/comsol_baseline.json`` when that
file exists -- written by ``scripts/measure_comsol_baseline.py`` from a real
timed solve -- and falls back to the placeholder constant below when it does
not. Every derived number is labelled MEASURED or ESTIMATE accordingly, in the
console output and in the exported JSON.
"""

from __future__ import annotations

import argparse
import json
import sys
import textwrap
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
# Used only when no measured baseline exists. Do not edit it by hand to make the
# speedup look better -- run scripts/measure_comsol_baseline.py instead, which
# writes a real timing that this script prefers automatically and stamps with the
# model, study, host and timestamp that produced it.
# --------------------------------------------------------------------------
COMSOL_SECONDS_PER_RUN = 240.0
COMSOL_REFERENCE_NOTE = (
    "Placeholder estimate for one COMSOL drift-diffusion aging run. NOT a "
    "measurement. Run scripts/measure_comsol_baseline.py on a machine with a "
    "reachable COMSOL licence to replace it with a timed solve; this script "
    "picks the result up automatically."
)

TABLE_DIR = ROOT / "outputs" / "tables"
BASELINE_PATH = TABLE_DIR / "comsol_baseline.json"


def comsol_reference() -> tuple[float, bool, str]:
    """The denominator of the speedup, and whether anyone actually measured it.

    Returns (seconds, measured, note). Prefers a real timing written by
    scripts/measure_comsol_baseline.py; falls back to the placeholder constant
    above and says so.
    """
    if BASELINE_PATH.exists():
        try:
            data = json.loads(BASELINE_PATH.read_text(encoding="utf-8"))
            seconds = float(data.get("seconds_per_condition") or data["seconds_total"])
            note = (
                f"Measured on {data.get('host', 'unknown host')} at "
                f"{data.get('measured_utc', 'unknown time')}: study "
                f"{data.get('study_tag')} ({data.get('study_label')}) of "
                f"{data.get('model_file')}, {data.get('seconds_total')} s total "
                f"across {data.get('parameter_points', 1)} parameter point(s)."
            )
            return seconds, True, note
        except (KeyError, ValueError, TypeError) as exc:
            print(f"[warn] {BASELINE_PATH.name} unreadable ({exc}); using the placeholder.")
    return COMSOL_SECONDS_PER_RUN, False, COMSOL_REFERENCE_NOTE


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
    comsol_s, measured, note = comsol_reference()
    speedup = comsol_s * 1000.0 / traj_ms if traj_ms > 0 else float("nan")

    print("\n" + "-" * 74)
    print("Speedup against the physics solver")
    print("-" * 74)
    status = "MEASURED" if measured else "ESTIMATE, NOT MEASURED"
    print(f"  COMSOL reference : {comsol_s:,.0f} s per run   [{status}]")
    print(f"  Surrogate        : {traj_ms:.2f} ms per equivalent trajectory   [MEASURED]")
    print(f"  Speedup          : {speedup:,.0f}x   [{status} denominator]")
    print()
    for line in textwrap.wrap(note, 70):
        print(f"  {line}")
    if not measured:
        print()
        print("  To replace the estimate with a real timing:")
        print("    python scripts/measure_comsol_baseline.py --mph <model.mph> --study <tag>")

    TABLE_DIR.mkdir(parents=True, exist_ok=True)
    table.to_csv(TABLE_DIR / "benchmark.csv", index=False)

    payload = {
        "generated_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "repeats": int(args.repeats),
        "surrogate_latency_ms": {k: v for k, v in stats.items()},
        "comsol_reference_seconds": comsol_s,
        "comsol_reference_measured": measured,
        "comsol_reference_note": note,
        "speedup": speedup,
        "speedup_is_estimate": not measured,
    }
    (TABLE_DIR / "benchmark.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")

    print(f"\nWrote {(TABLE_DIR / 'benchmark.csv').relative_to(ROOT)}")
    print("=" * 74)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
