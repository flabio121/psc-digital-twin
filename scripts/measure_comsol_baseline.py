"""Measure how long COMSOL actually takes, so the speedup stops being an estimate.

    python scripts/measure_comsol_baseline.py --mph "path/to/model.mph" --study std2

The surrogate side of the benchmark is measured on any machine. The COMSOL side
is not, because it needs the model, a COMSOL installation, and a license the
CI/dev machine usually cannot reach. Until this script has been run,
``scripts/benchmark.py`` reports the speedup against a declared placeholder and
labels every derived number as resting on an estimate.

Run this once, on a machine that can check out a COMSOL licence, and it writes
``outputs/tables/comsol_baseline.json``. ``benchmark.py`` picks that file up
automatically and relabels the speedup as MEASURED, recording which model,
which study, and which host produced it.

Requirements
------------
* COMSOL installed (this script finds ``comsolbatch.exe`` automatically).
* A reachable licence server. If you are off campus, connect to the VPN first --
  a licence failure is by far the most common reason this script stops, and it
  reports that case explicitly rather than as a generic error.
* The ``.mph`` model, and the tag of the study that corresponds to one aging run
  with its diagnostic J-V sweeps.

What is being measured
----------------------
One full solve of the named study: the unit of work the surrogate replaces.
Report it that way. A per-condition figure is only meaningful if the study
solves exactly one (illumination, temperature) condition; if the study contains
a parametric sweep, divide by the number of parameter points and say so.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import re
import subprocess
import sys
import time
import zipfile
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT_PATH = ROOT / "outputs" / "tables" / "comsol_baseline.json"

#: Where COMSOL installs live on each platform, newest first.
_SEARCH_ROOTS = (
    Path(r"C:\Program Files\COMSOL"),
    Path("/usr/local/comsol"),
    Path("/Applications"),
)

_LICENSE_HINT = (
    "COMSOL could not check out a licence.\n"
    "  This is almost always a network problem, not a model problem.\n"
    "  If your licence is served from a university host, connect to the VPN\n"
    "  and try again. The log line above names the host it tried to reach."
)


def find_comsolbatch(explicit: str | None = None) -> Path:
    """Locate the headless batch binary.

    Note the distinction that costs people an afternoon: ``comsol.exe batch``
    on Windows hands off to the GUI and returns success immediately without
    solving anything. ``comsolbatch.exe`` is the one that actually runs
    headless.
    """
    if explicit:
        path = Path(explicit)
        if not path.exists():
            raise FileNotFoundError(f"No COMSOL batch binary at {path}")
        return path

    names = ("comsolbatch.exe", "comsol batch", "comsolbatch")
    candidates: list[Path] = []
    for root in _SEARCH_ROOTS:
        if not root.exists():
            continue
        for name in names:
            candidates.extend(root.glob(f"**/bin/**/{name}"))
    if not candidates:
        raise FileNotFoundError(
            "Could not find comsolbatch. Pass --comsol with the full path, e.g.\n"
            r'  --comsol "C:\Program Files\COMSOL\COMSOL64\multiphysics\bin\win64\comsolbatch.exe"'
        )
    # Newest install wins: COMSOL63 < COMSOL64 lexicographically, which is what we want.
    return sorted(candidates)[-1]


def study_labels(mph: Path) -> dict[str, str]:
    """Read study tags and labels straight out of the .mph.

    A .mph is a ZIP holding an action log in ``dmodel.xml``. Parsing it is far
    cheaper than starting COMSOL just to ask what the studies are called, and it
    needs no licence. Labels are replayed in order, so the last one wins.
    """
    labels: dict[str, str] = {}
    try:
        with zipfile.ZipFile(mph) as archive:
            xml = archive.read("dmodel.xml").decode("utf-8", errors="replace")
    except (KeyError, zipfile.BadZipFile):
        return labels

    pattern = r't\(s\("/study/(std\d+)"\)\) m\(s\("label"\)\) s\("([^"]+)"\)'
    for match in re.finditer(pattern, xml):
        labels[match.group(1)] = match.group(2)
    return labels


def _digest(path: Path, limit: int = 8 << 20) -> str:
    """Short hash of the model's leading bytes, to pin which file was timed."""
    hasher = hashlib.sha256()
    with path.open("rb") as handle:
        hasher.update(handle.read(limit))
    return hasher.hexdigest()[:16]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--mph", required=True, help="path to the .mph model")
    parser.add_argument("--study", default=None, help="study tag to solve, e.g. std2")
    parser.add_argument("--comsol", default=None, help="path to comsolbatch executable")
    parser.add_argument("--parameter-points", type=int, default=1,
                        help="parameter points solved by the study, for the per-condition figure")
    parser.add_argument("--list-studies", action="store_true", help="print study tags and exit")
    parser.add_argument("--timeout", type=int, default=14400, help="seconds before giving up")
    args = parser.parse_args(argv)

    mph = Path(args.mph).expanduser()
    if not mph.exists():
        print(f"Model not found: {mph}")
        return 1

    labels = study_labels(mph)
    if args.list_studies or not args.study:
        print(f"\nStudies in {mph.name}:")
        for tag, label in sorted(labels.items()) or [("(none found)", "")]:
            print(f"  {tag:8s} {label}")
        if args.list_studies:
            return 0
        print("\nRe-run with --study <tag> to time one.")
        return 1

    try:
        batch = find_comsolbatch(args.comsol)
    except FileNotFoundError as exc:
        print(exc)
        return 1

    log = ROOT / "outputs" / "comsol_batch.log"
    log.parent.mkdir(parents=True, exist_ok=True)
    log.unlink(missing_ok=True)

    label = labels.get(args.study, "(unlabelled)")
    print("=" * 74)
    print("COMSOL baseline measurement")
    print("=" * 74)
    print(f"  binary : {batch}")
    print(f"  model  : {mph}")
    print(f"  study  : {args.study}  ({label})")
    print("\nSolving. This is the slow thing the surrogate exists to replace.\n")

    command = [
        str(batch),
        "-inputfile", str(mph),
        "-study", args.study,
        "-nosave",
        "-batchlog", str(log),
    ]

    started = time.perf_counter()
    try:
        proc = subprocess.run(command, capture_output=True, text=True, timeout=args.timeout)
    except subprocess.TimeoutExpired:
        print(f"Gave up after {args.timeout} s. Raise --timeout if the study is genuinely this slow.")
        return 1
    elapsed = time.perf_counter() - started

    log_text = log.read_text(encoding="utf-8", errors="replace") if log.exists() else ""
    combined = f"{proc.stdout}\n{proc.stderr}\n{log_text}"

    if "License error" in combined or "licence" in combined.lower() and "error" in combined.lower():
        print("\n".join(line for line in combined.splitlines() if line.strip())[-1200:])
        print("\n" + _LICENSE_HINT)
        return 1

    if "Error" in log_text and "Total time" not in log_text:
        print(log_text[-1500:])
        print("\nCOMSOL reported an error; no timing recorded.")
        return 1

    if elapsed < 20:
        print(combined[-800:])
        print(
            "\nThe solve returned in under 20 s, which almost certainly means it did not\n"
            "run. Check that you used comsolbatch (not `comsol batch`) and that the\n"
            "study tag exists."
        )
        return 1

    per_condition = elapsed / max(args.parameter_points, 1)
    payload = {
        "measured_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "seconds_total": round(elapsed, 2),
        "parameter_points": int(args.parameter_points),
        "seconds_per_condition": round(per_condition, 2),
        "study_tag": args.study,
        "study_label": label,
        "model_file": mph.name,
        "model_sha256_prefix": _digest(mph),
        "comsol_binary": str(batch),
        "host": platform.node(),
        "platform": platform.platform(),
        "measured": True,
    }

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUT_PATH.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    print(f"\nSolved in {elapsed:,.1f} s")
    if args.parameter_points > 1:
        print(f"  {per_condition:,.1f} s per condition across {args.parameter_points} parameter points")
    print(f"\nWrote {OUT_PATH.relative_to(ROOT)}")
    print("Re-run scripts/benchmark.py -- the speedup will now report as MEASURED.")
    print("=" * 74)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
