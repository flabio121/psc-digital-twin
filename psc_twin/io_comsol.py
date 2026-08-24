# Thesis/analysis/jv/io_comsol.py
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import re
from typing import Optional, List

import numpy as np
import pandas as pd


@dataclass
class JVTable:
    df: pd.DataFrame      # wide table: one row per time
    v_app: np.ndarray     # voltage vector aligned to J_ columns


def _read_comsol_csv(csv_path: Path) -> pd.DataFrame:
    """
    COMSOL exports often start with metadata lines beginning with '%'.
    One of those lines contains the real column header (comma-separated).
    Data begins on the next line (without header repeated).
    """
    lines = csv_path.read_text(encoding="utf-8", errors="replace").splitlines()

    header_line = None
    start_row = None
    for i, line in enumerate(lines[:400]):
        if line.startswith("%") and "," in line and ("V_app" in line):
            header_line = line.lstrip("%").strip()
            start_row = i + 1
            break

    if header_line is None or start_row is None:
        return pd.read_csv(csv_path)

    cols = [c.strip() for c in header_line.split(",")]
    return pd.read_csv(csv_path, skiprows=start_row, names=cols)


def _find_time_col(df: pd.DataFrame) -> str:
    for c in df.columns:
        if "tage" in c and "(h" in c:
            return c
    # fallback
    for c in df.columns:
        if "time" in c.lower() and "h" in c.lower():
            return c
    raise ValueError("Could not find time column like 'tage_hours (h)'.")


def _find_v_col(df: pd.DataFrame) -> Optional[str]:
    for c in df.columns:
        if ("V_app" in c) and ("(V" in c):
            return c
    return None


def _find_current_col_long(df: pd.DataFrame, vcol: str, tcol: str) -> str:
    # Prefer a column that mentions Area and known units
    for c in df.columns:
        if c in {vcol, tcol, "mode", "case_id"}:
            continue
        if ("Area" in c) and (("A/m^2" in c) or ("A/m2" in c) or ("mA/cm^2" in c) or ("mA/cm2" in c)):
            return c

    # fallback: choose last numeric-like column not in ids
    candidates = [c for c in df.columns if c not in {vcol, tcol, "mode", "case_id"}]
    if not candidates:
        raise ValueError("Could not identify current density column in LONG format.")
    return candidates[-1]


def load_jv_table(csv_path: Path) -> JVTable:
    """
    Returns JVTable(df, v_app) where df is WIDE:
      columns: case_id, (optional mode), tage_hours_h, J_000..J_NNN
      v_app: sorted voltages aligned to J_* columns

    Supports:
      - LONG: explicit columns 'V_app (V)' and a current density column
      - WIDE: columns like 'V_app=0.05 V, ...'
    """
    csv_path = Path(csv_path)
    raw = _read_comsol_csv(csv_path)

    # Normalize ids
    has_mode = "mode" in raw.columns
    has_case = "case_id" in raw.columns

    if not has_case:
        raise ValueError("Expected 'case_id' column in COMSOL export.")

    tcol = _find_time_col(raw)
    vcol = _find_v_col(raw)

    # -------------------------
    # LONG FORMAT: has V_app (V)
    # -------------------------
    if vcol is not None:
        jcol = _find_current_col_long(raw, vcol=vcol, tcol=tcol)

        df = raw.copy()
        df["tage_hours_h"] = pd.to_numeric(df[tcol], errors="coerce")
        df["V_app_V"] = pd.to_numeric(df[vcol], errors="coerce")
        j = pd.to_numeric(df[jcol], errors="coerce")

        # Units: A/m^2 -> mA/cm^2 (1 A/m^2 = 0.1 mA/cm^2)
        header = str(jcol)
        if ("A/m^2" in header) or ("A/m2" in header):
            df["J_mAcm2"] = 0.1 * j
        else:
            # assume already mA/cm^2
            df["J_mAcm2"] = j

        keep_cols = ["case_id"] + (["mode"] if has_mode else []) + ["tage_hours_h", "V_app_V", "J_mAcm2"]
        df = df[keep_cols].dropna(subset=["tage_hours_h", "V_app_V", "J_mAcm2"])

        # Pivot: one row per (case_id, mode, time), columns = V_app_V
        idx_cols = ["case_id"] + (["mode"] if has_mode else []) + ["tage_hours_h"]
        pv = (
            df.pivot_table(index=idx_cols, columns="V_app_V", values="J_mAcm2", aggfunc="mean")
              .sort_index(axis=1)
        )

        v_app = pv.columns.to_numpy(dtype=float)

        # Rename voltage columns to J_000..J_{N-1} to satisfy pipeline expectations
        j_cols = [f"J_{k:03d}" for k in range(len(v_app))]
        pv.columns = j_cols

        wide = pv.reset_index()
        return JVTable(df=wide, v_app=v_app)

    # -------------------------
    # WIDE FORMAT: voltages in headers (your existing style)
    # -------------------------
    df = raw.copy()
    df["tage_hours_h"] = pd.to_numeric(df[tcol], errors="coerce")

    # Identify all JV columns and parse V_app from header
    id_cols = ["case_id"] + (["mode"] if has_mode else []) + ["tage_hours_h"]
    value_cols = [c for c in df.columns if c not in set(id_cols + [tcol])]

    v_list: List[float] = []
    j_src_cols: List[str] = []

    for col in value_cols:
        m = re.search(r"V_app\s*=\s*([0-9]*\.?[0-9]+)", str(col))
        if not m:
            continue
        v_list.append(float(m.group(1)))
        j_src_cols.append(col)

    if not j_src_cols:
        raise ValueError(f"No wide-format JV columns found (expected 'V_app=...'). File: {csv_path.name}")

    # Sort by voltage and build J_ columns
    order = np.argsort(np.array(v_list))
    v_app = np.array(v_list, dtype=float)[order]
    j_src_cols = [j_src_cols[i] for i in order]

    out = df[id_cols].copy()
    for k, col in enumerate(j_src_cols):
        out[f"J_{k:03d}"] = pd.to_numeric(df[col], errors="coerce")

    out = out.dropna(subset=["tage_hours_h"])
    return JVTable(df=out, v_app=v_app)
