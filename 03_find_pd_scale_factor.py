#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
03_test_pd_unit_factor_candidates.py

Purpose
-------
Test possible unit-conversion factors for Pd.

This is NOT a station-by-station calibration. It is a unit-factor check.

Current Pd table may already contain:
    Pd_3s_cm = Pd_internal * CURRENT_FACTOR_USED_IN_PD_SCRIPT

This script reconstructs:
    Pd_internal = Pd_3s_cm / CURRENT_FACTOR_USED_IN_PD_SCRIPT

Then tests candidate factors:
    Pd_cm_candidate = Pd_internal * candidate_factor

Examples:
    1e-7 : internal amplitude is nm
    1e-6 : internal amplitude is 10 nm-like scale, or factor-10 response issue
    1e-5 : internal amplitude is 100 nm-like scale
    1e-4 : internal amplitude is micrometer
    1e-1 : internal amplitude is millimeter
    1e+2 : internal amplitude is meter

For each factor, compute:
    Mpd = 4.748 + 1.371*log10(Pd_cm) + 1.883*log10(R_km)

and compare the global offset:
    residual = Mpd - header_ML

A pure unit error should mostly appear as a nearly constant vertical offset
of Mpd - header_ML across stations/events:
    multiplying Pd by 10 changes Mpd by 1.371 magnitude units.

Outputs
-------
output/03_report_P_amp_filter_2sdt/pd_unit_factor_candidates/
    pd_unit_factor_candidate_summary.csv
    pd_unit_factor_candidate_detailed.csv
    pd_unit_factor_candidate_report.txt
    Mpd_residual_by_unit_factor.png
    Mpd_vs_header_best_factor.png
"""

from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# =====================================================================
# USER PARAMETERS
# =====================================================================

INPUT_CSV = Path(
    "output/03_report_P_amp_filter_2sdt/"
    "P_amp_tau_Z_3s_passed_2sdt.csv"
)

OUTPUT_DIR = Path(
    "output/03_report_P_amp_filter_2sdt/"
    "pd_unit_factor_candidates"
)

# Columns in current table.
PD_CURRENT_CM_COL = "Pd_3s_cm"
MAG_COL = "header_ML"
DIST_COL = "distance_km"

# Current factor used in the Pd script that made the table.
# In your current script:
#     NM_TO_CM = 1e-7
CURRENT_FACTOR_USED_IN_PD_SCRIPT = 1.0e-7

# Candidate factors from internal amplitude value to cm.
# Include pure unit candidates and nearby response-scale candidates.
CANDIDATE_FACTORS = [
    1.0e-9,
    1.0e-8,
    1.0e-7,   # internal unit = nm
    2.0e-7,
    3.0e-7,
    5.0e-7,
    1.0e-6,
    2.0e-6,
    5.0e-6,
    1.0e-5,
    1.0e-4,   # internal unit = micrometer
    1.0e-3,
    1.0e-2,
    1.0e-1,   # internal unit = millimeter
    1.0e0,
    1.0e1,
    1.0e2,    # internal unit = meter
]

# Wu & Zhao 2006 style Pd magnitude equation:
# Mpd = A + B*log10(Pd_cm) + C*log10(R_km)
WU_ZHAO_A = 4.748
WU_ZHAO_B = 1.371
WU_ZHAO_C = 1.883

# Residual for comparison.
# residual = Mpd - header_ML.
# A unit factor is suspicious if median residual is shifted by whole magnitude units.
RESIDUAL_COL = "Mpd_minus_header"

# Optional clipping only for summary statistics, not for table output.
CLIP_FOR_SUMMARY = True
RESIDUAL_CLIP_MAG = 5.0

MAKE_PLOTS = True


# =====================================================================
# HELPERS
# =====================================================================

def safe_float(s: pd.Series) -> pd.Series:
    return pd.to_numeric(s, errors="coerce").astype(float)


def calc_mpd(pd_cm: np.ndarray, r_km: np.ndarray) -> np.ndarray:
    return (
        WU_ZHAO_A
        + WU_ZHAO_B * np.log10(pd_cm)
        + WU_ZHAO_C * np.log10(r_km)
    )


def robust_stats(x: np.ndarray) -> dict:
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    if x.size == 0:
        return {
            "n": 0,
            "mean": np.nan,
            "median": np.nan,
            "std": np.nan,
            "mad": np.nan,
            "robust_sigma": np.nan,
            "median_abs": np.nan,
        }

    med = float(np.median(x))
    mad = float(np.median(np.abs(x - med)))

    return {
        "n": int(x.size),
        "mean": float(np.mean(x)),
        "median": med,
        "std": float(np.std(x, ddof=1)) if x.size > 1 else np.nan,
        "mad": mad,
        "robust_sigma": 1.4826 * mad,
        "median_abs": float(np.median(np.abs(x))),
    }


def classify_factor(factor: float) -> str:
    labels = {
        1.0e-9: "0.01 nm-like / 0.1 Angstrom-like",
        1.0e-8: "Angstrom-like",
        1.0e-7: "nm",
        1.0e-4: "micrometer",
        1.0e-1: "millimeter",
        1.0e2: "meter",
    }
    for k, v in labels.items():
        if np.isclose(factor, k, rtol=1e-12, atol=0):
            return v

    return "scale-test"


def plot_factor_summary(summary: pd.DataFrame, out_file: Path) -> None:
    if summary.empty:
        return

    x = np.log10(summary["candidate_factor_to_cm"].to_numpy(dtype=float))
    y = summary["median_Mpd_minus_header"].to_numpy(dtype=float)
    yabs = summary["median_abs_Mpd_minus_header"].to_numpy(dtype=float)

    fig, ax = plt.subplots(figsize=(8.2, 5.6))
    ax.plot(x, y, marker="o", label="median(Mpd - header ML)")
    ax.plot(x, yabs, marker="s", label="median absolute residual")
    ax.axhline(0.0, linewidth=1.0)
    ax.axvline(np.log10(CURRENT_FACTOR_USED_IN_PD_SCRIPT), linestyle="--",
               label=f"current factor = {CURRENT_FACTOR_USED_IN_PD_SCRIPT:g}")
    ax.set_xlabel("log10(candidate factor to cm)")
    ax.set_ylabel("Magnitude residual")
    ax.set_title("Pd unit-factor candidate test")
    ax.grid(True, alpha=0.3)
    ax.legend()

    fig.tight_layout()
    fig.savefig(out_file, dpi=250)
    plt.close(fig)


def plot_best_mpd_vs_header(detailed: pd.DataFrame, best_factor: float, out_file: Path) -> None:
    sub = detailed[np.isclose(detailed["candidate_factor_to_cm"], best_factor)]
    if sub.empty:
        return

    x = safe_float(sub[MAG_COL]).to_numpy()
    y = safe_float(sub["Mpd_candidate"]).to_numpy()
    ok = np.isfinite(x) & np.isfinite(y)

    if ok.sum() < 5:
        return

    fig, ax = plt.subplots(figsize=(6.8, 6.2))
    ax.scatter(x[ok], y[ok], s=20, alpha=0.70)

    lo = float(np.nanmin([x[ok].min(), y[ok].min()]))
    hi = float(np.nanmax([x[ok].max(), y[ok].max()]))
    pad = 0.25
    ax.plot([lo - pad, hi + pad], [lo - pad, hi + pad], linewidth=1.2, label="1:1")
    ax.set_xlim(lo - pad, hi + pad)
    ax.set_ylim(lo - pad, hi + pad)
    ax.set_xlabel("Header ML")
    ax.set_ylabel("Mpd candidate")
    ax.set_title(f"Best unit-factor candidate: {best_factor:g} cm/internal-unit")
    ax.grid(True, alpha=0.3)
    ax.legend()

    fig.tight_layout()
    fig.savefig(out_file, dpi=250)
    plt.close(fig)


# =====================================================================
# MAIN
# =====================================================================

def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    if not INPUT_CSV.exists():
        raise FileNotFoundError(f"Missing input table: {INPUT_CSV}")

    df = pd.read_csv(INPUT_CSV)

    for col in [PD_CURRENT_CM_COL, MAG_COL, DIST_COL]:
        if col not in df.columns:
            raise ValueError(f"Missing required column: {col}")

    pd_current_cm = safe_float(df[PD_CURRENT_CM_COL]).to_numpy()
    header_ml = safe_float(df[MAG_COL]).to_numpy()
    dist_km = safe_float(df[DIST_COL]).to_numpy()

    ok = (
        np.isfinite(pd_current_cm)
        & np.isfinite(header_ml)
        & np.isfinite(dist_km)
        & (pd_current_cm > 0)
        & (dist_km > 0)
    )

    base = df.loc[ok].copy()
    pd_current_cm = pd_current_cm[ok]
    header_ml = header_ml[ok]
    dist_km = dist_km[ok]

    if len(base) < 5:
        raise RuntimeError("Too few valid rows.")

    # Reconstruct the internal amplitude number used by the script.
    # If CURRENT_FACTOR_USED_IN_PD_SCRIPT=1e-7, this is Pd in the internal unit
    # interpreted previously as nm.
    pd_internal = pd_current_cm / float(CURRENT_FACTOR_USED_IN_PD_SCRIPT)

    detailed_rows = []
    summary_rows = []

    for factor in CANDIDATE_FACTORS:
        factor = float(factor)
        pd_candidate_cm = pd_internal * factor
        mpd = calc_mpd(pd_candidate_cm, dist_km)
        residual = mpd - header_ml

        stat_x = residual.copy()
        if CLIP_FOR_SUMMARY:
            stat_x = stat_x[np.abs(stat_x) <= float(RESIDUAL_CLIP_MAG)]

        stats = robust_stats(stat_x)

        summary_rows.append({
            "candidate_factor_to_cm": factor,
            "log10_candidate_factor": np.log10(factor),
            "factor_label": classify_factor(factor),
            "n_used_summary": stats["n"],
            "mean_Mpd_minus_header": stats["mean"],
            "median_Mpd_minus_header": stats["median"],
            "std_Mpd_minus_header": stats["std"],
            "mad_Mpd_minus_header": stats["mad"],
            "robust_sigma_Mpd_minus_header": stats["robust_sigma"],
            "median_abs_Mpd_minus_header": stats["median_abs"],
        })

        temp = base.copy()
        temp["candidate_factor_to_cm"] = factor
        temp["log10_candidate_factor"] = np.log10(factor)
        temp["factor_label"] = classify_factor(factor)
        temp["Pd_internal_value"] = pd_internal
        temp["Pd_candidate_cm"] = pd_candidate_cm
        temp["Mpd_candidate"] = mpd
        temp["Mpd_minus_header"] = residual
        detailed_rows.append(temp)

    summary = pd.DataFrame(summary_rows)
    detailed = pd.concat(detailed_rows, ignore_index=True)

    # Best factor by minimum absolute median residual, not by station-by-station fit.
    # This checks global unit offset only.
    best_idx = summary["median_abs_Mpd_minus_header"].astype(float).idxmin()
    best = summary.loc[best_idx]
    best_factor = float(best["candidate_factor_to_cm"])

    summary_csv = OUTPUT_DIR / "pd_unit_factor_candidate_summary.csv"
    detailed_csv = OUTPUT_DIR / "pd_unit_factor_candidate_detailed.csv"
    report_txt = OUTPUT_DIR / "pd_unit_factor_candidate_report.txt"

    summary.to_csv(summary_csv, index=False)
    detailed.to_csv(detailed_csv, index=False)

    with open(report_txt, "w") as f:
        f.write("Pd unit-factor candidate test\n")
        f.write("=============================\n\n")
        f.write(f"Input table: {INPUT_CSV}\n")
        f.write(f"Current Pd cm column: {PD_CURRENT_CM_COL}\n")
        f.write(f"Current factor used in Pd script: {CURRENT_FACTOR_USED_IN_PD_SCRIPT:g}\n")
        f.write(f"Valid rows: {len(base)}\n\n")
        f.write("Equation\n")
        f.write("--------\n")
        f.write(f"Mpd = {WU_ZHAO_A} + {WU_ZHAO_B}*log10(Pd_cm) + {WU_ZHAO_C}*log10(R_km)\n\n")
        f.write("Important interpretation\n")
        f.write("------------------------\n")
        f.write("A factor of 10 in Pd changes Mpd by 1.371 magnitude units.\n")
        f.write("This script tests global unit offsets only. It does not force every station to match header ML.\n\n")
        f.write("Best candidate by minimum median absolute residual\n")
        f.write("--------------------------------------------------\n")
        for col in summary.columns:
            f.write(f"{col}: {best[col]}\n")
        f.write("\nCandidate summary\n")
        f.write("-----------------\n")
        f.write(summary.to_string(index=False))
        f.write("\n")

    if MAKE_PLOTS:
        plot_factor_summary(summary, OUTPUT_DIR / "Mpd_residual_by_unit_factor.png")
        plot_best_mpd_vs_header(detailed, best_factor, OUTPUT_DIR / "Mpd_vs_header_best_factor.png")

    print("========== DONE ==========")
    print(f"Summary : {summary_csv}")
    print(f"Details : {detailed_csv}")
    print(f"Report  : {report_txt}")
    print("")
    print("Best candidate:")
    print(f"  factor_to_cm = {best_factor:g}")
    print(f"  label        = {best['factor_label']}")
    print(f"  median residual Mpd-header = {best['median_Mpd_minus_header']:.4f}")
    print(f"  median abs residual        = {best['median_abs_Mpd_minus_header']:.4f}")
    print("")
    print("Use this only as a unit-factor diagnostic, not as final Pd regional calibration.")


if __name__ == "__main__":
    main()
