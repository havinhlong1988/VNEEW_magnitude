#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
04_regress_logPd_M_R_with_residual.py

Fit the empirical 3-second Pd relation:

    log10(Pd) = A + B*M + C*log10(R) ± residual

where:
    Pd = zero-to-peak displacement amplitude in the first 3 seconds after P
    M  = header ML
    R  = distance in km

The residual is estimated as the scatter of:
    residual_i = log10(Pd_observed_i) - log10(Pd_predicted_i)

Main residual estimate:
    sigma_res = sqrt(SSE / (N - 3))

Robust residual estimate:
    sigma_robust = 1.4826 * median(abs(residual_i - median(residual_i)))

Default input:
    output/03_report_P_amp_filter_2sdt/P_amp_tau_Z_3s_passed_2sdt.csv

Default output:
    output/04_regression_logPd_M_R/
"""

from __future__ import annotations

from pathlib import Path
from typing import Tuple

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

OUTPUT_DIR = Path("output/04_regression_logPd_M_R")

# Regression columns.
PD_COL = "Pd_3s_cm"
M_COL = "header_ML"
R_COL = "distance_km"

# Unit label only for report/plot.
PD_UNIT_LABEL = "cm"

# Regression mode:
#   station      : each station-event row is one data point
#   event_mean   : average logPd, M, logR by event first
#   event_median : median logPd, M, logR by event first
DATA_MODE = "station"

# Optional residual clipping after initial fit.
APPLY_RESIDUAL_CLIPPING = False
RESIDUAL_SIGMA_CLIP = 3.0
MAX_CLIP_ITERATIONS = 3

# Estimate residual using:
#   "ols"    : sigma_res = sqrt(SSE / (N - p))
#   "std"    : standard deviation of residuals
#   "robust" : 1.4826 * MAD residual
RESIDUAL_FOR_EQUATION = "ols"

MAKE_PLOTS = True
PLOT_DPI = 300

MIN_PD = 0.0
MIN_DISTANCE_KM = 0.0


# =====================================================================
# HELPERS
# =====================================================================

def safe_float_series(s: pd.Series) -> pd.Series:
    return pd.to_numeric(s, errors="coerce").astype(float)


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
            "min": np.nan,
            "max": np.nan,
            "p05": np.nan,
            "p16": np.nan,
            "p84": np.nan,
            "p95": np.nan,
        }

    med = float(np.median(x))
    mad = float(np.median(np.abs(x - med)))

    return {
        "n": int(x.size),
        "mean": float(np.mean(x)),
        "median": med,
        "std": float(np.std(x, ddof=1)) if x.size > 1 else np.nan,
        "mad": mad,
        "robust_sigma": float(1.4826 * mad),
        "min": float(np.min(x)),
        "max": float(np.max(x)),
        "p05": float(np.percentile(x, 5)),
        "p16": float(np.percentile(x, 16)),
        "p84": float(np.percentile(x, 84)),
        "p95": float(np.percentile(x, 95)),
    }


def choose_equation_residual_sigma(stats: dict) -> float:
    mode = str(RESIDUAL_FOR_EQUATION).lower().strip()

    if mode == "robust":
        return float(stats.get("residual_robust_sigma", np.nan))
    if mode == "std":
        return float(stats.get("residual_std", np.nan))

    # default: OLS sigma = sqrt(SSE / dof)
    return float(stats.get("sigma_res_ols", np.nan))


def prepare_data(df: pd.DataFrame) -> pd.DataFrame:
    required = [PD_COL, M_COL, R_COL]
    for col in required:
        if col not in df.columns:
            raise ValueError(f"Missing required column: {col}")

    out = df.copy()
    out[PD_COL] = safe_float_series(out[PD_COL])
    out[M_COL] = safe_float_series(out[M_COL])
    out[R_COL] = safe_float_series(out[R_COL])

    ok = (
        np.isfinite(out[PD_COL])
        & np.isfinite(out[M_COL])
        & np.isfinite(out[R_COL])
        & (out[PD_COL] > float(MIN_PD))
        & (out[R_COL] > float(MIN_DISTANCE_KM))
    )

    out = out.loc[ok].copy()
    out["log10_Pd"] = np.log10(out[PD_COL].astype(float))
    out["log10_R"] = np.log10(out[R_COL].astype(float))

    out = out[
        np.isfinite(out["log10_Pd"])
        & np.isfinite(out[M_COL])
        & np.isfinite(out["log10_R"])
    ].copy()

    mode = DATA_MODE.lower().strip()
    if mode in ["event_mean", "event_median"]:
        if "event" not in out.columns:
            raise ValueError("DATA_MODE requires column 'event'.")

        agg_func = "mean" if mode == "event_mean" else "median"

        out = (
            out.groupby("event")
            .agg(
                log10_Pd=("log10_Pd", agg_func),
                header_ML=(M_COL, agg_func),
                distance_km=(R_COL, agg_func),
                log10_R=("log10_R", agg_func),
                n_station=("event", "count"),
            )
            .reset_index()
        )

    return out


def fit_ols(work: pd.DataFrame) -> Tuple[np.ndarray, dict, pd.DataFrame]:
    y = work["log10_Pd"].to_numpy(dtype=float)
    M = work[M_COL].to_numpy(dtype=float) if M_COL in work.columns else work["header_ML"].to_numpy(dtype=float)
    logR = work["log10_R"].to_numpy(dtype=float)

    # X = [1, M, log10(R)]
    X = np.column_stack([np.ones_like(y), M, logR])

    beta, _residuals_lstsq, rank, singular_values = np.linalg.lstsq(X, y, rcond=None)

    y_pred = X @ beta
    residual = y - y_pred

    n = int(len(y))
    p = int(X.shape[1])
    dof = max(n - p, 1)

    sse = float(np.sum(residual ** 2))
    mse = sse / dof
    sigma_res_ols = float(np.sqrt(mse))
    rmse = float(np.sqrt(np.mean(residual ** 2)))

    y_mean = float(np.mean(y))
    sst = float(np.sum((y - y_mean) ** 2))
    r2 = float(1.0 - sse / sst) if sst > 0 else np.nan
    adj_r2 = float(1.0 - (1.0 - r2) * (n - 1) / dof) if np.isfinite(r2) and n > p else np.nan

    try:
        cov_beta = mse * np.linalg.inv(X.T @ X)
        se_beta = np.sqrt(np.diag(cov_beta))
    except np.linalg.LinAlgError:
        se_beta = np.full(p, np.nan)

    rstats = robust_stats(residual)

    stats = {
        "n": n,
        "n_parameters": p,
        "dof": dof,
        "rank": int(rank),
        "A": float(beta[0]),
        "B": float(beta[1]),
        "C": float(beta[2]),
        "A_se": float(se_beta[0]) if np.isfinite(se_beta[0]) else np.nan,
        "B_se": float(se_beta[1]) if np.isfinite(se_beta[1]) else np.nan,
        "C_se": float(se_beta[2]) if np.isfinite(se_beta[2]) else np.nan,
        "sse": sse,
        "mse": float(mse),
        "sigma_res_ols": sigma_res_ols,
        "rmse_log10Pd": rmse,
        "r2": r2,
        "adj_r2": adj_r2,
        "singular_values": ";".join([f"{v:.8g}" for v in singular_values]),
    }

    for k, v in rstats.items():
        stats[f"residual_{k}"] = v

    sigma_for_equation = choose_equation_residual_sigma(stats)
    stats["sigma_for_equation"] = sigma_for_equation
    stats["residual_for_equation_mode"] = RESIDUAL_FOR_EQUATION

    pred = work.copy()
    pred["log10_Pd_pred"] = y_pred
    pred["log10_Pd_residual"] = residual
    pred["Pd_observed"] = 10.0 ** y
    pred["Pd_pred"] = 10.0 ** y_pred

    # Invert the fitted relation to estimate magnitude from observed Pd:
    #   log10(Pd) = A + B*M + C*log10(R)
    #   Mpd = (log10(Pd) - A - C*log10(R)) / B
    if np.isfinite(beta[1]) and abs(beta[1]) > 0:
        pred["Mpd_from_observed_Pd"] = (
            pred["log10_Pd"].astype(float) - float(beta[0]) - float(beta[2]) * pred["log10_R"].astype(float)
        ) / float(beta[1])
        pred["Mpd_minus_header_ML"] = pred["Mpd_from_observed_Pd"] - pred[M_COL].astype(float)
        pred["header_ML_minus_Mpd"] = pred[M_COL].astype(float) - pred["Mpd_from_observed_Pd"]

        # Convert log10(Pd) residual scatter to equivalent magnitude scatter.
        sigma_for_mpd = float(stats["sigma_for_equation"]) / abs(float(beta[1])) if np.isfinite(stats["sigma_for_equation"]) else np.nan
        pred["Mpd_pred_minus_1sigma"] = pred["Mpd_from_observed_Pd"] - sigma_for_mpd
        pred["Mpd_pred_plus_1sigma"] = pred["Mpd_from_observed_Pd"] + sigma_for_mpd
    else:
        pred["Mpd_from_observed_Pd"] = np.nan
        pred["Mpd_minus_header_ML"] = np.nan
        pred["header_ML_minus_Mpd"] = np.nan
        pred["Mpd_pred_minus_1sigma"] = np.nan
        pred["Mpd_pred_plus_1sigma"] = np.nan

    # ±1 sigma and ±2 sigma in log space, converted back to Pd units.
    pred["log10_Pd_pred_minus_1sigma"] = y_pred - sigma_for_equation
    pred["log10_Pd_pred_plus_1sigma"] = y_pred + sigma_for_equation
    pred["log10_Pd_pred_minus_2sigma"] = y_pred - 2.0 * sigma_for_equation
    pred["log10_Pd_pred_plus_2sigma"] = y_pred + 2.0 * sigma_for_equation

    pred["Pd_pred_minus_1sigma"] = 10.0 ** pred["log10_Pd_pred_minus_1sigma"]
    pred["Pd_pred_plus_1sigma"] = 10.0 ** pred["log10_Pd_pred_plus_1sigma"]
    pred["Pd_pred_minus_2sigma"] = 10.0 ** pred["log10_Pd_pred_minus_2sigma"]
    pred["Pd_pred_plus_2sigma"] = 10.0 ** pred["log10_Pd_pred_plus_2sigma"]

    # Multiplicative amplitude scatter factor.
    pred["residual_amplitude_factor"] = 10.0 ** residual

    return beta, stats, pred


def iterative_clip_and_fit(work: pd.DataFrame) -> Tuple[np.ndarray, dict, pd.DataFrame]:
    current = work.copy()
    beta = None
    stats = {}
    pred = current.copy()

    for i in range(int(MAX_CLIP_ITERATIONS) + 1):
        beta, stats, pred_used = fit_ols(current)

        residual = pred_used["log10_Pd_residual"].to_numpy(dtype=float)
        rstd = float(np.nanstd(residual, ddof=1)) if len(residual) > 1 else np.nan

        if (
            not APPLY_RESIDUAL_CLIPPING
            or i >= int(MAX_CLIP_ITERATIONS)
            or not np.isfinite(rstd)
            or rstd <= 0
        ):
            pred = pred_used
            stats["clip_iterations_done"] = i
            stats["n_after_clip"] = len(current)
            return beta, stats, pred

        keep = np.abs(residual) <= float(RESIDUAL_SIGMA_CLIP) * rstd

        if keep.all():
            pred = pred_used
            stats["clip_iterations_done"] = i
            stats["n_after_clip"] = len(current)
            return beta, stats, pred

        current = current.loc[keep].copy()

        if len(current) < 5:
            pred = pred_used
            stats["clip_iterations_done"] = i
            stats["n_after_clip"] = len(current)
            return beta, stats, pred

    return beta, stats, pred


def make_coefficient_table(stats: dict) -> pd.DataFrame:
    return pd.DataFrame([
        {
            "parameter": "A",
            "value": stats["A"],
            "standard_error": stats.get("A_se", np.nan),
            "description": "intercept",
        },
        {
            "parameter": "B",
            "value": stats["B"],
            "standard_error": stats.get("B_se", np.nan),
            "description": f"coefficient for {M_COL}",
        },
        {
            "parameter": "C",
            "value": stats["C"],
            "standard_error": stats.get("C_se", np.nan),
            "description": f"coefficient for log10({R_COL})",
        },
        {
            "parameter": "sigma_res",
            "value": stats["sigma_for_equation"],
            "standard_error": np.nan,
            "description": f"residual scatter in log10(Pd), mode={RESIDUAL_FOR_EQUATION}",
        },
        {
            "parameter": "sigma_res_ols",
            "value": stats["sigma_res_ols"],
            "standard_error": np.nan,
            "description": "sqrt(SSE/(N-3)) in log10(Pd)",
        },
        {
            "parameter": "sigma_res_std",
            "value": stats["residual_std"],
            "standard_error": np.nan,
            "description": "standard deviation of residuals in log10(Pd)",
        },
        {
            "parameter": "sigma_res_robust",
            "value": stats["residual_robust_sigma"],
            "standard_error": np.nan,
            "description": "1.4826*MAD residual in log10(Pd)",
        },
    ])


def make_equation_summary(stats: dict) -> pd.DataFrame:
    A = stats["A"]
    B = stats["B"]
    C = stats["C"]
    sigma = stats["sigma_for_equation"]

    return pd.DataFrame([
        {
            "equation": "log10(Pd) = A + B*M + C*log10(R) ± sigma_res",
            "Pd_column": PD_COL,
            "Pd_unit": PD_UNIT_LABEL,
            "M_column": M_COL,
            "R_column": R_COL,
            "A": A,
            "B": B,
            "C": C,
            "sigma_res": sigma,
            "sigma_res_mode": RESIDUAL_FOR_EQUATION,
            "sigma_res_ols": stats["sigma_res_ols"],
            "sigma_res_std": stats["residual_std"],
            "sigma_res_robust": stats["residual_robust_sigma"],
            "amplitude_factor_1sigma": 10.0 ** sigma if np.isfinite(sigma) else np.nan,
            "amplitude_factor_2sigma": 10.0 ** (2.0 * sigma) if np.isfinite(sigma) else np.nan,
            "sigma_Mpd_equivalent": sigma / abs(B) if np.isfinite(sigma) and np.isfinite(B) and abs(B) > 0 else np.nan,
            "n": stats["n"],
            "dof": stats["dof"],
            "r2": stats["r2"],
            "adj_r2": stats["adj_r2"],
            "rmse_log10Pd": stats["rmse_log10Pd"],
            "residual_median": stats["residual_median"],
            "residual_mean": stats["residual_mean"],
        }
    ])


def write_report(
    stats: dict,
    coef_csv: Path,
    equation_csv: Path,
    pred_csv: Path,
    report_txt: Path,
) -> None:
    A = stats["A"]
    B = stats["B"]
    C = stats["C"]
    sigma = stats["sigma_for_equation"]

    amp_factor_1 = 10.0 ** sigma if np.isfinite(sigma) else np.nan
    amp_factor_2 = 10.0 ** (2.0 * sigma) if np.isfinite(sigma) else np.nan

    with open(report_txt, "w") as f:
        f.write("Linear regression for 3-second peak displacement\n")
        f.write("================================================\n\n")

        f.write("Regression equation\n")
        f.write("-------------------\n")
        f.write("log10(Pd) = A + B*M + C*log10(R) ± sigma_res\n")
        f.write(f"Pd column: {PD_COL}\n")
        f.write(f"Pd unit: {PD_UNIT_LABEL}\n")
        f.write(f"M column: {M_COL}\n")
        f.write(f"R column: {R_COL} [km]\n")
        f.write(f"Data mode: {DATA_MODE}\n\n")

        f.write("Best-fit equation\n")
        f.write("-----------------\n")
        f.write(
            f"log10(Pd_{PD_UNIT_LABEL}) = "
            f"{A:.8f} + {B:.8f}*M + {C:.8f}*log10(R_km) "
            f"± {sigma:.8f}\n\n"
        )

        f.write("Residual meaning\n")
        f.write("----------------\n")
        f.write(f"sigma_res mode: {RESIDUAL_FOR_EQUATION}\n")
        f.write("sigma_res is in log10(Pd) units.\n")
        f.write(f"1-sigma amplitude factor = 10^sigma = {amp_factor_1:.6g}\n")
        f.write(f"2-sigma amplitude factor = 10^(2*sigma) = {amp_factor_2:.6g}\n")
        if np.isfinite(B) and abs(B) > 0 and np.isfinite(sigma):
            f.write(f"Equivalent sigma in Mpd = sigma_res / |B| = {sigma / abs(B):.6g}\n")
        f.write("\n")

        f.write("Magnitude inversion\n")
        f.write("-------------------\n")
        f.write("Mpd = (log10(Pd) - A - C*log10(R)) / B\n")
        f.write("This is used only for the Mpd versus header ML diagnostic plot.\n\n")

        f.write("Coefficients\n")
        f.write("------------\n")
        f.write(f"A = {A:.10f} ± {stats.get('A_se', np.nan):.10f}\n")
        f.write(f"B = {B:.10f} ± {stats.get('B_se', np.nan):.10f}\n")
        f.write(f"C = {C:.10f} ± {stats.get('C_se', np.nan):.10f}\n")
        f.write(f"sigma_res = {sigma:.10f}\n")
        f.write(f"sigma_res_ols = {stats['sigma_res_ols']:.10f}\n")
        f.write(f"sigma_res_std = {stats['residual_std']:.10f}\n")
        f.write(f"sigma_res_robust = {stats['residual_robust_sigma']:.10f}\n\n")

        f.write("Fit statistics\n")
        f.write("--------------\n")
        keys = [
            "n", "n_parameters", "dof", "rank",
            "r2", "adj_r2", "rmse_log10Pd",
            "residual_mean", "residual_median", "residual_std",
            "residual_mad", "residual_robust_sigma",
            "residual_min", "residual_p05", "residual_p16",
            "residual_p84", "residual_p95", "residual_max",
            "clip_iterations_done", "n_after_clip",
        ]

        for k in keys:
            if k in stats:
                f.write(f"{k}: {stats[k]}\n")

        f.write("\nOutput files\n")
        f.write("------------\n")
        f.write(f"Equation CSV: {equation_csv}\n")
        f.write(f"Coefficient CSV: {coef_csv}\n")
        f.write(f"Prediction CSV: {pred_csv}\n")


def plot_observed_vs_predicted(pred: pd.DataFrame, stats: dict, out_file: Path) -> None:
    x = pred["log10_Pd"].to_numpy(dtype=float)
    y = pred["log10_Pd_pred"].to_numpy(dtype=float)
    sigma = stats["sigma_for_equation"]

    ok = np.isfinite(x) & np.isfinite(y)
    if ok.sum() < 3:
        return

    residual = pred.loc[ok, "log10_Pd_residual"].to_numpy(dtype=float)

    fig, ax = plt.subplots(figsize=(6.8, 6.2))
    ax.scatter(x[ok], y[ok], s=20, alpha=0.75)

    lo = float(np.nanmin([np.nanmin(x[ok]), np.nanmin(y[ok])]))
    hi = float(np.nanmax([np.nanmax(x[ok]), np.nanmax(y[ok])]))
    pad = 0.15

    xs = np.array([lo - pad, hi + pad])
    ax.plot(xs, xs, linewidth=1.2, label="1:1")
    if np.isfinite(sigma):
        ax.plot(xs, xs + sigma, linestyle="--", linewidth=1.0, label="+1 sigma")
        ax.plot(xs, xs - sigma, linestyle="--", linewidth=1.0, label="-1 sigma")

    ax.set_xlim(lo - pad, hi + pad)
    ax.set_ylim(lo - pad, hi + pad)

    ax.set_xlabel(f"Observed log10(Pd [{PD_UNIT_LABEL}])")
    ax.set_ylabel(f"Predicted log10(Pd [{PD_UNIT_LABEL}])")
    ax.set_title(
        "Observed versus predicted log10(Pd)\n"
        f"sigma={sigma:.4f}, RMSE={np.sqrt(np.mean(residual**2)):.4f}, n={ok.sum()}"
    )
    ax.grid(True, alpha=0.3)
    ax.legend()

    fig.tight_layout()
    fig.savefig(out_file, dpi=int(PLOT_DPI))
    plt.close(fig)


def plot_residual_hist(pred: pd.DataFrame, stats: dict, out_file: Path) -> None:
    vals = pred["log10_Pd_residual"].to_numpy(dtype=float)
    vals = vals[np.isfinite(vals)]
    sigma = stats["sigma_for_equation"]

    if vals.size < 3:
        return

    fig, ax = plt.subplots(figsize=(7.2, 5.3))
    ax.hist(vals, bins=35, alpha=0.75)
    ax.axvline(0.0, linewidth=1.0, label="zero")
    ax.axvline(float(np.median(vals)), linestyle="--", linewidth=1.1, label="median")

    if np.isfinite(sigma):
        ax.axvline(+sigma, linestyle=":", linewidth=1.2, label="+1 sigma")
        ax.axvline(-sigma, linestyle=":", linewidth=1.2, label="-1 sigma")

    ax.set_xlabel("Observed - predicted log10(Pd)")
    ax.set_ylabel("Count")
    ax.set_title(
        "Regression residual histogram\n"
        f"sigma={sigma:.4f}, mean={np.mean(vals):.4f}, "
        f"median={np.median(vals):.4f}, std={np.std(vals, ddof=1):.4f}"
    )
    ax.grid(True, alpha=0.3)
    ax.legend()

    fig.tight_layout()
    fig.savefig(out_file, dpi=int(PLOT_DPI))
    plt.close(fig)


def plot_residual_vs_x(pred: pd.DataFrame, x_col: str, x_label: str, out_file: Path) -> None:
    if x_col not in pred.columns:
        return

    x = pd.to_numeric(pred[x_col], errors="coerce").to_numpy(dtype=float)
    y = pred["log10_Pd_residual"].to_numpy(dtype=float)

    ok = np.isfinite(x) & np.isfinite(y)
    if ok.sum() < 3:
        return

    fig, ax = plt.subplots(figsize=(7.2, 5.3))
    ax.scatter(x[ok], y[ok], s=20, alpha=0.75)
    ax.axhline(0.0, linewidth=1.0)

    ax.set_xlabel(x_label)
    ax.set_ylabel("Observed - predicted log10(Pd)")
    ax.set_title(f"Regression residual versus {x_label}")
    ax.grid(True, alpha=0.3)

    if ok.sum() >= 5:
        coef = np.polyfit(x[ok], y[ok], 1)
        xs = np.linspace(float(np.nanmin(x[ok])), float(np.nanmax(x[ok])), 100)
        ax.plot(xs, coef[0] * xs + coef[1], linewidth=1.2,
                label=f"trend: y={coef[0]:.4f}x+{coef[1]:.4f}")
        ax.legend()

    fig.tight_layout()
    fig.savefig(out_file, dpi=int(PLOT_DPI))
    plt.close(fig)



def plot_mpd_vs_header_ml(pred: pd.DataFrame, stats: dict, out_file: Path) -> None:
    """
    Plot magnitude estimated from observed Pd against header ML.

    Mpd = (log10(Pd) - A - C*log10(R)) / B
    """
    if "Mpd_from_observed_Pd" not in pred.columns or M_COL not in pred.columns:
        return

    x = pd.to_numeric(pred[M_COL], errors="coerce").to_numpy(dtype=float)
    y = pd.to_numeric(pred["Mpd_from_observed_Pd"], errors="coerce").to_numpy(dtype=float)
    r = pd.to_numeric(pred["distance_km"], errors="coerce").to_numpy(dtype=float) if "distance_km" in pred.columns else None

    ok = np.isfinite(x) & np.isfinite(y)
    if ok.sum() < 3:
        return

    residual = y[ok] - x[ok]

    B = float(stats.get("B", np.nan))
    sigma_logpd = float(stats.get("sigma_for_equation", np.nan))
    sigma_mpd = sigma_logpd / abs(B) if np.isfinite(sigma_logpd) and np.isfinite(B) and abs(B) > 0 else np.nan

    fig, ax = plt.subplots(figsize=(6.8, 6.2))

    if r is not None and np.isfinite(r[ok]).sum() >= 3:
        sc = ax.scatter(x[ok], y[ok], c=r[ok], s=22, alpha=0.78)
        cbar = fig.colorbar(sc, ax=ax)
        cbar.set_label("Distance [km]")
    else:
        ax.scatter(x[ok], y[ok], s=22, alpha=0.78)

    lo = float(np.nanmin([np.nanmin(x[ok]), np.nanmin(y[ok])]))
    hi = float(np.nanmax([np.nanmax(x[ok]), np.nanmax(y[ok])]))
    pad = 0.25

    xs = np.array([lo - pad, hi + pad])
    ax.plot(xs, xs, linewidth=1.2, label="1:1")

    if np.isfinite(sigma_mpd):
        ax.plot(xs, xs + sigma_mpd, linestyle="--", linewidth=1.0, label="+1 sigma")
        ax.plot(xs, xs - sigma_mpd, linestyle="--", linewidth=1.0, label="-1 sigma")

    ax.set_xlim(lo - pad, hi + pad)
    ax.set_ylim(lo - pad, hi + pad)
    ax.set_xlabel("Header ML")
    ax.set_ylabel("Mpd from observed Pd")
    ax.set_title(
        "Mpd from 3-s Pd versus header ML\n"
        f"median(Mpd-ML)={np.nanmedian(residual):.3f}, "
        f"mean={np.nanmean(residual):.3f}, "
        f"std={np.nanstd(residual, ddof=1):.3f}, n={ok.sum()}"
    )
    ax.grid(True, alpha=0.3)
    ax.legend()

    fig.tight_layout()
    fig.savefig(out_file, dpi=int(PLOT_DPI))
    plt.close(fig)


def plot_mpd_minus_ml_hist(pred: pd.DataFrame, out_file: Path) -> None:
    """
    Plot histogram of Mpd - header ML.
    """
    if "Mpd_minus_header_ML" not in pred.columns:
        return

    vals = pd.to_numeric(pred["Mpd_minus_header_ML"], errors="coerce").to_numpy(dtype=float)
    vals = vals[np.isfinite(vals)]

    if vals.size < 3:
        return

    fig, ax = plt.subplots(figsize=(7.2, 5.3))
    ax.hist(vals, bins=35, alpha=0.75)
    ax.axvline(0.0, linewidth=1.0, label="zero")
    ax.axvline(float(np.nanmedian(vals)), linestyle="--", linewidth=1.1, label="median")

    ax.set_xlabel("Mpd - header ML")
    ax.set_ylabel("Count")
    ax.set_title(
        "Mpd residual histogram\n"
        f"mean={np.nanmean(vals):.3f}, median={np.nanmedian(vals):.3f}, "
        f"std={np.nanstd(vals, ddof=1):.3f}, n={vals.size}"
    )
    ax.grid(True, alpha=0.3)
    ax.legend()

    fig.tight_layout()
    fig.savefig(out_file, dpi=int(PLOT_DPI))
    plt.close(fig)


def plot_mpd_minus_ml_vs_distance(pred: pd.DataFrame, out_file: Path) -> None:
    """
    Plot Mpd - header ML versus distance.
    """
    if "Mpd_minus_header_ML" not in pred.columns or "distance_km" not in pred.columns:
        return

    x = pd.to_numeric(pred["distance_km"], errors="coerce").to_numpy(dtype=float)
    y = pd.to_numeric(pred["Mpd_minus_header_ML"], errors="coerce").to_numpy(dtype=float)

    ok = np.isfinite(x) & np.isfinite(y)
    if ok.sum() < 3:
        return

    fig, ax = plt.subplots(figsize=(7.2, 5.3))
    ax.scatter(x[ok], y[ok], s=22, alpha=0.75)
    ax.axhline(0.0, linewidth=1.0)

    if ok.sum() >= 5:
        coef = np.polyfit(x[ok], y[ok], 1)
        xs = np.linspace(float(np.nanmin(x[ok])), float(np.nanmax(x[ok])), 100)
        ax.plot(xs, coef[0] * xs + coef[1], linewidth=1.2,
                label=f"trend: y={coef[0]:.4f}x+{coef[1]:.4f}")
        ax.legend()

    ax.set_xlabel("Distance [km]")
    ax.set_ylabel("Mpd - header ML")
    ax.set_title("Mpd residual versus distance")
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(out_file, dpi=int(PLOT_DPI))
    plt.close(fig)



def make_plots(pred: pd.DataFrame, stats: dict) -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    plot_observed_vs_predicted(
        pred,
        stats,
        OUTPUT_DIR / "logPd_observed_vs_predicted.png",
    )
    plot_residual_hist(
        pred,
        stats,
        OUTPUT_DIR / "logPd_residual_hist.png",
    )
    plot_residual_vs_x(
        pred,
        M_COL if M_COL in pred.columns else "header_ML",
        "Header ML",
        OUTPUT_DIR / "logPd_residual_vs_ML.png",
    )
    plot_residual_vs_x(
        pred,
        R_COL if R_COL in pred.columns else "distance_km",
        "Distance [km]",
        OUTPUT_DIR / "logPd_residual_vs_distance.png",
    )
    plot_residual_vs_x(
        pred,
        "log10_R",
        "log10(R [km])",
        OUTPUT_DIR / "logPd_residual_vs_logR.png",
    )

    plot_mpd_vs_header_ml(
        pred,
        stats,
        OUTPUT_DIR / "Mpd_from_Pd_vs_header_ML.png",
    )
    plot_mpd_minus_ml_hist(
        pred,
        OUTPUT_DIR / "Mpd_minus_header_ML_hist.png",
    )
    plot_mpd_minus_ml_vs_distance(
        pred,
        OUTPUT_DIR / "Mpd_minus_header_ML_vs_distance.png",
    )


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    if not INPUT_CSV.exists():
        raise FileNotFoundError(f"Missing input CSV: {INPUT_CSV}")

    print(f"[INFO] Reading: {INPUT_CSV}")
    df = pd.read_csv(INPUT_CSV)

    work = prepare_data(df)

    if len(work) < 5:
        raise RuntimeError(f"Too few valid rows for regression: {len(work)}")

    print(f"[INFO] Regression rows: {len(work)}")
    print(f"[INFO] DATA_MODE: {DATA_MODE}")
    print("[INFO] Fitting: log10(Pd) = A + B*M + C*log10(R) ± residual")

    _beta, stats, pred = iterative_clip_and_fit(work)

    coef_df = make_coefficient_table(stats)
    equation_df = make_equation_summary(stats)

    coef_csv = OUTPUT_DIR / "logPd_regression_coefficients.csv"
    equation_csv = OUTPUT_DIR / "logPd_regression_equation_summary.csv"
    pred_csv = OUTPUT_DIR / "logPd_regression_predictions.csv"
    report_txt = OUTPUT_DIR / "logPd_regression_coefficients.txt"

    coef_df.to_csv(coef_csv, index=False)
    equation_df.to_csv(equation_csv, index=False)
    pred.to_csv(pred_csv, index=False)

    write_report(stats, coef_csv, equation_csv, pred_csv, report_txt)

    if MAKE_PLOTS:
        print("[INFO] Making plots...")
        make_plots(pred, stats)

    A, B, C = stats["A"], stats["B"], stats["C"]
    sigma = stats["sigma_for_equation"]

    print("========== DONE ==========")
    print("Equation:")
    print(
        f"  log10(Pd_{PD_UNIT_LABEL}) = "
        f"{A:.8f} + {B:.8f}*M + {C:.8f}*log10(R_km) ± {sigma:.8f}"
    )
    print("")
    print(f"residual mode : {RESIDUAL_FOR_EQUATION}")
    print(f"sigma_res     : {sigma:.6f} log10(Pd)")
    print(f"amp factor 1σ : {10.0**sigma:.4f}")
    print(f"amp factor 2σ : {10.0**(2.0*sigma):.4f}")
    if np.isfinite(stats["B"]) and abs(stats["B"]) > 0:
        print(f"sigma Mpd     : {sigma/abs(stats['B']):.4f}")
    print("")
    print(f"n        : {stats['n']}")
    print(f"R2       : {stats['r2']:.4f}")
    print(f"adj R2   : {stats['adj_r2']:.4f}")
    print(f"RMSE     : {stats['rmse_log10Pd']:.4f}")
    print("")
    print(f"Report   : {report_txt}")
    print(f"Equation : {equation_csv}")
    print(f"Coef CSV : {coef_csv}")
    print(f"Pred CSV : {pred_csv}")

    if MAKE_PLOTS:
        print("Figures:")
        print(f"  {OUTPUT_DIR / 'logPd_observed_vs_predicted.png'}")
        print(f"  {OUTPUT_DIR / 'logPd_residual_hist.png'}")
        print(f"  {OUTPUT_DIR / 'logPd_residual_vs_ML.png'}")
        print(f"  {OUTPUT_DIR / 'logPd_residual_vs_distance.png'}")
        print(f"  {OUTPUT_DIR / 'logPd_residual_vs_logR.png'}")
        print(f"  {OUTPUT_DIR / 'Mpd_from_Pd_vs_header_ML.png'}")
        print(f"  {OUTPUT_DIR / 'Mpd_minus_header_ML_hist.png'}")
        print(f"  {OUTPUT_DIR / 'Mpd_minus_header_ML_vs_distance.png'}")


if __name__ == "__main__":
    main()
