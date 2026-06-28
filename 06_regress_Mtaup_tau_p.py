#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
06_regress_Mtaup_tau_p.py

Regression for Mtaup:

    Mtaup = A + B*log10(tau) ± sigma_res

where:
    tau = tau_p_max_3s_sec
    target magnitude = header_ML

Input:
    output/03_report_P_amp_filter_2sdt/P_amp_tau_Z_3s_passed_2sdt.csv

Outputs:
    output/06_regression_Mtaup_tau_p/
        Mtaup_regression_report.txt
        Mtaup_regression_equation_summary.csv
        Mtaup_regression_coefficients.csv
        Mtaup_regression_predictions.csv
        Mtaup_vs_header_ML.png
        Mtaup_residual_hist.png
        Mtaup_residual_vs_log10_tau.png
        Mtaup_residual_vs_distance.png
"""

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

OUTPUT_DIR = Path("output/06_regression_Mtaup_tau_p")

TAU_COL = "tau_p_max_3s_sec"
TAU_UNIT = "sec"
M_COL = "header_ML"
R_COL = "distance_km"

EQUATION_NAME = "Mtaup"

# station      : each station-event row is one regression datum
# event_mean   : average log10(tau), M, distance by event first
# event_median : median log10(tau), M, distance by event first
DATA_MODE = "station"

# residual in final equation:
#   ols    = sqrt(SSE/(N-2))
#   std    = standard deviation of residuals
#   robust = 1.4826*MAD of residuals
RESIDUAL_FOR_EQUATION = "ols"

APPLY_RESIDUAL_CLIPPING = False
RESIDUAL_SIGMA_CLIP = 3.0
MAX_CLIP_ITERATIONS = 3

MIN_TAU_SEC = 0.0

MAKE_PLOTS = True
PLOT_DPI = 300


# =====================================================================
# HELPERS
# =====================================================================

def safe_float_series(s):
    return pd.to_numeric(s, errors="coerce").astype(float)


def robust_stats(x):
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


def choose_sigma(stats):
    mode = str(RESIDUAL_FOR_EQUATION).lower().strip()

    if mode == "robust":
        return float(stats.get("residual_robust_sigma", np.nan))
    if mode == "std":
        return float(stats.get("residual_std", np.nan))

    return float(stats.get("sigma_res_ols", np.nan))


def prepare_data(df):
    required = [TAU_COL, M_COL]
    for col in required:
        if col not in df.columns:
            raise ValueError(f"Missing required column: {col}")

    out = df.copy()
    out[TAU_COL] = safe_float_series(out[TAU_COL])
    out[M_COL] = safe_float_series(out[M_COL])

    if R_COL in out.columns:
        out[R_COL] = safe_float_series(out[R_COL])
    else:
        out[R_COL] = np.nan

    ok = (
        np.isfinite(out[TAU_COL])
        & np.isfinite(out[M_COL])
        & (out[TAU_COL] > float(MIN_TAU_SEC))
    )

    out = out.loc[ok].copy()
    out["log10_tau"] = np.log10(out[TAU_COL].astype(float))
    out = out[np.isfinite(out["log10_tau"]) & np.isfinite(out[M_COL])].copy()

    mode = DATA_MODE.lower().strip()
    if mode in ["event_mean", "event_median"]:
        if "event" not in out.columns:
            raise ValueError("DATA_MODE requires column 'event'.")

        agg_func = "mean" if mode == "event_mean" else "median"

        out = (
            out.groupby("event")
            .agg(
                log10_tau=("log10_tau", agg_func),
                tau_value=(TAU_COL, agg_func),
                header_ML=(M_COL, agg_func),
                distance_km=(R_COL, agg_func),
                n_station=("event", "count"),
            )
            .reset_index()
        )

    return out


def fit_ols(work):
    y = work[M_COL].to_numpy(dtype=float) if M_COL in work.columns else work["header_ML"].to_numpy(dtype=float)
    logtau = work["log10_tau"].to_numpy(dtype=float)

    X = np.column_stack([np.ones_like(y), logtau])
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
        "A_se": float(se_beta[0]) if np.isfinite(se_beta[0]) else np.nan,
        "B_se": float(se_beta[1]) if np.isfinite(se_beta[1]) else np.nan,
        "sse": sse,
        "mse": float(mse),
        "sigma_res_ols": sigma_res_ols,
        "rmse_M": rmse,
        "r2": r2,
        "adj_r2": adj_r2,
        "singular_values": ";".join([f"{v:.8g}" for v in singular_values]),
    }

    for k, v in rstats.items():
        stats[f"residual_{k}"] = v

    stats["sigma_for_equation"] = choose_sigma(stats)
    stats["residual_for_equation_mode"] = RESIDUAL_FOR_EQUATION

    pred = work.copy()
    pred["M_observed"] = y
    pred["M_pred"] = y_pred
    pred["M_residual_observed_minus_pred"] = residual
    pred[f"{EQUATION_NAME}_pred"] = y_pred
    pred[f"{EQUATION_NAME}_minus_header_ML"] = y_pred - y
    pred[f"header_ML_minus_{EQUATION_NAME}"] = y - y_pred

    sigma = stats["sigma_for_equation"]
    pred[f"{EQUATION_NAME}_pred_minus_1sigma"] = y_pred - sigma
    pred[f"{EQUATION_NAME}_pred_plus_1sigma"] = y_pred + sigma
    pred[f"{EQUATION_NAME}_pred_minus_2sigma"] = y_pred - 2.0 * sigma
    pred[f"{EQUATION_NAME}_pred_plus_2sigma"] = y_pred + 2.0 * sigma

    return beta, stats, pred


def iterative_clip_and_fit(work):
    current = work.copy()
    beta = None
    stats = {}
    pred = current.copy()

    for i in range(int(MAX_CLIP_ITERATIONS) + 1):
        beta, stats, pred_used = fit_ols(current)

        residual = pred_used["M_residual_observed_minus_pred"].to_numpy(dtype=float)
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


def make_coefficient_table(stats):
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
            "description": f"coefficient for log10({TAU_COL})",
        },
        {
            "parameter": "sigma_res",
            "value": stats["sigma_for_equation"],
            "standard_error": np.nan,
            "description": f"magnitude residual scatter, mode={RESIDUAL_FOR_EQUATION}",
        },
        {
            "parameter": "sigma_res_ols",
            "value": stats["sigma_res_ols"],
            "standard_error": np.nan,
            "description": "sqrt(SSE/(N-2)) in magnitude units",
        },
        {
            "parameter": "sigma_res_std",
            "value": stats["residual_std"],
            "standard_error": np.nan,
            "description": "standard deviation of magnitude residuals",
        },
        {
            "parameter": "sigma_res_robust",
            "value": stats["residual_robust_sigma"],
            "standard_error": np.nan,
            "description": "1.4826*MAD of magnitude residuals",
        },
    ])


def make_equation_summary(stats):
    return pd.DataFrame([
        {
            "equation_name": EQUATION_NAME,
            "equation": "M_tau = A + B*log10(tau) ± sigma_res",
            "tau_column": TAU_COL,
            "tau_unit": TAU_UNIT,
            "M_column": M_COL,
            "A": stats["A"],
            "B": stats["B"],
            "sigma_res": stats["sigma_for_equation"],
            "sigma_res_mode": RESIDUAL_FOR_EQUATION,
            "sigma_res_ols": stats["sigma_res_ols"],
            "sigma_res_std": stats["residual_std"],
            "sigma_res_robust": stats["residual_robust_sigma"],
            "n": stats["n"],
            "dof": stats["dof"],
            "r2": stats["r2"],
            "adj_r2": stats["adj_r2"],
            "rmse_M": stats["rmse_M"],
            "residual_median": stats["residual_median"],
            "residual_mean": stats["residual_mean"],
        }
    ])


def write_report(stats, coef_csv, equation_csv, pred_csv, report_txt):
    A = stats["A"]
    B = stats["B"]
    sigma = stats["sigma_for_equation"]

    with open(report_txt, "w") as f:
        f.write(f"Linear regression for {EQUATION_NAME}\\n")
        f.write("=" * (22 + len(EQUATION_NAME)) + "\\n\\n")

        f.write("Regression equation\\n")
        f.write("-------------------\\n")
        f.write("M_tau = A + B*log10(tau) ± sigma_res\\n")
        f.write(f"tau column: {TAU_COL}\\n")
        f.write(f"tau unit: {TAU_UNIT}\\n")
        f.write(f"M column: {M_COL}\\n")
        f.write(f"Data mode: {DATA_MODE}\\n\\n")

        f.write("Best-fit equation\\n")
        f.write("-----------------\\n")
        f.write(
            f"{EQUATION_NAME} = {A:.8f} + {B:.8f}*log10({TAU_COL}) "
            f"± {sigma:.8f}\\n\\n"
        )

        f.write("Residual meaning\\n")
        f.write("----------------\\n")
        f.write(f"sigma_res mode: {RESIDUAL_FOR_EQUATION}\\n")
        f.write("sigma_res is in magnitude units.\\n\\n")

        f.write("Coefficients\\n")
        f.write("------------\\n")
        f.write(f"A = {A:.10f} ± {stats.get('A_se', np.nan):.10f}\\n")
        f.write(f"B = {B:.10f} ± {stats.get('B_se', np.nan):.10f}\\n")
        f.write(f"sigma_res = {sigma:.10f}\\n")
        f.write(f"sigma_res_ols = {stats['sigma_res_ols']:.10f}\\n")
        f.write(f"sigma_res_std = {stats['residual_std']:.10f}\\n")
        f.write(f"sigma_res_robust = {stats['residual_robust_sigma']:.10f}\\n\\n")

        f.write("Fit statistics\\n")
        f.write("--------------\\n")
        keys = [
            "n", "n_parameters", "dof", "rank",
            "r2", "adj_r2", "rmse_M",
            "residual_mean", "residual_median", "residual_std",
            "residual_mad", "residual_robust_sigma",
            "residual_min", "residual_p05", "residual_p16",
            "residual_p84", "residual_p95", "residual_max",
            "clip_iterations_done", "n_after_clip",
        ]

        for k in keys:
            if k in stats:
                f.write(f"{k}: {stats[k]}\\n")

        f.write("\\nOutput files\\n")
        f.write("------------\\n")
        f.write(f"Equation CSV: {equation_csv}\\n")
        f.write(f"Coefficient CSV: {coef_csv}\\n")
        f.write(f"Prediction CSV: {pred_csv}\\n")


def plot_mtau_vs_ml(pred, stats, out_file):
    x = pd.to_numeric(pred[M_COL], errors="coerce").to_numpy(dtype=float) if M_COL in pred.columns else pred["M_observed"].to_numpy(dtype=float)
    y = pred["M_pred"].to_numpy(dtype=float)
    dist = pd.to_numeric(pred[R_COL], errors="coerce").to_numpy(dtype=float) if R_COL in pred.columns else None

    ok = np.isfinite(x) & np.isfinite(y)
    if ok.sum() < 3:
        return

    residual = y[ok] - x[ok]
    sigma = float(stats.get("sigma_for_equation", np.nan))

    fig, ax = plt.subplots(figsize=(6.8, 6.2))

    if dist is not None and np.isfinite(dist[ok]).sum() >= 3:
        sc = ax.scatter(x[ok], y[ok], c=dist[ok], s=22, alpha=0.78)
        cbar = fig.colorbar(sc, ax=ax)
        cbar.set_label("Distance [km]")
    else:
        ax.scatter(x[ok], y[ok], s=22, alpha=0.78)

    lo = float(np.nanmin([np.nanmin(x[ok]), np.nanmin(y[ok])]))
    hi = float(np.nanmax([np.nanmax(x[ok]), np.nanmax(y[ok])]))
    pad = 0.25
    xs = np.array([lo - pad, hi + pad])

    ax.plot(xs, xs, linewidth=1.2, label="1:1")

    if np.isfinite(sigma):
        ax.plot(xs, xs + sigma, linestyle="--", linewidth=1.0, label="+1 sigma")
        ax.plot(xs, xs - sigma, linestyle="--", linewidth=1.0, label="-1 sigma")

    ax.set_xlim(lo - pad, hi + pad)
    ax.set_ylim(lo - pad, hi + pad)
    ax.set_xlabel("Header ML")
    ax.set_ylabel(EQUATION_NAME)
    ax.set_title(
        f"{EQUATION_NAME} versus header ML\\n"
        f"median({EQUATION_NAME}-ML)={np.nanmedian(residual):.3f}, "
        f"mean={np.nanmean(residual):.3f}, "
        f"std={np.nanstd(residual, ddof=1):.3f}, n={ok.sum()}"
    )
    ax.grid(True, alpha=0.3)
    ax.legend()

    fig.tight_layout()
    fig.savefig(out_file, dpi=int(PLOT_DPI))
    plt.close(fig)


def plot_residual_hist(pred, stats, out_file):
    # Positive means header ML is larger than M_tau.
    vals = pred["M_residual_observed_minus_pred"].to_numpy(dtype=float)
    vals = vals[np.isfinite(vals)]
    sigma = float(stats.get("sigma_for_equation", np.nan))

    if vals.size < 3:
        return

    fig, ax = plt.subplots(figsize=(7.2, 5.3))
    ax.hist(vals, bins=35, alpha=0.75)
    ax.axvline(0.0, linewidth=1.0, label="zero")
    ax.axvline(float(np.nanmedian(vals)), linestyle="--", linewidth=1.1, label="median")

    if np.isfinite(sigma):
        ax.axvline(+sigma, linestyle=":", linewidth=1.2, label="+1 sigma")
        ax.axvline(-sigma, linestyle=":", linewidth=1.2, label="-1 sigma")

    ax.set_xlabel(f"Header ML - {EQUATION_NAME}")
    ax.set_ylabel("Count")
    ax.set_title(
        f"{EQUATION_NAME} residual histogram\\n"
        f"mean={np.nanmean(vals):.3f}, median={np.nanmedian(vals):.3f}, "
        f"std={np.nanstd(vals, ddof=1):.3f}, n={vals.size}"
    )
    ax.grid(True, alpha=0.3)
    ax.legend()

    fig.tight_layout()
    fig.savefig(out_file, dpi=int(PLOT_DPI))
    plt.close(fig)


def plot_residual_vs_x(pred, x_col, x_label, out_file):
    if x_col not in pred.columns:
        return

    x = pd.to_numeric(pred[x_col], errors="coerce").to_numpy(dtype=float)
    y = pred["M_residual_observed_minus_pred"].to_numpy(dtype=float)

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

    ax.set_xlabel(x_label)
    ax.set_ylabel(f"Header ML - {EQUATION_NAME}")
    ax.set_title(f"Header ML - {EQUATION_NAME} versus {x_label}")
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(out_file, dpi=int(PLOT_DPI))
    plt.close(fig)


def make_plots(pred, stats):
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    plot_mtau_vs_ml(
        pred,
        stats,
        OUTPUT_DIR / f"{EQUATION_NAME}_vs_header_ML.png",
    )
    plot_residual_hist(
        pred,
        stats,
        OUTPUT_DIR / f"{EQUATION_NAME}_residual_hist.png",
    )
    plot_residual_vs_x(
        pred,
        "log10_tau",
        f"log10({TAU_COL})",
        OUTPUT_DIR / f"{EQUATION_NAME}_residual_vs_log10_tau.png",
    )
    plot_residual_vs_x(
        pred,
        R_COL,
        "Distance [km]",
        OUTPUT_DIR / f"{EQUATION_NAME}_residual_vs_distance.png",
    )


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    if not INPUT_CSV.exists():
        raise FileNotFoundError(f"Missing input CSV: {INPUT_CSV}")

    print(f"[INFO] Reading: {INPUT_CSV}")
    df = pd.read_csv(INPUT_CSV)

    work = prepare_data(df)

    if len(work) < 5:
        raise RuntimeError(f"Too few valid rows for regression: {len(work)}")

    print(f"[INFO] Regression rows: {len(work)}")
    print(f"[INFO] Fitting: {EQUATION_NAME} = A + B*log10({TAU_COL}) ± residual")

    _beta, stats, pred = iterative_clip_and_fit(work)

    coef_df = make_coefficient_table(stats)
    equation_df = make_equation_summary(stats)

    coef_csv = OUTPUT_DIR / f"{EQUATION_NAME}_regression_coefficients.csv"
    equation_csv = OUTPUT_DIR / f"{EQUATION_NAME}_regression_equation_summary.csv"
    pred_csv = OUTPUT_DIR / f"{EQUATION_NAME}_regression_predictions.csv"
    report_txt = OUTPUT_DIR / f"{EQUATION_NAME}_regression_report.txt"

    coef_df.to_csv(coef_csv, index=False)
    equation_df.to_csv(equation_csv, index=False)
    pred.to_csv(pred_csv, index=False)

    write_report(stats, coef_csv, equation_csv, pred_csv, report_txt)

    if MAKE_PLOTS:
        print("[INFO] Making plots...")
        make_plots(pred, stats)

    A = stats["A"]
    B = stats["B"]
    sigma = stats["sigma_for_equation"]

    print("========== DONE ==========")
    print("Equation:")
    print(f"  {EQUATION_NAME} = {A:.8f} + {B:.8f}*log10({TAU_COL}) ± {sigma:.8f}")
    print("")
    print(f"residual mode : {RESIDUAL_FOR_EQUATION}")
    print(f"sigma_res     : {sigma:.6f} magnitude")
    print(f"n             : {stats['n']}")
    print(f"R2            : {stats['r2']:.4f}")
    print(f"adj R2        : {stats['adj_r2']:.4f}")
    print(f"RMSE          : {stats['rmse_M']:.4f}")
    print("")
    print(f"Report        : {report_txt}")
    print(f"Equation CSV  : {equation_csv}")
    print(f"Coef CSV      : {coef_csv}")
    print(f"Pred CSV      : {pred_csv}")

    if MAKE_PLOTS:
        print("Figures:")
        print(f"  {OUTPUT_DIR / f'{EQUATION_NAME}_vs_header_ML.png'}")
        print(f"  {OUTPUT_DIR / f'{EQUATION_NAME}_residual_hist.png'}")
        print(f"  {OUTPUT_DIR / f'{EQUATION_NAME}_residual_vs_log10_tau.png'}")
        print(f"  {OUTPUT_DIR / f'{EQUATION_NAME}_residual_vs_distance.png'}")


if __name__ == "__main__":
    main()
