#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from pathlib import Path
import numpy as np
import pandas as pd
import pygmt

# ============================================================
# USER PARAMETERS
# ============================================================
INPUT_CSV = Path("output/event_magnitude_summary.csv")
OUTPUT_FIG = Path("figures/00_input_magnitude_summary_pygmt.png")
OUTPUT_FIG.parent.mkdir(parents=True, exist_ok=True)

COL_EVENT_ID   = "ID"
COL_HEADER_MAG = "mean_header_mag"
COL_MEAN_Z_MAG = "magnitudez"
COL_STD_Z_MAG  = "magnitudezstd"

FIG_W = "14c"
FIG_H = "10c"
SYMBOL = "c0.35c"
PEN_SYMBOL = "0.6p,black"
PEN_ERROR = "0.8p,black"

CAT_COLORS = {
    "M3": "dodgerblue3",
    "M4": "orange",
    "M5": "tomato",
    "M6": "purple",
}

# ============================================================
# HELPERS
# ============================================================
def classify_mag(m):
    if pd.isna(m):
        return None
    if 3.0 <= m < 4.0:
        return "M3"
    elif 4.0 <= m < 5.0:
        return "M4"
    elif 5.0 <= m < 6.0:
        return "M5"
    elif m >= 6.0:
        return "M6"
    else:
        return "M3"

# ============================================================
# READ DATA
# ============================================================
if not INPUT_CSV.exists():
    raise FileNotFoundError(f"Input CSV not found: {INPUT_CSV}")

df = pd.read_csv(INPUT_CSV)

required = [COL_HEADER_MAG, COL_MEAN_Z_MAG, COL_STD_Z_MAG]
missing = [c for c in required if c not in df.columns]
if missing:
    raise ValueError(f"Missing required columns: {missing}\nAvailable columns: {list(df.columns)}")

plot_df = df.copy()
plot_df[COL_HEADER_MAG] = pd.to_numeric(plot_df[COL_HEADER_MAG], errors="coerce")
plot_df[COL_MEAN_Z_MAG] = pd.to_numeric(plot_df[COL_MEAN_Z_MAG], errors="coerce")
plot_df[COL_STD_Z_MAG]  = pd.to_numeric(plot_df[COL_STD_Z_MAG], errors="coerce").fillna(0.0)

plot_df = plot_df.dropna(subset=[COL_HEADER_MAG, COL_MEAN_Z_MAG]).copy()
plot_df["mag_cat"] = plot_df[COL_HEADER_MAG].apply(classify_mag)

if plot_df.empty:
    raise ValueError("No valid data to plot after filtering.")

# ============================================================
# REGION
# ============================================================
xmin = np.floor(plot_df[COL_HEADER_MAG].min() * 10) / 10 - 0.1
xmax = np.ceil(plot_df[COL_HEADER_MAG].max() * 10) / 10 + 0.1

ymin_data = (plot_df[COL_MEAN_Z_MAG] - plot_df[COL_STD_Z_MAG]).min()
ymax_data = (plot_df[COL_MEAN_Z_MAG] + plot_df[COL_STD_Z_MAG]).max()

ymin = np.floor(ymin_data * 10) / 10 - 0.1
ymax = np.ceil(ymax_data * 10) / 10 + 0.1
region = [xmin, xmax, ymin, ymax]

# ============================================================
# PLOT
# ============================================================
pygmt.config(
    FONT_LABEL="12p,Times-Bold,black",
    FONT_ANNOT_PRIMARY="10p,Times-Roman,black",
    FONT_TITLE="14p,Times-Bold,black",
    MAP_FRAME_TYPE="plain",
)

fig = pygmt.Figure()

fig.basemap(
    region=region,
    projection=f"X{FIG_W}/{FIG_H}",
    frame=[
        "WSen",
        'xaf0.5+l"Header magnitude"',
        'yaf0.5+l"Mean Z magnitude"',
        # '+t"Header magnitude vs mean Z magnitude"',
    ],
)

# 1:1 line
ref_min = max(xmin, ymin)
ref_max = min(xmax, ymax)
if ref_max > ref_min:
    fig.plot(
        x=[ref_min, ref_max],
        y=[ref_min, ref_max],
        pen="1.0p,gray50,--",
    )

# Plot by category
for cat in ["M3", "M4", "M5", "M6"]:
    sub = plot_df[plot_df["mag_cat"] == cat]
    if sub.empty:
        continue

    # 3 columns: x, y, yerr
    data_cat = np.column_stack([
        sub[COL_HEADER_MAG].to_numpy(),
        sub[COL_MEAN_Z_MAG].to_numpy(),
        sub[COL_STD_Z_MAG].to_numpy(),
    ])

    fig.plot(
        data=data_cat,
        style=SYMBOL,
        fill=CAT_COLORS[cat],
        pen=PEN_SYMBOL,
        error_bar=f"y+{PEN_ERROR}",
        label=cat,
    )

fig.legend(position="JTR+jTR+o0.2c", box="+gwhite+p0.8p")
fig.savefig(OUTPUT_FIG, dpi=300)
print(f"[OK] Saved figure: {OUTPUT_FIG}")