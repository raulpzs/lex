# script for generating plots for LawExpression indices

from pathlib import Path
import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt

import numpy as np

import matplotlib.colors as mcolors

# ----------------------------
# CONFIG
# ----------------------------

DATA_PATH = Path("data/raw/country_year_final_panel_full_new.csv")
OUTPUT_DIR = Path("outputs")

START_YEAR = 1976
END_YEAR = 2025

WORLD_URL = "https://naturalearth.s3.amazonaws.com/110m_cultural/ne_110m_admin_0_countries.zip"

EXCLUDE_ENTITIES = {
    "ASEAN",
    "African Union",
    "European Union",
    "OEA",
    "World",
    "OECD"
}


# ----------------------------
# Load data
# ----------------------------

def load_data():
    df = pd.read_csv(DATA_PATH)

    df = df.loc[:, ~df.columns.str.contains("^Unnamed")]
    df["year"] = pd.to_numeric(df["year"], errors="coerce")

    # Remove non-country entities
    df = df[~df["COUNTRY"].isin(EXCLUDE_ENTITIES)]

    # Clean iso3
    df["iso3"] = df["iso3"].str.strip().str.upper()

    return df


# ----------------------------
# Cumulative transformation
# ----------------------------

def add_cumulative(df):
    df = df.sort_values(["iso3", "year"])

    df["cum_weighted_total"] = (
        df.groupby("iso3")["wdj_expression"]
        .cumsum()
    )

    return df


# ----------------------------
# Time filter (AFTER cumulative)
# ----------------------------

def filter_time(df):
    return df[
        (df["year"] >= START_YEAR) &
        (df["year"] <= END_YEAR)
    ]


# ----------------------------
# Validation
# ----------------------------

def validate_iso3(df, world):
    valid_iso3 = set(world["iso_a3"])

    invalid = df.loc[~df["iso3"].isin(valid_iso3), "iso3"].unique()

    if len(invalid) > 0:
        print("\nInvalid ISO3 codes (not in map):")
        print(invalid)


# ----------------------------
# Maps
# ----------------------------

def plot_coverage_map(df, world):
    covered = df["iso3"].dropna().unique()

    world["in_data"] = world["iso_a3"].isin(covered)

    fig, ax = plt.subplots(figsize=(12, 6))
    world.plot(column="in_data", ax=ax, legend=True)

    ax.set_title("Data Coverage Map")
    ax.axis("off")

    plt.savefig(OUTPUT_DIR / "coverage_map.png", dpi=300, bbox_inches="tight")
    plt.close()

# raw version to show skewness in data
def plot_weighted_map_raw(df, world):

    # --- Aggregate ---
    avg_df = (
        df.groupby("iso3")["wdj_expression"]
        .mean()
        .reset_index()
    )

    merged = world.merge(
        avg_df,
        left_on="iso_a3",
        right_on="iso3",
        how="left"
    )

    # --- TRUE data range (no clipping) ---
    vmin = merged["wdj_expression"].min()
    vmax = merged["wdj_expression"].max()

    norm = mcolors.TwoSlopeNorm(vmin=vmin, vcenter=0, vmax=vmax)

    # --- Plot ---
    fig, ax = plt.subplots(figsize=(12, 6))

    merged.plot(
        column="wdj_expression",
        ax=ax,
        cmap="coolwarm",
        norm=norm,
        legend=False,  # manual legend
        linewidth=0,
        missing_kwds={
            "color": "#f9f9f9",
            "hatch": "///",
            "edgecolor": "#cccccc",
            "linewidth": 0.3,
            "label": "No data"
        }
    )

    # --- Borders overlay ---
    world.boundary.plot(
        ax=ax,
        color="black",
        linewidth=0.3
    )

    # --- Manual colorbar ---
    sm = plt.cm.ScalarMappable(cmap="coolwarm", norm=norm)
    sm.set_array([])

    cbar = fig.colorbar(sm, ax=ax)

    # --- Balanced ticks: 3 per side + zero ---
    n_per_side = 3
    
    neg_ticks = np.linspace(vmin, 0, n_per_side + 1)[:-1]  # drop 0
    pos_ticks = np.linspace(0, vmax, n_per_side + 1)[1:]   # drop 0
    
    ticks = np.concatenate([neg_ticks, [0], pos_ticks])
    
    cbar.set_ticks(ticks)
    cbar.ax.set_yticklabels([f"{t:.2f}" for t in ticks])

    cbar.set_label("Weighted De Jure Score")

    # --- Title / layout ---
    ax.set_title("Average Weighted De Jure (1976–2025) — Raw Scale")
    ax.axis("off")

    plt.savefig(OUTPUT_DIR / "weighted_total_map_raw.png", dpi=300, bbox_inches="tight")
    plt.close()

# clipping and forced symmetry around zero
def plot_weighted_map(df, world):

    # --- Aggregate ---
    avg_df = (
        df.groupby("iso3")["wdj_expression"]
        .mean()
        .reset_index()
    )

    merged = world.merge(
        avg_df,
        left_on="iso_a3",
        right_on="iso3",
        how="left"
    )

    # --- Quantile clipping ---
    vmin_q = merged["wdj_expression"].quantile(0.05)
    vmax_q = merged["wdj_expression"].quantile(0.95)

    # --- Enforce symmetry around 0 ---
    abs_max = max(abs(vmin_q), abs(vmax_q))
    vmin, vmax = -abs_max, abs_max

    norm = mcolors.TwoSlopeNorm(vmin=vmin, vcenter=0, vmax=vmax)

    # --- Plot ---
    fig, ax = plt.subplots(figsize=(12, 6))

    merged.plot(
        column="wdj_expression",
        ax=ax,
        cmap="coolwarm",
        norm=norm,
        legend=False,  # <-- turn off GeoPandas legend
        linewidth=0,
        missing_kwds={
            "color": "#f9f9f9",
            "hatch": "///",
            "edgecolor": "#cccccc",
            "linewidth": 0.3,
            "label": "No data"
        }
    )

    # --- Borders overlay ---
    world.boundary.plot(
        ax=ax,
        color="black",
        linewidth=0.3
    )

    # --- Manual colorbar ---
    sm = plt.cm.ScalarMappable(cmap="coolwarm", norm=norm)
    sm.set_array([])

    cbar = fig.colorbar(sm, ax=ax)

    # Symmetric ticks centered at 0
    ticks = np.linspace(vmin, vmax, 7)
    cbar.set_ticks(ticks)
    cbar.ax.set_yticklabels([f"{t:.2f}" for t in ticks])

    cbar.set_label("Weighted De Jure Score")

    # --- Title / layout ---
    ax.set_title("Average Weighted De Jure (1976–2025)")
    ax.axis("off")

    plt.savefig(OUTPUT_DIR / "weighted_total_map.png", dpi=300, bbox_inches="tight")
    plt.close()




# ----------------------------
# Global trends
# ----------------------------

def plot_global_trends(df):
    cols = [
        "wdj_citizen",
        "wdj_govprot",
        "wdj_intermediaries",
        "wdj_press"
    ]

    global_avg = df.groupby("year")[cols].mean().reset_index()

    fig, ax = plt.subplots(figsize=(10, 6))

    for col in cols:
        ax.plot(global_avg["year"], global_avg[col], label=col)

    ax.set_title("Global Trends (Weighted Components)")
    ax.set_xlabel("Year")
    ax.set_ylabel("Average")
    ax.legend()

    plt.savefig(OUTPUT_DIR / "global_trends.png", dpi=300, bbox_inches="tight")
    plt.close()


# ----------------------------
# Country trends (UPDATED)
# ----------------------------

def plot_country_trends(df):
    fig, ax = plt.subplots(figsize=(12, 8))

    for country, group in df.groupby("COUNTRY"):
        ax.plot(
            group["year"],
            group["cum_weighted_total"],
            alpha=0.25
        )

    ax.set_title("Country Trends: Cumulative Weighted De Jure Total")
    ax.set_xlabel("Year")
    ax.set_ylabel("Cumulative Value")

    plt.savefig(OUTPUT_DIR / "country_trends.png", dpi=300, bbox_inches="tight")
    plt.close()


# ----------------------------
# Run
# ----------------------------

def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    df = load_data()
    df = add_cumulative(df)
    df = filter_time(df)

    # Load world map (updated method)
    world = gpd.read_file(WORLD_URL)
    world.columns = world.columns.str.lower()  # normalize column names

    validate_iso3(df, world)

    plot_coverage_map(df, world)
    plot_weighted_map_raw(df, world)
    plot_weighted_map(df, world)
    plot_global_trends(df)
    plot_country_trends(df)


if __name__ == "__main__":
    main()
