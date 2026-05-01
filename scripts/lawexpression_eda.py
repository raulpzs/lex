# doing basic EDA on country-year dataset

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# ----------------------------
# PATH/CONFIG
# ----------------------------

base_dir = Path(__file__).resolve().parent.parent

data_dir = base_dir / "data" / "raw"
data_path = data_dir / "country_year_final_panel_full_new_legalrulecount.csv"

output_dir = base_dir / "outputs"

# INDICES
indices = ["wdj_expression", 
           "wdj_citizen",
          "wdj_intermediaries",
          "wdj_press",
          "wdj_govprot",
          ]

# ----------------------------
# LOADING DATA
# ----------------------------

df = pd.read_csv(data_path)

df = df.drop(df.columns[0], axis=1)

# ----------------------------
# DESCRIPTIVE STATS TABLE
# ----------------------------

desc = df[indices].describe().T

# Keep only relevant stats
desc = desc[["mean", "std", "min", "25%", "50%", "75%", "max"]]

# Add missingness + N
desc["n_obs"] = df[indices].notna().sum()
desc["missing_share"] = df[indices].isna().mean()

# will need to update these with official labels
pretty_names = {
    "wdj_expression": "Weighted De Jure Expression",
    "wdj_citizen": "Weighted Citizen Expression",
    "wdj_intermediaries": "Weighted Intermediaries Expression",
    "wdj_press": "Weighted Press Freedom",
    "wdj_govprot": "Weighted Government Protection"
}

desc.index = desc.index.map(pretty_names)

# Reorder columns
desc = desc[[
    "n_obs", "missing_share",
    "mean", "std",
    "min", "25%", "50%", "75%", "max"
]]

# Round for presentation
desc = desc.round(3)

print(desc)

# saving
latex_path = output_dir / "descriptive_stats.tex"

desc.to_latex(
    latex_path,
    float_format = "%.3f",
    caption = "Descriptive statistics for the Weighted De Jure indices.",
    label = "tab:desc_stats",
    bold_rows = True
)

# ----------------------------
# SLICING DATA
# ----------------------------

if "C_DISINFO_GEN" not in df.columns:
    raise ValueError('Column "C_DISINFO_GEN" not found in the dataframe.')

start_col = df.columns.get_loc("C_DISINFO_GEN") # this is the first of the rules variables

# looking at the non-rule columns
print(df.columns.tolist()[:start_col])

df_vars = df.iloc[:, start_col:].select_dtypes(include=[np.number]) # updated with location instead

# taking a look at the shape
print(f"Shape of dataframe: {df_vars.shape}")

# ----------------------------
# NO-VARIANCE VARIABLES
# ----------------------------

all_zeros = df_vars.columns[
    df_vars.fillna(0).eq(0).all(axis=0)
]

print("Number of all-zero columns:", len(all_zeros))
print(all_zeros.tolist()[:10])

zero_share = (df_vars == 0).mean()

# looking at near-zero variance
threshold = 0.99
mostly_zero_cols = zero_share[zero_share >= threshold]

print(f"Number of columns with ≥{threshold*100:.0f}% zeros:", len(mostly_zero_cols))
print(mostly_zero_cols.index.tolist()[:10])

# ----------------------------
# VARIANCE SUMMARY
# ----------------------------

variation_summary = pd.DataFrame({
    "zero_share": zero_share,
    "mean": df_vars.mean(numeric_only=True),
    "std": df_vars.std(numeric_only=True),
    "n_unique": df_vars.nunique()
}).sort_values("zero_share", ascending=False)

print(variation_summary.head(20))

# ----------------------------
# MOST COMMON (LATEX TABLE)
# ----------------------------

nonzero_counts = (df_vars != 0).sum(axis=0)

threshold = 30

top = (
    nonzero_counts
    .sort_values(ascending=False)
    .head(threshold)
)

# Convert to dataframe
top_df = top.reset_index()
top_df.columns = ["variable", "nonzero_count"]

# Add share of observations
n_obs = len(df_vars)
top_df["nonzero_share"] = (top_df["nonzero_count"] / n_obs).round(3)

latex_path = output_dir / "top_nonzero_variables.tex"

top_df.to_latex(
    latex_path,
    index=False,
    float_format="%.3f",
    caption=f"Top {threshold} variables by number of non-zero observations",
    label="tab:top_nonzero",
    longtable=False
)

print(f"LaTeX table saved to: {latex_path}")

# ----------------------------
# ZERO-SHARE GRAPH
# ----------------------------

# Histogram of zero-share
hist_path = output_dir / "zero_share_histogram.png"


plt.figure()
zero_share.hist(bins=30)
plt.xlabel("Proportion of zeros")
plt.ylabel("Number of variables")
plt.title("Distribution of zero-share across variables")

plt.tight_layout()  # prevents label cutoff
plt.savefig(hist_path, dpi=300)
plt.close()  # important when running in scripts

# ----------------------------
# COLUMN SUM GRAPH
# ----------------------------

# Histogram of values by column
col_sums = df_vars.sum(axis=0)

hist_path = output_dir / "column_sum_histogram.png"

plt.figure()
col_sums.hist(bins=40)
plt.xlabel("Column Sum")
plt.ylabel("Number of variables")
plt.title("Distribution of column sums")

plt.tight_layout()
plt.savefig(hist_path, dpi=300)
plt.close()

# ----------------------------
# DISTRIBUTION OF DISTRIBUTIONS
# ----------------------------

# Core stats
means = df_vars.mean(numeric_only=True)
stds = df_vars.std(numeric_only=True)
medians = df_vars.median(numeric_only=True)

# Sparsity
zero_share = (df_vars == 0).mean()

# Conditional mean (non-zero only)
cond_means = df_vars.replace(0, np.nan).mean()

# ----------------------------
# COMBINED SUMMARY TABLE
# ----------------------------

dist_summary = pd.DataFrame({
    "mean": means,
    "std": stds,
    "median": medians,
    "zero_share": zero_share,
    "cond_mean": cond_means
}).sort_values("mean", ascending=False)

# Save summary
dist_summary_path = output_dir / "variable_distribution_summary.csv"

dist_summary.to_csv(dist_summary_path)

print("Top variables by mean:\n")
print(dist_summary.head(10))

# ----------------------------
# PLOTTING
# ----------------------------

def save_hist(series, title, xlabel, filename, bins=40):
    plt.figure()
    series.hist(bins=bins)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel("Number of variables")
    plt.tight_layout()
    plt.savefig(output_dir / filename, dpi=300)
    plt.close()

# Core distributions
save_hist(
    means, 
    "Distribution of Variable Means", 
    "Mean", 
    "dist_means.png"
)

save_hist(
    stds, 
    "Distribution of Variable Standard Deviations", 
    "Std Dev", 
    "dist_stds.png"
)

save_hist(
    medians, 
    "Distribution of Variable Medians", 
    "Median", 
    "dist_medians.png"
)

# Conditional means (key insight)
save_hist(
    cond_means.dropna(),
    "Distribution of Conditional Means (Non-zero Only)",
    "Conditional Mean",
    "dist_cond_means.png"
)