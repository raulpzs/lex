# pca analysis of expression data

import pandas as pd
from pathlib import Path

from sklearn.preprocessing import MaxAbsScaler
from sklearn.decomposition import TruncatedSVD

import numpy as np

import matplotlib.pyplot as plt

# =========================
# Paths
# =========================

DATA_PATH = Path("data/raw/country_year_final_panel_full_new_legalrulecount.csv")
OUTPUT_DIR = Path("outputs")
LATEX_DIR = OUTPUT_DIR / "tables"
LATEX_DIR.mkdir(parents = True, exist_ok = True)

# =========================
# Loading data
# =========================

df = pd.read_csv(DATA_PATH)

print(df.head())

print(f"Size of original dataframe: {df.shape}")

print(df.describe())

# =========================
# Looking at min and max across columns
# =========================

# looking at the min and max values across columns
cols = df.columns[5:]

top_max = df[cols].max().nlargest(10)
print(top_max)

top_min = df[cols].min().nsmallest(10)
print(top_min)

start_col = "C_DISINFO_GEN"

start_idx = df.columns.get_loc(start_col)
df_cols = df.iloc[:, start_idx:]

print(df_cols.head())

# =========================
# Dropping no-variance cols
# =========================

X = df_cols.loc[:, df_cols.var() > 0]

print(f"Size of reduced dataframe: {X.shape}")

# dropping near-zero variation columns (SAVED FOR LATER)
# X = X.loc[:, (X != 0).mean() > 0.05]

# =========================
# Looking at missing data
# =========================

X_na = X[X.isna().any(axis=1)]
print(X_na.head(10))

# looks like OAS data has missing values, so dropping all NaN rows
X = X.dropna()

# =========================
# Scaling (preprocessing)
# =========================

X_scaled = MaxAbsScaler().fit_transform(X)

# =========================
# SVD
# =========================

svd = TruncatedSVD(n_components = 20, random_state = 42)
X_svd = svd.fit_transform(X_scaled)

explained = svd.explained_variance_ratio_

loadings = pd.DataFrame(
    svd.components_.T,
    index = X.columns,
    columns = [f"PC{i+1}" for i in range(svd.n_components)]
)

# =========================
# Weighted contribution scores
# =========================

# weight loadings by explained variance
weights = svd.explained_variance_ratio_

# squared loadings (contribution to variance)

contrib = (loadings ** 2) @ weights

contrib = contrib.sort_values(ascending=False)

print(contrib)

# =========================
# Explained variance
# =========================

cum_explained = np.cumsum(explained)

print(explained[:10])
print(cum_explained[:10])

# =========================
# Scree plot
# =========================

plt.figure(figsize=(12,8))
plt.plot(explained, marker = 'o')
plt.title("Scree plot")
plt.xlabel("Component")
plt.ylabel("Explained variance")

plt.savefig(OUTPUT_DIR / "svd_scree_plot.png", dpi=300, bbox_inches="tight")
plt.close()

# =========================
# Explained variance plot
# =========================

plt.figure(figsize=(12,8))
plt.plot(range(1, len(explained)+1), cum_explained, marker='o', linestyle='-')
plt.axhline(y=0.8, color='r', linestyle='--', label='80%')
plt.axhline(y=0.9, color='g', linestyle='--', label='90%')
plt.axhline(y=0.95, color='b', linestyle='--', label='95%')
plt.xlabel('Number of Components')
plt.ylabel('Cumulative Explained Variance')
plt.title('Cumulative Variance Explained by SVD Components')
plt.xticks(range(1, len(explained)+1, 2))
plt.legend()
plt.grid(True)
plt.savefig(OUTPUT_DIR / "svd_cumsum_plot.png", dpi=300, bbox_inches="tight")
plt.close()

# =========================
# Inspecting PC1
# =========================

# looking at which variables are driving the first component
print(f"Variables contributing most to PC1:\n{loadings["PC1"].sort_values(ascending=False).head(10)}")

print(f"Variables contributing most to PC1 by absolute value:\n{loadings["PC1"].abs().sort_values(ascending=False).head(10)}")

# checking correlation between PC1 and totals
pc1 = X_svd[:, 0] # pulling out the PC1 values for each row of X

# grabbing the legal_rule_count_total columns from the original df, then filtering down based on X index
aligned_total = df.loc[X.index, "legal_rule_count_total"]

# calculating correlation
corr = np.corrcoef(pc1, aligned_total)[0,1]

print(f"Correlation: {corr:.3f}")

# =========================
# Examining top 5 PC
# =========================

# Top variables per component
for i in range(5):
    print(f"\nPC{i+1}")
    print(loadings.iloc[:, i].sort_values(ascending=False).head(5))
    print(loadings.iloc[:, i].sort_values().head(5))

# checking for correlation between components
print(f"Correlation between components is: {np.corrcoef(X_svd.T)}")

# =========================
# Exporting LaTeX tables for loadings
# =========================

def make_pc_loading_table(loadings_df: pd.DataFrame, pc: str, top_n: int = 10) -> pd.DataFrame:
    """
    Create a table with top positive and negative loadings for one PC.
    """
    s = loadings_df[pc].sort_values(ascending=False)

    top_pos = s.head(top_n).reset_index()
    top_pos.columns = ["variable", "loading"]

    top_neg = s.tail(top_n).sort_values().reset_index()
    top_neg.columns = ["variable", "loading"]

    top_pos["direction"] = "positive"
    top_neg["direction"] = "negative"

    out = pd.concat([top_pos, top_neg], ignore_index=True)
    out["abs_loading"] = out["loading"].abs()
    out = out[["direction", "variable", "loading", "abs_loading"]]

    return out


def export_pc_loading_latex(
    loadings_df: pd.DataFrame,
    latex_dir: Path,
    n_components: int = 5,
    top_n: int = 10,
    float_format: str = "%.4f"
) -> None:
    """
    Export one LaTeX table per principal component.
    """
    for i in range(n_components):
        pc = f"PC{i+1}"
        table_df = make_pc_loading_table(loadings_df, pc=pc, top_n=top_n)

        latex_str = table_df.to_latex(
            index=False,
            escape=False,
            float_format=float_format.__mod__,
            caption=f"Top positive and negative loadings for {pc}",
            label=f"tab:{pc.lower()}_loadings",
            column_format="llrr"
        )

        out_path = latex_dir / f"{pc.lower()}_loadings.tex"
        out_path.write_text(latex_str, encoding="utf-8")


# export first 5 PCs
export_pc_loading_latex(
    loadings_df=loadings,
    latex_dir=LATEX_DIR,
    n_components=5,
    top_n=10
)

# optional: also export full loading matrix as one LaTeX table
full_loadings_path = LATEX_DIR / "all_loadings.tex"
full_loadings_path.write_text(
    loadings.reset_index().rename(columns={"index": "variable"}).to_latex(
        index=False,
        escape=False,
        float_format="%.4f".__mod__,
        caption="Full loading matrix for all principal components",
        label="tab:all_loadings"
    ),
    encoding="utf-8"
)
