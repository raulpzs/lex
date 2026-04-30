# doing basic EDA on country-year dataset

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

base_dir = Path(__file__).resolve().parent.parent

data_dir = base_dir / "data"
data_path = data_dir / "country_year_index_v2.csv"

output_dir = base_dir / "outputs"

df = pd.read_csv(data_path)

df = df.drop(df.columns[0], axis=1)

print(df.columns.tolist()[:6])

df_vars = df.iloc[:, 5:]

# taking a look at the shape
print(f"Shape of dataframe: {df_vars.shape}")

all_zeros = df_vars.columns[
    df_vars.fillna(0).eq(0).all(axis=0)
]

print("Number of all-zero columns:", len(all_zeros))
print(all_zeros.tolist()[:10])

zero_share = (df_vars == 0).mean()

threshold = 0.99
mostly_zero_cols = zero_share[zero_share >= threshold]

print(f"Number of columns with ≥{threshold*100:.0f}% zeros:", len(mostly_zero_cols))
print(mostly_zero_cols.index.tolist()[:10])

variation_summary = pd.DataFrame({
    "zero_share": zero_share,
    "mean": df_vars.mean(numeric_only=True),
    "std": df_vars.std(numeric_only=True),
    "n_unique": df_vars.nunique()
}).sort_values("zero_share", ascending=False)

print(variation_summary.head(20))

# Show most common columns
nonzero_counts = (df_vars != 0).sum(axis=0)

threshold = 30

top = (
    nonzero_counts
    .sort_values(ascending=False)
    .head(30)
)

print(f"Top {threshold} columns by number of non-zero rows:\n")
for col, count in top.items():
    print(f"{col:<40} {count}")

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