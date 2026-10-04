import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.stats as stats
import seaborn as sns

# Set style for publication-quality visualizations
sns.set_theme(style="whitegrid", palette="muted")
plt.rcParams["font.size"] = 10
plt.rcParams["figure.titlesize"] = 14

# ---------------------------------------------------------
# 1. LOAD DATASET & DATA PREPARATION
# Replace 'your_dataset.csv' with your actual file path
# ---------------------------------------------------------
# df = pd.read_csv('your_dataset.csv')

# Demo dataset generation (Customer Churn / Spending context)
np.random.seed(42)
n_samples = 500
df = pd.DataFrame(
    {
        "Age": np.random.normal(40, 12, n_samples).astype(int),
        "Annual_Income_k": np.random.normal(65, 20, n_samples),
        "Spending_Score": np.random.normal(50, 25, n_samples),
        "Tenure_Months": np.random.randint(1, 72, n_samples),
        "Segment": np.random.choice(["Basic", "Silver", "Gold"], size=n_samples),
        "Churn": np.random.choice([0, 1], size=n_samples, p=[0.75, 0.25]),
    }
)

# Clean boundary values after generation
df["Spending_Score"] = df["Spending_Score"].clip(1, 100)
df["Annual_Income_k"] = df["Annual_Income_k"].clip(15, 150)

print("Dataset Head:")
print(df.head())

# ---------------------------------------------------------
# 2. ADVANCED ANALYSIS 1: CORRELATION HEATMAP
# ---------------------------------------------------------
plt.figure(figsize=(8, 6))
numeric_df = df.select_dtypes(include=[np.number])
corr_matrix = numeric_df.corr(method="pearson")

# Generate mask for upper triangle
mask = np.triu(np.ones_like(corr_matrix, dtype=bool))

sns.heatmap(
    corr_matrix,
    mask=mask,
    annot=True,
    fmt=".2f",
    cmap="coolwarm",
    vmin=-1,
    vmax=1,
    linewidths=0.5,
    cbar_kws={"shrink": 0.8},
)
plt.title("Correlation Matrix of Numerical Variables", pad=15, fontweight="bold")
plt.tight_layout()
plt.savefig("fig1_correlation_heatmap.png", dpi=300)
plt.show()

# ---------------------------------------------------------
# 3. ADVANCED ANALYSIS 2: MULTI-FACETED DISTRIBUTION & OUTLIERS
# ---------------------------------------------------------
plt.figure(figsize=(10, 5))
sns.boxplot(
    x="Segment",
    y="Spending_Score",
    hue="Churn",
    data=df,
    palette={0: "#2ecc71", 1: "#e74c3c"},
)
plt.title(
    "Spending Score Distribution Across Customer Segments by Churn Status",
    fontweight="bold",
)
plt.xlabel("Customer Segment")
plt.ylabel("Spending Score (1-100)")
plt.legend(title="Churned", labels=["Retained (0)", "Churned (1)"])
plt.tight_layout()
plt.savefig("fig2_segment_churn_boxplot.png", dpi=300)
plt.show()

# ---------------------------------------------------------
# 4. ADVANCED ANALYSIS 3: PAIRPLOT / MULTI-VARIABLE PATTERNS
# ---------------------------------------------------------
pairplot = sns.pairplot(
    df,
    vars=["Age", "Annual_Income_k", "Spending_Score"],
    hue="Churn",
    palette={0: "#3498db", 1: "#e74c3c"},
    diag_kind="kde",
    plot_kws={"alpha": 0.6, "s": 30},
)
pairplot.fig.suptitle(
    "Pairwise Scatter Plots & Kernel Density Estimations", y=1.02, fontweight="bold"
)
pairplot.savefig("fig3_pairplot_kde.png", dpi=300)
plt.show()

# ---------------------------------------------------------
# 5. HYPOTHESIS TESTING
# Test if mean Income differs significantly between Churned & Retained customers
# ---------------------------------------------------------
income_retained = df[df["Churn"] == 0]["Annual_Income_k"]
income_churned = df[df["Churn"] == 1]["Annual_Income_k"]

# Independent Two-Sample T-Test
t_stat, p_val = stats.ttest_ind(
    income_retained, income_churned, equal_var=False
)

print("\n--- HYPOTHESIS TEST RESULTS ---")
print(f"T-statistic: {t_stat:.4f}")
print(f"P-value: {p_val:.4f}")

if p_val < 0.05:
    print(
        "Result: Statistically significant difference in Annual Income between churned and retained customers (Reject H0)."
    )
else:
    print(
        "Result: No statistically significant difference in Annual Income found (Fail to reject H0)."
    )