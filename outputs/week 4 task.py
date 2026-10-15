"""
Week 4: Data Exploration Review and Optimization Template
--------------------------------------------------------
Edit INPUT_FILE and OUTPUT_DIR before running.
Install dependencies:
    python -m pip install pandas matplotlib seaborn
Run:
    python week4_eda_optimization_template.py

This script does not assume that all missing values or outliers are errors.
It creates a profile and generic charts, and leaves domain-specific cleaning
decisions to the analyst.
"""
from pathlib import Path
from time import perf_counter
import json
import logging

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

INPUT_FILE = Path("data/retail_sales_eda_dataset.csv")
OUTPUT_DIR = Path("outputs")
FIGURE_DIR = OUTPUT_DIR / "figures"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
FIGURE_DIR.mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)
sns.set_theme(style="whitegrid", context="notebook")


def load_data(path: Path) -> pd.DataFrame:
    """Load a CSV file and fail clearly if the path does not exist."""
    if not path.exists():
        raise FileNotFoundError(
            f"Input file not found: {path.resolve()}. "
            "Update INPUT_FILE to the location of your dataset."
        )
    start = perf_counter()
    frame = pd.read_csv(path, low_memory=False)
    logging.info("Loaded %s rows x %s columns in %.3fs",
                 frame.shape[0], frame.shape[1], perf_counter() - start)
    if frame.empty:
        raise ValueError("The input dataset contains no rows.")
    return frame


def normalize_column_names(frame: pd.DataFrame) -> pd.DataFrame:
    """Standardize names; check for collisions after normalization."""
    result = frame.copy()
    result.columns = (
        result.columns.astype(str)
        .str.strip()
        .str.lower()
        .str.replace(r"\s+", "_", regex=True)
        .str.replace(r"[^0-9a-zA-Z_]", "", regex=True)
    )
    if result.columns.duplicated().any():
        duplicates = result.columns[result.columns.duplicated()].tolist()
        raise ValueError(f"Column-name collisions after normalization: {duplicates}")
    return result


def profile_data(frame: pd.DataFrame) -> dict:
    """Return data-quality metrics without modifying the input."""
    missing_count = frame.isna().sum()
    missing_pct = (missing_count / len(frame) * 100).round(2)
    profile = {
        "rows": int(frame.shape[0]),
        "columns": int(frame.shape[1]),
        "duplicate_rows": int(frame.duplicated().sum()),
        "columns_detail": {},
    }
    for col in frame.columns:
        profile["columns_detail"][str(col)] = {
            "dtype": str(frame[col].dtype),
            "missing_count": int(missing_count[col]),
            "missing_percent": float(missing_pct[col]),
            "unique_non_null": int(frame[col].nunique(dropna=True)),
        }
    return profile


def save_profile(frame: pd.DataFrame, profile: dict) -> None:
    (OUTPUT_DIR / "data_quality_profile.json").write_text(
        json.dumps(profile, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    missing = pd.DataFrame({
        "missing_count": frame.isna().sum(),
        "missing_percent": (frame.isna().mean() * 100).round(2),
        "dtype": frame.dtypes.astype(str),
        "unique_non_null": frame.nunique(dropna=True),
    }).sort_values("missing_percent", ascending=False)
    missing.to_csv(OUTPUT_DIR / "column_quality_summary.csv", index=True)
    frame.describe(include="all").transpose().to_csv(
        OUTPUT_DIR / "descriptive_statistics.csv", index=True
    )


def create_basic_charts(frame: pd.DataFrame) -> None:
    """Create generic charts; review whether each is meaningful for your data."""
    numeric_cols = frame.select_dtypes(include="number").columns.tolist()
    categorical_cols = frame.select_dtypes(
        include=["object", "category", "bool"]
    ).columns.tolist()

    for col in numeric_cols[:8]:
        values = frame[col].dropna()
        if values.empty:
            continue
        fig, ax = plt.subplots(figsize=(8, 4.5))
        sns.histplot(values, kde=(values.nunique() > 10), ax=ax)
        ax.set_title(f"Distribution of {col}")
        ax.set_xlabel(col)
        ax.set_ylabel("Count")
        fig.tight_layout()
        fig.savefig(FIGURE_DIR / f"distribution_{col}.png", dpi=160)
        plt.close(fig)

    for col in categorical_cols[:8]:
        counts = frame[col].astype("string").fillna("<missing>").value_counts().head(15)
        if counts.empty:
            continue
        fig, ax = plt.subplots(figsize=(9, 5))
        sns.barplot(x=counts.values, y=counts.index.astype(str), ax=ax)
        ax.set_title(f"Most frequent values in {col} (top 15)")
        ax.set_xlabel("Count")
        ax.set_ylabel(col)
        fig.tight_layout()
        safe_name = "".join(ch if ch.isalnum() or ch in "_-" else "_" for ch in col)
        fig.savefig(FIGURE_DIR / f"categories_{safe_name}.png", dpi=160)
        plt.close(fig)

    if len(numeric_cols) >= 2:
        corr = frame[numeric_cols].corr(numeric_only=True)
        if not corr.empty:
            fig, ax = plt.subplots(figsize=(max(7, len(corr) * 0.8),
                                            max(5, len(corr) * 0.65)))
            sns.heatmap(corr, cmap="vlag", center=0, annot=(len(corr) <= 10),
                        fmt=".2f", ax=ax)
            ax.set_title("Correlation matrix for numeric variables")
            fig.tight_layout()
            fig.savefig(FIGURE_DIR / "numeric_correlation_heatmap.png", dpi=160)
            plt.close(fig)


def main() -> None:
    raw = load_data(INPUT_FILE)
    original_profile = profile_data(raw)
    save_profile(raw, original_profile)

    # Normalizing names is usually safe, but always inspect the resulting names.
    cleaned = normalize_column_names(raw)

    # Intentionally do not automatically drop duplicates, impute nulls,
    # or remove outliers. Apply documented, domain-specific rules here.
    cleaned.to_csv(OUTPUT_DIR / "working_copy.csv", index=False)
    create_basic_charts(cleaned)

    summary = {
        "input_file": str(INPUT_FILE),
        "original_shape": list(raw.shape),
        "working_copy_shape": list(cleaned.shape),
        "exact_duplicate_rows_before_cleaning": int(raw.duplicated().sum()),
        "missing_cells_before_cleaning": int(raw.isna().sum().sum()),
        "notes": [
            "No rows were automatically removed.",
            "No missing values were automatically imputed.",
            "Review the profile and charts before applying domain-specific rules.",
        ],
    }
    (OUTPUT_DIR / "run_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    logging.info("Finished. Review outputs in: %s", OUTPUT_DIR.resolve())


if __name__ == "__main__":
    main()
