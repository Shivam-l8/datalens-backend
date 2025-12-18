from __future__ import annotations

import base64
import io
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from .schemas import (
    AnalysisResult,
    AnalysisSummary,
    ChartData,
    InsightBlock,
    RecommendationBlock,
)


def detect_column_roles(df: pd.DataFrame) -> Dict[str, List[str]]:
    """Detect numeric, categorical, and datetime-like columns."""
    numeric_cols = df.select_dtypes(include=["number"]).columns.tolist()
    cat_cols = df.select_dtypes(include=["object", "category", "bool"]).columns.tolist()

    datetime_cols: List[str] = []
    for col in df.columns:
        if pd.api.types.is_datetime64_any_dtype(df[col]):
            datetime_cols.append(col)
        else:
            # Try parsing if looks like date-like
            sample = df[col].dropna().astype(str).head(20)
            if not sample.empty:
                parsed = pd.to_datetime(sample, errors="coerce", infer_datetime_format=True)
                if parsed.notna().mean() > 0.8:
                    datetime_cols.append(col)

    # Ensure uniqueness
    datetime_cols = list(dict.fromkeys(datetime_cols))
    numeric_cols = [c for c in numeric_cols if c not in datetime_cols]

    return {
        "numeric": numeric_cols,
        "categorical": cat_cols,
        "datetime": datetime_cols,
    }


def basic_clean(df: pd.DataFrame) -> pd.DataFrame:
    """Simple, explainable cleaning suitable for an MVP."""
    df_clean = df.copy()

    # Drop exact duplicates
    df_clean = df_clean.drop_duplicates()

    roles = detect_column_roles(df_clean)
    num_cols = roles["numeric"]
    cat_cols = roles["categorical"]

    for col in num_cols:
        if df_clean[col].isna().any():
            df_clean[col] = df_clean[col].fillna(df_clean[col].median())

    for col in cat_cols:
        if df_clean[col].isna().any():
            df_clean[col] = df_clean[col].fillna("Missing")

    # Convert datetime-like columns explicitly
    for col in roles["datetime"]:
        df_clean[col] = pd.to_datetime(df_clean[col], errors="coerce", infer_datetime_format=True)

    return df_clean


def dataframe_to_base64_csv(df: pd.DataFrame) -> str:
    buffer = io.StringIO()
    df.to_csv(buffer, index=False)
    return base64.b64encode(buffer.getvalue().encode("utf-8")).decode("utf-8")


def compute_summary_statistics(df: pd.DataFrame, roles: Dict[str, List[str]]) -> Dict[str, Dict]:
    summary: Dict[str, Dict] = {}

    if roles["numeric"]:
        summary["numeric"] = df[roles["numeric"]].describe().T.to_dict(orient="index")

    if roles["categorical"]:
        top_values: Dict[str, Dict] = {}
        for col in roles["categorical"]:
            vc = df[col].value_counts(dropna=False).head(10)
            top_values[col] = vc.to_dict()
        summary["categorical"] = top_values

    return summary


def compute_correlations(df: pd.DataFrame, roles: Dict[str, List[str]]) -> Dict[str, float]:
    """Return flattened correlation pairs for numeric columns."""
    numeric = roles["numeric"]
    if len(numeric) < 2:
        return {}
    corr = df[numeric].corr()
    pairs: Dict[str, float] = {}
    for i, col_i in enumerate(numeric):
        for j, col_j in enumerate(numeric):
            if j <= i:
                continue
            val = corr.loc[col_i, col_j]
            if not np.isnan(val):
                key = f"{col_i}__vs__{col_j}"
                pairs[key] = float(val)
    # Keep strongest correlations
    return dict(sorted(pairs.items(), key=lambda kv: abs(kv[1]), reverse=True)[:20])


def generate_histogram_chart(df: pd.DataFrame, col: str) -> ChartData:
    data = df[col].dropna()
    counts, bin_edges = np.histogram(data, bins=10)
    return ChartData(
        id=f"hist_{col}",
        type="histogram",
        title=f"Distribution of {col}",
        description=f"Histogram showing how {col} is distributed across observations.",
        data={
            "bins": bin_edges.tolist(),
            "counts": counts.tolist(),
            "xLabel": col,
            "yLabel": "Count",
        },
    )


def generate_categorical_bar_chart(df: pd.DataFrame, col: str) -> ChartData:
    vc = df[col].value_counts().head(10)
    return ChartData(
        id=f"bar_{col}",
        type="bar",
        title=f"Top categories in {col}",
        description=f"Bar chart of the top categories in {col}.",
        data={
            "labels": vc.index.astype(str).tolist(),
            "values": vc.values.tolist(),
            "xLabel": col,
            "yLabel": "Count",
        },
    )


def generate_time_trend_chart(df: pd.DataFrame, date_col: str, roles: Dict[str, List[str]]) -> ChartData | None:
    if not roles["numeric"]:
        return None
    num_col = roles["numeric"][0]
    ts = (
        df[[date_col, num_col]]
        .dropna()
        .sort_values(date_col)
        .groupby(date_col)[num_col]
        .mean()
    )
    if ts.empty:
        return None
    return ChartData(
        id=f"trend_{date_col}_{num_col}",
        type="line",
        title=f"Trend of {num_col} over {date_col}",
        description=f"Average {num_col} over time based on {date_col}.",
        data={
            "x": ts.index.astype(str).tolist(),
            "y": ts.values.tolist(),
            "xLabel": date_col,
            "yLabel": f"Avg {num_col}",
        },
    )


def generate_correlation_heatmap_chart(df: pd.DataFrame, roles: Dict[str, List[str]]) -> ChartData | None:
    numeric = roles["numeric"]
    if len(numeric) < 2:
        return None

    corr = df[numeric].corr()
    return ChartData(
        id="corr_heatmap",
        type="heatmap",
        title="Correlation between numeric features",
        description="Correlation matrix of numeric columns. Darker colors indicate stronger relationships.",
        data={
            "matrix": corr.values.tolist(),
            "labels": corr.columns.tolist(),
        },
    )


def infer_outcome_candidates(df: pd.DataFrame, roles: Dict[str, List[str]]) -> List[str]:
    """Heuristic: outcome-like columns often binary, rates, or targets."""
    candidates: List[str] = []

    for col in roles["categorical"]:
        nunique = df[col].nunique(dropna=True)
        if 2 <= nunique <= 5:
            candidates.append(col)

    for col in roles["numeric"]:
        if any(
            key in col.lower()
            for key in ["price", "churn", "attrition", "target", "label", "score", "rate"]
        ):
            candidates.append(col)

    # Deduplicate while preserving order
    seen = set()
    ordered: List[str] = []
    for c in candidates:
        if c not in seen:
            seen.add(c)
            ordered.append(c)
    return ordered


def generate_insights_and_recommendations(
    df: pd.DataFrame,
    roles: Dict[str, List[str]],
    correlations: Dict[str, float],
) -> Tuple[List[InsightBlock], List[RecommendationBlock], str]:
    insights: List[InsightBlock] = []
    recommendations: List[RecommendationBlock] = []
    lines: List[str] = []

    outcome_candidates = infer_outcome_candidates(df, roles)
    roles_text = []
    if roles["numeric"]:
        roles_text.append(f"{len(roles['numeric'])} numeric columns")
    if roles["categorical"]:
        roles_text.append(f"{len(roles['categorical'])} categorical columns")
    if roles["datetime"]:
        roles_text.append(f"{len(roles['datetime'])} date/time columns")
    lines.append(
        "Dataset overview: "
        + ", ".join(roles_text)
        + f", {df.shape[0]} rows and {df.shape[1]} columns."
    )

    if outcome_candidates:
        insights.append(
            InsightBlock(
                title="Potential outcome variables detected",
                detail=(
                    "The following columns behave like outcomes or target variables "
                    f"based on their distribution or naming: {', '.join(outcome_candidates)}."
                ),
            )
        )
        lines.append(
            f"Outcome-like columns detected: {', '.join(outcome_candidates)}. "
            "These are good candidates for retention, churn, or pricing outcomes depending on context."
        )

    # Strong correlations
    strong_corrs = {k: v for k, v in correlations.items() if abs(v) >= 0.4}
    if strong_corrs:
        corr_text_parts = []
        for pair, val in strong_corrs.items():
            a, b = pair.split("__vs__")
            corr_text_parts.append(f"{a} vs {b} (corr={val:.2f})")
        text = "; ".join(corr_text_parts)
        insights.append(
            InsightBlock(
                title="Strong relationships between numeric drivers",
                detail=(
                    "Several numeric features move together strongly, which may indicate shared drivers "
                    f"or causal relationships in the underlying process: {text}."
                ),
            )
        )
        lines.append(
            "Strong numeric relationships: "
            + text
            + ". These features are promising drivers to monitor or use in predictive models."
        )

    # Missingness
    missing_counts = df.isna().sum()
    high_missing = missing_counts[missing_counts > 0].sort_values(ascending=False).head(5)
    if not high_missing.empty:
        cols = ", ".join(f"{c} ({int(v)} missing)" for c, v in high_missing.items())
        insights.append(
            InsightBlock(
                title="Data quality considerations",
                detail=(
                    "Some columns have notable missing values, which may mask patterns or introduce bias "
                    f"if not handled carefully: {cols}."
                ),
            )
        )
        recommendations.append(
            RecommendationBlock(
                title="Prioritize data quality improvements",
                action=(
                    f"Review collection processes for {', '.join(high_missing.index.tolist())} and "
                    "either improve capture rates or formally define how to handle missing values in analysis "
                    "and reporting."
                ),
            )
        )
        lines.append(
            "Data quality: several columns show missing data "
            f"({cols}). Improving capture here would strengthen downstream analyses."
        )

    # Generic but data-tied recommendations
    if roles["datetime"] and roles["numeric"]:
        recommendations.append(
            RecommendationBlock(
                title="Monitor trends over time",
                action=(
                    f"Use the detected date column(s) ({', '.join(roles['datetime'])}) with key numeric metrics "
                    "to build time-series dashboards that highlight seasonality, growth, or emerging risks."
                ),
            )
        )

    if outcome_candidates and roles["categorical"]:
        recommendations.append(
            RecommendationBlock(
                title="Segment outcomes across key groups",
                action=(
                    f"Compare {', '.join(outcome_candidates)} across important segments such as "
                    f"{', '.join(roles['categorical'][:3])} to identify high-risk groups or high-value segments."
                ),
            )
        )

    if not recommendations:
        recommendations.append(
            RecommendationBlock(
                title="Refine business questions using this dataset",
                action=(
                    "Use the identified numeric trends and categorical segments to frame sharper business questions, "
                    "such as which groups are driving volatility and which metrics best represent success."
                ),
            )
        )

    # Combine into plain-text narrative
    lines.append("")
    lines.append("Key recommendations:")
    for rec in recommendations:
        lines.append(f"- {rec.title}: {rec.action}")

    full_text = "\n".join(lines)
    return insights, recommendations, full_text


def run_automated_eda(df: pd.DataFrame) -> AnalysisResult:
    roles = detect_column_roles(df)
    df_clean = basic_clean(df)

    summary = AnalysisSummary(
        dataset_shape={"rows": int(df_clean.shape[0]), "columns": int(df_clean.shape[1])},
        column_overview={col: str(dtype) for col, dtype in df_clean.dtypes.items()},
        missing_summary=df_clean.isna().sum().to_dict(),
        detected_roles=roles,
    )

    summary_stats = compute_summary_statistics(df_clean, roles)
    correlations = compute_correlations(df_clean, roles)

    charts: List[ChartData] = []
    # Up to 2 numeric histograms
    for col in roles["numeric"][:2]:
        charts.append(generate_histogram_chart(df_clean, col))
    # Up to 2 categorical bar charts
    for col in roles["categorical"][:2]:
        charts.append(generate_categorical_bar_chart(df_clean, col))
    # Time trend
    if roles["datetime"]:
        trend_chart = generate_time_trend_chart(df_clean, roles["datetime"][0], roles)
        if trend_chart:
            charts.append(trend_chart)
    # Correlation heatmap
    corr_chart = generate_correlation_heatmap_chart(df_clean, roles)
    if corr_chart:
        charts.append(corr_chart)

    insights, recommendations, insight_text = generate_insights_and_recommendations(
        df_clean,
        roles,
        correlations,
    )

    cleaned_csv_b64 = dataframe_to_base64_csv(df_clean)

    return AnalysisResult(
        summary=summary,
        summary_statistics=summary_stats,
        correlations=correlations,
        key_insights=insights,
        recommendations=recommendations,
        charts=charts,
        cleaned_csv_base64=cleaned_csv_b64,
        insight_text=insight_text,
    )


