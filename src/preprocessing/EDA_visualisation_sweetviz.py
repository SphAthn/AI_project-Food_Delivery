"""
Author: Baozhi Li
Flinders FAN: li2777
Date of Completion: Oct 7th, 2026
"""


"""
Visualisation Module
This module contains helper functions to generate Sweetviz EDA reports
for the Zomato delivery dataset.
"""
 
import os
import re
 
import pandas as pd
import sweetviz as sv
 
DEFAULT_TARGET = "Time_taken(min)"
 
 
def _resolve_column(df: pd.DataFrame, name: str) -> str:
    """Finds a column even if spacing/case differs (e.g. 'Time_taken (min)')."""
    if name in df.columns:
        return name
    key = re.sub(r"[^a-z0-9]", "", name.lower())
    for col in df.columns:
        if re.sub(r"[^a-z0-9]", "", str(col).lower()) == key:
            return col
    raise KeyError(f"Column '{name}' not found. Available: {list(df.columns)}")
 
 
def _prepare(df: pd.DataFrame, target: str = None):
    """Copies the dataframe and makes the target numeric (e.g. '(min) 24' -> 24)."""
    df_sv = df.copy()
    target_feat = None
    if target:
        target_feat = _resolve_column(df_sv, target)
        if not pd.api.types.is_numeric_dtype(df_sv[target_feat]):
            digits = df_sv[target_feat].astype(str).str.extract(r"(\d+\.?\d*)")[0]
            df_sv[target_feat] = pd.to_numeric(digits, errors="coerce")
    return df_sv, target_feat
 
 
def _save(report, output_path: str, open_browser: bool) -> str:
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    report.show_html(filepath=output_path, open_browser=open_browser)
    print(f"Sweetviz report saved to: {output_path}")
    return output_path
 
 
def generate_sweetviz_report(
    df: pd.DataFrame,
    output_path: str = "reports/Zomato_Sweetviz_Report.html",
    target: str = DEFAULT_TARGET,
    pairwise_analysis: str = "auto",
    open_browser: bool = False,
) -> str:
    """
    Generates a Sweetviz EDA report for one dataframe.
 
    Args:
        df: Dataframe to analyse (the cleaned, non-encoded one is the most readable).
        output_path: Where to save the HTML report.
        target: Numeric target feature (None to disable).
        pairwise_analysis: 'auto', 'on' or 'off' ('off' is faster on big data).
        open_browser: Open the report in the browser when done.
 
    Returns:
        Path of the HTML report.
    """
    df_sv, target_feat = _prepare(df, target)
    report = sv.analyze(df_sv, target_feat=target_feat, pairwise_analysis=pairwise_analysis)
    return _save(report, output_path, open_browser)
 
 
def generate_comparison_report(
    df_a: pd.DataFrame,
    df_b: pd.DataFrame,
    output_path: str = "reports/Zomato_Comparison_Report.html",
    names: tuple = ("Dataset A", "Dataset B"),
    target: str = DEFAULT_TARGET,
    pairwise_analysis: str = "auto",
    open_browser: bool = False,
) -> str:
    """
    Generates a Sweetviz comparison report between two dataframes
    (e.g. raw vs cleaned, or train vs test).
    """
    a, target_feat = _prepare(df_a, target)
    b, _ = _prepare(df_b, target)
    report = sv.compare(
        [a, names[0]], [b, names[1]], target_feat=target_feat, pairwise_analysis=pairwise_analysis
    )
    return _save(report, output_path, open_browser)
 
 
def generate_subgroup_report(
    df: pd.DataFrame,
    column: str,
    value,
    output_path: str = "reports/Zomato_Subgroup_Report.html",
    target: str = DEFAULT_TARGET,
    pairwise_analysis: str = "auto",
    open_browser: bool = False,
) -> str:
    """
    Compares one subgroup against the rest (e.g. column='Festival', value='Yes'
    or column='City', value='Urban').
    """
    df_sv, target_feat = _prepare(df, target)
    col = _resolve_column(df_sv, column)
    mask = df_sv[col] == value
    report = sv.compare(
        [df_sv[mask], f"{col} = {value}"],
        [df_sv[~mask], "Rest"],
        target_feat=target_feat,
        pairwise_analysis=pairwise_analysis,
    )
    return _save(report, output_path, open_browser)
 
 
if __name__ == "__main__":
    # Run from the project root: python src/EDA_visualisation_sweetviz.py
    cleaned = pd.read_csv("data/Zomato_Cleaned.csv")
    generate_sweetviz_report(cleaned, "reports/zomato_eda.html")
