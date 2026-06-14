"""Plotly chart factory for consistent, branded visualizations."""
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
import numpy as np

# ── Brand palette ──────────────────────────────────────────────────────────
PRIMARY   = "#6C63FF"
SECONDARY = "#48CAE4"
SUCCESS   = "#06D6A0"
DANGER    = "#EF476F"
WARNING   = "#FFD166"
BG        = "rgba(0,0,0,0)"
GRID      = "rgba(255,255,255,0.06)"
TEXT      = "#E0E0E0"

LAYOUT = dict(
    paper_bgcolor=BG, plot_bgcolor=BG,
    font=dict(color=TEXT, family="Inter, sans-serif"),
    margin=dict(l=20, r=20, t=40, b=20),
    legend=dict(bgcolor="rgba(255,255,255,0.05)", bordercolor="rgba(255,255,255,0.1)"),
)


def model_comparison_bar(results: dict) -> go.Figure:
    names   = list(results.keys())
    metrics = ["accuracy", "auc", "f1"]
    colors  = [PRIMARY, SECONDARY, SUCCESS]
    fig = go.Figure()
    for i, m in enumerate(metrics):
        fig.add_trace(go.Bar(
            name=m.upper(), x=names, y=[results[n][m] for n in names],
            marker_color=colors[i], opacity=0.88,
            text=[f"{results[n][m]:.3f}" for n in names],
            textposition="outside",
        ))
    fig.update_layout(**LAYOUT, title="Model Performance Comparison",
                      barmode="group", yaxis=dict(range=[0.85, 1.01], gridcolor=GRID),
                      xaxis=dict(gridcolor=GRID))
    return fig


def grade_distribution(df: pd.DataFrame) -> go.Figure:
    fig = go.Figure()
    for label, color, name in [(1, SUCCESS, "Pass"), (0, DANGER, "Fail")]:
        sub = df[df["passed"] == label]["final_grade"]
        fig.add_trace(go.Histogram(x=sub, name=name, marker_color=color,
                                   opacity=0.75, nbinsx=20))
    fig.update_layout(**LAYOUT, title="Final Grade Distribution by Outcome",
                      barmode="overlay", xaxis_title="Grade (0–20)",
                      yaxis=dict(gridcolor=GRID), xaxis=dict(gridcolor=GRID))
    return fig


def risk_scatter(df: pd.DataFrame) -> go.Figure:
    sample = df.sample(min(500, len(df)), random_state=42)
    fig = px.scatter(sample, x="absences", y="final_grade",
                     color="passed", symbol="passed",
                     color_discrete_map={1: SUCCESS, 0: DANGER},
                     labels={"passed": "Outcome", "absences": "Absences", "final_grade": "Final Grade"},
                     title="Absences vs Final Grade")
    fig.update_layout(**LAYOUT, xaxis=dict(gridcolor=GRID), yaxis=dict(gridcolor=GRID))
    return fig


def feature_importance_bar(feat_df: pd.DataFrame) -> go.Figure:
    feat_df = feat_df.sort_values("SHAP Importance")
    fig = go.Figure(go.Bar(
        x=feat_df["SHAP Importance"], y=feat_df["Feature"],
        orientation="h", marker_color=PRIMARY,
        marker=dict(line=dict(width=0)),
    ))
    fig.update_layout(**LAYOUT, title="SHAP Feature Importance",
                      xaxis=dict(gridcolor=GRID, title="Mean |SHAP Value|"),
                      yaxis=dict(gridcolor=GRID))
    return fig


def study_time_pass_rate(df: pd.DataFrame) -> go.Figure:
    grouped = df.groupby("study_time")["passed"].mean().reset_index()
    fig = go.Figure(go.Bar(
        x=grouped["study_time"].map({1:"<2h",2:"2-5h",3:"5-10h",4:">10h"}),
        y=grouped["passed"],
        marker_color=[PRIMARY, SECONDARY, SUCCESS, WARNING],
        text=[f"{v:.1%}" for v in grouped["passed"]],
        textposition="outside",
    ))
    fig.update_layout(**LAYOUT, title="Pass Rate by Study Time",
                      yaxis=dict(range=[0, 1.1], tickformat=".0%", gridcolor=GRID),
                      xaxis=dict(title="Weekly Study Hours", gridcolor=GRID))
    return fig


def correlation_heatmap(df: pd.DataFrame) -> go.Figure:
    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    corr = df[num_cols].corr()
    fig = go.Figure(go.Heatmap(
        z=corr.values, x=corr.columns, y=corr.columns,
        colorscale=[[0, DANGER], [0.5, "rgba(30,30,60,1)"], [1, SUCCESS]],
        zmid=0, text=np.round(corr.values, 2), texttemplate="%{text}",
    ))
    fig.update_layout(**LAYOUT, title="Feature Correlation Matrix",
                      height=600, margin=dict(l=100, r=20, t=40, b=100))
    return fig


def gauge_chart(value: float, title: str) -> go.Figure:
    color = SUCCESS if value >= 0.7 else (WARNING if value >= 0.5 else DANGER)
    fig = go.Figure(go.Indicator(
        mode="gauge+number", value=value * 100,
        title={"text": title, "font": {"color": TEXT, "size": 14}},
        number={"suffix": "%", "font": {"color": color, "size": 28}},
        gauge=dict(
            axis=dict(range=[0, 100], tickcolor=TEXT),
            bar=dict(color=color),
            bgcolor="rgba(255,255,255,0.05)",
            steps=[
                dict(range=[0, 50], color="rgba(239,71,111,0.15)"),
                dict(range=[50, 75], color="rgba(255,209,102,0.15)"),
                dict(range=[75, 100], color="rgba(6,214,160,0.15)"),
            ],
            threshold=dict(line=dict(color="white", width=2), value=70),
        ),
    ))
    fig.update_layout(paper_bgcolor=BG, font=dict(color=TEXT), height=220,
                      margin=dict(l=20, r=20, t=50, b=20))
    return fig


def absence_impact(df: pd.DataFrame) -> go.Figure:
    bins = pd.cut(df["absences"], bins=[-1,2,7,15,50],
                  labels=["Low (0-2)", "Medium (3-7)", "High (8-15)", "Very High (>15)"])
    grouped = df.groupby(bins, observed=True)["passed"].agg(["mean","count"]).reset_index()
    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=grouped["absences"].astype(str), y=grouped["mean"],
        name="Pass Rate", marker_color=PRIMARY,
        text=[f"{v:.1%}" for v in grouped["mean"]], textposition="outside",
    ))
    fig.update_layout(**LAYOUT, title="Pass Rate by Absence Level",
                      yaxis=dict(range=[0,1.15], tickformat=".0%", gridcolor=GRID),
                      xaxis=dict(gridcolor=GRID))
    return fig
