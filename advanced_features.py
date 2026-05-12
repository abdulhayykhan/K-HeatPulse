from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
import streamlit as st

try:
    import shap                
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False

try:
    import plotly.graph_objects as go                
    PLOTLY_AVAILABLE = True
except ImportError:
    go = None
    PLOTLY_AVAILABLE = False

try:
    from statsmodels.tsa.seasonal import STL                
    STATSMODELS_AVAILABLE = True
except ImportError:
    STL = None
    STATSMODELS_AVAILABLE = False

FEATURE_COLUMNS = [
    "year",
    "month",
    "day",
    "dayofweek",
    "is_weekend",
    "latitude",
    "longitude",
    "elevation",
    "tmin",
    "tmax",
    "tavg",
    "prcp",
    "wspd",
    "humidity",
    "pressure",
    "dew_point",
    "cloud_cover",
    "temp_range",
]
CATEGORICAL_COLUMNS = ["season", "wind_category", "rainfall_intensity"]
ALL_FEATURES = FEATURE_COLUMNS + CATEGORICAL_COLUMNS
TRANSPARENT_BG = "rgba(0,0,0,0)"


def _season_from_month(month: int) -> str:
    if month in {12, 1, 2}:
        return "Winter"
    if month in {3, 4, 5}:
        return "Spring"
    if month in {6, 7, 8}:
        return "Summer"
    return "Autumn"


def _risk_colour_for_probability(probability: float) -> str:
    if probability >= 0.7:
        return "#ef4444"
    if probability >= 0.45:
        return "#f97316"
    return "#22c55e"


def render_xai_panel(bundle: Any) -> None:
    st.divider()
    st.subheader("Feature D: Explainable AI")
    st.caption(
        "SHAP decomposes today's Random Forest prediction into feature-level contributions."
    )

    if not SHAP_AVAILABLE:
        st.info("SHAP is not installed in this environment.")
        return
    if not PLOTLY_AVAILABLE:
        st.info("Plotly is not installed in this environment.")
        return

    if st.button("Explain Today's Prediction", key="shap_btn", type="primary"):
        with st.spinner("Computing SHAP values..."):
            _compute_and_render_shap(bundle)


def _cached_shap_values(
    rf_model,
    background_hash: str,
    background_matrix: np.ndarray,
    explain_matrix: np.ndarray,
):
    explainer = shap.TreeExplainer(
        rf_model,
        data=background_matrix,
        feature_perturbation="interventional",
    )
    shap_values = explainer.shap_values(explain_matrix)
    if isinstance(shap_values, list):
        shap_values = shap_values[1]
    expected_value = explainer.expected_value
    if isinstance(expected_value, (list, np.ndarray)):
        expected_value = expected_value[1]
    return shap_values, float(expected_value)


def _compute_and_render_shap(bundle: Any) -> None:
    frame = bundle.frame
    rf_pipeline = bundle.rf_pipeline
    rf_model = rf_pipeline.named_steps["model"]
    preprocessor = rf_pipeline.named_steps["preprocessor"]

    train_frame = bundle.split["train"]
    background_frame = train_frame[ALL_FEATURES].dropna()
    if background_frame.empty:
        st.warning("Not enough complete training rows to compute SHAP values.")
        return

    background_sample = background_frame.sample(
        n=min(200, len(background_frame)), random_state=42
    )
    background_matrix = preprocessor.transform(background_sample)
    if hasattr(background_matrix, "toarray"):
        background_matrix = background_matrix.toarray()
    background_matrix = np.asarray(background_matrix, dtype=np.float32)

    latest = frame.sort_values("date").iloc[[-1]]
    explain_matrix_raw = preprocessor.transform(latest[ALL_FEATURES])
    if hasattr(explain_matrix_raw, "toarray"):
        explain_matrix_raw = explain_matrix_raw.toarray()
    explain_matrix = np.asarray(explain_matrix_raw, dtype=np.float32)

    try:
        feature_names = preprocessor.get_feature_names_out()
    except AttributeError:
        feature_names = [f"f{i}" for i in range(explain_matrix.shape[1])]

    background_hash = f"{background_matrix.shape}:{background_matrix.mean().round(4)}"
    shap_values, base_value = _cached_shap_values(
        rf_model,
        background_hash,
        background_matrix,
        explain_matrix,
    )

    shap_array = np.asarray(shap_values)
    if shap_array.ndim == 3:
        shap_row = shap_array[0, :, 1] if shap_array.shape[-1] > 1 else shap_array[0, :, 0]
    elif shap_array.ndim == 2:
        shap_row = shap_array[0]
    else:
        shap_row = shap_array.reshape(-1)

    shap_row = np.asarray(shap_row, dtype=np.float32).reshape(-1)
    if len(feature_names) != len(shap_row):
        limit = min(len(feature_names), len(shap_row))
        feature_names = list(feature_names)[:limit]
        shap_row = shap_row[:limit]

    shap_frame = pd.DataFrame({"feature": feature_names, "shap_value": shap_row})
    shap_frame["abs_value"] = shap_frame["shap_value"].abs()
    shap_frame = shap_frame.nlargest(15, "abs_value").sort_values("shap_value")

    colours = ["#ef4444" if value > 0 else "#3b82f6" for value in shap_frame["shap_value"]]
    fig = go.Figure(
        go.Bar(
            x=shap_frame["shap_value"],
            y=shap_frame["feature"],
            orientation="h",
            marker_color=colours,
            text=[f"{value:+.3f}" for value in shap_frame["shap_value"]],
            textposition="outside",
            hovertemplate="<b>%{y}</b><br>SHAP contribution: %{x:+.4f}<extra></extra>",
        )
    )

    prediction_probability = float(rf_pipeline.predict_proba(latest[ALL_FEATURES])[:, 1][0])
    fig.update_layout(
        title={
            "text": (
                "SHAP Feature Contributions — Today's Heat Risk: "
                f"<b>{prediction_probability:.1%}</b>"
                f" (base rate: {base_value:.1%})"
            ),
            "font": {"size": 14},
        },
        xaxis_title="SHAP value (impact on model output)",
        yaxis_title="",
        height=500,
        plot_bgcolor=TRANSPARENT_BG,
        paper_bgcolor=TRANSPARENT_BG,
        font={"size": 12},
        xaxis={"zeroline": True, "zerolinecolor": "#6b7280", "zerolinewidth": 1.5},
        margin={"l": 200, "r": 80, "t": 60, "b": 40},
    )
    st.plotly_chart(fig, use_container_width=True)

    top_positive = shap_frame[shap_frame["shap_value"] > 0].nlargest(1, "abs_value")
    top_negative = shap_frame[shap_frame["shap_value"] < 0].nlargest(1, "abs_value")

    summary_parts: list[str] = []
    if not top_positive.empty:
        feature_name = top_positive.iloc[0]["feature"].replace("num__", "").replace("cat__", "")
        summary_parts.append(f"**{feature_name}** is the biggest driver *increasing* risk today")
    if not top_negative.empty:
        feature_name = top_negative.iloc[0]["feature"].replace("num__", "").replace("cat__", "")
        summary_parts.append(f"**{feature_name}** is the biggest driver *reducing* risk")
    if summary_parts:
        st.info(" | ".join(summary_parts))


def render_whatif_simulator(bundle: Any) -> None:
    st.divider()
    st.subheader("Feature E: What-If Heat Scenario Simulator")
    st.caption(
        "Adjust conditions to simulate hypothetical Karachi weather and compare both model outputs."
    )

    frame = bundle.frame

    def _safe_bounds(column: str, low_q: float = 0.01, high_q: float = 0.99):
        series = frame[column].dropna()
        return float(series.quantile(low_q)), float(series.quantile(high_q)), float(series.median())

    t_lo, t_hi, t_med = _safe_bounds("tavg")
    h_lo, h_hi, h_med = _safe_bounds("humidity")
    w_lo, w_hi, w_med = _safe_bounds("wspd")
    p_lo, p_hi, p_med = _safe_bounds("pressure")
    dp_lo, dp_hi, dp_med = _safe_bounds("dew_point")

    slider_col, output_col = st.columns([2, 3])

    with slider_col:
        st.markdown("**Set conditions:**")
        sim_tavg = st.slider("Average Temperature (°C)", float(t_lo), float(t_hi), float(t_med), 0.5, key="sim_tavg")
        sim_humidity = st.slider("Humidity (%)", float(h_lo), float(h_hi), float(h_med), 1.0, key="sim_humidity")
        sim_wspd = st.slider("Wind Speed (km/h)", float(w_lo), float(w_hi), float(w_med), 0.5, key="sim_wspd")
        sim_pressure = st.slider("Pressure (hPa)", float(p_lo), float(p_hi), float(p_med), 0.5, key="sim_pressure")
        sim_dew = st.slider("Dew Point (°C)", float(dp_lo), float(dp_hi), float(dp_med), 0.5, key="sim_dew")

        today = pd.Timestamp.now(tz="Asia/Karachi").normalize().tz_localize(None)
        sim_month = st.selectbox(
            "Month",
            list(range(1, 13)),
            index=today.month - 1,
            format_func=lambda month: [
                "Jan",
                "Feb",
                "Mar",
                "Apr",
                "May",
                "Jun",
                "Jul",
                "Aug",
                "Sep",
                "Oct",
                "Nov",
                "Dec",
            ][month - 1],
            key="sim_month",
        )

    def _build_sim_row() -> pd.DataFrame:
        base = frame.sort_values("date").iloc[-1].copy()
        month_rows = frame[frame["month"] == sim_month]
        monthly_median = month_rows[FEATURE_COLUMNS].median(numeric_only=True)

        for column in FEATURE_COLUMNS:
            if column in monthly_median and pd.notna(monthly_median[column]):
                base[column] = monthly_median[column]

        base["tavg"] = sim_tavg
        base["tmin"] = sim_tavg - 4.0
        base["tmax"] = sim_tavg + 6.0
        base["temp_range"] = base["tmax"] - base["tmin"]
        base["humidity"] = sim_humidity
        base["wspd"] = sim_wspd
        base["pressure"] = sim_pressure
        base["dew_point"] = sim_dew
        base["month"] = sim_month
        base["day"] = 15
        base["dayofweek"] = 2
        base["is_weekend"] = 0
        base["year"] = today.year
        base["season"] = _season_from_month(sim_month)
        return pd.DataFrame([base])

    sim_row = _build_sim_row()

    with output_col:
        rf_prob, dl_prob = _run_sim_predictions(bundle, sim_row)
        _render_gauge_row(rf_prob, dl_prob)
        _render_risk_interpretation(rf_prob, dl_prob, sim_humidity)


def _run_sim_predictions(bundle: Any, sim_row: pd.DataFrame) -> tuple[float, float | None]:
    try:
        rf_prob = float(bundle.rf_pipeline.predict_proba(sim_row[ALL_FEATURES])[:, 1][0])
    except Exception:
        rf_prob = 0.5

    dl_prob = None
    if bundle.dl_model is not None and bundle.dl_preprocessor is not None:
        try:
            transformed = bundle.dl_preprocessor.transform(sim_row[ALL_FEATURES])
            if hasattr(transformed, "toarray"):
                transformed = transformed.toarray()
            transformed = np.asarray(transformed, dtype=np.float32)
            dl_prob = float(bundle.dl_model.predict(transformed, verbose=0).ravel()[0])
        except Exception:
            dl_prob = None

    return rf_prob, dl_prob


def _render_gauge_row(rf_prob: float, dl_prob: float | None) -> None:
    if not PLOTLY_AVAILABLE:
        st.metric("RF Heat Risk", f"{rf_prob:.1%}")
        if dl_prob is not None:
            st.metric("NN Heat Risk", f"{dl_prob:.1%}")
        return

    def _gauge(probability: float, label: str):
        fig = go.Figure(
            go.Indicator(
                mode="gauge+number+delta",
                value=round(probability * 100, 1),
                number={"suffix": "%", "font": {"size": 36}},
                delta={
                    "reference": 50,
                    "increasing": {"color": "#ef4444"},
                    "decreasing": {"color": "#22c55e"},
                },
                gauge={
                    "axis": {"range": [0, 100], "ticksuffix": "%"},
                    "bar": {"color": _risk_colour_for_probability(probability), "thickness": 0.3},
                    "steps": [
                        {"range": [0, 45], "color": "#dcfce7"},
                        {"range": [45, 70], "color": "#fef9c3"},
                        {"range": [70, 100], "color": "#fee2e2"},
                    ],
                    "threshold": {
                        "line": {"color": "#1f2937", "width": 3},
                        "thickness": 0.8,
                        "value": 50,
                    },
                },
                title={"text": label, "font": {"size": 14}},
            )
        )
        fig.update_layout(height=260, margin={"t": 40, "b": 10, "l": 20, "r": 20}, paper_bgcolor=TRANSPARENT_BG)
        return fig

    if dl_prob is not None:
        left_col, right_col = st.columns(2)
        with left_col:
            st.plotly_chart(_gauge(rf_prob, "Random Forest"), use_container_width=True)
        with right_col:
            st.plotly_chart(_gauge(dl_prob, "Neural Network"), use_container_width=True)
    else:
        st.plotly_chart(_gauge(rf_prob, "Random Forest"), use_container_width=True)
        st.info("Neural Network unavailable because TensorFlow is not installed.")


def _render_risk_interpretation(rf_prob: float, dl_prob: float | None, humidity: float) -> None:
    avg_prob = float(np.mean([probability for probability in [rf_prob, dl_prob] if probability is not None]))

    if avg_prob >= 0.7:
        tier, icon, level = "HIGH RISK", "🔴", "error"
        advice = "Both models signal severe heat conditions."
    elif avg_prob >= 0.45:
        tier, icon, level = "MODERATE RISK", "🟡", "warning"
        advice = "Heat conditions look borderline and should be monitored closely."
    else:
        tier, icon, level = "LOW RISK", "🟢", "success"
        advice = "Conditions look relatively benign for Karachi norms."

    if humidity >= 75:
        advice += " Humidity is also elevated."

    getattr(st, level)(f"{icon} **{tier}** — {advice}")

    if dl_prob is not None:
        gap = abs(rf_prob - dl_prob)
        if gap < 0.10:
            st.success(f"Models agree closely (gap: {gap:.1%}).")
        elif gap < 0.25:
            st.warning(f"Models diverge slightly (gap: {gap:.1%}).")
        else:
            st.error(f"Models strongly disagree (gap: {gap:.1%}). Treat the result with caution.")


def render_trend_decomposition(frame: pd.DataFrame) -> None:
    st.divider()
    st.subheader("Feature F: 25-Year Heat Trend Decomposition")
    st.caption(
        "STL decomposition separates Karachi's monthly maximum temperature into observed, trend, seasonal, and residual components."
    )

    if not STATSMODELS_AVAILABLE:
        st.info("statsmodels is not installed in this environment.")
        return
    if not PLOTLY_AVAILABLE:
        st.info("Plotly is not installed in this environment.")
        return

    monthly = (
        frame.dropna(subset=["tmax"])
        .groupby(["year", "month"])["tmax"]
        .mean()
        .reset_index()
    )
    monthly["date"] = pd.to_datetime(monthly[["year", "month"]].assign(day=1))
    monthly = monthly.sort_values("date").reset_index(drop=True)

    full_index = pd.date_range(monthly["date"].min(), monthly["date"].max(), freq="MS")
    series = monthly.set_index("date")["tmax"].reindex(full_index).interpolate("linear")

    if len(series) < 24:
        st.warning("Not enough monthly data for seasonal decomposition.")
        return

    result = STL(series, period=12, robust=True).fit()
    dates = series.index
    residual = result.resid

    figure = go.Figure()
    figure.add_trace(
        go.Scatter(
            x=dates,
            y=series.values,
            mode="lines",
            name="Observed",
            line={"color": "#6366f1", "width": 1.5},
            opacity=0.7,
            yaxis="y1",
        )
    )
    figure.add_trace(
        go.Scatter(
            x=dates,
            y=result.trend,
            mode="lines",
            name="Trend",
            line={"color": "#ef4444", "width": 2.5},
            yaxis="y2",
        )
    )
    figure.add_trace(
        go.Scatter(
            x=dates,
            y=result.seasonal,
            mode="lines",
            name="Seasonal",
            line={"color": "#f97316", "width": 1.5},
            fill="tozeroy",
            fillcolor="rgba(249,115,22,0.12)",
            yaxis="y3",
        )
    )
    figure.add_trace(
        go.Bar(
            x=dates,
            y=residual,
            name="Residual / Anomaly",
            marker_color=["#ef4444" if value > 0 else "#3b82f6" for value in residual],
            opacity=0.8,
            yaxis="y4",
        )
    )

    heatwave_date = pd.Timestamp("2015-06-01")
    if dates.min() <= heatwave_date <= dates.max():
        heatwave_index = series.index.get_indexer([heatwave_date], method="nearest")[0]
        figure.add_annotation(
            x=heatwave_date,
            y=residual.iloc[heatwave_index],
            yref="y4",
            text="2015 Heatwave",
            showarrow=True,
            arrowhead=2,
            arrowcolor="#ef4444",
            font={"color": "#ef4444", "size": 11},
            bgcolor="rgba(239,68,68,0.1)",
            bordercolor="#ef4444",
        )

    valid_trend = result.trend.dropna()
    if len(valid_trend) >= 2:
        years_span = (valid_trend.index[-1] - valid_trend.index[0]).days / 365.25
        trend_change = valid_trend.iloc[-1] - valid_trend.iloc[0]
        slope_per_decade = (trend_change / years_span) * 10 if years_span > 0 else 0.0
        sign = "+" if slope_per_decade >= 0 else ""
        figure.add_annotation(
            x=0.01,
            y=0.97,
            xref="paper",
            yref="paper",
            text=f"<b>Trend: {sign}{slope_per_decade:.2f}°C / decade</b>",
            showarrow=False,
            font={"size": 13, "color": "#ef4444"},
            bgcolor="rgba(239,68,68,0.08)",
            bordercolor="#ef4444",
            borderwidth=1,
        )

    figure.update_layout(
        title="Karachi Monthly Maximum Temperature — STL Decomposition",
        height=700,
        plot_bgcolor=TRANSPARENT_BG,
        paper_bgcolor=TRANSPARENT_BG,
        hovermode="x unified",
        legend={"orientation": "h", "y": 1.06},
        yaxis={"title": "Observed (°C)", "domain": [0.76, 1.0], "gridcolor": "#e5e7eb"},
        yaxis2={"title": "Trend (°C)", "domain": [0.52, 0.74], "gridcolor": "#e5e7eb"},
        yaxis3={"title": "Seasonal (°C)", "domain": [0.28, 0.50], "gridcolor": "#e5e7eb", "zeroline": True},
        yaxis4={"title": "Residual (°C)", "domain": [0.0, 0.26], "gridcolor": "#e5e7eb", "zeroline": True, "zerolinecolor": "#9ca3af"},
        xaxis={"domain": [0, 1], "anchor": "y4"},
        margin={"t": 80, "b": 40, "l": 70, "r": 30},
    )
    st.plotly_chart(figure, use_container_width=True)

    if len(valid_trend) >= 2:
        years_span = (valid_trend.index[-1] - valid_trend.index[0]).days / 365.25
        total_change = valid_trend.iloc[-1] - valid_trend.iloc[0]
        slope_per_decade = (total_change / years_span) * 10 if years_span > 0 else 0.0
        peak_anomaly_month = residual.abs().idxmax().strftime("%B %Y")
        peak_anomaly_value = residual.abs().max()

        metric_col_1, metric_col_2, metric_col_3 = st.columns(3)
        metric_col_1.metric("Total Trend Change", f"{total_change:+.2f}°C", f"over {years_span:.0f} years")
        metric_col_2.metric("Rate of Warming", f"{slope_per_decade:+.2f}°C/decade", "trend slope")
        metric_col_3.metric("Largest Anomaly", f"±{peak_anomaly_value:.2f}°C", peak_anomaly_month)

        if slope_per_decade > 0.3:
            st.warning(
                f"Karachi's warming trend is {slope_per_decade:+.2f}°C/decade, which is well above the global land surface average."
            )
