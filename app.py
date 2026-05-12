from __future__ import annotations

import importlib
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import requests
import seaborn as sns
import streamlit as st
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.utils.class_weight import compute_class_weight

def load_tensorflow_backend() -> tuple[Any | None, Any | None]:
    try:
        tf_module = importlib.import_module("tensorflow")
    except Exception:
        return None, None
    return tf_module, getattr(tf_module, "keras", None)


tf, keras = load_tensorflow_backend()

from advanced_features import (
    render_trend_decomposition,
    render_whatif_simulator,
    render_xai_panel,
)


BASE_DIR = Path(__file__).resolve().parent
HISTORICAL_PATH = BASE_DIR / "pakistan_weather_2000_2024.csv"
RECENT_PATH = BASE_DIR / "pakistan_weather_data-Sep2024-Oct2025.csv"

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
TARGET_COLUMN = "is_hot_day"

RANDOM_FOREST_MODEL_NAME = "Random Forest"
NEURAL_NETWORK_MODEL_NAME = "Neural Network"

MONTH_NAMES = {
    1: "Jan",
    2: "Feb",
    3: "Mar",
    4: "Apr",
    5: "May",
    6: "Jun",
    7: "Jul",
    8: "Aug",
    9: "Sep",
    10: "Oct",
    11: "Nov",
    12: "Dec",
}


@dataclass
class ModelBundle:
    frame: pd.DataFrame
    split: dict[str, Any]
    rf_pipeline: Pipeline
    dl_preprocessor: ColumnTransformer | None
    dl_model: Any | None
    dl_history: Any | None
    dl_metrics: dict[str, float] | None
    rf_metrics: dict[str, float]


sns.set_theme(style="whitegrid")


def make_one_hot_encoder() -> OneHotEncoder:
    try:
        return OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    except TypeError:
        return OneHotEncoder(handle_unknown="ignore", sparse=False)


def parse_date_column(frame: pd.DataFrame) -> pd.DataFrame:
    frame = frame.copy()
    frame["date"] = pd.to_datetime(frame["date"], errors="coerce")
    return frame


def normalize_karachi_frame(frame: pd.DataFrame) -> pd.DataFrame:
    frame = frame.copy()
    frame = frame[frame["city"].astype(str).str.lower() == "karachi"].copy()
    frame = frame.sort_values("date").reset_index(drop=True)
    frame = frame.ffill()
    frame["year"] = frame["date"].dt.year
    frame["month"] = frame["date"].dt.month
    frame["day"] = frame["date"].dt.day
    frame["dayofweek"] = frame["date"].dt.dayofweek
    frame["is_weekend"] = (frame["dayofweek"] >= 5).astype(int)
    frame["temp_range"] = pd.to_numeric(frame["temp_range"], errors="coerce")
    frame[TARGET_COLUMN] = pd.to_numeric(frame[TARGET_COLUMN], errors="coerce").fillna(0).astype(int)
    for column in FEATURE_COLUMNS + CATEGORICAL_COLUMNS:
        if column not in frame.columns:
            frame[column] = np.nan
    return frame


def load_merged_data() -> pd.DataFrame:
    historical = pd.read_csv(HISTORICAL_PATH)
    recent = pd.read_csv(RECENT_PATH)
    historical = historical[historical["city"].astype(str).str.lower() == "karachi"].copy()
    recent = recent[recent["city"].astype(str).str.lower() == "karachi"].copy()
    historical = parse_date_column(historical)
    recent = parse_date_column(recent)
    if recent["date"].dt.tz is not None:
        recent["date"] = recent["date"].dt.tz_localize(None)
    merged = pd.concat([historical, recent], ignore_index=True)
    merged = normalize_karachi_frame(merged)
    return merged


def build_rf_pipeline() -> Pipeline:
    numeric_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
        ]
    )
    categorical_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", make_one_hot_encoder()),
        ]
    )
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, FEATURE_COLUMNS),
            ("cat", categorical_transformer, CATEGORICAL_COLUMNS),
        ]
    )
    model = RandomForestClassifier(
        n_estimators=300,
        random_state=42,
        class_weight="balanced_subsample",
        min_samples_leaf=2,
        max_features="sqrt",
        n_jobs=-1,
    )
    return Pipeline(steps=[("preprocessor", preprocessor), ("model", model)], memory=None)


def build_dl_preprocessor() -> ColumnTransformer:
    numeric_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
    categorical_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", make_one_hot_encoder()),
        ]
    )
    return ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, FEATURE_COLUMNS),
            ("cat", categorical_transformer, CATEGORICAL_COLUMNS),
        ]
    )


def build_dl_model(input_dim: int):
    if keras is None:
        return None
    model = keras.Sequential(
        [
            keras.layers.Input(shape=(input_dim,)),
            keras.layers.Dense(64, activation="relu"),
            keras.layers.Dropout(0.25),
            keras.layers.Dense(32, activation="relu"),
            keras.layers.Dropout(0.15),
            keras.layers.Dense(1, activation="sigmoid"),
        ]
    )
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.001),
        loss="binary_crossentropy",
        metrics=["accuracy", keras.metrics.AUC(name="auc")],
    )
    return model


def chronological_split(frame: pd.DataFrame, test_size: float = 0.2) -> tuple[pd.DataFrame, pd.DataFrame]:
    cutoff = int(len(frame) * (1 - test_size))
    train = frame.iloc[:cutoff].copy()
    test = frame.iloc[cutoff:].copy()
    return train, test


@st.cache_resource(show_spinner=False)
def train_models() -> ModelBundle:
    frame = load_merged_data()
    usable = frame.dropna(subset=[TARGET_COLUMN]).copy()
    usable[TARGET_COLUMN] = usable[TARGET_COLUMN].astype(int)
    train_frame, test_frame = chronological_split(usable, test_size=0.2)

    rf_pipeline = build_rf_pipeline()
    rf_pipeline.fit(train_frame[FEATURE_COLUMNS + CATEGORICAL_COLUMNS], train_frame[TARGET_COLUMN])
    rf_predictions = rf_pipeline.predict(test_frame[FEATURE_COLUMNS + CATEGORICAL_COLUMNS])
    rf_probabilities = rf_pipeline.predict_proba(test_frame[FEATURE_COLUMNS + CATEGORICAL_COLUMNS])[:, 1]

    rf_metrics = {
        "accuracy": float(accuracy_score(test_frame[TARGET_COLUMN], rf_predictions)),
        "auc": float(roc_auc_score(test_frame[TARGET_COLUMN], rf_probabilities)),
    }

    dl_preprocessor = None
    dl_model = None
    dl_history = None
    dl_metrics = None

    if keras is not None and tf is not None:
        dl_preprocessor = build_dl_preprocessor()
        x_train = dl_preprocessor.fit_transform(train_frame[FEATURE_COLUMNS + CATEGORICAL_COLUMNS])
        x_test = dl_preprocessor.transform(test_frame[FEATURE_COLUMNS + CATEGORICAL_COLUMNS])
        y_train = train_frame[TARGET_COLUMN].astype(float).to_numpy()
        y_test = test_frame[TARGET_COLUMN].astype(float).to_numpy()

        if hasattr(x_train, "toarray"):
            x_train = x_train.toarray()
        if hasattr(x_test, "toarray"):
            x_test = x_test.toarray()

        x_train = np.asarray(x_train, dtype=np.float32)
        x_test = np.asarray(x_test, dtype=np.float32)

        class_labels = np.unique(y_train)
        weights = compute_class_weight(class_weight="balanced", classes=class_labels, y=y_train)
        class_weight = {int(label): float(weight) for label, weight in zip(class_labels, weights)}

        dl_model = build_dl_model(x_train.shape[1])
        if dl_model is not None:
            callbacks = [
                keras.callbacks.EarlyStopping(monitor="val_loss", patience=10, restore_best_weights=True)
            ]
            dl_history = dl_model.fit(
                x_train,
                y_train,
                validation_split=0.2,
                epochs=60,
                batch_size=32,
                verbose=0,
                class_weight=class_weight,
                callbacks=callbacks,
            )
            dl_probabilities = dl_model.predict(x_test, verbose=0).ravel()
            dl_predictions = (dl_probabilities >= 0.5).astype(int)
            dl_metrics = {
                "accuracy": float(accuracy_score(y_test, dl_predictions)),
                "auc": float(roc_auc_score(y_test, dl_probabilities)),
            }

    split = {
        "train": train_frame,
        "test": test_frame,
    }

    return ModelBundle(
        frame=usable,
        split=split,
        rf_pipeline=rf_pipeline,
        dl_preprocessor=dl_preprocessor,
        dl_model=dl_model,
        dl_history=dl_history,
        dl_metrics=dl_metrics,
        rf_metrics=rf_metrics,
    )


@st.cache_data(ttl=600, show_spinner=False)
def get_live_karachi_weather() -> dict[str, Any]:
    try:
        api_key = st.secrets["VISUAL_CROSSING_API_KEY"]
    except (KeyError, FileNotFoundError):
        return {"ok": False, "error": "Missing VISUAL_CROSSING_API_KEY in Streamlit secrets."}

    url = (
        "https://weather.visualcrossing.com/VisualCrossingWebServices/rest/services/timeline/"
        f"Karachi,PK/today?unitGroup=metric&key={api_key}&contentType=json"
    )
    response = requests.get(url, timeout=20)
    response.raise_for_status()
    payload = response.json()
    current = payload.get("currentConditions", {})
    return {
        "ok": True,
        "humidity": current.get("humidity"),
        "wspd": current.get("windspeed"),
        "pressure": current.get("pressure"),
        "temp": current.get("temp"),
        "conditions": current.get("conditions", "Unknown"),
        "source": payload.get("resolvedAddress", "Karachi,PK"),
    }


def season_from_month(month: int) -> str:
    if month in {12, 1, 2}:
        return "Winter"
    if month in {3, 4, 5}:
        return "Spring"
    if month in {6, 7, 8}:
        return "Summer"
    return "Autumn"


def build_live_feature_row(frame: pd.DataFrame, live_weather: dict[str, Any]) -> pd.DataFrame:
    today = pd.Timestamp.now(tz="Asia/Karachi").normalize().tz_localize(None)
    latest = frame.sort_values("date").iloc[-1].copy()
    month_rows = frame[frame["month"] == today.month]
    monthly_baseline = month_rows[FEATURE_COLUMNS + CATEGORICAL_COLUMNS].median(numeric_only=True)

    live_row = latest.copy()
    for column in FEATURE_COLUMNS:
        if column in monthly_baseline and pd.notna(monthly_baseline[column]):
            live_row[column] = monthly_baseline[column]

    live_row["date"] = today
    live_row["year"] = today.year
    live_row["month"] = today.month
    live_row["day"] = today.day
    live_row["dayofweek"] = today.dayofweek
    live_row["is_weekend"] = int(today.dayofweek >= 5)
    live_row["season"] = season_from_month(today.month)
    live_row["humidity"] = live_weather.get("humidity", live_row.get("humidity"))
    live_row["wspd"] = live_weather.get("wspd", live_row.get("wspd"))
    live_row["pressure"] = live_weather.get("pressure", live_row.get("pressure"))
    live_row["tavg"] = live_weather.get("temp", live_row.get("tavg"))
    return pd.DataFrame([live_row])


def probability_to_label(probability: float) -> str:
    if probability >= 0.7:
        return "High"
    if probability >= 0.45:
        return "Moderate"
    return "Low"


def render_metrics_cards(live_weather: dict[str, Any]) -> None:
    columns = st.columns(4)
    columns[0].metric("Humidity", f"{live_weather.get('humidity', 'N/A')} %")
    columns[1].metric("Wind Speed", f"{live_weather.get('wspd', 'N/A')} km/h")
    columns[2].metric("Pressure", f"{live_weather.get('pressure', 'N/A')} hPa")
    columns[3].metric("Condition", f"{live_weather.get('conditions', 'N/A')}")



def build_fallback_feature_row(frame: pd.DataFrame) -> pd.DataFrame:
    latest = frame.sort_values("date").iloc[[-1]].copy()
    latest.loc[:, "season"] = season_from_month(int(latest["month"].iloc[0]))
    return latest


def render_sidebar(bundle: ModelBundle) -> None:
    frame = bundle.frame
    train_frame = bundle.split["train"]
    test_frame = bundle.split["test"]
    with st.sidebar:
        st.header("Dataset Snapshot")
        st.write(f"Rows after Karachi filter: {len(frame):,}")
        st.write(f"Date span: {frame['date'].min().date()} to {frame['date'].max().date()}")
        st.write(f"Training rows: {len(train_frame):,}")
        st.write(f"Testing rows: {len(test_frame):,}")
        if keras is None:
            st.warning("TensorFlow/Keras is not available in this Python environment. The ML brain is active; the DL brain needs TensorFlow support.")


def render_live_pulse_section() -> None:
    st.subheader("Feature A: Live Pulse")
    if st.button("Get Live Karachi Stats", type="primary"):
        try:
            live_weather = get_live_karachi_weather()
            if not live_weather.get("ok"):
                st.warning(live_weather.get("error", "Live weather unavailable."))
            else:
                render_metrics_cards(live_weather)
                st.success(f"Live source: {live_weather.get('source')}")
        except Exception as exc:
            st.error(f"Could not fetch live Karachi weather: {exc}")


def get_prediction_row(frame: pd.DataFrame) -> pd.DataFrame:
    try:
        live_weather_state = get_live_karachi_weather()
    except Exception:
        live_weather_state = {"ok": False}

    if live_weather_state and live_weather_state.get("ok"):
        return build_live_feature_row(frame, live_weather_state)
    return build_fallback_feature_row(frame)


def run_ml_prediction(bundle: ModelBundle, live_row: pd.DataFrame) -> None:
    try:
        ml_prob = float(bundle.rf_pipeline.predict_proba(live_row[FEATURE_COLUMNS + CATEGORICAL_COLUMNS])[:, 1][0])
        ml_pred = int(ml_prob >= 0.5)
        st.metric("ML Heat Risk", f"{ml_prob:.1%}", delta=probability_to_label(ml_prob))
        st.write("Random Forest prediction:", "Heatwave risk likely" if ml_pred else "Heatwave risk lower")
    except Exception as exc:
        st.error(f"ML prediction failed: {exc}")


def run_dl_prediction(bundle: ModelBundle, live_row: pd.DataFrame) -> None:
    if bundle.dl_model is None or bundle.dl_preprocessor is None:
        st.warning("TensorFlow/Keras is unavailable here, so the DL brain cannot run in this environment.")
        return

    try:
        dl_features = bundle.dl_preprocessor.transform(live_row[FEATURE_COLUMNS + CATEGORICAL_COLUMNS])
        if hasattr(dl_features, "toarray"):
            dl_features = dl_features.toarray()
        dl_features = np.asarray(dl_features, dtype=np.float32)
        dl_prob = float(bundle.dl_model.predict(dl_features, verbose=0).ravel()[0])
        dl_pred = int(dl_prob >= 0.5)
        st.metric("DL Heat Risk", f"{dl_prob:.1%}", delta=probability_to_label(dl_prob))
        st.write("Neural network prediction:", "Heatwave risk likely" if dl_pred else "Heatwave risk lower")
    except Exception as exc:
        st.error(f"DL prediction failed: {exc}")


def render_duel_section(bundle: ModelBundle, frame: pd.DataFrame) -> None:
    st.divider()
    st.subheader("Feature B: The Duel")
    col_ml, col_dl = st.columns(2)
    live_row = get_prediction_row(frame)

    with col_ml:
        if st.button("Predict with ML"):
            run_ml_prediction(bundle, live_row)

    with col_dl:
        if st.button("Predict with DL"):
            run_dl_prediction(bundle, live_row)


def render_heat_mapping(frame: pd.DataFrame) -> None:
    st.divider()
    st.subheader("Feature C: Heat Mapping")
    latest_year = int(frame["year"].max())
    heatmap_source = frame[frame["year"].isin([2000, latest_year])].copy()
    heatmap_source["month_name"] = heatmap_source["month"].map(MONTH_NAMES)
    heatmap_table = (
        heatmap_source.groupby(["month_name", "year"], as_index=False)["tmax"].mean()
        .pivot(index="month_name", columns="year", values="tmax")
        .reindex([MONTH_NAMES[i] for i in range(1, 13)])
    )

    if latest_year < 2025:
        st.info(f"The provided Karachi data only reaches {latest_year}, so the comparison uses the latest available year instead of 2025.")

    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(heatmap_table, annot=True, fmt=".1f", cmap="YlOrRd", linewidths=0.5, ax=ax)
    ax.set_title(f"Karachi Average Maximum Temperature: 2000 vs {latest_year}")
    ax.set_xlabel("Year")
    ax.set_ylabel("Month")
    st.pyplot(fig, clear_figure=True)


def render_model_comparison_and_preview(bundle: ModelBundle, frame: pd.DataFrame) -> None:
    st.divider()
    left, right = st.columns(2)
    with left:
        st.subheader("Model Comparison")
        comparison_rows = [
            {"Model": RANDOM_FOREST_MODEL_NAME, "Accuracy": bundle.rf_metrics["accuracy"], "ROC AUC": bundle.rf_metrics["auc"]},
        ]
        if bundle.dl_metrics is not None:
            comparison_rows.append(
                {"Model": NEURAL_NETWORK_MODEL_NAME, "Accuracy": bundle.dl_metrics["accuracy"], "ROC AUC": bundle.dl_metrics["auc"]}
            )
        st.dataframe(pd.DataFrame(comparison_rows).round(3), use_container_width=True)

    with right:
        st.subheader("Preview")
        st.dataframe(frame[["date", "tmax", "humidity", "pressure", "is_hot_day"]].tail(12), use_container_width=True)


def main() -> None:
    st.set_page_config(page_title="K-HeatPulse", page_icon="\U0001F525", layout="wide")
    st.title("K-HeatPulse (Karachi Heat Oracle)")
    st.caption("Quarter-century Karachi heat risk dashboard with live Visual Crossing weather data.")

    try:
        bundle = train_models()
    except Exception as exc:
        st.error(f"Training failed: {exc}")
        st.stop()

    render_sidebar(bundle)
    render_live_pulse_section()
    render_duel_section(bundle, bundle.frame)
    render_heat_mapping(bundle.frame)
    render_model_comparison_and_preview(bundle, bundle.frame)
    render_xai_panel(bundle)
    render_whatif_simulator(bundle)
    render_trend_decomposition(bundle.frame)


if __name__ == "__main__":
    main()
