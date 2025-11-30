# streamlit_app.py
import math
from pathlib import Path

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st
from sklearn.neighbors import LocalOutlierFactor

try:
    import stumpy
except ImportError:  # pylint: disable=ungrouped-imports
    st.warning("Install stumpy for Matrix Profile support: pip install stumpy")
    stumpy = None

DATA_PATH = Path("Khondaker data request.xlsm")
DATASETS = {
    "CHW hourly": {"sheet": "CHW hourly", "drop": ["Date"], "datetime_index": True},
    "CHW totals": {"sheet": "CHW totals", "drop": [], "datetime_index": False},
    "MTHW hourly": {"sheet": "MTHW hourly", "drop": ["Date"], "datetime_index": True},
    "MTHW totals": {"sheet": "MTHW totals", "drop": [], "datetime_index": False},
    "ELEC hourly": {"sheet": "ELEC hourly", "drop": ["Date"], "datetime_index": True},
    "ELEC totals": {"sheet": "ELEC totals", "drop": [], "datetime_index": False},
}

st.set_page_config(page_title="Energy Anomaly Explorer", layout="wide")

@st.cache_data
def load_dataset(name: str) -> pd.DataFrame:
    """Load the requested dataset and ensure timestamps are parsed."""
    spec = DATASETS[name]
    df = pd.read_excel(
        DATA_PATH,
        sheet_name=spec["sheet"],
        index_col="Timestamp" if spec["datetime_index"] else None,
    )
    for col in spec["drop"]:
        if col in df.columns:
            df = df.drop(columns=col)
    if "Timestamp" in df.columns:
        df["Timestamp"] = pd.to_datetime(df["Timestamp"], errors="coerce")
        df = df.set_index("Timestamp")
    if isinstance(df.index, pd.DatetimeIndex):
        df = df.sort_index()
    return df


def detect_lof(series: pd.Series, contamination: float, n_neighbors: int) -> pd.Index:
    """Local Outlier Factor anomaly detection."""
    nn = min(n_neighbors, max(2, len(series) - 1))
    lof = LocalOutlierFactor(n_neighbors=nn, contamination=contamination)
    labels = lof.fit_predict(series.to_numpy().reshape(-1, 1))
    return series.index[labels == -1]


def detect_matrix_profile(series: pd.Series, window: int, top_k: int) -> pd.Index:
    """Matrix Profile (discords)."""
    if stumpy is None:
        st.error("Matrix Profile requires stumpy (pip install stumpy).")
        return pd.Index([])
    window = max(4, min(window, len(series) // 2))
    mp = stumpy.stump(series.to_numpy(dtype="float64"), m=window)
    discord_idx = mp[:, 0].argsort()[::-1][:top_k]
    # Starts correspond to subsequence start positions; convert to timestamps.
    return series.index[discord_idx].sort_values()


def plot_series(series: pd.Series, anomalies: pd.Index) -> plt.Figure:
    """Scatter plot with anomalies highlighted."""
    fig, ax = plt.subplots(figsize=(12, 4))
    ax.scatter(series.index, series.values, s=12, label="Data", color="tab:blue")
    if len(anomalies):
        ax.scatter(
            anomalies,
            series.loc[anomalies],
            s=48,
            color="tab:red",
            label="Anomaly",
            zorder=3,
        )
    ax.set_title(series.name or "Selected series")
    ax.set_xlabel("Timestamp" if isinstance(series.index, pd.DatetimeIndex) else "Index")
    ax.set_ylabel("Value")
    if isinstance(series.index, pd.DatetimeIndex):
        locator = mdates.AutoDateLocator(minticks=8, maxticks=60)
        ax.xaxis.set_major_locator(locator)
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m-%d\n%H:%M"))
        ax.tick_params(axis="x", rotation=45)
    ax.legend()
    fig.tight_layout()
    return fig


st.title("Energy Anomaly Explorer")

with st.sidebar:
    st.header("Configuration")
    dataset_name = st.selectbox("Dataset", list(DATASETS))
    df = load_dataset(dataset_name)
    numeric_cols = df.select_dtypes(include="number").columns
    if len(numeric_cols) == 0:
        st.error("Selected dataset has no numeric columns.")
        st.stop()
    column = st.selectbox("Column", numeric_cols)
    resample = st.selectbox(
        "Resample (optional)",
        ["None", "15min", "H", "D"],
        help="Applies only if the index is datetime.",
    )
    method = st.radio(
        "Anomaly detector",
        ("None", "Local Outlier Factor", "Matrix Profile"),
    )
    sample_pct = st.slider(
        "Sample size (%)",
        min_value=5,
        max_value=100,
        value=100,
        step=5,
        help="Trim data from the end to speed up experiments.",
    )

series = df[column]
if resample != "None" and isinstance(series.index, pd.DatetimeIndex):
    series = series.resample(resample).mean()
series = series.dropna()
sample_n = max(1, math.ceil(len(series) * sample_pct / 100))
series = series.iloc[:sample_n]

# ... existing imports ...
from datetime import datetime

# after series selection
series = df[column]
if not isinstance(series.index, pd.DatetimeIndex):
    st.warning("This column has no datetime index; date/time filtering is disabled.")
else:
    min_time = series.index.min()
    max_time = series.index.max()
    start = st.sidebar.date_input("Start date", min_time.date(), min_value=min_time.date(), max_value=max_time.date())
    end = st.sidebar.date_input("End date", max_time.date(), min_value=min_time.date(), max_value=max_time.date())
    start_time = st.sidebar.time_input("Start time", min_time.time())
    end_time = st.sidebar.time_input("End time", max_time.time())
    start_dt = datetime.combine(start, start_time)
    end_dt = datetime.combine(end, end_time)
    if start_dt >= end_dt:
        st.error("Start must be earlier than end.")
    else:
        series = series.loc[(series.index >= start_dt) & (series.index <= end_dt)]

anomalies = pd.Index([])
if method == "Local Outlier Factor" and len(series) > 5:
    contamination = st.sidebar.slider("LOF contamination", 0.01, 0.20, 0.05, 0.01)
    neighbors = st.sidebar.slider("LOF neighbors", 10, 400, 100, 10)
    anomalies = detect_lof(series, contamination, neighbors)
elif method == "Matrix Profile" and len(series) > 10:
    window = st.sidebar.slider(
        "Window size", 10, max(20, len(series) // 2), min(100, len(series) // 3), 5
    )
    top_k = st.sidebar.slider(
        "Top discord count", 1, min(20, len(series) // 2), 5, 1
    )
    anomalies = detect_matrix_profile(series, window, top_k)

st.subheader(f"{dataset_name} · {column}")
st.pyplot(plot_series(series, anomalies))

if len(anomalies):
    display_df = (
        series.loc[anomalies]
        .sort_index()
        .to_frame(name=column)
        .reset_index()
    )
    name_col = "Timestamp" if isinstance(series.index, pd.DatetimeIndex) else "Index"
    display_df.rename(columns={display_df.columns[0]: name_col}, inplace=True)

    st.subheader("Detected anomalies")
    st.dataframe(display_df, use_container_width=True)
else:
    st.caption("No anomalies detected with current parameters.")