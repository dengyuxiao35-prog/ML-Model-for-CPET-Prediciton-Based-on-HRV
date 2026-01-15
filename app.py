import json
import re
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

# ==========================================
# Helpers
# ==========================================
_HAN_RE = re.compile(r"[\u4e00-\u9fff]")


def _contains_han(text: str) -> bool:
    return bool(_HAN_RE.search(str(text)))


def normalize_input_columns(df_in: pd.DataFrame) -> pd.DataFrame:
    df = df_in.copy()
    df.columns = [str(c).strip() for c in df.columns]

    rename: dict[str, str] = {}

    # Time
    for c in df.columns:
        if c in {"Time", "TIME", "time", "Time_s", "time_s"}:
            rename[c] = "Time"
            break

    # RR (respiratory rate)
    if "RR" not in df.columns:
        for c in df.columns:
            if c in {"RR", "rr", "RF", "rf"}:
                rename[c] = "RR"
                break
            if ("RF" in c or "rf" in c) and _contains_han(c):
                rename[c] = "RR"
                break

    # DFA alpha1
    if "DFA_alpha1" not in df.columns:
        for c in df.columns:
            if c in {"DFA_alpha1", "DFA_Alpha1", "dfa_alpha1", "DFAα1"}:
                rename[c] = "DFA_alpha1"
                break

    # Spectral powers
    if "HF_power" not in df.columns:
        for c in df.columns:
            if c in {"HF_power", "HF power"}:
                rename[c] = "HF_power"
                break
    if "VLF_power" not in df.columns:
        for c in df.columns:
            if c in {"VLF_power", "VLF power"}:
                rename[c] = "VLF_power"
                break

    # Optional: ground-truth stage column (renamed for display/validation)
    if "TrueStage" not in df.columns:
        for c in ["TrueStage", "True_Stage", "true_stage", "Stage_true", "stage_true", "Stage", "\u9636\u6bb5"]:
            if c in df.columns:
                rename[c] = "TrueStage"
                break

    df = df.rename(columns=rename)
    return df


def infer_rr_training_token(feature_names: list[str]) -> str | None:
    han_features = [f for f in feature_names if _contains_han(f)]
    if not han_features:
        return None
    base = [f for f in han_features if "_" not in f]
    return min(base or han_features, key=len)


def expected_to_canonical_feature(expected_name: str, rr_token: str | None) -> str:
    name = expected_name
    if rr_token:
        name = name.replace(rr_token, "RR")
    name = name.replace("HF power", "HF_power")
    name = name.replace("VLF power", "VLF_power")
    name = name.replace("DFAα1", "DFA_alpha1")
    return name


@st.cache_resource
def load_resources():
    try:
        base_dir = Path(__file__).resolve().parent
        rf_model = joblib.load(base_dir / "rf_vts_model.pkl")
        scaler = joblib.load(base_dir / "scaler.pkl")
        return rf_model, scaler, None
    except Exception as e:
        return None, None, str(e)


# ==========================================
# Page setup
# ==========================================
st.set_page_config(
    page_title="AI-CPET Assessment System",
    page_icon="🏃‍♂️",
    layout="wide",
)

st.markdown(
    """
<style>
    .main-header {font-size: 2rem; font-weight: bold; color: #0e1117;}
    .result-box {padding: 20px; background-color: #f0f2f6; border-radius: 10px; margin-bottom: 20px;}
</style>
""",
    unsafe_allow_html=True,
)


# ==========================================
# Sidebar: participant info
# ==========================================
with st.sidebar:
    st.image("https://img.icons8.com/color/96/000000/heart-monitor.png", width=60)
    st.header("1. Participant Info")
    st.markdown("Please input subject demographics:")

    gender_input = st.selectbox("Gender", ["Male", "Female"])
    age = st.number_input("Age", 18, 80, 25)
    height = st.number_input("Height (cm)", 140.0, 220.0, 175.0)
    weight = st.number_input("Weight (kg)", 40.0, 150.0, 70.0)
    hr_rest = st.number_input("Resting HR (bpm)", 30, 120, 60)

    bmi = weight / ((height / 100) ** 2)
    st.info(f"📊 Calculated BMI: **{bmi:.1f}** kg/m²")

    # Gender encoding for VT model (Male=0, Female=1)
    gender_val = 1 if gender_input == "Female" else 0
    # Gender encoding for VO2peak regression formula (Male=1, Female=0)
    gender_male = 1 if gender_input == "Male" else 0

    st.markdown("---")
    st.markdown("**Model Info:**")
    st.caption("• VTs: Random Forest (+Hybrid Scaler)")
    st.caption("• VO2peak: Linear Regression Formula")


# ==========================================
# Main: file upload
# ==========================================
st.markdown(
    '<div class="main-header">AI-Based CPET Analysis Platform</div>',
    unsafe_allow_html=True,
)
st.markdown(
    """
This tool uses machine learning to detect **Ventilatory Thresholds (VT1/VT2)** and calculate **Peak Oxygen Uptake ($VO_{2peak}$)**.
"""
)

st.warning(
    "📋 Requirement: Upload **5s-interpolated** data (time in seconds).\n\n"
    "Required columns (supports common aliases):\n"
    "- `Time`\n"
    "- `HR`\n"
    "- `RR` (respiratory rate; `RF` is also accepted)\n"
    "- `RMSSD`\n"
    "- `DFA_alpha1` (or `DFAα1`)\n"
    "- `MeanRRi`\n"
    "- `SD1`, `SD2`\n"
    "- `HF_power`/`HF power`, `VLF_power`/`VLF power`\n"
)

with st.expander("📦 Example Data (Download)", expanded=False):
    base_dir = Path(__file__).resolve().parent
    examples_dir = base_dir / "examples"

    sample_in = examples_dir / "sample_input.csv"
    sample_summary = examples_dir / "sample_output_summary.json"

    if sample_in.exists():
        st.download_button(
            "Download `sample_input.csv`",
            data=sample_in.read_bytes(),
            file_name="sample_input.csv",
            mime="text/csv",
        )

    if sample_summary.exists():
        try:
            summary_obj = json.loads(sample_summary.read_text(encoding="utf-8"))
            demo_settings = summary_obj.get("demo_settings", {})
            sample_output = summary_obj.get("sample_output", {})

            st.caption("Use these settings in the sidebar for the expected results:")
            st.json(demo_settings)
            st.caption("Expected output summary (from the provided model artifacts):")
            st.json(sample_output)
        except Exception:
            st.caption("`sample_output_summary.json` exists but failed to parse.")

uploaded_file = st.file_uploader("📂 Upload Excel or CSV File", type=["xlsx", "xls", "csv"])

if uploaded_file:
    try:
        if uploaded_file.name.lower().endswith(".csv"):
            df_raw = pd.read_csv(uploaded_file)
        else:
            df_raw = pd.read_excel(uploaded_file)

        st.success(f"✅ Data Loaded Successfully: {len(df_raw)} time points.")

        df_norm = normalize_input_columns(df_raw)
        with st.expander("Input Preview (normalized)", expanded=False):
            preview_cols = [
                c
                for c in [
                    "Time",
                    "HR",
                    "RR",
                    "RMSSD",
                    "DFA_alpha1",
                    "MeanRRi",
                    "SD1",
                    "SD2",
                    "HF_power",
                    "VLF_power",
                    "TrueStage",
                ]
                if c in df_norm.columns
            ]
            st.dataframe(df_norm[preview_cols].head())

        rf_model, scaler, error_msg = load_resources()
        if error_msg:
            st.error(f"❌ Load Error: {error_msg}")
            st.warning("Please make sure the repo includes both `rf_vts_model.pkl` and `scaler.pkl`.")
            st.stop()

        if st.button("🚀 Start AI Analysis", type="primary"):
            with st.spinner("Processing features, scaling data & predicting..."):
                SD1_STD_6_REL_SESSION_MEAN = 0.380222622896499
                SD1_STD_6_REL_SESSION_SCALE = 0.5544652392014172

                def build_vt_features(df_in: pd.DataFrame) -> tuple[pd.DataFrame, str]:
                    X1 = df_in.copy()

                    if "Time" not in X1.columns:
                        raise ValueError("Missing time column: expected `Time`.")

                    required_signals = [
                        "HR",
                        "RMSSD",
                        "DFA_alpha1",
                        "RR",
                        "MeanRRi",
                        "SD1",
                        "SD2",
                        "HF_power",
                        "VLF_power",
                    ]

                    missing = [c for c in required_signals if c not in X1.columns]
                    if missing:
                        raise ValueError(f"Missing required columns: {missing}")

                    keep_cols = ["Time", *required_signals]
                    if "TrueStage" in X1.columns:
                        keep_cols.append("TrueStage")
                    X1 = X1.loc[:, keep_cols].copy()

                    X1["Time"] = pd.to_numeric(X1["Time"], errors="coerce")
                    X1 = X1.dropna(subset=["Time"]).sort_values("Time").reset_index(drop=True)

                    for col in required_signals:
                        X1[col] = pd.to_numeric(X1[col], errors="coerce")

                    # Inject static features (from sidebar)
                    X1["Age"] = age
                    X1["Gender"] = gender_val
                    X1["Height"] = height
                    X1["Weight"] = weight
                    X1["BMI"] = bmi

                    # Rolling features (window=6,12; min_periods=1; std uses ddof=1 -> NaN at first row)
                    window_sizes = [6, 12]
                    rolling_features: list[str] = []
                    for f in required_signals:
                        for w in window_sizes:
                            col_mean = f"{f}_mean_{w}"
                            col_std = f"{f}_std_{w}"
                            X1[col_mean] = X1[f].rolling(window=w, min_periods=1).mean()
                            X1[col_std] = X1[f].rolling(window=w, min_periods=1).std()
                            rolling_features.extend([col_mean, col_std])

                    # Relative features baseline: Time <= 120 seconds (match training)
                    all_cols_to_normalize = required_signals + rolling_features
                    baseline_mask = X1["Time"] <= 120
                    if baseline_mask.any():
                        baseline = X1.loc[baseline_mask, all_cols_to_normalize].mean()
                    else:
                        baseline = X1.loc[:0, all_cols_to_normalize].mean()

                    for f in all_cols_to_normalize:
                        X1[f"{f}_rel_session"] = X1[f] / (baseline[f] + 1e-6)

                    return X1, "Time"

                X, time_col = build_vt_features(df_norm)
                df_proc = X

                # ==========================================
                # Feature alignment + hybrid scaling
                # ==========================================
                scaler_features = list(scaler.feature_names_in_)
                model_features = list(rf_model.feature_names_in_)
                rr_token = infer_rr_training_token(scaler_features)

                scaler_features_canon = [expected_to_canonical_feature(f, rr_token) for f in scaler_features]

                missing_feats = [c for c in scaler_features_canon if c not in X.columns]
                if missing_feats:
                    st.error(f"❌ Missing engineered features required by the model: {missing_feats}")
                    st.stop()

                X_for_scaler = X.loc[:, scaler_features_canon]
                valid_mask = ~X_for_scaler.isna().any(axis=1) & ~X["SD1_std_6_rel_session"].isna()
                valid_idx = X_for_scaler.index[valid_mask]
                if len(valid_idx) == 0:
                    st.error("❌ Not enough valid rows to run the model (check missing/NaN values).")
                    st.stop()

                try:
                    X_scaled_np = scaler.transform(X_for_scaler.loc[valid_idx].to_numpy(dtype=float))
                except Exception as e:
                    st.error(f"Scaler Error: {e}")
                    st.stop()

                scaler_idx = {f: i for i, f in enumerate(scaler_features)}
                cols = []
                for feat in model_features:
                    if feat in scaler_idx:
                        cols.append(X_scaled_np[:, scaler_idx[feat]])
                    elif feat == "SD1_std_6_rel_session":
                        raw = X.loc[valid_idx, "SD1_std_6_rel_session"].to_numpy(dtype=float)
                        cols.append(
                            (raw - SD1_STD_6_REL_SESSION_MEAN) / (SD1_STD_6_REL_SESSION_SCALE + 1e-12)
                        )
                    else:
                        st.error(f"Model feature missing: {feat}")
                        st.stop()

                X_final_model_np = np.column_stack(cols)

                # ==========================================
                # Predict stages
                # ==========================================
                pred_stages_valid = rf_model.predict(X_final_model_np)

                # Expand back to full length and fill missing edges (e.g., first-row std NaNs)
                pred_full = pd.Series(index=X.index, dtype=float)
                pred_full.loc[valid_idx] = pred_stages_valid.astype(float)
                pred_full = pred_full.bfill().ffill().astype(int)

                # Smooth (mode filter) to reduce isolated jitter
                def mode_1d(a: pd.Series) -> int:
                    m = a.mode()
                    return int(m.iloc[0]) if len(m) else int(a.iloc[0])

                smooth_stages = (
                    pred_full.rolling(window=12, center=True, min_periods=1)
                    .apply(mode_1d)
                    .bfill()
                    .ffill()
                    .astype(int)
                )
                df_proc["PredStage"] = smooth_stages

                # Optional: validate against ground-truth stage column if user uploads labeled data
                if "TrueStage" in df_proc.columns:
                    with st.expander("🧪 Validation (ground truth detected)", expanded=False):
                        st.caption(
                            "Detected `TrueStage` in the uploaded file. This column is NOT used for prediction; "
                            "it is only used here to compare with the model output."
                        )

                        y_true = pd.to_numeric(df_proc["TrueStage"], errors="coerce")
                        eval_mask = ~y_true.isna()
                        if eval_mask.sum() == 0:
                            st.warning("Ground truth column contains no valid numeric labels; skip validation.")
                        else:
                            y_true_i = y_true[eval_mask].astype(int)
                            y_pred_raw = pred_full[eval_mask].astype(int)
                            y_pred_smooth = df_proc.loc[eval_mask, "PredStage"].astype(int)

                            acc_raw = float((y_true_i == y_pred_raw).mean())
                            acc_smooth = float((y_true_i == y_pred_smooth).mean())

                            c1v, c2v = st.columns(2)
                            c1v.metric("Stage Accuracy (Raw)", f"{acc_raw:.4f}")
                            c2v.metric("Stage Accuracy (Smoothed)", f"{acc_smooth:.4f}")

                            cm = pd.crosstab(
                                y_true_i,
                                y_pred_smooth,
                                rownames=["True"],
                                colnames=["Pred"],
                                dropna=False,
                            )
                            st.dataframe(cm)

                            def extract_vt_points(stage_series: pd.Series):
                                vt1 = None
                                vt2 = None

                                idx_1 = (
                                    stage_series.index[stage_series == 1].min()
                                    if (stage_series == 1).any()
                                    else None
                                )
                                if idx_1 is not None and idx_1 > 0:
                                    before = stage_series.loc[: idx_1 - 1]
                                    idx_vt1 = (
                                        before.index[before == 0].max() if (before == 0).any() else None
                                    )
                                    if idx_vt1 is not None:
                                        vt1 = (
                                            float(df_proc.loc[idx_vt1, time_col]),
                                            float(df_proc.loc[idx_vt1, "HR"]),
                                        )

                                idx_2 = (
                                    stage_series.index[stage_series == 2].min()
                                    if (stage_series == 2).any()
                                    else None
                                )
                                if idx_2 is not None and idx_2 > 0:
                                    before = stage_series.loc[: idx_2 - 1]
                                    idx_vt2 = (
                                        before.index[before == 1].max() if (before == 1).any() else None
                                    )
                                    if idx_vt2 is not None:
                                        vt2 = (
                                            float(df_proc.loc[idx_vt2, time_col]),
                                            float(df_proc.loc[idx_vt2, "HR"]),
                                        )
                                return vt1, vt2

                            if y_true.isna().any():
                                st.info("Ground truth stage has NaNs; skip VT1/VT2 comparison.")
                            else:
                                vt1_true, vt2_true = extract_vt_points(y_true.astype(int))
                                vt1_pred, vt2_pred = extract_vt_points(df_proc["PredStage"].astype(int))

                                rows = [
                                    {
                                        "VT": "VT1",
                                        "True_Time(s)": None if not vt1_true else vt1_true[0],
                                        "Pred_Time(s)": None if not vt1_pred else vt1_pred[0],
                                        "True_HR(bpm)": None if not vt1_true else vt1_true[1],
                                        "Pred_HR(bpm)": None if not vt1_pred else vt1_pred[1],
                                    },
                                    {
                                        "VT": "VT2",
                                        "True_Time(s)": None if not vt2_true else vt2_true[0],
                                        "Pred_Time(s)": None if not vt2_pred else vt2_pred[0],
                                        "True_HR(bpm)": None if not vt2_true else vt2_true[1],
                                        "Pred_HR(bpm)": None if not vt2_pred else vt2_pred[1],
                                    },
                                ]
                                st.dataframe(pd.DataFrame(rows))

                # VT extraction: VT1 = last Stage 0 before Stage 1; VT2 = last Stage 1 before Stage 2
                vt1_res = None
                vt2_res = None

                idx_stage1 = (
                    df_proc.index[df_proc["PredStage"] == 1].min()
                    if (df_proc["PredStage"] == 1).any()
                    else None
                )
                if idx_stage1 is not None and idx_stage1 > 0:
                    before_1 = df_proc.loc[: idx_stage1 - 1]
                    idx_vt1 = (
                        before_1.index[before_1["PredStage"] == 0].max()
                        if (before_1["PredStage"] == 0).any()
                        else None
                    )
                    if idx_vt1 is not None:
                        vt1_res = {
                            "Time": float(df_proc.loc[idx_vt1, time_col]),
                            "HR": float(df_proc.loc[idx_vt1, "HR"]),
                        }

                idx_stage2 = (
                    df_proc.index[df_proc["PredStage"] == 2].min()
                    if (df_proc["PredStage"] == 2).any()
                    else None
                )
                if idx_stage2 is not None and idx_stage2 > 0:
                    before_2 = df_proc.loc[: idx_stage2 - 1]
                    idx_vt2 = (
                        before_2.index[before_2["PredStage"] == 1].max()
                        if (before_2["PredStage"] == 1).any()
                        else None
                    )
                    if idx_vt2 is not None:
                        vt2_res = {
                            "Time": float(df_proc.loc[idx_vt2, time_col]),
                            "HR": float(df_proc.loc[idx_vt2, "HR"]),
                        }

                # VO2peak
                peak_df = df_proc.tail(6).mean(numeric_only=True)
                val_rmssd_peak = float(peak_df["RMSSD"]) if "RMSSD" in peak_df else 0.0
                val_rr_peak = float(peak_df["RR"]) if "RR" in peak_df else 0.0
                val_hr_peak = float(peak_df["HR"]) if "HR" in peak_df else 0.0

                pred_vo2 = (
                    -2.3123
                    + (0.530595 * gender_male)
                    + (0.039042 * val_rmssd_peak)
                    + (0.028138 * age)
                    + (0.025320 * weight)
                    + (0.013507 * val_rr_peak)
                    - (0.010645 * hr_rest)
                    + (0.010629 * height)
                    + (0.003778 * val_hr_peak)
                )
                if pred_vo2 < 0:
                    pred_vo2 = 0.5

                # ==========================================
                # Report
                # ==========================================
                st.divider()
                st.subheader("📊 Analysis Report")

                c1, c2, c3 = st.columns(3)
                c1.metric("Predicted VO2peak", f"{pred_vo2:.2f} L/min")

                if vt1_res:
                    c2.metric("VT1", f"{vt1_res['HR']:.0f} bpm", f"Time: {vt1_res['Time']} s")
                else:
                    c2.metric("VT1", "Not Detected")

                if vt2_res:
                    c3.metric("VT2", f"{vt2_res['HR']:.0f} bpm", f"Time: {vt2_res['Time']} s")
                else:
                    c3.metric("VT2", "Not Detected")

                st.markdown("### Physiological Response")
                fig, ax = plt.subplots(figsize=(12, 5))
                ax.plot(df_proc[time_col], df_proc["HR"], "k-", label="Heart Rate", linewidth=2)

                ax.fill_between(
                    df_proc[time_col],
                    0,
                    220,
                    where=df_proc["PredStage"] == 0,
                    color="#eaffea",
                    alpha=0.6,
                    label="Zone 1",
                )
                ax.fill_between(
                    df_proc[time_col],
                    0,
                    220,
                    where=df_proc["PredStage"] == 1,
                    color="#fff9c4",
                    alpha=0.6,
                    label="Zone 2",
                )
                ax.fill_between(
                    df_proc[time_col],
                    0,
                    220,
                    where=df_proc["PredStage"] == 2,
                    color="#ffebee",
                    alpha=0.6,
                    label="Zone 3",
                )

                if vt1_res:
                    ax.axvline(vt1_res["Time"], color="blue", linestyle="--", label="VT1")
                if vt2_res:
                    ax.axvline(vt2_res["Time"], color="red", linestyle="--", label="VT2")

                ax.set_ylim(bottom=float(df_proc["HR"].min()) * 0.9, top=float(df_proc["HR"].max()) * 1.1)
                ax.legend(loc="upper left")
                st.pyplot(fig)

                export_cols = ["Time", "HR", "PredStage"]
                if "TrueStage" in df_proc.columns:
                    export_cols = ["Time", "HR", "TrueStage", "PredStage"]
                res_csv = df_proc.loc[:, export_cols].to_csv(index=False).encode("utf-8")
                st.download_button(
                    "📥 Download Result CSV",
                    data=res_csv,
                    file_name="cpet_results.csv",
                    mime="text/csv",
                )

    except Exception as e:
        st.error(f"⚠️ Program Error: {e}")
