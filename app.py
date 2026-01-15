import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
from pathlib import Path

# ==========================================
# 0. 页面配置
# ==========================================
st.set_page_config(
    page_title="AI-CPET Assessment System",
    page_icon="🏃‍♂️",
    layout="wide"
)

# CSS 美化
st.markdown("""
<style>
    .main-header {font-size: 2rem; font-weight: bold; color: #0e1117;}
    .result-box {padding: 20px; background-color: #f0f2f6; border-radius: 10px; margin-bottom: 20px;}
</style>
""", unsafe_allow_html=True)

# ==========================================
# 1. 侧边栏：受试者信息
# ==========================================
with st.sidebar:
    st.image("https://img.icons8.com/color/96/000000/heart-monitor.png", width=60)
    st.header("1. Participant Info")
    st.markdown("Please input subject demographics:")
    
    # 输入控件
    gender_input = st.selectbox("Gender (性别)", ["Male", "Female"])
    age = st.number_input("Age (年龄)", 18, 80, 25)
    height = st.number_input("Height (身高 cm)", 140.0, 220.0, 175.0)
    weight = st.number_input("Weight (体重 kg)", 40.0, 150.0, 70.0)
    hr_rest = st.number_input("Resting HR (静息心率 bpm)", 30, 120, 60)
    
    # 自动计算 BMI
    bmi = weight / ((height/100)**2)
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
# 2. 主界面：文件上传
# ==========================================
st.markdown('<div class="main-header">AI-Based CPET Analysis Platform</div>', unsafe_allow_html=True)
st.markdown("""
This tool uses machine learning to detect **Ventilatory Thresholds (VT1/VT2)** and calculate **Peak Oxygen Uptake ($VO_{2peak}$)**.
""")

st.warning(
    "📋 Requirement: Upload **5s-interpolated** data (time in seconds).\n\n"
    "Required columns (supports common aliases):\n"
    "- `Time`/`TIME`\n"
    "- `HR`\n"
    "- `RF` (or `RF（呼吸频率）`)\n"
    "- `RMSSD`\n"
    "- `DFA_alpha1` (or `DFAα1`)\n"
    "- `MeanRRi`\n"
    "- `SD1`, `SD2`\n"
    "- `HF_power`/`HF power`, `VLF_power`/`VLF power`\n"
)

uploaded_file = st.file_uploader("📂 Upload Excel or CSV File", type=["xlsx", "xls", "csv"])

if uploaded_file:
    try:
        # 读取数据
        if uploaded_file.name.endswith('csv'):
            df = pd.read_csv(uploaded_file)
        else:
            df = pd.read_excel(uploaded_file)
            
        st.success(f"✅ Data Loaded Successfully: {len(df)} time points.")
        
        with st.expander("查看原始数据 (Raw Data Preview)"):
            st.dataframe(df.head())

        # ==========================================
        # 3. 加载模型和标准化器
        # ==========================================
        @st.cache_resource
        def load_resources():
            try:
                base_dir = Path(__file__).resolve().parent
                # 加载分类模型
                rf = joblib.load(base_dir / 'rf_vts_model.pkl')
                # 加载标准化器
                scaler = joblib.load(base_dir / 'scaler.pkl')
                return rf, scaler, None 
            except FileNotFoundError as e:
                return None, None, str(e)
            except Exception as e:
                return None, None, str(e)
            
        rf_model, scaler, error_msg = load_resources()

        if error_msg:
            st.error(f"❌ Load Error: {error_msg}")
            st.warning("Please make sure you uploaded BOTH 'rf_vts_model.pkl' AND 'scaler.pkl' to GitHub.")
        
        elif rf_model and scaler:
            if st.button("🚀 Start AI Analysis", type="primary"):
                with st.spinner("Processing features, Scaling data & Predicting..."):
                    
                    # ==========================================
                    # 4. 特征工程 (Feature Engineering)
                    # ==========================================
                    SD1_STD_6_REL_SESSION_MEAN = 0.380222622896499
                    SD1_STD_6_REL_SESSION_SCALE = 0.5544652392014172

                    def normalize_columns(df_in: pd.DataFrame) -> pd.DataFrame:
                        X0 = df_in.copy()
                        rename_dict = {
                            # time
                            'Time': 'TIME',
                            'time': 'TIME',
                            # respiration frequency
                            'RF': 'RF（呼吸频率）',
                            'rf': 'RF（呼吸频率）',
                            # DFA a1
                            'DFA_alpha1': 'DFAα1',
                            'DFA_Alpha1': 'DFAα1',
                            'dfa_alpha1': 'DFAα1',
                            # spectral powers
                            'LF_power': 'LF power',
                            'HF_power': 'HF power',
                            'VLF_power': 'VLF power',
                        }
                        for old, new in rename_dict.items():
                            if old in X0.columns and new not in X0.columns:
                                X0.rename(columns={old: new}, inplace=True)
                        return X0

                    def build_vt_features(df_in: pd.DataFrame) -> tuple[pd.DataFrame, str]:
                        X1 = normalize_columns(df_in)

                        time_col = 'TIME' if 'TIME' in X1.columns else None
                        if not time_col:
                            raise ValueError("Missing time column: expected `Time` or `TIME`.")

                        X1[time_col] = pd.to_numeric(X1[time_col], errors='coerce')
                        X1 = X1.dropna(subset=[time_col]).sort_values(time_col).reset_index(drop=True)

                        # Inject static features (from sidebar)
                        X1['Age'] = age
                        X1['Gender'] = gender_val
                        X1['Height'] = height
                        X1['Weight'] = weight
                        X1['BMI'] = bmi

                        # Minimal required dynamic signals for this VT model
                        required_signals = [
                            'HR',
                            'RMSSD',
                            'DFAα1',
                            'RF（呼吸频率）',
                            'MeanRRi',
                            'SD1',
                            'SD2',
                            'HF power',
                            'VLF power',
                        ]

                        missing = [c for c in required_signals if c not in X1.columns]
                        if missing:
                            raise ValueError(f"Missing required columns: {missing}")

                        # Ensure numeric (DO NOT fill std NaNs with 0; match training preprocessing)
                        for col in required_signals:
                            X1[col] = pd.to_numeric(X1[col], errors='coerce')

                        # Rolling features (window=6,12; min_periods=1; std uses pandas default ddof=1 -> NaN at first row)
                        window_sizes = [6, 12]
                        base_signals = required_signals  # only compute what's needed

                        rolling_features: list[str] = []
                        for f in base_signals:
                            for w in window_sizes:
                                col_mean = f'{f}_mean_{w}'
                                col_std = f'{f}_std_{w}'
                                X1[col_mean] = X1[f].rolling(window=w, min_periods=1).mean()
                                X1[col_std] = X1[f].rolling(window=w, min_periods=1).std()
                                rolling_features.extend([col_mean, col_std])

                        # Relative features baseline: TIME <= 120 seconds (match training)
                        all_cols_to_normalize = base_signals + rolling_features
                        baseline_df = X1[X1[time_col] <= 120][all_cols_to_normalize].mean()
                        for f in all_cols_to_normalize:
                            base_val = baseline_df[f]
                            X1[f'{f}_rel_session'] = X1[f] / (base_val + 1e-6)

                        return X1, time_col

                    X, time_col = build_vt_features(df)
                    df_proc = X  # use the same, time-sorted frame for plotting/export

                    # ==========================================
                    # 5. 特征对齐与“混合”标准化 (Hybrid Scaling)
                    # ==========================================
                    
                    scaler_features = list(scaler.feature_names_in_)
                    model_features = list(rf_model.feature_names_in_)

                    # 3. 准备喂给 Scaler 的数据
                    X_for_scaler = X.reindex(columns=scaler_features).copy()

                    # 4. 执行标准化
                    try:
                        # Drop rows with NaNs (match training: scaler/model never saw NaN rows)
                        valid_mask = ~X_for_scaler.isna().any(axis=1) & ~X['SD1_std_6_rel_session'].isna()
                        valid_idx = X_for_scaler.index[valid_mask]
                        if len(valid_idx) == 0:
                            st.error("❌ Not enough valid rows to run the model (check missing/NaN values in required columns).")
                            st.stop()

                        X_for_scaler_clean = X_for_scaler.loc[valid_idx]
                        X_scaled_np = scaler.transform(X_for_scaler_clean)
                        X_scaled_df = pd.DataFrame(X_scaled_np, columns=scaler_features, index=valid_idx)
                    except Exception as e:
                        st.error(f"Scaler Error: {e}")
                        st.stop()
                        
                    # 5. 拼装给 Model 的最终数据
                    # Note: scaler/model feature lists differ by 1 feature:
                    # - scaler has `RMSSD_std_12`
                    # - model has `SD1_std_6_rel_session`
                    # The RF model was trained on z-scored features, so we manually z-score SD1_std_6_rel_session
                    # using training-set statistics.
                    X_final_model = pd.DataFrame(index=valid_idx)
                    for feat in model_features:
                        if feat in X_scaled_df.columns:
                            X_final_model[feat] = X_scaled_df[feat]
                        elif feat == 'SD1_std_6_rel_session':
                            raw = X.loc[valid_idx, feat]
                            X_final_model[feat] = (raw - SD1_STD_6_REL_SESSION_MEAN) / (SD1_STD_6_REL_SESSION_SCALE + 1e-12)
                        else:
                            st.error(f"Model feature missing: {feat}")
                            st.stop()

                    # ==========================================
                    # 6. 执行预测
                    # ==========================================
                    
                    # --- VTs ---
                    pred_stages_valid = rf_model.predict(X_final_model)

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
                    df_proc['Stage'] = smooth_stages

                    # VT extraction (match notebook logic): VT1 = last point in Stage 0 before Stage 1; VT2 = last point in Stage 1 before Stage 2
                    vt1_res = None
                    vt2_res = None

                    idx_stage1 = df_proc.index[df_proc['Stage'] == 1].min() if (df_proc['Stage'] == 1).any() else None
                    if idx_stage1 is not None and idx_stage1 > 0:
                        before_1 = df_proc.loc[:idx_stage1 - 1]
                        idx_vt1 = before_1.index[before_1['Stage'] == 0].max() if (before_1['Stage'] == 0).any() else None
                        if idx_vt1 is not None:
                            vt1_res = {'Time': float(df_proc.loc[idx_vt1, time_col]), 'HR': float(df_proc.loc[idx_vt1, 'HR'])}

                    idx_stage2 = df_proc.index[df_proc['Stage'] == 2].min() if (df_proc['Stage'] == 2).any() else None
                    if idx_stage2 is not None and idx_stage2 > 0:
                        before_2 = df_proc.loc[:idx_stage2 - 1]
                        idx_vt2 = before_2.index[before_2['Stage'] == 1].max() if (before_2['Stage'] == 1).any() else None
                        if idx_vt2 is not None:
                            vt2_res = {'Time': float(df_proc.loc[idx_vt2, time_col]), 'HR': float(df_proc.loc[idx_vt2, 'HR'])}

                    # --- VO2peak ---
                    peak_df = df_proc.tail(6).mean() 
                    val_RMSSD_peak = peak_df['RMSSD'] if 'RMSSD' in peak_df else 0
                    val_RF_peak    = peak_df['RF（呼吸频率）'] if 'RF（呼吸频率）' in peak_df else 0
                    val_HR_peak    = peak_df['HR'] if 'HR' in peak_df else 0
                    
                    pred_vo2 = (
                        -2.3123 
                        + (0.530595 * gender_male) 
                        + (0.039042 * val_RMSSD_peak) 
                        + (0.028138 * age) 
                        + (0.025320 * weight) 
                        + (0.013507 * val_RF_peak) 
                        - (0.010645 * hr_rest) 
                        + (0.010629 * height) 
                        + (0.003778 * val_HR_peak)
                    )
                    if pred_vo2 < 0: pred_vo2 = 0.5 

                    # ==========================================
                    # 7. 结果展示
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
                    plot_time_col = 'Time' if 'Time' in df_proc.columns else ('TIME' if 'TIME' in df_proc.columns else time_col)
                    ax.plot(df_proc[plot_time_col], df_proc['HR'], 'k-', label='Heart Rate', linewidth=2)
                    
                    ax.fill_between(df_proc[plot_time_col], 0, 220, where=df_proc['Stage']==0, color='#eaffea', alpha=0.6, label='Zone 1')
                    ax.fill_between(df_proc[plot_time_col], 0, 220, where=df_proc['Stage']==1, color='#fff9c4', alpha=0.6, label='Zone 2')
                    ax.fill_between(df_proc[plot_time_col], 0, 220, where=df_proc['Stage']==2, color='#ffebee', alpha=0.6, label='Zone 3')
                    
                    if vt1_res: ax.axvline(vt1_res['Time'], color='blue', linestyle='--', label='VT1')
                    if vt2_res: ax.axvline(vt2_res['Time'], color='red', linestyle='--', label='VT2')
                    
                    ax.set_ylim(bottom=min(df_proc['HR'])*0.9, top=max(df_proc['HR'])*1.1)
                    ax.legend(loc='upper left')
                    st.pyplot(fig)
                    
                    export_time_col = plot_time_col
                    res_csv = df_proc[[export_time_col, 'HR', 'Stage']].to_csv(index=False).encode('utf-8')
                    st.download_button("📥 Download Result CSV", data=res_csv, file_name="cpet_results.csv", mime="text/csv")

    except Exception as e:
        st.error(f"⚠️ Program Error: {e}")
