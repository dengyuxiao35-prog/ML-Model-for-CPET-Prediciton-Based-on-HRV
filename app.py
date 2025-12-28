import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt

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
    
    # 性别数值化 (Male=0, Female=1)
    gender_val = 1 if gender_input == "Female" else 0
    
    st.markdown("---")
    st.markdown("**Model Info:**")
    st.caption("• VTs: Random Forest (+Scaler)")
    st.caption("• VO2peak: Linear Regression Formula")

# ==========================================
# 2. 主界面：文件上传
# ==========================================
st.markdown('<div class="main-header">AI-Based CPET Analysis Platform</div>', unsafe_allow_html=True)
st.markdown("""
This tool uses machine learning to detect **Ventilatory Thresholds (VT1/VT2)** and calculate **Peak Oxygen Uptake ($VO_{2peak}$)**.
""")

st.warning("📋 Requirement: Upload 5s-interpolated data. Must contain `Time`, `HR`, `RF`, `RMSSD`, `DFA_alpha1`.")

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
        # 3. 加载模型和标准化器 (已修复 Bug)
        # ==========================================
        @st.cache_resource
        def load_resources():
            try:
                # 加载分类模型
                rf = joblib.load('rf_vts_model.pkl')
                # 加载标准化器
                scaler = joblib.load('scaler.pkl')
                # ✅ 修复点：这里必须返回 3 个值，与 except 保持一致
                return rf, scaler, None 
            except FileNotFoundError as e:
                # 失败时返回 3 个值
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
                    X = df.copy()
                    
                    # (A) 列名强制标准化
                    rename_dict = {
                        'RF': 'RF（呼吸频率）',
                        'rf': 'RF（呼吸频率）',
                        'DFA_alpha1': 'DFAα1',
                        'DFA_Alpha1': 'DFAα1',
                        'dfa_alpha1': 'DFAα1',
                        'MeanRRi': 'MeanRRi', 
                        'LF_power': 'LF power',
                        'HF_power': 'HF power', 
                        'VLF_power': 'VLF power'
                    }
                    for old, new in rename_dict.items():
                        if old in X.columns:
                            X.rename(columns={old: new}, inplace=True)
                    
                    # (B) 注入静态特征
                    X['Age'] = age
                    X['Gender'] = gender_val
                    X['Height'] = height
                    X['Weight'] = weight
                    X['BMI'] = bmi
                    
                    # (C) 生成复杂的动态特征
                    base_signals = ['HR', 'RMSSD', 'DFAα1', 'RF（呼吸频率）', 'MeanRRi', 'SD1', 'SD2', 'VLF power', 'HF power', 'LF power']
                    
                    # 1. 确保数值型
                    for col in base_signals:
                        if col in X.columns:
                            X[col] = pd.to_numeric(X[col], errors='coerce').fillna(method='ffill').fillna(0)
                        else:
                            X[col] = 0
                    
                    # 2. 生成滚动特征
                    rolling_cols = []
                    for col in base_signals:
                        # Window 6
                        mean_6 = f'{col}_mean_6'
                        std_6  = f'{col}_std_6'
                        X[mean_6] = X[col].rolling(window=6, min_periods=1).mean()
                        X[std_6]  = X[col].rolling(window=6, min_periods=1).std().fillna(0)
                        
                        # Window 12
                        mean_12 = f'{col}_mean_12'
                        std_12  = f'{col}_std_12'
                        X[mean_12] = X[col].rolling(window=12, min_periods=1).mean()
                        X[std_12]  = X[col].rolling(window=12, min_periods=1).std().fillna(0)
                        
                        rolling_cols.extend([mean_6, std_6, mean_12, std_12])

                    # 3. 生成相对特征
                    all_cols_to_normalize = base_signals + rolling_cols
                    baseline_df = X.iloc[:12][all_cols_to_normalize].mean()
                    
                    for col in all_cols_to_normalize:
                        rel_col_name = f'{col}_rel_session'
                        base_val = baseline_df[col]
                        if base_val == 0 or pd.isna(base_val): base_val = 1.0
                        X[rel_col_name] = X[col] / base_val

                    # ==========================================
                    # 5. 特征对齐与标准化 (Standardization)
                    # ==========================================
                    final_feature_list = [
                        'HR_mean_6', 'RMSSD_mean_12', 'DFAα1_mean_6_rel_session', 'RF（呼吸频率）', 
                        'HR_std_12', 'RMSSD_std_12', 'DFAα1_std_6_rel_session', 'DFAα1_std_12', 
                        'MeanRRi_std_6', 'SD2_std_12_rel_session', 'RF（呼吸频率）_std_12', 
                        'HR_rel_session', 'SD1_mean_12_rel_session', 'VLF power_mean_12_rel_session', 
                        'RF（呼吸频率）_rel_session', 'HR_std_12_rel_session', 'SD1_std_12_rel_session', 
                        'MeanRRi_std_6_rel_session', 'HF power_std_12_rel_session', 
                        'RF（呼吸频率）_std_6_rel_session', 'RF（呼吸频率）_std_12_rel_session', 
                        'Height', 'Weight', 'Age', 'Gender', 'BMI'
                    ]
                    
                    # 提取数据
                    X_model_input = pd.DataFrame()
                    for feat in final_feature_list:
                        if feat in X.columns:
                            X_model_input[feat] = X[feat]
                        else:
                            X_model_input[feat] = 0
                    X_model_input.fillna(0, inplace=True)
                    
                    # 执行标准化
                    try:
                        X_scaled_array = scaler.transform(X_model_input)
                        X_ready = pd.DataFrame(X_scaled_array, columns=final_feature_list)
                    except Exception as e:
                        st.error(f"Scaler Error: {e}")
                        st.stop()

                    # ==========================================
                    # 6. 执行预测
                    # ==========================================
                    
                    # --- VTs ---
                    pred_stages = rf_model.predict(X_ready)
                    smooth_stages = pd.Series(pred_stages).rolling(window=12, center=True).apply(lambda x: x.mode()[0] if not x.mode().empty else x[0]).fillna(method='bfill').fillna(method='ffill')
                    df['Stage'] = smooth_stages

                    vt1_idx = df[df['Stage'] == 1].index.min()
                    vt2_idx = df[df['Stage'] == 2].index.min()
                    
                    vt1_res = {'Time': df.loc[vt1_idx, 'Time'], 'HR': df.loc[vt1_idx, 'HR']} if pd.notna(vt1_idx) else None
                    vt2_res = {'Time': df.loc[vt2_idx, 'Time'], 'HR': df.loc[vt2_idx, 'HR']} if pd.notna(vt2_idx) else None

                    # --- VO2peak ---
                    peak_df = df.tail(6).mean() 
                    val_RMSSD_peak = peak_df['RMSSD'] if 'RMSSD' in peak_df else 0
                    val_RF_peak    = peak_df['RF（呼吸频率）'] if 'RF（呼吸频率）' in peak_df else 0
                    val_HR_peak    = peak_df['HR'] if 'HR' in peak_df else 0
                    
                    pred_vo2 = (
                        -2.3123 
                        + (0.530595 * gender_val) 
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
                    
                    c1, c2,