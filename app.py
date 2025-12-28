import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

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
    
    # 性别数值化 (根据您提供的模型训练逻辑: 男=0, 女=1)
    # 如果您的公式里 Gender=1 代表男性，请这里改成: 1 if gender_input == "Male" else 0
    gender_val = 1 if gender_input == "Female" else 0
    
    st.markdown("---")
    st.markdown("**Model Info:**")
    st.caption("• VTs: Random Forest Classifier")
    st.caption("• VO2peak: Linear Regression Formula")

# ==========================================
# 2. 主界面：文件上传
# ==========================================
st.markdown('<div class="main-header">AI-Based CPET Analysis Platform</div>', unsafe_allow_html=True)
st.markdown("""
This tool uses machine learning to detect **Ventilatory Thresholds (VT1/VT2)** and calculate **Peak Oxygen Uptake ($VO_{2peak}$)**.
""")

st.warning("📋 Requirement: Upload 5s-interpolated data. Must contain columns like `Time`, `HR`, `RF` (or `RF（呼吸频率）`), `RMSSD`.")

uploaded_file = st.file_uploader("📂 Upload Excel or CSV File", type=["xlsx", "xls", "csv"])

if uploaded_file:
    try:
        # 读取数据
        if uploaded_file.name.endswith('csv'):
            df = pd.read_csv(uploaded_file)
        else:
            df = pd.read_excel(uploaded_file)
            
        st.success(f"✅ Data Loaded Successfully: {len(df)} time points.")
        
        # 简单预览
        with st.expander("查看原始数据 (Raw Data Preview)"):
            st.dataframe(df.head())

        # ==========================================
        # 3. 加载模型 (只加载 VT 分类模型)
        # ==========================================
        @st.cache_resource
        def load_rf_model():
            try:
                # 只需要加载这个文件了，vo2_regression_model.pkl 不需要了
                rf = joblib.load('rf_vts_model.pkl')
                return rf
            except FileNotFoundError:
                st.error("❌ 'rf_vts_model.pkl' not found! Please upload it to your GitHub/Folder.")
                return None
            
        rf_model = load_rf_model()

        if rf_model:
            if st.button("🚀 Start AI Analysis", type="primary"):
                with st.spinner("Processing signals & Calculating VO2max..."):
                    
                    # ==========================================
                    # 4. 特征工程 (Feature Engineering)
                    # ==========================================
                    X = df.copy()
                    
                    # (A) 列名标准化 (兼容中文列名)
                    # 这样无论上传的是 RF 还是 RF（呼吸频率），后面都叫 RF
                    col_mapping = {
                        'RF（呼吸频率）': 'RF', 
                        'DFA_alpha1': 'DFAα1', 
                        'LF_power': 'LF power', 
                        'HF_power': 'HF power', 
                        'VLF_power': 'VLF power'
                    }
                    X.rename(columns=col_mapping, inplace=True)
                    
                    # (B) 注入静态特征 (用于 RF 分类模型)
                    X['Age'] = age
                    X['Gender'] = gender_val
                    X['Height'] = height
                    X['Weight'] = weight
                    X['BMI'] = bmi
                    X['HRrest'] = hr_rest
                    
                    # (C) 自动生成动态特征 (滚动/相对)
                    base_signals = ['HR', 'RF', 'MeanRRi', 'RMSSD', 'LF power', 'HF power', 'VLF power', 'DFAα1', 'SD1', 'SD2']
                    
                    for col in base_signals:
                        if col in X.columns:
                            # 转数值，防报错
                            X[col] = pd.to_numeric(X[col], errors='coerce').fillna(0)
                            
                            # 滚动特征
                            X[f'{col}_mean_6'] = X[col].rolling(window=6, min_periods=1).mean()
                            X[f'{col}_std_6'] = X[col].rolling(window=6, min_periods=1).std().fillna(0)
                            X[f'{col}_mean_12'] = X[col].rolling(window=12, min_periods=1).mean()
                            X[f'{col}_std_12'] = X[col].rolling(window=12, min_periods=1).std().fillna(0)
                            
                            # 相对特征
                            baseline = hr_rest if col == 'HR' else (X[col].iloc[:6].mean() if X[col].iloc[:6].mean() != 0 else 1.0)
                            X[f'{col}_rel_session'] = X[col] / baseline
                        else:
                            # 缺失填0
                            for suffix in ['_mean_6', '_std_6', '_mean_12', '_std_12', '_rel_session']:
                                X[f'{col}{suffix}'] = 0

                    # (D) 特征对齐 (准备给随机森林吃)
                    try:
                        if hasattr(rf_model, "feature_names_in_"):
                            model_features = rf_model.feature_names_in_
                        else:
                            # 兜底：用所有数值列
                            model_features = X.select_dtypes(include=[np.number]).columns
                        
                        # 补齐缺失列
                        for f in model_features:
                            if f not in X.columns:
                                X[f] = 0
                        X_final = X[model_features].fillna(0)
                        
                    except Exception as e:
                        st.error(f"Feature processing error: {e}")
                        st.stop()

                    # ==========================================
                    # 5. 执行预测 (Prediction)
                    # ==========================================
                    
                    # --- 任务 1: VTs 识别 (随机森林) ---
                    pred_stages = rf_model.predict(X_final)
                    # 平滑处理
                    smooth_stages = pd.Series(pred_stages).rolling(window=12, center=True).apply(lambda x: x.mode()[0] if not x.mode().empty else x[0]).fillna(method='bfill').fillna(method='ffill')
                    df['Stage'] = smooth_stages

                    # 提取时间点
                    vt1_idx = df[df['Stage'] == 1].index.min()
                    vt2_idx = df[df['Stage'] == 2].index.min()
                    
                    vt1_res = {'Time': df.loc[vt1_idx, 'Time'], 'HR': df.loc[vt1_idx, 'HR']} if pd.notna(vt1_idx) else None
                    vt2_res = {'Time': df.loc[vt2_idx, 'Time'], 'HR': df.loc[vt2_idx, 'HR']} if pd.notna(vt2_idx) else None

                    # --- 任务 2: VO2peak 计算 (使用您的线性公式) ---
                    # 1. 提取峰值数据 (最后 30s 均值)
                    # 注意：如果文件最后几行是恢复期，可能需要调整这里取值逻辑
                    peak_df = df.tail(6).mean() 
                    
                    # 2. 提取公式需要的变量
                    # 如果列名不存在，默认为 0
                    val_RMSSD_peak = peak_df['RMSSD'] if 'RMSSD' in peak_df else 0
                    val_RF_peak    = peak_df['RF'] if 'RF' in peak_df else 0
                    val_HR_peak    = peak_df['HR'] if 'HR' in peak_df else 0
                    
                    # 3. 代入公式 (直接计算)
                    # VO2max (L/min) = -2.3123 
                    #                + (0.530595 * Gender) 
                    #                + (0.039042 * RMSSD_peak_mean)
                    #                + (0.028138 * Age) 
                    #                + (0.025320 * Weight)
                    #                + (0.013507 * RF_peak_mean)
                    #                - (0.010645 * HRrest) 
                    #                + (0.010629 * Height) 
                    #                + (0.003778 * HR_peak_mean)
                    
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
                    
                    # 防止出现负数 (兜底)
                    if pred_vo2 < 0: pred_vo2 = 0.5 

                    # ==========================================
                    # 6. 结果展示
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
                    ax.plot(df['Time'], df['HR'], 'k-', label='Heart Rate', linewidth=2)
                    
                    # 颜色分区
                    ax.fill_between(df['Time'], 0, 220, where=df['Stage']==0, color='#eaffea', alpha=0.6, label='Zone 1')
                    ax.fill_between(df['Time'], 0, 220, where=df['Stage']==1, color='#fff9c4', alpha=0.6, label='Zone 2')
                    ax.fill_between(df['Time'], 0, 220, where=df['Stage']==2, color='#ffebee', alpha=0.6, label='Zone 3')
                    
                    if vt1_res: ax.axvline(vt1_res['Time'], color='blue', linestyle='--', label='VT1')
                    if vt2_res: ax.axvline(vt2_res['Time'], color='red', linestyle='--', label='VT2')
                    
                    ax.set_ylim(bottom=min(df['HR'])*0.9, top=max(df['HR'])*1.1)
                    ax.legend(loc='upper left')
                    st.pyplot(fig)
                    
                    res_csv = df[['Time', 'HR', 'Stage']].to_csv(index=False).encode('utf-8')
                    st.download_button("📥 Download Results", data=res_csv, file_name="results.csv", mime="text/csv")

    except Exception as e:
        st.error(f"⚠️ Error: {e}")