import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

# ==========================================
# 0. 页面配置 (设置网页标题和图标)
# ==========================================
st.set_page_config(
    page_title="AI-CPET Assessment System",
    page_icon="🏃‍♂️",
    layout="wide"
)

# 美化样式的 CSS
st.markdown("""
<style>
    .main-header {font-size: 2rem; font-weight: bold; color: #0e1117;}
    .result-box {padding: 20px; background-color: #f0f2f6; border-radius: 10px; margin-bottom: 20px;}
</style>
""", unsafe_allow_html=True)

# ==========================================
# 1. 侧边栏：受试者基本信息输入
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
    
    # 性别转换 (模型训练时: Male=0, Female=1 或者反过来，这里假设 Male=1, Female=0，请根据您实际情况微调)
    # 根据您之前的代码：'女': 1, '男': 0 -> Female=1, Male=0
    gender_val = 1 if gender_input == "Female" else 0
    
    st.markdown("---")
    st.markdown("**Model Info:**")
    st.caption("• VTs: Random Forest Classifier")
    st.caption("• VO2peak: Gradient Boosting / MLR")

# ==========================================
# 2. 主界面：文件上传区域
# ==========================================
st.markdown('<div class="main-header">AI-Based CPET Analysis Platform</div>', unsafe_allow_html=True)
st.markdown("""
This tool uses machine learning to detect **Ventilatory Thresholds (VT1/VT2)** and predict **Peak Oxygen Uptake ($VO_{2peak}$)** from non-invasive physiological signals.
""")

st.warning("📋 Requirement: Upload 5s-interpolated data containing at least: `Time`, `HR`, `RF`, `RMSSD`, `DFA_alpha1`.")

uploaded_file = st.file_uploader("📂 Upload Excel or CSV File", type=["xlsx", "xls", "csv"])

if uploaded_file:
    try:
        # 读取数据
        if uploaded_file.name.endswith('csv'):
            df = pd.read_csv(uploaded_file)
        else:
            df = pd.read_excel(uploaded_file)
            
        st.success(f"✅ Data Loaded Successfully: {len(df)} time points.")
        
        # 简单的数据预览
        with st.expander("查看原始数据 (Raw Data Preview)"):
            st.dataframe(df.head())

        # ==========================================
        # 3. 加载模型 (Load Models)
        # ==========================================
        @st.cache_resource
        def load_models():
            # 确保这两个文件在同一文件夹下
            try:
                rf = joblib.load('rf_vts_model.pkl')
                vo2 = joblib.load('vo2_regression_model.pkl')
                return rf, vo2
            except FileNotFoundError:
                st.error("❌ Model files not found! Please check if .pkl files are in the folder.")
                return None, None
            
        rf_model, vo2_model = load_models()

        if rf_model and vo2_model:
            if st.button("🚀 Start AI Analysis", type="primary"):
                with st.spinner("Processing signals & Computing features..."):
                    
                    # ==========================================
                    # 4. 特征工程 (Feature Engineering)
                    # ==========================================
                    X = df.copy()
                    
                    # (A) 注入静态特征
                    X['Age'] = age
                    X['Gender'] = gender_val
                    X['Height'] = height
                    X['Weight'] = weight
                    X['BMI'] = bmi
                    X['HRrest'] = hr_rest # 确保也有这个列，以防回归模型需要
                    
                    # (B) 自动生成动态特征 (Rolling & Relative)
                    # 定义 10 个基座信号 (如果 Excel 里有就算，没有就填 0)
                    base_signals = ['HR', 'RF', 'MeanRRi', 'RMSSD', 'LF power', 'HF power', 'VLF power', 'DFAα1', 'SD1', 'SD2']
                    # 兼容中文列名或英文列名
                    col_mapping = {'RF（呼吸频率）': 'RF', 'DFA_alpha1': 'DFAα1', 'LF_power': 'LF power', 'HF_power': 'HF power', 'VLF_power': 'VLF power'}
                    X.rename(columns=col_mapping, inplace=True)

                    for col in base_signals:
                        if col in X.columns:
                            # 确保是数值类型
                            X[col] = pd.to_numeric(X[col], errors='coerce').fillna(0)
                            
                            # 滚动平均 (30s 和 60s)
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

                    # (C) 特征对齐 (Feature Alignment)
                    # 自动获取分类模型需要的特征名
                    try:
                        if hasattr(rf_model, "feature_names_in_"):
                            model_features = rf_model.feature_names_in_
                        else:
                            # 如果模型没保存特征名，使用所有数值列兜底
                            st.warning("Model feature names not found. Using all available numeric columns.")
                            model_features = X.select_dtypes(include=[np.number]).columns
                        
                        # 补齐缺失特征并排序
                        for f in model_features:
                            if f not in X.columns:
                                X[f] = 0
                        X_final = X[model_features].fillna(0)
                        
                    except Exception as e:
                        st.error(f"Feature processing error: {e}")
                        st.stop()

                    # ==========================================
                    # 5. 模型预测 (Prediction)
                    # ==========================================
                    
                    # --- 任务 1: VTs 识别 (分类) ---
                    pred_stages = rf_model.predict(X_final)
                    # 平滑处理 (使用众数滤波，窗口60s)
                    smooth_stages = pd.Series(pred_stages).rolling(window=12, center=True).apply(lambda x: x.mode()[0] if not x.mode().empty else x[0]).fillna(method='bfill').fillna(method='ffill')
                    df['Stage'] = smooth_stages

                    # 提取 VT1/VT2 时间点
                    # 逻辑: 找到第一个进入阶段 1 的点作为 VT1，第一个进入阶段 2 的点作为 VT2
                    vt1_idx = df[df['Stage'] == 1].index.min()
                    vt2_idx = df[df['Stage'] == 2].index.min()
                    
                    vt1_res = {'Time': df.loc[vt1_idx, 'Time'], 'HR': df.loc[vt1_idx, 'HR']} if pd.notna(vt1_idx) else None
                    vt2_res = {'Time': df.loc[vt2_idx, 'Time'], 'HR': df.loc[vt2_idx, 'HR']} if pd.notna(vt2_idx) else None

                    # --- 任务 2: VO2peak 预测 (回归) ---
                    # 构造回归输入：静态特征 + 峰值时刻(最后30s)均值
                    peak_df = df.tail(6).mean() # 取最后 6 个点 (30s)
                    
                    # 构建回归模型的输入字典 (必须与训练时的列名一致)
                    # 您的回归代码中用了: Age, Height, Weight, Gender, HRrest, 以及 _peak_mean 结尾的动态特征
                    vo2_input = {
                        'Age': age, 'Height': height, 'Weight': weight, 'Gender': gender_val, 'HRrest': hr_rest
                    }
                    # 自动把所有动态列加上 _peak_mean 后缀加入
                    for col in base_signals:
                        col_key = f'{col}_peak_mean'
                        # 如果原始数据里有这个列，就取均值，否则0
                        vo2_input[col_key] = peak_df[col] if col in peak_df else 0
                        
                    vo2_input_df = pd.DataFrame([vo2_input])
                    
                    # 对齐回归特征
                    if hasattr(vo2_model, "feature_names_in_"):
                        reg_feats = vo2_model.feature_names_in_
                        for f in reg_feats:
                            if f not in vo2_input_df.columns:
                                vo2_input_df[f] = 0
                        vo2_input_df = vo2_input_df[reg_feats]
                    
                    pred_vo2 = vo2_model.predict(vo2_input_df)[0]

                    # ==========================================
                    # 6. 结果展示 (Results)
                    # ==========================================
                    st.divider()
                    st.subheader("📊 Analysis Report")
                    
                    # 1. 核心指标卡片
                    c1, c2, c3 = st.columns(3)
                    c1.metric("Predicted VO2peak", f"{pred_vo2:.2f} L/min", help="Predicted Aerobic Capacity")
                    
                    if vt1_res:
                        c2.metric("VT1 (Aerobic Threshold)", f"{vt1_res['HR']:.0f} bpm", f"Time: {vt1_res['Time']} s")
                    else:
                        c2.metric("VT1", "Not Detected")
                        
                    if vt2_res:
                        c3.metric("VT2 (Anaerobic Threshold)", f"{vt2_res['HR']:.0f} bpm", f"Time: {vt2_res['Time']} s")
                    else:
                        c3.metric("VT2", "Not Detected")

                    # 2. 可视化图表
                    st.markdown("### Physiological Response & Intensity Zones")
                    
                    fig, ax = plt.subplots(figsize=(12, 5))
                    # 绘制心率
                    ax.plot(df['Time'], df['HR'], 'k-', label='Heart Rate', linewidth=2)
                    
                    # 绘制背景颜色分区
                    # Zone 1 (Pre-VT1): Green
                    ax.fill_between(df['Time'], 0, 220, where=df['Stage']==0, color='#eaffea', alpha=0.6, label='Zone 1 (Recovery/Light)')
                    # Zone 2 (VT1-VT2): Yellow
                    ax.fill_between(df['Time'], 0, 220, where=df['Stage']==1, color='#fff9c4', alpha=0.6, label='Zone 2 (Threshold)')
                    # Zone 3 (Post-VT2): Red
                    ax.fill_between(df['Time'], 0, 220, where=df['Stage']==2, color='#ffebee', alpha=0.6, label='Zone 3 (High Intensity)')
                    
                    # 标记 VT 线
                    if vt1_res:
                        ax.axvline(vt1_res['Time'], color='blue', linestyle='--', linewidth=2, label='VT1 Boundary')
                    if vt2_res:
                        ax.axvline(vt2_res['Time'], color='red', linestyle='--', linewidth=2, label='VT2 Boundary')
                    
                    ax.set_xlabel("Time (seconds)", fontsize=12)
                    ax.set_ylabel("Heart Rate (bpm)", fontsize=12)
                    ax.set_ylim(bottom=min(df['HR'])*0.9, top=max(df['HR'])*1.1)
                    ax.legend(loc='upper left')
                    ax.grid(True, linestyle=':', alpha=0.3)
                    
                    st.pyplot(fig)
                    
                    # 3. 下载结果
                    res_csv = df[['Time', 'HR', 'Stage']].to_csv(index=False).encode('utf-8')
                    st.download_button("📥 Download Analysis Results (CSV)", data=res_csv, file_name="cpet_analysis_result.csv", mime="text/csv")

    except Exception as e:
        st.error(f"⚠️ An error occurred: {e}")
        st.info("Please ensure your uploaded file contains the required columns (HR, RF, RMSSD, DFA_alpha1, etc.)")