{\rtf1\ansi\ansicpg936\cocoartf2867
\cocoatextscaling0\cocoaplatform0{\fonttbl\f0\fswiss\fcharset0 Helvetica;}
{\colortbl;\red255\green255\blue255;}
{\*\expandedcolortbl;;}
\paperw11900\paperh16840\margl1440\margr1440\vieww11520\viewh8400\viewkind0
\pard\tx720\tx1440\tx2160\tx2880\tx3600\tx4320\tx5040\tx5760\tx6480\tx7200\tx7920\tx8640\pardirnatural\partightenfactor0

\f0\fs24 \cf0 import streamlit as st\
import pandas as pd\
import numpy as np\
import joblib\
import matplotlib.pyplot as plt\
import seaborn as sns\
\
# ==========================================\
# 0. \uc0\u39029 \u38754 \u37197 \u32622  (\u35774 \u32622 \u32593 \u39029 \u26631 \u39064 \u21644 \u22270 \u26631 )\
# ==========================================\
st.set_page_config(\
    page_title="AI-CPET Assessment System",\
    page_icon="\uc0\u55356 \u57283 \u8205 \u9794 \u65039 ",\
    layout="wide"\
)\
\
# \uc0\u32654 \u21270 \u26679 \u24335 \u30340  CSS\
st.markdown("""\
<style>\
    .main-header \{font-size: 2rem; font-weight: bold; color: #0e1117;\}\
    .result-box \{padding: 20px; background-color: #f0f2f6; border-radius: 10px; margin-bottom: 20px;\}\
</style>\
""", unsafe_allow_html=True)\
\
# ==========================================\
# 1. \uc0\u20391 \u36793 \u26639 \u65306 \u21463 \u35797 \u32773 \u22522 \u26412 \u20449 \u24687 \u36755 \u20837 \
# ==========================================\
with st.sidebar:\
    st.image("https://img.icons8.com/color/96/000000/heart-monitor.png", width=60)\
    st.header("1. Participant Info")\
    st.markdown("Please input subject demographics:")\
    \
    # \uc0\u36755 \u20837 \u25511 \u20214 \
    gender_input = st.selectbox("Gender (\uc0\u24615 \u21035 )", ["Male", "Female"])\
    age = st.number_input("Age (\uc0\u24180 \u40836 )", 18, 80, 25)\
    height = st.number_input("Height (\uc0\u36523 \u39640  cm)", 140.0, 220.0, 175.0)\
    weight = st.number_input("Weight (\uc0\u20307 \u37325  kg)", 40.0, 150.0, 70.0)\
    hr_rest = st.number_input("Resting HR (\uc0\u38745 \u24687 \u24515 \u29575  bpm)", 30, 120, 60)\
    \
    # \uc0\u33258 \u21160 \u35745 \u31639  BMI\
    bmi = weight / ((height/100)**2)\
    st.info(f"\uc0\u55357 \u56522  Calculated BMI: **\{bmi:.1f\}** kg/m\'b2")\
    \
    # \uc0\u24615 \u21035 \u36716 \u25442  (\u27169 \u22411 \u35757 \u32451 \u26102 : Male=0, Female=1 \u25110 \u32773 \u21453 \u36807 \u26469 \u65292 \u36825 \u37324 \u20551 \u35774  Male=1, Female=0\u65292 \u35831 \u26681 \u25454 \u24744 \u23454 \u38469 \u24773 \u20917 \u24494 \u35843 )\
    # \uc0\u26681 \u25454 \u24744 \u20043 \u21069 \u30340 \u20195 \u30721 \u65306 '\u22899 ': 1, '\u30007 ': 0 -> Female=1, Male=0\
    gender_val = 1 if gender_input == "Female" else 0\
    \
    st.markdown("---")\
    st.markdown("**Model Info:**")\
    st.caption("\'95 VTs: Random Forest Classifier")\
    st.caption("\'95 VO2peak: Gradient Boosting / MLR")\
\
# ==========================================\
# 2. \uc0\u20027 \u30028 \u38754 \u65306 \u25991 \u20214 \u19978 \u20256 \u21306 \u22495 \
# ==========================================\
st.markdown('<div class="main-header">AI-Based CPET Analysis Platform</div>', unsafe_allow_html=True)\
st.markdown("""\
This tool uses machine learning to detect **Ventilatory Thresholds (VT1/VT2)** and predict **Peak Oxygen Uptake ($VO_\{2peak\}$)** from non-invasive physiological signals.\
""")\
\
st.warning("\uc0\u55357 \u56523  Requirement: Upload 5s-interpolated data containing at least: `Time`, `HR`, `RF`, `RMSSD`, `DFA_alpha1`.")\
\
uploaded_file = st.file_uploader("\uc0\u55357 \u56514  Upload Excel or CSV File", type=["xlsx", "xls", "csv"])\
\
if uploaded_file:\
    try:\
        # \uc0\u35835 \u21462 \u25968 \u25454 \
        if uploaded_file.name.endswith('csv'):\
            df = pd.read_csv(uploaded_file)\
        else:\
            df = pd.read_excel(uploaded_file)\
            \
        st.success(f"\uc0\u9989  Data Loaded Successfully: \{len(df)\} time points.")\
        \
        # \uc0\u31616 \u21333 \u30340 \u25968 \u25454 \u39044 \u35272 \
        with st.expander("\uc0\u26597 \u30475 \u21407 \u22987 \u25968 \u25454  (Raw Data Preview)"):\
            st.dataframe(df.head())\
\
        # ==========================================\
        # 3. \uc0\u21152 \u36733 \u27169 \u22411  (Load Models)\
        # ==========================================\
        @st.cache_resource\
        def load_models():\
            # \uc0\u30830 \u20445 \u36825 \u20004 \u20010 \u25991 \u20214 \u22312 \u21516 \u19968 \u25991 \u20214 \u22841 \u19979 \
            try:\
                rf = joblib.load('rf_vts_model.pkl')\
                vo2 = joblib.load('vo2_regression_model.pkl')\
                return rf, vo2\
            except FileNotFoundError:\
                st.error("\uc0\u10060  Model files not found! Please check if .pkl files are in the folder.")\
                return None, None\
            \
        rf_model, vo2_model = load_models()\
\
        if rf_model and vo2_model:\
            if st.button("\uc0\u55357 \u56960  Start AI Analysis", type="primary"):\
                with st.spinner("Processing signals & Computing features..."):\
                    \
                    # ==========================================\
                    # 4. \uc0\u29305 \u24449 \u24037 \u31243  (Feature Engineering)\
                    # ==========================================\
                    X = df.copy()\
                    \
                    # (A) \uc0\u27880 \u20837 \u38745 \u24577 \u29305 \u24449 \
                    X['Age'] = age\
                    X['Gender'] = gender_val\
                    X['Height'] = height\
                    X['Weight'] = weight\
                    X['BMI'] = bmi\
                    X['HRrest'] = hr_rest # \uc0\u30830 \u20445 \u20063 \u26377 \u36825 \u20010 \u21015 \u65292 \u20197 \u38450 \u22238 \u24402 \u27169 \u22411 \u38656 \u35201 \
                    \
                    # (B) \uc0\u33258 \u21160 \u29983 \u25104 \u21160 \u24577 \u29305 \u24449  (Rolling & Relative)\
                    # \uc0\u23450 \u20041  10 \u20010 \u22522 \u24231 \u20449 \u21495  (\u22914 \u26524  Excel \u37324 \u26377 \u23601 \u31639 \u65292 \u27809 \u26377 \u23601 \u22635  0)\
                    base_signals = ['HR', 'RF', 'MeanRRi', 'RMSSD', 'LF power', 'HF power', 'VLF power', 'DFA\uc0\u945 1', 'SD1', 'SD2']\
                    # \uc0\u20860 \u23481 \u20013 \u25991 \u21015 \u21517 \u25110 \u33521 \u25991 \u21015 \u21517 \
                    col_mapping = \{'RF\uc0\u65288 \u21628 \u21560 \u39057 \u29575 \u65289 ': 'RF', 'DFA_alpha1': 'DFA\u945 1', 'LF_power': 'LF power', 'HF_power': 'HF power', 'VLF_power': 'VLF power'\}\
                    X.rename(columns=col_mapping, inplace=True)\
\
                    for col in base_signals:\
                        if col in X.columns:\
                            # \uc0\u30830 \u20445 \u26159 \u25968 \u20540 \u31867 \u22411 \
                            X[col] = pd.to_numeric(X[col], errors='coerce').fillna(0)\
                            \
                            # \uc0\u28378 \u21160 \u24179 \u22343  (30s \u21644  60s)\
                            X[f'\{col\}_mean_6'] = X[col].rolling(window=6, min_periods=1).mean()\
                            X[f'\{col\}_std_6'] = X[col].rolling(window=6, min_periods=1).std().fillna(0)\
                            X[f'\{col\}_mean_12'] = X[col].rolling(window=12, min_periods=1).mean()\
                            X[f'\{col\}_std_12'] = X[col].rolling(window=12, min_periods=1).std().fillna(0)\
                            \
                            # \uc0\u30456 \u23545 \u29305 \u24449 \
                            baseline = hr_rest if col == 'HR' else (X[col].iloc[:6].mean() if X[col].iloc[:6].mean() != 0 else 1.0)\
                            X[f'\{col\}_rel_session'] = X[col] / baseline\
                        else:\
                            # \uc0\u32570 \u22833 \u22635 0\
                            for suffix in ['_mean_6', '_std_6', '_mean_12', '_std_12', '_rel_session']:\
                                X[f'\{col\}\{suffix\}'] = 0\
\
                    # (C) \uc0\u29305 \u24449 \u23545 \u40784  (Feature Alignment)\
                    # \uc0\u33258 \u21160 \u33719 \u21462 \u20998 \u31867 \u27169 \u22411 \u38656 \u35201 \u30340 \u29305 \u24449 \u21517 \
                    try:\
                        if hasattr(rf_model, "feature_names_in_"):\
                            model_features = rf_model.feature_names_in_\
                        else:\
                            # \uc0\u22914 \u26524 \u27169 \u22411 \u27809 \u20445 \u23384 \u29305 \u24449 \u21517 \u65292 \u20351 \u29992 \u25152 \u26377 \u25968 \u20540 \u21015 \u20828 \u24213 \
                            st.warning("Model feature names not found. Using all available numeric columns.")\
                            model_features = X.select_dtypes(include=[np.number]).columns\
                        \
                        # \uc0\u34917 \u40784 \u32570 \u22833 \u29305 \u24449 \u24182 \u25490 \u24207 \
                        for f in model_features:\
                            if f not in X.columns:\
                                X[f] = 0\
                        X_final = X[model_features].fillna(0)\
                        \
                    except Exception as e:\
                        st.error(f"Feature processing error: \{e\}")\
                        st.stop()\
\
                    # ==========================================\
                    # 5. \uc0\u27169 \u22411 \u39044 \u27979  (Prediction)\
                    # ==========================================\
                    \
                    # --- \uc0\u20219 \u21153  1: VTs \u35782 \u21035  (\u20998 \u31867 ) ---\
                    pred_stages = rf_model.predict(X_final)\
                    # \uc0\u24179 \u28369 \u22788 \u29702  (\u20351 \u29992 \u20247 \u25968 \u28388 \u27874 \u65292 \u31383 \u21475 60s)\
                    smooth_stages = pd.Series(pred_stages).rolling(window=12, center=True).apply(lambda x: x.mode()[0] if not x.mode().empty else x[0]).fillna(method='bfill').fillna(method='ffill')\
                    df['Stage'] = smooth_stages\
\
                    # \uc0\u25552 \u21462  VT1/VT2 \u26102 \u38388 \u28857 \
                    # \uc0\u36923 \u36753 : \u25214 \u21040 \u31532 \u19968 \u20010 \u36827 \u20837 \u38454 \u27573  1 \u30340 \u28857 \u20316 \u20026  VT1\u65292 \u31532 \u19968 \u20010 \u36827 \u20837 \u38454 \u27573  2 \u30340 \u28857 \u20316 \u20026  VT2\
                    vt1_idx = df[df['Stage'] == 1].index.min()\
                    vt2_idx = df[df['Stage'] == 2].index.min()\
                    \
                    vt1_res = \{'Time': df.loc[vt1_idx, 'Time'], 'HR': df.loc[vt1_idx, 'HR']\} if pd.notna(vt1_idx) else None\
                    vt2_res = \{'Time': df.loc[vt2_idx, 'Time'], 'HR': df.loc[vt2_idx, 'HR']\} if pd.notna(vt2_idx) else None\
\
                    # --- \uc0\u20219 \u21153  2: VO2peak \u39044 \u27979  (\u22238 \u24402 ) ---\
                    # \uc0\u26500 \u36896 \u22238 \u24402 \u36755 \u20837 \u65306 \u38745 \u24577 \u29305 \u24449  + \u23792 \u20540 \u26102 \u21051 (\u26368 \u21518 30s)\u22343 \u20540 \
                    peak_df = df.tail(6).mean() # \uc0\u21462 \u26368 \u21518  6 \u20010 \u28857  (30s)\
                    \
                    # \uc0\u26500 \u24314 \u22238 \u24402 \u27169 \u22411 \u30340 \u36755 \u20837 \u23383 \u20856  (\u24517 \u39035 \u19982 \u35757 \u32451 \u26102 \u30340 \u21015 \u21517 \u19968 \u33268 )\
                    # \uc0\u24744 \u30340 \u22238 \u24402 \u20195 \u30721 \u20013 \u29992 \u20102 : Age, Height, Weight, Gender, HRrest, \u20197 \u21450  _peak_mean \u32467 \u23614 \u30340 \u21160 \u24577 \u29305 \u24449 \
                    vo2_input = \{\
                        'Age': age, 'Height': height, 'Weight': weight, 'Gender': gender_val, 'HRrest': hr_rest\
                    \}\
                    # \uc0\u33258 \u21160 \u25226 \u25152 \u26377 \u21160 \u24577 \u21015 \u21152 \u19978  _peak_mean \u21518 \u32512 \u21152 \u20837 \
                    for col in base_signals:\
                        col_key = f'\{col\}_peak_mean'\
                        # \uc0\u22914 \u26524 \u21407 \u22987 \u25968 \u25454 \u37324 \u26377 \u36825 \u20010 \u21015 \u65292 \u23601 \u21462 \u22343 \u20540 \u65292 \u21542 \u21017 0\
                        vo2_input[col_key] = peak_df[col] if col in peak_df else 0\
                        \
                    vo2_input_df = pd.DataFrame([vo2_input])\
                    \
                    # \uc0\u23545 \u40784 \u22238 \u24402 \u29305 \u24449 \
                    if hasattr(vo2_model, "feature_names_in_"):\
                        reg_feats = vo2_model.feature_names_in_\
                        for f in reg_feats:\
                            if f not in vo2_input_df.columns:\
                                vo2_input_df[f] = 0\
                        vo2_input_df = vo2_input_df[reg_feats]\
                    \
                    pred_vo2 = vo2_model.predict(vo2_input_df)[0]\
\
                    # ==========================================\
                    # 6. \uc0\u32467 \u26524 \u23637 \u31034  (Results)\
                    # ==========================================\
                    st.divider()\
                    st.subheader("\uc0\u55357 \u56522  Analysis Report")\
                    \
                    # 1. \uc0\u26680 \u24515 \u25351 \u26631 \u21345 \u29255 \
                    c1, c2, c3 = st.columns(3)\
                    c1.metric("Predicted VO2peak", f"\{pred_vo2:.2f\} L/min", help="Predicted Aerobic Capacity")\
                    \
                    if vt1_res:\
                        c2.metric("VT1 (Aerobic Threshold)", f"\{vt1_res['HR']:.0f\} bpm", f"Time: \{vt1_res['Time']\} s")\
                    else:\
                        c2.metric("VT1", "Not Detected")\
                        \
                    if vt2_res:\
                        c3.metric("VT2 (Anaerobic Threshold)", f"\{vt2_res['HR']:.0f\} bpm", f"Time: \{vt2_res['Time']\} s")\
                    else:\
                        c3.metric("VT2", "Not Detected")\
\
                    # 2. \uc0\u21487 \u35270 \u21270 \u22270 \u34920 \
                    st.markdown("### Physiological Response & Intensity Zones")\
                    \
                    fig, ax = plt.subplots(figsize=(12, 5))\
                    # \uc0\u32472 \u21046 \u24515 \u29575 \
                    ax.plot(df['Time'], df['HR'], 'k-', label='Heart Rate', linewidth=2)\
                    \
                    # \uc0\u32472 \u21046 \u32972 \u26223 \u39068 \u33394 \u20998 \u21306 \
                    # Zone 1 (Pre-VT1): Green\
                    ax.fill_between(df['Time'], 0, 220, where=df['Stage']==0, color='#eaffea', alpha=0.6, label='Zone 1 (Recovery/Light)')\
                    # Zone 2 (VT1-VT2): Yellow\
                    ax.fill_between(df['Time'], 0, 220, where=df['Stage']==1, color='#fff9c4', alpha=0.6, label='Zone 2 (Threshold)')\
                    # Zone 3 (Post-VT2): Red\
                    ax.fill_between(df['Time'], 0, 220, where=df['Stage']==2, color='#ffebee', alpha=0.6, label='Zone 3 (High Intensity)')\
                    \
                    # \uc0\u26631 \u35760  VT \u32447 \
                    if vt1_res:\
                        ax.axvline(vt1_res['Time'], color='blue', linestyle='--', linewidth=2, label='VT1 Boundary')\
                    if vt2_res:\
                        ax.axvline(vt2_res['Time'], color='red', linestyle='--', linewidth=2, label='VT2 Boundary')\
                    \
                    ax.set_xlabel("Time (seconds)", fontsize=12)\
                    ax.set_ylabel("Heart Rate (bpm)", fontsize=12)\
                    ax.set_ylim(bottom=min(df['HR'])*0.9, top=max(df['HR'])*1.1)\
                    ax.legend(loc='upper left')\
                    ax.grid(True, linestyle=':', alpha=0.3)\
                    \
                    st.pyplot(fig)\
                    \
                    # 3. \uc0\u19979 \u36733 \u32467 \u26524 \
                    res_csv = df[['Time', 'HR', 'Stage']].to_csv(index=False).encode('utf-8')\
                    st.download_button("\uc0\u55357 \u56549  Download Analysis Results (CSV)", data=res_csv, file_name="cpet_analysis_result.csv", mime="text/csv")\
\
    except Exception as e:\
        st.error(f"\uc0\u9888 \u65039  An error occurred: \{e\}")\
        st.info("Please ensure your uploaded file contains the required columns (HR, RF, RMSSD, DFA_alpha1, etc.)")}