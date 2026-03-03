import streamlit as st
import pandas as pd
import numpy as np
from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score

# 設定網頁標題與樣式
st.set_page_config(page_title="酒類資料集預測系統", layout="wide")

# 套用自定義 CSS 以提升美感
st.markdown("""
<style>
    .main {
        background-color: #f8f9fa;
    }
    .stButton>button {
        width: 100%;
        border-radius: 5px;
        height: 3em;
        background-color: #007bff;
        color: white;
        font-weight: bold;
    }
    .stButton>button:hover {
        background-color: #0056b3;
        border-color: #0056b3;
    }
    .reportview-container .main .block-container{
        padding-top: 2rem;
    }
    h1 {
        color: #1a1a1a;
        font-family: 'Inter', sans-serif;
    }
</style>
""", unsafe_allow_html=True)

# 1. 載入資料集
@st.cache_data
def load_data():
    wine = load_wine()
    df = pd.DataFrame(wine.data, columns=wine.feature_names)
    df['target'] = wine.target
    return wine, df

wine_data, df = load_data()

# 2. 左側 Sidebar
st.sidebar.title("🛠️ 設定與資訊")

# 模型選擇下拉選單
model_choice = st.sidebar.selectbox(
    "選擇預測模型",
    ("KNN", "羅吉斯迴歸", "XGBoost", "隨機森林")
)

st.sidebar.markdown("---")
st.sidebar.subheader("🍷 酒類資料集資訊")
st.sidebar.info(f"""
**資料集名稱**: Wine dataset
**樣本數**: {len(df)}
**特徵數**: {len(wine_data.feature_names)}
**類別數**: {len(wine_data.target_names)} ({', '.join(wine_data.target_names)})
""")

# 3. 右側 Main 區
st.title("🍷 酒類品質預測系統")

col1, col2 = st.columns([1, 1])

with col1:
    st.subheader("📋 資料集前 5 筆內容")
    st.dataframe(df.head(), use_container_width=True)

with col2:
    st.subheader("📊 特徵統計值資訊")
    st.dataframe(df.describe().T, use_container_width=True)

st.markdown("---")

# 4. 預測邏輯
st.subheader("🚀 模型預測")

if st.button("開始進行預測"):
    with st.spinner('正在訓練模型中...'):
        # 準備資料
        X = df.drop('target', axis=1)
        y = df['target']
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # 根據選擇初始化模型
        if model_choice == "KNN":
            model = KNeighborsClassifier(n_neighbors=5)
        elif model_choice == "羅吉斯迴歸":
            model = LogisticRegression(max_iter=5000)
        elif model_choice == "XGBoost":
            model = XGBClassifier(use_label_encoder=False, eval_metric='mlogloss')
        elif model_choice == "隨機森林":
            model = RandomForestClassifier(n_estimators=100, random_state=42)

        # 訓練模型
        model.fit(X_train, y_train)
        
        # 預測
        y_pred = model.predict(X_test)
        acc = accuracy_score(y_test, y_pred)

        # 顯示結果
        st.success(f"### 預測完成！")
        
        m_col1, m_col2 = st.columns(2)
        m_col1.metric("使用模型", model_choice)
        m_col2.metric("預測準確度 (Accuracy)", f"{acc:.2%}")

        # 顯示部分預測比較
        results_df = pd.DataFrame({
            '實際類別': [wine_data.target_names[i] for i in y_test[:10]],
            '預測類別': [wine_data.target_names[i] for i in y_pred[:10]]
        })
        st.write("#### 前 10 筆測試資料預測對照：")
        st.table(results_df)

        if acc > 0.9:
            st.balloons()
