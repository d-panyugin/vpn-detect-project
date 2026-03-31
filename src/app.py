import streamlit as st
import os
import sys
import pandas as pd
import numpy as np
import plotly.express as px
from pathlib import Path

# --- CRITICAL FIX: Добавляем папку текущего файла (src) в path ---
# Это позволяет импортировать core.py независимо от того, откуда запущен скрипт
sys.path.insert(0, str(Path(__file__).parent.resolve()))

from core import load_model_pipeline

# --- STYLE ---
st.set_page_config(layout="wide", page_icon="🛡️")

st.markdown("""
<style>
    .main { background-color: #0e1117; }
    h1 {
        font-family: 'Helvetica', sans-serif; font-weight: 700;
        background: linear-gradient(90deg, #00ff41, #008F11);
        -webkit-background-clip: text; -webkit-text-fill-color: transparent;
        text-align: center; font-size: 3rem; margin-bottom: 1rem;
    }
    div[data-testid="stMetricValue"] { color: #e0e0e0; font-family: 'Courier New', monospace; }
</style>
""", unsafe_allow_html=True)

# --- LOGIC ---
model_path = os.environ.get("VPN_MODEL_PATH")
data_path = os.environ.get("DATA_PATH")

st.markdown("<h1>VPN traffic detector</h1>", unsafe_allow_html=True)

if not data_path:
    st.warning("DATA_PATH не установлен. Проверьте run.py")
else:
    try:
        # Проверка файла
        if not os.path.exists(data_path):
            st.error(f"❌ Файл данных не найден по пути: {data_path}")
        else:
            df = pd.read_csv(data_path).replace(-1, 0)
            df['label'] = df['traffic_type'].str.contains('VPN', na=False)

            if model_path:
                # Загрузка модели
                data = load_model_pipeline(model_path)
                model = data['model']
                features = data['features']
                algo_name = data.get('algo_name', 'Unknown')
                importances = data.get('importances', np.array([]))

                st.info(f"📂 Модель: **{algo_name.upper()}**")

                # Проверка фичей
                missing_feats = set(features) - set(df.columns)
                if missing_feats:
                    st.error(f"⚠️ В датасете не хватает колонок: {missing_feats}")
                else:
                    X = df[features]
                    y = df['label']
                    
                    from sklearn.model_selection import train_test_split
                    _, X_test, _, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
                    
                    y_pred = model.predict(X_test)
                    
                    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report
                    acc = accuracy_score(y_test, y_pred)
                    prec = precision_score(y_test, y_pred, zero_division=0)
                    rec = recall_score(y_test, y_pred, zero_division=0)
                    f1 = f1_score(y_test, y_pred, zero_division=0)
                    report = classification_report(y_test, y_pred)

                    # Metrics
                    m1, m2, m3, m4 = st.columns(4)
                    m1.metric("Accuracy", f"{acc:.2%}")
                    m2.metric("Precision", f"{prec:.2%}")
                    m3.metric("Recall", f"{rec:.2%}")
                    m4.metric("F1 Score", f"{f1:.2%}")

                    st.divider()

                    # Feature Importance
                    if len(importances) > 0:
                        feat_df = pd.DataFrame({'Feature': features, 'Importance': importances})
                        feat_df = feat_df.sort_values(by='Importance', ascending=True).tail(10)

                        fig = px.bar(
                            feat_df, x='Importance', y='Feature', orientation='h', 
                            color='Importance', color_continuous_scale=['#1a1a2e', '#00ff41']
                        )
                        fig.update_layout(
                            plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)', 
                            font_color='#e0e0e0', margin=dict(l=0, r=0, t=0, b=0),
                            yaxis=dict(title_font=dict(size=14), tickfont=dict(size=12)),
                            xaxis=dict(title_font=dict(size=14), tickfont=dict(size=12))
                        )
                        st.plotly_chart(fig, width='stretch')

                    with st.expander("Classification Report"):
                        st.code(report, language='text')
            else:
                st.warning("Модель не загружена. Передайте `--model <путь>` в аргументах запуска.")

    except Exception as e:
        st.error(f"Ошибка: {e}")
        import traceback
        st.text(traceback.format_exc())