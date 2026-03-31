import streamlit as st
import os
import sys
import argparse
import pandas as pd
import numpy as np
import plotly.express as px
from pathlib import Path
import warnings
import json
from datetime import datetime

# УБИРАЕМ ВСЕ ПРЕДУПРЕЖДЕНИЯ
warnings.filterwarnings("ignore")

# Добавляем папку текущего файла (src) в path
sys.path.insert(0, str(Path(__file__).parent.resolve()))

from core import load_model_pipeline

# --- STYLE & CONFIG ---
st.set_page_config(layout="wide", page_icon="🔥")

COLOR_1 = "#F7B267" 
COLOR_2 = "#F79D65"
COLOR_3 = "#F4845F"
COLOR_4 = "#F27059"
COLOR_5 = "#F25C54" 
COLOR_TEXT = "#39393A" 
HEADER_GRADIENT = f"linear-gradient(90deg, {COLOR_1}, {COLOR_2}, {COLOR_3}, {COLOR_4}, {COLOR_5})"

st.markdown(f"""
<style>
    .main {{ background-color: #F0F2F6; color: {COLOR_TEXT}; }}
    h1 {{
        font-family: 'Helvetica', sans-serif; font-weight: 800;
        background: {HEADER_GRADIENT};
        -webkit-background-clip: text; -webkit-text-fill-color: transparent;
        text-align: center; font-size: 2.5rem; margin-bottom: 0.5rem; line-height: 1.2;
    }}
    .model-subtitle {{ font-size: 0.6em; opacity: 0.8; font-weight: 600; -webkit-text-fill-color: {COLOR_TEXT}; }}
    div[data-testid="stMetricLabel"] > div {{ color: #666666 !important; font-size: 14px; font-weight: 600; text-transform: uppercase; }}
    div[data-testid="stMetricValue"] {{ color: {COLOR_TEXT} !important; font-family: 'Helvetica', sans-serif; font-size: 32px; font-weight: 800; }}
    div[data-testid="stMetricDelta"] > div > span {{ color: {COLOR_5} !important; font-weight: bold; }}
    .block-container {{ padding-top: 2rem; padding-bottom: 2rem; }}
</style>
""", unsafe_allow_html=True)

# --- ПАРСИНГ АРГУМЕНТОВ КОМАНДНОЙ СТРОКИ ---
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, required=True, help='Путь к файлу модели (.pkl)')
    parser.add_argument('--data', type=str, required=True, help='Путь к файлу данных (.csv)')
    # Streamlit передает аргументы после '--' через sys.argv
    return parser.parse_args(sys.argv[1:])

try:
    args = parse_args()
    model_path = args.model
    data_path = args.data
except SystemExit:
    st.error("❌ Ошибка запуска. Не указаны обязательные аргументы.")
    st.info("Используйте формат: `streamlit run src/app.py -- --model models/model.pkl --data data/data.csv`")
    st.stop()

# --- CACHING ---
@st.cache_resource
def get_model(_path):
    return load_model_pipeline(_path)

@st.cache_data
def get_data(_path):
    return pd.read_csv(_path).replace(-1, 0)

# --- LOGIC ---

algo_name_display = "UNKNOWN"
# Проверяем существование файла модели для красивого отображения ошибки или названия
if os.path.exists(model_path):
    try:
        m_data = get_model(model_path)
        algo_name_display = m_data.get('algo_name', 'Unknown').upper()
    except:
        algo_name_display = "LOADING ERROR"
else:
    st.error(f"❌ Модель не найдена по пути: `{model_path}`")
    st.stop()

st.markdown(f"<h1>VPN TRAFFIC DETECTOR <br><span class='model-subtitle'>// {algo_name_display}</span></h1>", unsafe_allow_html=True)

tab1, tab2 = st.tabs(["📊 Результаты (Metrics)", "💾 Сохранить & Визуализация"])

# Проверяем данные
if not os.path.exists(data_path):
    st.error(f"❌ Файл данных не найден по пути: `{data_path}`")
else:
    try:
        df = get_data(data_path)
        df['label'] = df['traffic_type'].str.contains('VPN', na=False)
        
        data = get_model(model_path)
        model = data['model']
        features = data['features']
        importances = data.get('importances', np.array([]))

        missing_feats = set(features) - set(df.columns)
        
        if missing_feats:
            st.error(f"⚠️ В датасете не хватает колонок: {missing_feats}")
        else:
            from sklearn.model_selection import train_test_split
            _, X_test, _, y_test = train_test_split(df[features], df['label'], test_size=0.3, random_state=42)
            y_pred = model.predict(X_test)
            
            from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report
            acc = accuracy_score(y_test, y_pred)
            prec = precision_score(y_test, y_pred, zero_division=0)
            rec = recall_score(y_test, y_pred, zero_division=0)
            f1 = f1_score(y_test, y_pred, zero_division=0)
            report = classification_report(y_test, y_pred)

            with tab1:
                m1, m2, m3, m4 = st.columns(4)
                m1.metric("Accuracy", f"{acc:.2%}")
                m2.metric("Precision", f"{prec:.2%}")
                m3.metric("Recall", f"{rec:.2%}")
                m4.metric("F1 Score", f"{f1:.2%}")
                with st.expander("Подробнее (Classification Report)"):
                    st.code(report, language='text')

            with tab2:
                col_save, col_graph = st.columns([1, 2])
                
                with col_save:
                    st.subheader("💾 Сохранить в аналитику")
                    run_name = st.text_input("Название эксперимента", placeholder="Например: RF v2 depth=15")
                    save_btn = st.button("Сохранить (JSON)", type="primary")
                    
                    if save_btn:
                        if not run_name:
                            st.error("Введите название!")
                        else:
                            results_dir = "results"
                            os.makedirs(results_dir, exist_ok=True)
                            
                            fi_list = []
                            if len(importances) > 0:
                                for f, imp in zip(features, importances):
                                    fi_list.append({"feature": f, "importance": float(imp)})
                            
                            result_data = {
                                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                                "model_name": algo_name_display,
                                "run_name": run_name,
                                "metrics": {
                                    "accuracy": acc,
                                    "precision": prec,
                                    "recall": rec,
                                    "f1_score": f1
                                },
                                "feature_importance_full": fi_list
                            }
                            
                            safe_name = "".join([c for c in run_name if c.isalnum() or c in (' ', '-', '_')]).strip()
                            timestamp_str = datetime.now().strftime('%Y%m%d_%H%M%S')
                            filename = f"{results_dir}/{timestamp_str}_{safe_name}.json"
                            
                            with open(filename, "w", encoding='utf-8') as f:
                                json.dump(result_data, f, ensure_ascii=False, indent=4)
                            
                            st.success(f"✅ Сохранено:\n`{filename}`")

                with col_graph:
                    st.subheader("Top 10 Features (Preview)")
                    if len(importances) > 0:
                        feat_df = pd.DataFrame({'Feature': features, 'Importance': importances})
                        feat_df = feat_df.sort_values(by='Importance', ascending=True).tail(10)
                        custom_colors = [COLOR_5, COLOR_4, COLOR_3, COLOR_2, COLOR_1]
                        
                        fig = px.bar(
                            feat_df, x='Importance', y='Feature', orientation='h', 
                            color='Importance', color_continuous_scale=custom_colors
                        )
                        fig.update_layout(
                            plot_bgcolor='rgba(255,255,255,0)', paper_bgcolor='rgba(255,255,255,0)', 
                            font_color=COLOR_TEXT, margin=dict(l=0, r=0, t=0, b=0),
                            coloraxis_showscale=False
                        )
                        st.plotly_chart(fig, use_container_width=True)

    except Exception as e:
        st.error(f"Ошибка: {e}")
        import traceback
        st.text(traceback.format_exc())