
import streamlit as st
import pandas as pd
import plotly.express as px
import os
import glob
import json
from datetime import datetime

# --- CONFIG ---
st.set_page_config(page_title="Model Analytics", layout="wide", page_icon="📊")

# Стиль (аналогичный app.py)
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
        text-align: center; font-size: 2.5rem; margin-bottom: 1rem;
    }}
</style>
""", unsafe_allow_html=True)

RESULTS_DIR = "results"

st.markdown("<h1>📊 АНАЛИТИКА МОДЕЛЕЙ (Папка results)</h1>", unsafe_allow_html=True)

# --- ЗАГРУЗКА ДАННЫХ ИЗ ПАПКИ ---
@st.cache_data
def load_results_from_folder(folder):
    if not os.path.exists(folder):
        return pd.DataFrame()
    
    files = sorted(glob.glob(f"{folder}/*.json"))
    data = []
    
    for file_path in files:
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data.append(json.load(f))
        except Exception as e:
            st.warning(f"Не удалось прочитать {file_path}: {e}")
            
    return pd.DataFrame(data)

df = load_results_from_folder(RESULTS_DIR)

if df.empty:
    st.info("Папка `results` пуста или не существует. Сохраните результаты в app.py.")
else:
    # Преобразуем timestamp для корректной сортировки
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df = df.sort_values(by='timestamp', ascending=False)
    
    # --- 1. ТАБЛИЦА ---
    st.subheader("📋 История запусков")
    display_df = df[['timestamp', 'model_name', 'run_name', 'accuracy', 'precision', 'recall', 'f1_score']].copy()
    display_df['timestamp'] = display_df['timestamp'].dt.strftime('%Y-%m-%d %H:%M')
    st.dataframe(display_df, use_container_width=True)
    
    st.divider()

    # --- 2. СРАВНЕНИЕ МЕТРИК ---
    st.subheader("📈 Сравнение метрик")
    metrics = ['accuracy', 'precision', 'recall', 'f1_score']
    df_melted = df.melt(id_vars=['run_name', 'model_name'], value_vars=metrics, 
                        var_name='Metric', value_name='Score')

    fig = px.bar(df_melted, x='run_name', y='Score', color='Metric', 
                 barmode='group',
                 color_discrete_map={
                     'accuracy': COLOR_1,
                     'precision': COLOR_2,
                     'recall': COLOR_3,
                     'f1_score': COLOR_5
                 },
                 hover_data=['model_name'])
    
    fig.update_layout(
        plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)', 
        font_color=COLOR_TEXT,
        xaxis_title="Эксперимент",
        yaxis_title="Значение",
        legend_title="Метрика"
    )
    st.plotly_chart(fig, use_container_width=True)

    # --- 3. ТОП F1 ---
    st.subheader("🏆 Топ-5 моделей по F1 Score")
    top_f1 = df.sort_values(by='f1_score', ascending=False).head(5)
    
    fig_f1 = px.bar(top_f1, x='f1_score', y='run_name', orientation='h',
                    color='f1_score', 
                    color_continuous_scale=[COLOR_5, COLOR_1],
                    text='f1_score')
    
    fig_f1.update_layout(
        plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)', 
        font_color=COLOR_TEXT,
        xaxis_title="F1 Score",
        yaxis_title="Эксперимент",
        showlegend=False,
        coloraxis_showscale=False
    )
    fig_f1.update_traces(texttemplate='%{text:.4f}', textposition='outside')
    st.plotly_chart(fig_f1, use_container_width=True)
