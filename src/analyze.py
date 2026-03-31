import streamlit as st
import pandas as pd
import plotly.express as px
import os
import glob
import json
from datetime import datetime

# --- CONFIG ---
st.set_page_config(page_title="Model Analytics", layout="wide", page_icon="📊")

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

st.markdown("<h1>📊 АНАЛИТИКА МОДЕЛЕЙ</h1>", unsafe_allow_html=True)

@st.cache_data
def load_results_from_folder(folder):
    if not os.path.exists(folder):
        return pd.DataFrame()
    
    files = sorted(glob.glob(f"{folder}/*.json"), reverse=True) # Новые сверху
    data = []
    for file_path in files:
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = json.load(f)
                # Flatten metrics for the main table
                flat = {
                    'file': os.path.basename(file_path),
                    'timestamp': content.get('timestamp'),
                    'model_name': content.get('model_name'),
                    'run_name': content.get('run_name'),
                    'accuracy': content.get('metrics', {}).get('accuracy'),
                    'precision': content.get('metrics', {}).get('precision'),
                    'recall': content.get('metrics', {}).get('recall'),
                    'f1_score': content.get('metrics', {}).get('f1_score'),
                    'raw_data': content # Храним полные данные для детализации
                }
                data.append(flat)
        except Exception as e:
            pass
    return pd.DataFrame(data)

df = load_results_from_folder(RESULTS_DIR)

if df.empty:
    st.info("Папка `results` пуста.")
else:
    # --- 1. ГЛАВНАЯ ТАБЛИЦА ---
    st.subheader("📋 История запусков")
    display_cols = ['timestamp', 'model_name', 'run_name', 'accuracy', 'precision', 'recall', 'f1_score']
    st.dataframe(df[display_cols], use_container_width=True)
    
    st.divider()
    
    # --- 2. ВЫБОР ЗАПУСКА ДЛЯ ДЕТАЛЬНОГО ПРОСМОТРА ---
    st.subheader("🔍 Детальный анализ запуска")
    
    # Создаем читаемую опцию для selectbox
    df['select_option'] = df['timestamp'] + " | " + df['run_name']
    selected_option = st.selectbox("Выберите эксперимент:", df['select_option'].tolist())
    
    # Находим соответствующие данные
    selected_row = df[df['select_option'] == selected_option].iloc[0]
    raw_data = selected_row['raw_data']
    
    col_a, col_b = st.columns(2)
    
    with col_a:
        st.metric("F1 Score", f"{selected_row['f1_score']:.4f}")
        st.metric("Accuracy", f"{selected_row['accuracy']:.4f}")
        
    with col_b:
        st.metric("Precision", f"{selected_row['precision']:.4f}")
        st.metric("Recall", f"{selected_row['recall']:.4f}")

    st.divider()
    
    # --- 3. ПОЛНЫЙ СПИСОК ФИЧ ---
    st.subheader(f"📐 Feature Importance: {selected_row['run_name']}")
    
    if 'feature_importance_full' in raw_data and len(raw_data['feature_importance_full']) > 0:
        fi_df = pd.DataFrame(raw_data['feature_importance_full'])
        fi_df = fi_df.sort_values(by='importance', ascending=False)
        
        st.write(f"Всего признаков: {len(fi_df)}")
        st.dataframe(fi_df, use_container_width=True)
        
        # График топ-20
        fig = px.bar(fi_df.head(20), x='importance', y='feature', orientation='h',
                     color='importance', color_continuous_scale=[COLOR_5, COLOR_1])
        fig.update_layout(plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)', font_color=COLOR_TEXT)
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("Данные о важности признаков не сохранены в этом файле.")