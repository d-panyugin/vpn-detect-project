# app.py
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

# --- 1. Настройки и Инициализация ---
warnings.filterwarnings("ignore")
# Добавляем корень проекта в путь, чтобы видеть папку src
sys.path.insert(0, str(Path(__file__).parent.resolve()))

from src.core import load_model_pipeline

# --- 2. Загрузка внешних ресурсов (CSS) ---
def load_css(file_path):
    """Загружает стили из внешнего файла."""
    if os.path.exists(file_path):
        with open(file_path, "r", encoding="utf-8") as f:
            st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)
    else:
        st.warning(f"CSS файл не найден: {file_path}")

load_css("src/style.css")

# --- 3. Парсинг аргументов командной строки ---
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, required=True, help='Путь к файлу модели (.pkl)')
    parser.add_argument('--data', type=str, required=True, help='Путь к файлу данных (.csv)')
    # sys.argv[1:] отбрасывает аргументы самого streamlit
    return parser.parse_args(sys.argv[1:])

# Безопасное получение аргументов
try:
    args = parse_args()
    model_path = args.model
    data_path = args.data
except SystemExit:
    st.error("❌ Ошибка запуска. Укажите параметры: `streamlit run app.py -- --model model.pkl --data data.csv`")
    st.stop()

# --- 4. Кэширование ресурсов (Производительность) ---
# Модель загружается один раз и кэшируется
@st.cache_resource
def get_model(_path):
    return load_model_pipeline(_path)

# Данные кэшируются, чтобы не перечитывать при каждом действии в UI
@st.cache_data
def get_data(_path):
    # -1 заменяем на 0 (специфика данных)
    return pd.read_csv(_path).replace(-1, 0)

# --- 5. ОСНОВНАЯ ЛОГИКА ---

# Проверка существования модели
if not os.path.exists(model_path):
    st.error(f"❌ Файл модели не найден: {model_path}")
    st.stop()

# Загрузка модели
m_data = get_model(model_path)
pipeline = m_data['model'] # Это sklearn.Pipeline
algo_name = m_data.get('algo_name', 'Unknown').upper()
train_dataset_name = Path(m_data.get('dataset_path', 'Unknown')).name

# Отрисовка заголовка
st.markdown(f"<h1>VPN TRAFFIC DETECTOR <br><span class='model-subtitle'>// {algo_name}</span></h1>", unsafe_allow_html=True)
st.markdown(f"""
<div style="text-align: center; margin-bottom: 20px; color: #666;">
    🎓 <b>Train Dataset:</b> {train_dataset_name} <br>
    🔍 <b>Test Dataset:</b> {Path(data_path).name}
</div>
""", unsafe_allow_html=True)

# Вкладки
tab1, tab2 = st.tabs(["📊 Результаты (Metrics)", "💾 Сохранить & Визуализация"])

# Проверка данных
if not os.path.exists(data_path):
    st.error(f"❌ Файл данных не найден: {data_path}")
    st.stop()

try:
    df = get_data(data_path)
    
    # 1. Подготовка таргета (True/False)
    df['label'] = df['traffic_type'].str.contains('VPN', na=False)
    y_true = df['label']
    
    # 2. Подготовка входных данных (X)
    model_features = m_data['features'] 
    
    missing_cols = set(model_features) - set(df.columns)
    if missing_cols:
        st.error(f"⚠️ В тестовом датасете не хватает колонок, которые были в обучении: {missing_cols}")
        st.info("Убедитесь, что структура CSV совпадает с обучающей выборкой.")
        st.stop()
    
    X_input = df[model_features]
    
    # 3. Предсказание
    # Pipeline автоматически выполнит:
    # - Удаление ненужных колонок (DataPreprocessor)
    # - Генерацию новых фичей (DataPreprocessor)
    # - Масштабирование (StandardScaler)
    # - Предсказание (Classifier)
    # Это и есть преимущество Pipeline перед набором скриптов.
    
    y_pred = pipeline.predict(X_input)
    
    # 4. Расчет метрик
    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
    
    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    
    try:
        y_proba = pipeline.predict_proba(X_input)[:, 1]
        roc_auc = roc_auc_score(y_true, y_proba)
    except:
        roc_auc = 0.0

    cm = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel()
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0

    # --- ВКЛАДКА 1: МЕТРИКИ ---
    with tab1:
        m1, m2, m3, m4, m5, m6 = st.columns(6)
        m1.metric("Accuracy", f"{acc:.2%}")
        m2.metric("Precision", f"{prec:.2%}")
        m3.metric("Recall", f"{rec:.2%}")
        m4.metric("F1 Score", f"{f1:.2%}")
        m5.metric("AUC-ROC", f"{roc_auc:.4f}")
        m6.metric("Specificity", f"{specificity:.2%}")

        with st.expander("Classification Report"):
            from sklearn.metrics import classification_report
            st.code(classification_report(y_true, y_pred), language='text')

    # --- ВКЛАДКА 2: СОХРАНЕНИЕ И ГРАФИКИ ---
    with tab2:
        col_save, col_graph = st.columns([1, 2])
        
        with col_save:
            st.subheader("💾 Сохранить результат")
            run_name = st.text_input("Название прогона", placeholder="Test Run #1")
            save_btn = st.button("Сохранить в JSON", type="primary")
            
            if save_btn:
                if not run_name:
                    st.error("Введите название!")
                else:
                    results_dir = "results"
                    os.makedirs(results_dir, exist_ok=True)
                    
                    # Подготовка данных для сохранения
                    fi_list = []
                    processed_features = m_data.get('processed_features', [])
                    importances = m_data.get('importances', np.array([]))
                    
                    if len(importances) > 0 and len(importances) == len(processed_features):
                        for f, imp in zip(processed_features, importances):
                            fi_list.append({"feature": f, "importance": float(imp)})
                    
                    result_data = {
                        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                        "model_name": algo_name,
                        "run_name": run_name,
                        "train_dataset": train_dataset_name,
                        "test_dataset": Path(data_path).name,
                        "metrics": {
                            "accuracy": acc, "precision": prec, "recall": rec, 
                            "f1_score": f1, "roc_auc": roc_auc, "specificity": specificity
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
            st.subheader("Top 10 Features")
            processed_features = m_data.get('processed_features', [])
            importances = m_data.get('importances', np.array([]))

            if len(importances) > 0 and len(importances) == len(processed_features):
                feat_df = pd.DataFrame({'Feature': processed_features, 'Importance': importances})
                feat_df = feat_df.sort_values(by='Importance', ascending=True).tail(10)
                
                # Цвета из CSS
                custom_colors = ["#F25C54", "#F27059", "#F4845F", "#F79D65", "#F7B267"]
                fig = px.bar(feat_df, x='Importance', y='Feature', orientation='h', color='Importance', color_continuous_scale=custom_colors)
                
                fig.update_layout(
                    plot_bgcolor='rgba(0,0,0,0)', 
                    paper_bgcolor='rgba(0,0,0,0)', 
                    font_color="#39393A", 
                    margin=dict(l=0, r=0, t=0, b=0), 
                    coloraxis_showscale=False
                )
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("Важность признаков недоступна (возможно, использовался PCA, или модель не поддерживает оценку).")

except Exception as e:
    st.error(f"Критическая ошибка: {e}")
    import traceback
    st.text(traceback.format_exc())