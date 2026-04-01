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

warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).parent.resolve()))
from core import load_model_pipeline

# ... (CSS и COLORS остаются без изменений) ...
COLOR_TEXT = "#39393A"
st.markdown(f"""<style>... CSS ...</style>""", unsafe_allow_html=True)

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, required=True, help='Путь к модели (.pkl)')
    parser.add_argument('--data', type=str, required=True, help='Путь к данным для теста (.csv)')
    return parser.parse_args(sys.argv[1:])

try:
    args = parse_args()
    model_path = args.model
    data_path = args.data
except SystemExit:
    st.error("❌ Укажите --model и --data")
    st.stop()

@st.cache_resource
def get_model(_path):
    return load_model_pipeline(_path)

@st.cache_data
def get_data(_path):
    return pd.read_csv(_path).replace(-1, 0)

# --- LOGIC ---

if os.path.exists(model_path):
    m_data = get_model(model_path)
    algo_name_display = m_data.get('algo_name', 'Unknown').upper()
    # Берем имя файла, на котором учились
    train_dataset_name = Path(m_data.get('dataset_path', 'Unknown')).name
else:
    st.error(f"❌ Модель не найдена: {model_path}")
    st.stop()

# Заголовок с именами датасетов
st.markdown(f"<h1>VPN TRAFFIC DETECTOR <br><span class='model-subtitle'>// {algo_name_display}</span></h1>", unsafe_allow_html=True)
st.markdown(f"""
<div style="text-align: center; margin-bottom: 20px; color: #666;">
    🎓 <b>Train Dataset:</b> {train_dataset_name} <br>
    🔍 <b>Test Dataset:</b> {Path(data_path).name}
</div>
""", unsafe_allow_html=True)

tab1, tab2 = st.tabs(["📊 Результаты (Metrics)", "💾 Сохранить & Визуализация"])

if not os.path.exists(data_path):
    st.error(f"❌ Файл данных не найден: {data_path}")
else:
    try:
        # Просто загружаем, не трогая структуру (duration должен быть уже удален на этапе подготовки)
        df = get_data(data_path)
        df['label'] = df['traffic_type'].str.contains('VPN', na=False)
        
        model = m_data['model']
        features = m_data['features']
        importances = m_data.get('importances', np.array([]))

        missing_feats = set(features) - set(df.columns)
        
        if missing_feats:
            st.error(f"⚠️ В тестовом датасете не хватает колонок: {missing_feats}")
            st.stop()
        else:
            from sklearn.model_selection import train_test_split
            # Для теста берем часть из загруженного файла
            _, X_test, _, y_test = train_test_split(df[features], df['label'], test_size=0.3, random_state=42)
            y_pred = model.predict(X_test)
            
            from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
            
            acc = accuracy_score(y_test, y_pred)
            prec = precision_score(y_test, y_pred, zero_division=0)
            rec = recall_score(y_test, y_pred, zero_division=0)
            f1 = f1_score(y_test, y_pred, zero_division=0)
            
            try:
                y_proba = model.predict_proba(X_test)[:, 1]
                roc_auc = roc_auc_score(y_test, y_proba)
            except:
                roc_auc = 0.0

            cm = confusion_matrix(y_test, y_pred)
            tn, fp, fn, tp = cm.ravel()
            specificity = tn / (tn + fp) if (tn + fp) > 0 else 0

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
                    st.code(classification_report(y_test, y_pred), language='text')

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
                            
                            fi_list = []
                            if len(importances) > 0:
                                for f, imp in zip(features, importances):
                                    fi_list.append({"feature": f, "importance": float(imp)})
                            
                            result_data = {
                                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                                "model_name": algo_name_display,
                                "run_name": run_name,
                                "train_dataset": train_dataset_name,   # Сохраняем имя файла обучения
                                "test_dataset": Path(data_path).name,  # Сохраняем имя файла теста
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
                    if len(importances) > 0:
                        feat_df = pd.DataFrame({'Feature': features, 'Importance': importances})
                        feat_df = feat_df.sort_values(by='Importance', ascending=True).tail(10)
                        custom_colors = ["#F25C54", "#F27059", "#F4845F", "#F79D65", "#F7B267"]
                        fig = px.bar(feat_df, x='Importance', y='Feature', orientation='h', color='Importance', color_continuous_scale=custom_colors)
                        fig.update_layout(plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)', font_color="#39393A", margin=dict(l=0, r=0, t=0, b=0), coloraxis_showscale=False)
                        st.plotly_chart(fig, use_container_width=True)

    except Exception as e:
        st.error(f"Ошибка: {e}")
        import traceback
        st.text(traceback.format_exc())