# src/analyze.py
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import os
import glob
import json
import joblib
import shap
import matplotlib.pyplot as plt
import warnings
import sys

from pathlib import Path
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

warnings.filterwarnings("ignore")

# --- PATH FIX ---
sys.path.insert(0, str(Path(__file__).parent.parent.resolve()))

from src.core import load_model_pipeline

# --- CONFIGURATION ---
st.set_page_config(page_title="VPN Detector Hub", layout="wide")

# Load CSS
def load_css(file_name):
    try:
        with open(file_name) as f:
            st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)
    except:
        pass

load_css("src/style.css")

def get_files(folder, extension):
    """Scans project folder for files with specific extension."""
    target_path = Path(__file__).parent.parent / folder
    if not target_path.exists():
        return []
    return [f.name for f in target_path.glob(f"*{extension}")]

@st.cache_data
def load_history(folder):
    """Loads JSON results."""
    target_path = Path(__file__).parent.parent / folder
    if not target_path.exists(): return pd.DataFrame()
    
    files = sorted(target_path.glob("*.json"), reverse=True)
    data = []
    for f in files:
        try:
            with open(f, 'r', encoding='utf-8') as file:
                c = json.load(file)
                m = c.get('metrics', {})
                data.append({
                    'Timestamp': c.get('timestamp'),
                    'Model': c.get('model_name'),
                    'Run': c.get('run_name'),
                    'Accuracy': m.get('accuracy', 0),
                    'F1': m.get('f1_score', 0),
                    'ROC-AUC': m.get('roc_auc', 0),
                    'Raw': c
                })
        except: continue
    return pd.DataFrame(data)

# --- UI ---
st.title("VPN Traffic Analysis Hub")

tab_history, tab_comparison, tab_research = st.tabs(["Experiment History", "Model Comparison", "Research Lab"])

# ====================== #
# TAB 1: HISTORY (JSON)  #
# ====================== #
with tab_history:
    st.header("Experiment History")
    RESULTS_DIR = "results"

    df_hist = load_history(RESULTS_DIR)
    if df_hist.empty:
        st.info("No saved reports found in /results directory.")
    else:
        st.dataframe(df_hist.drop(columns=['Raw']), use_container_width=True)
        
        st.divider()
        selected_run = st.selectbox("Select Run for Details", df_hist['Run'].tolist())
        raw = df_hist[df_hist['Run'] == selected_run].iloc[0]['Raw']
        
        # --- Feature Importance ---
        if raw.get('feature_importance_full'):
            st.subheader("Feature Importance")
            fi = pd.DataFrame(raw['feature_importance_full']).sort_values('importance', ascending=False).head(20)
            fig = px.bar(fi, x='importance', y='feature', orientation='h', color_discrete_sequence=['#3366CC'])
            st.plotly_chart(fig, use_container_width=True)
            st.divider()

        col1, col2 = st.columns(2)
        with col1:
            st.metric("Accuracy", f"{raw['metrics']['accuracy']:.4f}")
            st.metric("F1 Score", f"{raw['metrics']['f1_score']:.4f}")
        with col2:
            with st.expander("Raw Configuration"):
                st.json(raw)
# ==========================================
# TAB 2: MODEL COMPARISON (FIXED)
# ==========================================
with tab_comparison:
    st.header("Model Comparison (A vs B)")
    st.caption("Select two models and a dataset to compare performance.")
    
    # --- SETTINGS ---
    with st.expander("Configuration", expanded=True):
        col_m1, col_m2, col_d = st.columns(3)
        
        # Auto-discover files
        models_list = get_files("models", ".pkl")
        data_list = get_files("data/processed", ".csv")
        
        with col_m1:
            model_a_name = st.selectbox("Model A (Champion)", models_list)
            if not models_list:
                st.warning("No models found in /models folder")
                
        with col_m2:
            idx_b = 1 if len(models_list) > 1 else 0
            model_b_name = st.selectbox("Model B (Challenger)", models_list, index=idx_b)
            
        with col_d:
            data_name = st.selectbox("Dataset", data_list)
            if not data_list:
                st.warning("No CSV data found in /data folder")

        win_metric = st.selectbox("Winning Metric", ["f1", "accuracy", "precision", "recall", "roc_auc"])
        btn_compare = st.button("Run Comparison", type="primary")
        
    # --- SESSION STATE INIT ---
    # Инициализируем переменную сессии, если её нет
    if 'comp_results' not in st.session_state:
        st.session_state.comp_results = None

    # --- BUTTON LOGIC ---
    if btn_compare:
        # 1. Validation Check
        if not model_a_name or not model_b_name or not data_name:
            st.error("Please select both models and a dataset before running.")
        else:
            st.toast("Process started...", icon="ℹ️")
            
            # Локальные переменные для расчетов
            error_msg = None
            m_a, m_b = None, None
            X_a, X_b = None, None
            pred_a, pred_b = np.array([]), np.array([])
            proba_a, proba_b = np.array([]), np.array([])
            met_a, met_b = {}, {}
            df = pd.DataFrame()
            y_true = None
            val_a, val_b = 0.0, 0.0
            winner = "A"

            # 3. ВСЕ ВЫЧИСЛЕНИЯ (внутри спиннера)
            with st.spinner("Processing data, comparing models, and detecting anomalies..."):
                try:
                    path_a = Path("models") / model_a_name
                    path_b = Path("models") / model_b_name
                    path_data = Path("data/processed") / data_name
                    
                    m_a = load_model_pipeline(path_a)
                    m_b = load_model_pipeline(path_b)
                    df = pd.read_csv(path_data).replace(-1, 0)
                    
                    if 'traffic_type' in df.columns:
                        y_true = df['traffic_type'].str.contains('VPN', na=False)

                    X_a = df[m_a['features']]
                    X_b = df[m_b['features']]
                    
                    pred_a = m_a['model'].predict(X_a)
                    pred_b = m_b['model'].predict(X_b)
                    
                    try:
                        proba_a = m_a['model'].predict_proba(X_a)[:, 1]
                        proba_b = m_b['model'].predict_proba(X_b)[:, 1]
                    except:
                        proba_a = np.array([0.5]*len(df))
                        proba_b = np.array([0.5]*len(df))

                    def get_metrics(y, p, prob):
                        if y is None: return {}
                        return {
                            "accuracy": accuracy_score(y, p),
                            "f1": f1_score(y, p, zero_division=0),
                            "precision": precision_score(y, p, zero_division=0),
                            "recall": recall_score(y, p, zero_division=0),
                            "roc_auc": roc_auc_score(y, prob) if prob is not None else 0.0
                        }
                    
                    met_a = get_metrics(y_true, pred_a, proba_a)
                    met_b = get_metrics(y_true, pred_b, proba_b) 
                    
                    val_a = met_a.get(win_metric, 0.0)
                    val_b = met_b.get(win_metric, 0.0)
                    winner = "A" if val_a >= val_b else "B"

                except Exception as e:
                    error_msg = str(e)
                    import traceback
                    st.error(f"Error: {traceback.format_exc()}")

            # 4. СОХРАНЕНИЕ В SESSION STATE
            if not error_msg:
                st.session_state.comp_results = {
                    'm_a': m_a, 'm_b': m_b,
                    'X_a': X_a, 'X_b': X_b,
                    'df': df, 'y_true': y_true,
                    'proba_a': proba_a, 'proba_b': proba_b,
                    'pred_a': pred_a, 'pred_b': pred_b,
                    'met_a': met_a, 'met_b': met_b,
                    'val_a': val_a, 'val_b': val_b,
                    'winner': winner, 'win_metric': win_metric,
                    'model_a_name': model_a_name, 'model_b_name': model_b_name
                }
            else:
                st.error(f"Critical Error: {error_msg}")
                st.session_state.comp_results = None

    # --- ОТРИСОВКА (Проверяет сессию, а не кнопку) ---
    # Теперь этот блок выполняется всегда, если данные есть в сессии
    if st.session_state.comp_results:
        res = st.session_state.comp_results
        
        # Распаковка для удобства
        m_a, m_b = res['m_a'], res['m_b']
        met_a, met_b = res['met_a'], res['met_b']
        val_a, val_b = res['val_a'], res['val_b']
        winner = res['winner']
        win_metric = res['win_metric']
        
        # 7. DISPLAY CARDS
        st.markdown(f"### Winner by {win_metric.upper()}: <span style='color:#2ecc71; font-weight:bold'>MODEL {winner}</span>", unsafe_allow_html=True)
        
        col_ca, col_cb = st.columns(2)
        
        def render_card(title, metrics, is_win, win_val, opp_val):
            css_class = "model-card winner-card" if is_win else "model-card loser-card"
            delta = win_val - opp_val
            sign = "+" if delta > 0 else ""
            color = "#2ecc71" if delta >= 0 else "#e74c3c"
            
            st.markdown(f"""
            <div class="{css_class}">
                <h3>{title}</h3>
                <div style='font-size: 2rem; font-weight: bold; margin: 10px 0;'>{metrics.get(win_metric, 0.0):.4f}</div>
                <div style='color: {color}; font-weight: 500;'>{sign}{delta:.4f} ({win_metric.upper()})</div>
                <hr style='border: 0; border-top: 1px solid #eee; margin: 15px 0;'>
                <div style='display: grid; grid-template-columns: 1fr 1fr 1fr; gap: 5px; font-size: 0.85rem; color: #555;'>
                    <div>Acc: {metrics.get('accuracy',0):.3f}</div>
                    <div>Prec: {metrics.get('precision',0):.3f}</div>
                    <div>Rec: {metrics.get('recall',0):.3f}</div>
                </div>
            </div>
            """, unsafe_allow_html=True)

        with col_ca:
            render_card(f"Model A: {m_a['algo_name']}", met_a, winner=="A", val_a, val_b)
        with col_cb:
            render_card(f"Model B: {m_b['algo_name']}", met_b, winner=="B", val_b, val_a)

        # 8. DETAILED ANALYTICS TABS
        t_prob, t_diff, t_xai = st.tabs(["Probability Distribution", "Disagreements / Anomalies", "XAI Explanation"])
        
        with t_prob:
            st.subheader("Prediction Confidence")
            df_plot = pd.DataFrame({'Model A': res['proba_a'], 'Model B': res['proba_b']}).melt(var_name='Model', value_name='Probability')
            fig = px.histogram(df_plot, x='Probability', color='Model', barmode='overlay', 
                            opacity=0.6, nbins=50, 
                            color_discrete_map={'Model A':'#3498db', 'Model B':'#e74c3c'})
            st.plotly_chart(fig, use_container_width=True)
        
        with t_diff:
            st.subheader("Disagreements and Potential Anomalies")
            mask = res['pred_a'] != res['pred_b']
            disagree_count = mask.sum()
            st.write(f"Found {disagree_count} disagreements (potential anomalies) out of {len(res['df'])} rows.")
            
            if disagree_count > 0:
                diff_df = res['df'][mask].copy()
                diff_df['Model_A_Pred'] = res['pred_a'][mask]
                diff_df['Model_B_Pred'] = res['pred_b'][mask]
                if res['y_true'] is not None:
                    diff_df['Real_Label'] = res['y_true'][mask]
                
                st.dataframe(diff_df.head(20), use_container_width=True)

        with t_xai:
            st.subheader("Explainable AI (SHAP)")
            st.caption("Explain a specific prediction from the selected Winner Model.")
            
            winner_model = m_a if winner == "A" else m_b
            winner_X = res['X_a'] if winner == "A" else res['X_b']
            
            if winner_X is not None and len(winner_X) > 0:
                row_idx = st.slider("Select Row Index to Explain", 0, len(winner_X)-1, 0)
                
                if st.button("Generate Explanation"):
                    with st.spinner("Calculating SHAP values (this may take a moment)..."):
                        try:
                            # 1. Обертка с фиксом конвертации типов
                            def model_wrapper(x):
                                # Если SHAP передает numpy массив, превращаем его в DataFrame
                                # чтобы пайплайн мог использовать .drop() и другие методы pandas
                                if isinstance(x, np.ndarray):
                                    x = pd.DataFrame(x, columns=winner_X.columns)
                                return winner_model['model'].predict_proba(x)[:, 1]

                            # 2. Используем KernelExplainer для стабильности с пайплайнами
                            # Ограничиваем количество фоновых данных для скорости
                            background_data = shap.sample(winner_X, 50) 
                            explainer = shap.KernelExplainer(model_wrapper, background_data)
                            
                            # 3. Вычисляем SHAP для одной строки
                            shap_values = explainer(winner_X.iloc[[row_idx]])

                            fig, ax = plt.subplots()
                            shap.plots.waterfall(shap_values[0], show=False)
                            st.pyplot(fig)
                            plt.close(fig)
                        except Exception as e:
                            st.error(f"SHAP generation failed: {e}")

# ======================== #
# TAB 3: RESEARCH GRAPH    #
# ======================== #
with tab_research:
    st.header("Model Ranking & Improvement Trend")
    
    df_hist = load_history("results")
    
    if not df_hist.empty:
        # 1. Выбор метрики
        metric = st.selectbox("Select Metric to Sort By", ["F1", "Accuracy", "ROC-AUC"], index=0)
        
        # 2. Достаем имя датасета
        df_hist['Dataset_Name'] = df_hist['Raw'].apply(lambda x: x.get('train_dataset', 'unknown'))
        
        # 3. Сортировка (Рейтинг)
        df_sorted = df_hist.sort_values(by=metric).reset_index(drop=True)
        
        # 4. Подписи
        def get_profile(run_name):
            parts = run_name.split('_')
            return parts[-1] if len(parts) > 1 else 'default'
            
        df_sorted['Label'] = df_sorted['Model'] + " (" + df_sorted['Run'].apply(get_profile) + ")"
        
        # 5. График
        # trendline_scope='traces' -> создает линию тренда ДЛЯ КАЖДОЙ МОДЕЛИ (цвет совпадает)
        fig = px.scatter(
            df_sorted,
            x=df_sorted.index,
            y=metric,
            text='Label',
            title=f"Model Performance Ladder ({metric})",
            color='Model',
            trendline="ols",
            trendline_scope="traces", # Ключевой момент: отдельный тренд для каждой группы
            hover_data=['Dataset_Name']
        )
        
        # 6. Настройка линий и точек через цикл
        for trace in fig.data:
            # Если в имени есть "Trend" — это линия аппроксимации
            if "Trend" in trace.name:
                trace.update(
                    mode='lines',         # Только линия
                    line=dict(width=4),   # Жирная
                    opacity=0.5           # Чуть прозрачнее, чтобы не путать с данными
                )
            # Иначе это основные данные (точки)
            else:
                trace.update(
                    mode='lines+markers+text', # Соединяем точки + сами точки + подписи
                    textposition='top center',
                    textfont=dict(size=14, color='black'), # Крупный черный текст
                    marker=dict(size=12, opacity=0.9),
                    line=dict(width=2)
                )

        fig.update_layout(
            xaxis_title="Experiment Rank (Worst -> Best)",
            yaxis_title=metric,
            showlegend=True
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
    else:
        st.info("No data. Train models first.")