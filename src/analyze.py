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
import plotly.graph_objects as go
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
# TAB 3: RESEARCH LAB      #
# ======================== #
with tab_research:
    st.header("Research Lab: Архитектура vs Данные")
    
    df_hist = load_history("results")
    
    if not df_hist.empty:
        metrics_list = ['accuracy', 'precision', 'recall', 'f1_score']
        for m in metrics_list:
            df_hist[m] = df_hist['Raw'].apply(lambda x: x['metrics'].get(m, 0))
            
        df_hist['Profile'] = df_hist['Run'].apply(lambda x: x.split('_', 1)[1] if '_' in x else 'default')
        df_hist['Dataset'] = df_hist['Raw'].apply(lambda x: x.get('train_dataset', 'unknown'))
        df_hist['Timestamp'] = pd.to_datetime(df_hist['Timestamp'], errors='coerce')
        df_hist = df_hist.dropna(subset=['Timestamp']).sort_values('Timestamp').reset_index(drop=True)

        def get_model_family(model_name):
            model_upper = model_name.upper()
            if 'XGB' in model_upper: return 'XGBoost'
            if 'STACK' in model_upper: return 'Стекинг'
            if 'GB' in model_upper: return 'Gradient Boosting'
            if 'LR' in model_upper: return 'Линейные (LR)'
            if 'RF' in model_upper or 'DT' in model_upper or 'BAG' in model_upper: return 'Деревья (RF/DT)'
            return model_name

        df_hist['Model_Family'] = df_hist['Model'].apply(get_model_family)
        df_hist['Utopia_Dist'] = np.sqrt((1 - df_hist['precision'])**2 + (1 - df_hist['recall'])**2)

        # ГЛОБАЛЬНЫЙ ФИЛЬТР: Убираем LR из анализа архитектур, чтобы не ломать масштаб
        df_hist_no_lr = df_hist[df_hist['Model_Family'] != 'Линейные (LR)'].copy()

        # Вспомогательная функция для центрирования
        def render_tall_chart(fig, height=700, width=800):
            fig.update_layout(height=height, width=width)
            col1, col2, col3 = st.columns([1, 3, 1])
            with col2:
                st.plotly_chart(fig, use_container_width=False)

        # ========================================== #
        # ГРАФИК 1: Архитектура vs Данные (F1) - БЕЗ LR
        # ========================================== #
        st.subheader("Доказательство тезиса: Архитектура первична, данные вторичны")
        st.caption("Каждая точка — запуск с профилем препроцессинга. Компактный кластер = модель игнорирует шум в данных.")

        fig_f1 = px.strip(
            df_hist_no_lr, 
            x="Model_Family", 
            y="f1_score", 
            color="Profile", 
            stripmode="overlay",
            title="Распределение F1 Score по Семействам (цвет = профиль)"
        )
        fig_f1.add_trace(px.box(df_hist_no_lr, x="Model_Family", y="f1_score").data[0])
        fig_f1.update_traces(marker=dict(opacity=0.7, size=8), selector=dict(type='box'))
        render_tall_chart(fig_f1, height=700, width=800)
        st.divider()

        # ========================================== #
        # ГРАФИК 2: Микроскоп P vs R (Квадрат 1:1)
        # ========================================== #
        st.subheader("Матрица ошибок: Микроскоп Precision vs Recall")
        st.caption("Физические оси равны (1:1). Масштаб 0.8–1.0. Диагональ — баланс смещения ошибок (FP vs FN).")

        fig_pr = px.scatter(
            df_hist_no_lr, 
            x="recall", 
            y="precision", 
            color="Model_Family",  
            symbol="Profile",     
            hover_data={"Model": True, "Run": True}
        )
        
        fig_pr.update_xaxes(range=[0.8, 1.0], dtick=0.05)
        fig_pr.update_yaxes(range=[0.8, 1.0], dtick=0.05)

        fig_pr.add_shape(type="line", x0=0.8, y0=0.8, x1=1.0, y1=1.0,
                         line=dict(color="gray", width=2, dash="dash"))
        
        fig_pr.update_traces(marker=dict(size=12))
        
        fig_pr.update_layout(
            width=700, 
            height=700, 
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
            margin=dict(l=60, r=20, b=40, t=80)
        )
        
        col1, col2, col3 = st.columns([1, 3, 1])
        with col2:
            st.plotly_chart(fig_pr, use_container_width=False)
        st.divider()

        # ========================================== #
        # ГРАФИК 3: Boxplot компромисса - БЕЗ LR
        # ========================================== #
        st.subheader("Метрика компромисса: $D = \\sqrt{(1-P)^2 + (1-R)^2}$")
        st.caption("Низкий и узкий ящик = модель стабильна и близка к идеалу.")

        fig_dist = px.strip(df_hist_no_lr, x="Model_Family", y="Utopia_Dist", color="Profile")
        fig_dist.add_trace(px.box(df_hist_no_lr, x="Model_Family", y="Utopia_Dist").data[0])
        fig_dist.update_traces(marker=dict(size=8, opacity=0.7), selector=dict(type='box'))
        fig_dist.update_yaxes(title_text="Distance to (1, 1) [меньше = лучше]")
        render_tall_chart(fig_dist, height=700, width=800)
        st.divider()

        # ========================================== #
        # ГРАФИК 4: Журнал исследований (SOTA Progression)
        # ========================================== #
        st.subheader("Журнал исследований: Эволюция SOTA (State-of-the-Art)")
        st.caption("Линия показывает рекордную метрику F1 на момент эксперимента. Скачки наверх — смена архитектуры, давшая прорыв.")

        df_sota = df_hist_no_lr.sort_values('Timestamp').copy()
        df_sota['SOTA_F1'] = df_sota['f1_score'].cummax()
        
        df_sota['Is_New_Record'] = df_sota['f1_score'] >= df_sota['SOTA_F1']
        df_records = df_sota[df_sota['Is_New_Record']].drop_duplicates(subset=['SOTA_F1'], keep='first')

        fig_journal = px.scatter(
            df_sota,
            x="Timestamp",
            y="f1_score",
            color="Model_Family",
            symbol="Profile",
            hover_data=["Run", "Model"],
            title="История экспериментов: Фоновый шум vs Рекорды",
            opacity=0.4 
        )
        
        # Используем импортированный go
        fig_journal.add_trace(go.Scatter(
            x=df_records['Timestamp'],
            y=df_records['SOTA_F1'],
            line=dict(color="black", width=3, dash="dash"),
            mode="lines+markers",
            name="SOTA F1 Record",
            marker=dict(size=10, symbol="star", color="black")
        ))
        
        for _, row in df_records.iterrows():
            fig_journal.add_annotation(
                x=row['Timestamp'], y=row['SOTA_F1'],
                text=f"{row['Model_Family']}<br>({row['Profile']})",
                showarrow=True, arrowhead=1, ax=20, ay=-30,
                font=dict(size=10, color="black", family="Courier New")
            )

        fig_journal.update_yaxes(title_text="F1 Score")
        render_tall_chart(fig_journal, height=700, width=800)
        st.divider()

        # ========================================== #
        # БЛОК 5: Инъекция аномалий (All vs All Matrix)
        # ========================================== #
        st.subheader("Стресс-тестирование: All vs All Матрица Робастности")
        st.caption("Автоматический прогон искажений с разной силой. Показывает, при каком уровне яда архитектура начинает сыпаться.")
        
        with st.expander("Настройки стресс-теста", expanded=True):
            stress_data_name = st.selectbox("Целевой датасет", get_files("data/processed", ".csv"), key="stress_data_sel")
            scenario = st.selectbox("Сценарий искажения", [
                "Сценарий 1: Инфляция максимальных значений (max_*)",
                "Сценарий 2: Дефляция минимальных значений (min_*)"
            ], key="stress_scen_sel")

        models_list = get_files("models", ".pkl")
        
        if stress_data_name and models_list:
            if st.button("Запустить мульти-стресс-тест", type="primary"):
                with st.spinner("Вычисление деградации на 3 уровнях искажения..."):
                    path_d = Path("data/processed") / stress_data_name
                    df_clean = pd.read_csv(path_d).replace(-1, 0)
                    y_true = df_clean['traffic_type'].str.contains('VPN', na=False) if 'traffic_type' in df_clean.columns else None
                    
                    results_stress = []
                    multipliers = [2.0, 5.0, 10.0]
                    
                    for model_name in models_list:
                        try:
                            model_data = load_model_pipeline(Path("models") / model_name)
                            X_clean = df_clean[model_data['features']]
                            
                            pred_clean = model_data['model'].predict(X_clean)
                            f1_clean = f1_score(y_true, pred_clean, zero_division=0) if y_true is not None else 0.0
                            
                            algo_key = model_name.replace('.pkl', '').split('_')[0]
                            family = get_model_family(algo_key)
                            
                            for mult in multipliers:
                                df_distorted = df_clean.copy()
                                if "Сценарий 1" in scenario:
                                    max_cols = [c for c in X_clean.columns if c.startswith('max_')]
                                    for col in max_cols:
                                        df_distorted[col] = np.maximum(X_clean[col] * mult, X_clean[col].mean())
                                else:
                                    min_cols = [c for c in X_clean.columns if c.startswith('min_')]
                                    for col in min_cols:
                                        df_distorted[col] = np.minimum(X_clean[col] / mult, X_clean[col].mean())
                                        
                                X_distorted = df_distorted[model_data['features']]
                                pred_distorted = model_data['model'].predict(X_distorted)
                                f1_distorted = f1_score(y_true, pred_distorted, zero_division=0) if y_true is not None else 0.0
                                
                                results_stress.append({
                                    "Family": family,
                                    "Multiplier": f"x{int(mult)}",
                                    "Delta_F1": f1_clean - f1_distorted
                                })
                        except Exception as e:
                            pass

                    df_res = pd.DataFrame(results_stress)
                    
                    if not df_res.empty:
                        pivot_stress = df_res.groupby(['Family', 'Multiplier'])['Delta_F1'].mean().reset_index()
                        
                        col_order = [f"x{int(m)}" for m in multipliers]
                        heatmap_data = pivot_stress.pivot(index='Family', columns='Multiplier', values='Delta_F1')
                        heatmap_data = heatmap_data.reindex(columns=col_order)

                        fig_stress = px.imshow(
                            heatmap_data.values,
                            labels=dict(x="Сила искажения", y="Семейство", color="Деградация F1"),
                            x=heatmap_data.columns,
                            y=heatmap_data.index,
                            text_auto=".4f",
                            aspect="auto",
                            color_continuous_scale="Reds"
                        )
                        fig_stress.update_xaxes(side="top")
                        render_tall_chart(fig_stress, height=600, width=800)
    else:
        st.info("Нет данных. Сначала обучите модели.")