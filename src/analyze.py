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
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import MinMaxScaler
from scipy.spatial import ConvexHull

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
    target_path = Path(__file__).parent.parent / folder
    if not target_path.exists():
        return []
    return [f.name for f in target_path.glob(f"*{extension}")]

@st.cache_data
def load_history(folder):
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

# Вспомогательная функция для отрисовки мягких облаков-кластеров
def add_cluster_hulls(fig, df, x_col, y_col, cluster_col='Cluster'):
    for cluster_id in df[cluster_col].unique():
        if cluster_id == -1: continue
        cluster_df = df[df[cluster_col] == cluster_id]
        if len(cluster_df) < 3: continue
        
        points = cluster_df[[x_col, y_col]].values
        try:
            hull = ConvexHull(points)
            hull_points = points[hull.vertices].tolist()
            hull_points.append(hull_points[0])
            
            x_hull, y_hull = zip(*hull_points)
            fig.add_trace(go.Scatter(
                x=x_hull, y=y_hull, fill='toself',
                fillcolor='rgba(180, 180, 180, 0.1)',
                line=dict(color='rgba(100, 100, 100, 0.3)', width=1, dash='dot'),
                name=f'Кластер {cluster_id}', hoverinfo='skip'
            ))
        except Exception:
            pass

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
# TAB 2: MODEL COMPARISON
# ==========================================
with tab_comparison:
    st.header("Model Comparison (A vs B)")
    st.caption("Select two models and a dataset to compare performance.")
    
    with st.expander("Configuration", expanded=True):
        col_m1, col_m2, col_d = st.columns(3)
        models_list = get_files("models", ".pkl")
        data_list = get_files("data/processed", ".csv")
        
        with col_m1:
            model_a_name = st.selectbox("Model A (Champion)", models_list)
        with col_m2:
            idx_b = 1 if len(models_list) > 1 else 0
            model_b_name = st.selectbox("Model B (Challenger)", models_list, index=idx_b)
        with col_d:
            data_name = st.selectbox("Dataset", data_list)

        win_metric = st.selectbox("Winning Metric", ["f1", "accuracy", "precision", "recall", "roc_auc"])
        btn_compare = st.button("Run Comparison", type="primary")
        
    if 'comp_results' not in st.session_state:
        st.session_state.comp_results = None

    if btn_compare:
        if not model_a_name or not model_b_name or not data_name:
            st.error("Please select both models and a dataset before running.")
        else:
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
                            "accuracy": accuracy_score(y, p), "f1": f1_score(y, p, zero_division=0),
                            "precision": precision_score(y, p, zero_division=0), "recall": recall_score(y, p, zero_division=0),
                            "roc_auc": roc_auc_score(y, prob) if prob is not None else 0.0
                        }
                    
                    met_a = get_metrics(y_true, pred_a, proba_a)
                    met_b = get_metrics(y_true, pred_b, proba_b) 
                    
                    val_a = met_a.get(win_metric, 0.0)
                    val_b = met_b.get(win_metric, 0.0)
                    winner = "A" if val_a >= val_b else "B"

                    st.session_state.comp_results = {
                        'm_a': m_a, 'm_b': m_b, 'X_a': X_a, 'X_b': X_b,
                        'df': df, 'y_true': y_true, 'proba_a': proba_a, 'proba_b': proba_b,
                        'pred_a': pred_a, 'pred_b': pred_b, 'met_a': met_a, 'met_b': met_b,
                        'val_a': val_a, 'val_b': val_b, 'winner': winner, 'win_metric': win_metric,
                        'model_a_name': model_a_name, 'model_b_name': model_b_name
                    }
                except Exception as e:
                    import traceback
                    st.error(f"Error: {traceback.format_exc()}")
                    st.session_state.comp_results = None

    if st.session_state.comp_results:
        res = st.session_state.comp_results
        m_a, m_b = res['m_a'], res['m_b']
        met_a, met_b = res['met_a'], res['met_b']
        val_a, val_b = res['val_a'], res['val_b']
        winner = res['winner']
        win_metric = res['win_metric']
        
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

        t_prob, t_diff, t_xai = st.tabs(["Probability Distribution", "Disagreements / Anomalies", "XAI Explanation"])
        
        with t_prob:
            df_plot = pd.DataFrame({'Model A': res['proba_a'], 'Model B': res['proba_b']}).melt(var_name='Model', value_name='Probability')
            fig = px.histogram(df_plot, x='Probability', color='Model', barmode='overlay', opacity=0.6, nbins=50, color_discrete_map={'Model A':'#3498db', 'Model B':'#e74c3c'})
            st.plotly_chart(fig, use_container_width=True)
        
        with t_diff:
            mask = res['pred_a'] != res['pred_b']
            if mask.sum() > 0:
                diff_df = res['df'][mask].copy()
                diff_df['Model_A_Pred'] = res['pred_a'][mask]
                diff_df['Model_B_Pred'] = res['pred_b'][mask]
                if res['y_true'] is not None: diff_df['Real_Label'] = res['y_true'][mask]
                st.dataframe(diff_df.head(20), use_container_width=True)

        with t_xai:
            winner_model = m_a if winner == "A" else m_b
            winner_X = res['X_a'] if winner == "A" else res['X_b']
            if winner_X is not None and len(winner_X) > 0:
                row_idx = st.slider("Select Row Index to Explain", 0, len(winner_X)-1, 0)
                if st.button("Generate Explanation"):
                    with st.spinner("Calculating SHAP values..."):
                        try:
                            def model_wrapper(x):
                                if isinstance(x, np.ndarray): x = pd.DataFrame(x, columns=winner_X.columns)
                                return winner_model['model'].predict_proba(x)[:, 1]
                            background_data = shap.sample(winner_X, 50) 
                            explainer = shap.KernelExplainer(model_wrapper, background_data)
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
        
        def count_important_features(raw_data):
            fi = raw_data.get('feature_importance_full', [])
            if not fi: return 0
            return sum(1 for f in fi if f.get('importance', 0) > 0.01)
        df_hist['Num_Features'] = df_hist['Raw'].apply(count_important_features)
        
        df_hist = df_hist.dropna(subset=['Timestamp']).sort_values('Timestamp').reset_index(drop=True)

        def get_model_family(model_name):
            model_upper = model_name.upper()
            if 'XGB' in model_upper: return 'XGBoost'
            if 'STACK' in model_upper: return 'Стекинг'
            if 'BAG' in model_upper or 'RF' in model_upper or 'DT' in model_upper: return 'Деревья и Бэггинг'
            if 'GB' in model_upper: return 'Gradient Boosting'
            if 'LR' in model_upper: return 'Линейные (LR)'
            return model_name

        df_hist['Model_Family'] = df_hist['Model'].apply(get_model_family)
        df_hist_no_lr = df_hist[df_hist['Model_Family'] != 'Линейные (LR)'].copy()

        family_shapes = {'XGBoost': 'circle', 'Стекинг': 'star', 'Gradient Boosting': 'square', 'Деревья и Бэггинг': 'diamond'}
        family_colors = {'XGBoost': '#636EFA', 'Стекинг': '#FF6692', 'Gradient Boosting': '#FFA15A', 'Деревья и Бэггинг': '#B6E880'}

        def render_tall_chart(fig, height=700, width=800):
            fig.update_layout(height=height, width=width)
            col1, col2, col3 = st.columns([1, 3, 1])
            with col2: st.plotly_chart(fig, use_container_width=False)

        # ========================================== #
        # ГРАФИК 1: Архитектура vs данные
        # ========================================== #
        st.subheader("Доказательство тезиса: Архитектура первична, данные вторичны")
        st.caption("Жирная линия — медиана. Цвет точки — отклонение от медианы (Зеленый = лучше, Красный = профиль вредит). Узкий разброс = модель стабильна.")

        families = df_hist_no_lr['Model_Family'].unique().tolist()
        family_to_num = {f: i for i, f in enumerate(families)}
        
        df_hist_no_lr['f1_median'] = df_hist_no_lr.groupby('Model_Family')['f1_score'].transform('median')
        df_hist_no_lr['f1_delta'] = df_hist_no_lr['f1_score'] - df_hist_no_lr['f1_median']
        
        np.random.seed(42)
        df_hist_no_lr['x_jitter'] = df_hist_no_lr['Model_Family'].map(family_to_num) + np.random.uniform(-0.2, 0.2, len(df_hist_no_lr))

        fig_f1 = go.Figure()

        for family, x_num in family_to_num.items():
            med = df_hist_no_lr[df_hist_no_lr['Model_Family'] == family]['f1_median'].iloc[0]
            fig_f1.add_trace(go.Scatter(
                x=[x_num - 0.3, x_num + 0.3], y=[med, med],
                mode='lines', line=dict(color='black', width=4), showlegend=False
            ))

        fig_f1.add_trace(go.Scatter(
            x=df_hist_no_lr['x_jitter'], y=df_hist_no_lr['f1_score'], mode='markers',
            marker=dict(
                size=12, color=df_hist_no_lr['f1_delta'], colorscale='RdYlGn',
                line=dict(width=1, color='black'),
                symbol=df_hist_no_lr['Model_Family'].map(family_shapes).fillna('circle'),
                colorbar=dict(title="Delta F1", thickness=10, len=0.5)
            ),
            hovertext=df_hist_no_lr.apply(lambda x: f"{x['Model']}<br>Profile: {x['Profile']}<br>F1: {x['f1_score']:.4f}", axis=1)
        ))

        fig_f1.update_xaxes(tickvals=list(family_to_num.values()), ticktext=list(family_to_num.keys()), title_text="Семейство алгоритмов")
        fig_f1.update_yaxes(title_text="F1 Score")
        render_tall_chart(fig_f1, height=700, width=800)
        st.divider()

        # ========================================== #
        # ГРАФИК 2: Precision vs Recall (Density + Adaptive Jitter)
        # ========================================== #
        st.subheader("Матрица ошибок: Trade-off Precision vs Recall")
        st.caption("Фон — плотность моделей. Джиттер и прозрачность помогают рассмотреть слипшиеся точки. Топ-3 по F1 выделены.")

        scale_r = df_hist_no_lr['recall'].std() * 0.2
        scale_p = df_hist_no_lr['precision'].std() * 0.2
        np.random.seed(42)
        df_hist_no_lr['recall_j'] = df_hist_no_lr['recall'] + np.random.normal(0, max(scale_r, 0.005), len(df_hist_no_lr))
        df_hist_no_lr['precision_j'] = df_hist_no_lr['precision'] + np.random.normal(0, max(scale_p, 0.005), len(df_hist_no_lr))

        fig_pr = go.Figure()

        fig_pr.add_trace(go.Histogram2dContour(
            x=df_hist_no_lr['recall'], 
            y=df_hist_no_lr['precision'],
            colorscale='Blues',
            showscale=False,
            opacity=0.4,
            hoverinfo='skip'
        ))

        fig_pr.add_shape(type="rect", x0=0.95, y0=0.95, x1=1.01, y1=1.01, fillcolor="rgba(46, 204, 113, 0.15)", line_width=0)
        fig_pr.add_annotation(x=0.98, y=0.955, text="Utopia Zone", showarrow=False, font=dict(color="green", size=10))

        fig_pr.add_trace(go.Scatter(
            x=df_hist_no_lr['recall_j'], y=df_hist_no_lr['precision_j'], mode='markers',
            marker=dict(size=8, color='rgba(150, 150, 150, 0.6)', symbol=df_hist_no_lr['Model_Family'].map(family_shapes).fillna('circle')),
            hovertext=df_hist_no_lr['Model'], showlegend=False
        ))

        if len(df_hist_no_lr) >= 3:
            top_3 = df_hist_no_lr.nlargest(3, 'f1_score')
            fig_pr.add_trace(go.Scatter(
                x=top_3['recall_j'], y=top_3['precision_j'], mode='markers+text',
                marker=dict(size=14, color='#2ecc71', symbol=top_3['Model_Family'].map(family_shapes).fillna('circle'), line=dict(width=2, color='black')),
                text=top_3['Model'], textposition='top center', textfont=dict(color='black', size=11, weight='bold'),
                hovertext=top_3['Run'], name='Top 3 F1'
            ))

        min_r = 0.8
        min_p = 0.8
        fig_pr.update_xaxes(range=[min_r, 1.005], dtick=0.05, constrain="domain", title_text="Recall")
        fig_pr.update_yaxes(range=[min_p, 1.005], dtick=0.05, scaleanchor="x", scaleratio=1, constrain="domain", title_text="Precision")
        
        fig_pr.add_shape(type="line", x0=min_r, y0=min_p, x1=1.005, y1=1.005, line=dict(color="gray", width=2, dash="dash"))
        fig_pr.update_layout(width=750, height=700, legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1))
        
        col1, col2, col3 = st.columns([1, 3, 1])
        with col2: st.plotly_chart(fig_pr, use_container_width=False)
        st.divider()
        # ========================================== #
        # ГРАФИК 3: Heatmap (DELTA F1 - Убийца шоколадки)
        # ========================================== #
        st.subheader("Влияние препроцессинга: Delta F1 от профиля данных")
        st.caption("Показывает не средний скор, а НАСКОЛЬКО профиль улучшает/ухудшает модель относительно её базовой линии. Красный = вредит, Зеленый = помогает. Пустые клетки = нет эксперимента.")

        pivot_df = df_hist_no_lr.groupby(['Model_Family', 'Profile'])['f1_score'].mean().reset_index()
        heatmap_f1 = pivot_df.pivot(index='Model_Family', columns='Profile', values='f1_score')
        
        # 1. Удаляем строки/столбцы, где вообще нет данных (полные NaN)
        heatmap_f1 = heatmap_f1.dropna(axis=0, how='all').dropna(axis=1, how='all')
        
        # 2. Считаем базовую линию (среднее по доступным профилям для модели)
        baseline = heatmap_f1.mean(axis=1)
        
        # 3. Считаем Дельту (отклонение от базовой линии)
        heatmap_delta = heatmap_f1.sub(baseline, axis=0)
        
        # 4. Сортировка по качеству (baseline)
        heatmap_delta = heatmap_delta.loc[baseline.sort_values(ascending=False).index]

        # 5. Формируем текст (показываем саму дельту со знаком + или -)
        text_matrix = heatmap_delta.applymap(lambda x: f"{x:+.4f}" if not pd.isna(x) else "")
        
        # Выделяем лучший профиль (максимальная дельта)
        for family in heatmap_delta.index:
            valid_deltas = heatmap_delta.loc[family].dropna()
            if not valid_deltas.empty:
                best_prof = valid_deltas.idxmax()
                val = valid_deltas[best_prof]
                text_matrix.loc[family, best_prof] = f"⭐ {val:+.4f}"

        # Строим график
        fig_prof = px.imshow(
            heatmap_delta.values, 
            x=heatmap_delta.columns, 
            y=heatmap_delta.index,
            color_continuous_scale='RdYlGn', # Красный - Плохо, Зеленый - Хорошо
            color_continuous_midpoint=0, # ИСПРАВЛЕНИЕ ТУТ: центрируем шкалу относительно 0
            aspect="auto"
        )
        
        fig_prof.update_traces(
            text=text_matrix.values, 
            texttemplate="%{text}", 
            textfont=dict(size=12),
            hoverongaps=False # Не наводить мышку на пустые клетки
        )

        fig_prof.update_layout(coloraxis_colorbar=dict(title="Δ F1", thickness=15))
        fig_prof.update_xaxes(side="top", title_text="Профиль данных (Препроцессинг)")
        fig_prof.update_yaxes(title_text="Семейство алгоритмов")
        render_tall_chart(fig_prof, height=500, width=800)
        st.divider()

        # ========================================== #
        # ГРАФИК 4: F1 vs количество признаков (Тренд)
        # ========================================== #
        st.subheader("Архитектурная спарсификация: Efficiency (Качество / Сложность)")
        st.caption("Красная линия — тренд. Ищите модели выше линии (дают больше качества за меньшее число признаков).")

        fig_sparse = go.Figure()

        if len(df_hist_no_lr) >= 4:
            x_sp = df_hist_no_lr['Num_Features'].values
            y_sp = df_hist_no_lr['f1_score'].values
            try:
                z = np.polyfit(x_sp, y_sp, 2)
                p = np.poly1d(z)
                x_trend = np.linspace(x_sp.min(), x_sp.max(), 100)
                fig_sparse.add_trace(go.Scatter(x=x_trend, y=p(x_trend), mode='lines', name='Trend', line=dict(color='red', width=2, dash='dash')))
            except: pass

        for family_name, group_df in df_hist_no_lr.groupby('Model_Family'):
            fig_sparse.add_trace(go.Scatter(
                x=group_df['Num_Features'], y=group_df['f1_score'], mode='markers',
                marker=dict(size=12, color=family_colors.get(family_name, '#999999'), line=dict(width=1, color='black'), symbol=family_shapes.get(family_name, 'circle')),
                name=family_name, hovertext=group_df['Model']
            ))

        fig_sparse.update_xaxes(title_text="Количество значимых признаков")
        fig_sparse.update_yaxes(title_text="F1 Score")
        render_tall_chart(fig_sparse, height=600, width=800)
        st.divider()

        # ========================================== #
        # ГРАФИК 5: Граница Парето
        # ========================================== #
        st.subheader("Идеальный компромисс: Pareto Front & Optimal Choice")
        st.caption("Серые модели можно игнорировать. Звезда — точка излома (Optimal), где дальнейшее усложнение модели не окупается.")

        pareto_df = df_hist_no_lr.sort_values(by='Num_Features')
        best_f1 = -1
        pareto_points = []
        for _, row in pareto_df.iterrows():
            if row['f1_score'] > best_f1:
                pareto_points.append(row)
                best_f1 = row['f1_score']
                
        if pareto_points:
            pareto_df_final = pd.DataFrame(pareto_points)
            fig_pareto = go.Figure()
            
            fig_pareto.add_trace(go.Scatter(
                x=df_hist_no_lr['Num_Features'], y=df_hist_no_lr['f1_score'], mode='markers',
                marker=dict(size=8, color='rgba(200,200,200,0.3)'), name='Sub-optimal', hovertext=df_hist_no_lr['Model'], showlegend=False
            ))
                
            fig_pareto.add_trace(go.Scatter(
                x=pareto_df_final['Num_Features'], y=pareto_df_final['f1_score'], mode='lines+markers+text',
                line=dict(color='#2ecc71', width=3),
                marker=dict(size=10, color='#2ecc71', line=dict(width=1, color='black')),
                text=pareto_df_final['Model'], textposition='top center', textfont=dict(size=10, weight='bold'),
                name='Pareto Front', hovertext=pareto_df_final['Run']
            ))

            if len(pareto_df_final) >= 2:
                pareto_df_final['f1_diff'] = pareto_df_final['f1_score'].diff().fillna(0)
                pareto_df_final['feat_diff'] = pareto_df_final['Num_Features'].diff().fillna(1)
                pareto_df_final['efficiency'] = pareto_df_final.apply(lambda x: x['f1_diff'] / x['feat_diff'] if x['feat_diff'] > 0 else 0, axis=1)
                if len(pareto_df_final) > 1:
                    elbow_idx = pareto_df_final['efficiency'].iloc[1:].idxmin()
                    elbow_row = pareto_df_final.loc[elbow_idx]
                    fig_pareto.add_trace(go.Scatter(
                        x=[elbow_row['Num_Features']], y=[elbow_row['f1_score']], mode='markers',
                        marker=dict(size=20, color='red', symbol='star', line=dict(width=2, color='darkred')),
                        name='Optimal Choice', hovertext=elbow_row['Model']
                    ))
            
            fig_pareto.update_xaxes(title_text="Количество значимых признаков")
            fig_pareto.update_yaxes(title_text="F1 Score")
            fig_pareto.update_layout(legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1))
            render_tall_chart(fig_pareto, height=600, width=800)
        st.divider()

        # ========================================== #
        # ГРАФИК 6: Latency vs F1 (Production Score)
        # ========================================== #
        st.subheader("Production Ready: Время отклика vs Качество")
        st.caption("Ось X — логарифмическая. Золотая звезда — модель с максимальным Production Score (F1 / Latency).")

        data_list = get_files("data/processed", ".csv")
        models_list = get_files("models", ".pkl")
        
        if data_list and models_list:
            latency_data_name = st.selectbox("Датасет для бенчмарка задержки", data_list, key="latency_data_sel")
            
            if st.button("Запустить замер задержки (Inference Time)", type="primary"):
                with st.spinner("Прогоняем модели на 1 строке данных 100 раз..."):
                    import time
                    path_d = Path("data/processed") / latency_data_name
                    df_bench = pd.read_csv(path_d).replace(-1, 0).head(1)
                    
                    latency_results = []
                    for model_name in models_list:
                        try:
                            model_data = load_model_pipeline(Path("models") / model_name)
                            X_bench = df_bench[model_data['features']]
                            for _ in range(10): model_data['model'].predict(X_bench)
                            start_time = time.perf_counter()
                            for _ in range(100): model_data['model'].predict(X_bench)
                            elapsed_ms = (time.perf_counter() - start_time) / 100 * 1000
                            
                            hist_row = df_hist_no_lr[df_hist_no_lr['Run'] == model_name.replace('.pkl', '')]
                            f1_val = hist_row['f1_score'].iloc[0] if not hist_row.empty else 0
                            fam = hist_row['Model_Family'].iloc[0] if not hist_row.empty else get_model_family(model_name)
                            n_feat = hist_row['Num_Features'].iloc[0] if not hist_row.empty else len(model_data['features'])
                            
                            latency_results.append({"Model": model_name.replace('.pkl', ''), "Latency_ms": elapsed_ms, "F1": f1_val, "Family": fam, "Num_Features": n_feat})
                        except: pass
                    
                    if latency_results:
                        df_lat = pd.DataFrame(latency_results)
                        df_lat['Score'] = df_lat['F1'] / (df_lat['Latency_ms'] + 1e-6)
                        
                        fig_lat = go.Figure()

                        fig_lat.add_vrect(x0=0, x1=5, fillcolor="rgba(46, 204, 113, 0.1)", line_width=0)
                        fig_lat.add_annotation(x=2, y=df_lat['F1'].max(), text="Real-time", showarrow=False, font=dict(color="green", size=10))
                        fig_lat.add_vrect(x0=5, x1=50, fillcolor="rgba(241, 196, 15, 0.1)", line_width=0)
                        fig_lat.add_annotation(x=15, y=df_lat['F1'].max(), text="API", showarrow=False, font=dict(color="orange", size=10))
                        fig_lat.add_vrect(x0=50, x1=df_lat['Latency_ms'].max()+10, fillcolor="rgba(231, 76, 60, 0.1)", line_width=0)
                        fig_lat.add_annotation(x=60, y=df_lat['F1'].max(), text="Batch", showarrow=False, font=dict(color="red", size=10))

                        fig_lat.add_trace(go.Scatter(
                            x=df_lat['Latency_ms'], y=df_lat['F1'], mode='markers',
                            marker=dict(
                                size=df_lat['Num_Features'].apply(lambda x: max(8, x/2)),
                                color=df_lat['Family'].map(family_colors),
                                symbol=df_lat['Family'].map(family_shapes).fillna('circle'),
                                line=dict(width=1, color='black'), opacity=0.8
                            ),
                            hovertext=df_lat.apply(lambda x: f"{x['Model']}<br>Feats: {x['Num_Features']}<br>Score: {x['Score']:.2f}", axis=1),
                            name='Models'
                        ))

                        champion = df_lat.loc[df_lat['Score'].idxmax()]
                        fig_lat.add_trace(go.Scatter(
                            x=[champion['Latency_ms']], y=[champion['F1']], mode='markers+text',
                            marker=dict(size=20, color='gold', symbol='star', line=dict(width=2, color='black')),
                            text=['PROD CANDIDATE'], textposition='top center', textfont=dict(size=12, weight='bold'),
                            showlegend=False
                        ))

                        fig_lat.update_xaxes(type='log', title_text="Время инференса (мс, лог. шкала)")
                        fig_lat.update_yaxes(title_text="F1 Score")
                        render_tall_chart(fig_lat, height=600, width=800)
        st.divider()

        # ========================================== #
        # БЛОК 7: Инъекция аномалий
        # ========================================== #
        st.subheader("Стресс-тестирование: All vs All Матрица Робастности")
        st.caption("Автоматический прогон искажений. Показывает, при каком уровне яда архитектура начинает сыпаться.")
        
        with st.expander("Настройки стресс-теста", expanded=True):
            stress_data_name = st.selectbox("Целевой датасет", data_list, key="stress_data_sel")
            scenario = st.selectbox("Сценарий искажения", [
                "Сценарий 1: Инфляция максимальных значений (max_*)",
                "Сценарий 2: Дефляция минимальных значений (min_*)"
            ], key="stress_scen_sel")

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
                                    for col in max_cols: df_distorted[col] = np.maximum(X_clean[col] * mult, X_clean[col].mean())
                                else:
                                    min_cols = [c for c in X_clean.columns if c.startswith('min_')]
                                    for col in min_cols: df_distorted[col] = np.minimum(X_clean[col] / mult, X_clean[col].mean())
                                        
                                X_distorted = df_distorted[model_data['features']]
                                pred_distorted = model_data['model'].predict(X_distorted)
                                f1_distorted = f1_score(y_true, pred_distorted, zero_division=0) if y_true is not None else 0.0
                                
                                results_stress.append({"Family": family, "Multiplier": f"x{int(mult)}", "Delta_F1": f1_clean - f1_distorted})
                        except: pass

                    df_res = pd.DataFrame(results_stress)
                    if not df_res.empty:
                        pivot_stress = df_res.groupby(['Family', 'Multiplier'])['Delta_F1'].mean().reset_index()
                        col_order = [f"x{int(m)}" for m in multipliers]
                        heatmap_data = pivot_stress.pivot(index='Family', columns='Multiplier', values='Delta_F1').reindex(columns=col_order)
                        fig_stress = px.imshow(
                            heatmap_data.values, labels=dict(x="Сила искажения", y="Семейство", color="Деградация F1"),
                            x=heatmap_data.columns, y=heatmap_data.index, text_auto=".4f", aspect="auto", color_continuous_scale="Reds"
                        )
                        fig_stress.update_xaxes(side="top")
                        render_tall_chart(fig_stress, height=600, width=800)
    else:
        st.info("Нет данных. Сначала обучите модели.")
