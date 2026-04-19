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
            # BAG проверяется ДО GB, чтобы bag_dt не попал в Gradient Boosting
            if 'BAG' in model_upper or 'RF' in model_upper or 'DT' in model_upper: return 'Деревья и Бэггинг'
            if 'GB' in model_upper: return 'Gradient Boosting'
            if 'LR' in model_upper: return 'Линейные (LR)'
            return model_name

        df_hist['Model_Family'] = df_hist['Model'].apply(get_model_family)
        df_hist_no_lr = df_hist[df_hist['Model_Family'] != 'Линейные (LR)'].copy()

        # Маппинг фигур и цветов
        family_shapes = {'XGBoost': 'circle', 'Стекинг': 'star', 'Gradient Boosting': 'square', 'Деревья и Бэггинг': 'diamond'}
        family_colors_sparse = {'XGBoost': '#636EFA', 'Стекинг': '#FF6692', 'Gradient Boosting': '#FFA15A', 'Деревья и Бэггинг': '#B6E880'}
        profiles = df_hist_no_lr['Profile'].unique().tolist()
        profile_colors = px.colors.qualitative.Plotly

        def render_tall_chart(fig, height=700, width=800):
            fig.update_layout(height=height, width=width)
            col1, col2, col3 = st.columns([1, 3, 1])
            with col2: st.plotly_chart(fig, use_container_width=False)

        # ========================================== #
        # ГРАФИК 1: Архитектура vs Данные
        # ========================================== #
        st.subheader("Доказательство тезиса: Архитектура первична, данные вторичны")
        st.caption("Форма — семейство алгоритмов. Цвет — профиль препроцессинга. Пунктирная линия разделяет стандартные (справа) и консервативные (слева) архитектуры.")

        families = df_hist_no_lr['Model_Family'].unique().tolist()
        family_to_num = {f: i for i, f in enumerate(families)}
        np.random.seed(42)
        
        # Разделение джиттера: стандартные вправо, консервативные влево
        def get_jitter(row):
            x_num = family_to_num[row['Model_Family']]
            if 'conservative' in row['Model'].lower():
                return x_num - np.random.uniform(0.05, 0.3)
            else:
                return x_num + np.random.uniform(0.05, 0.3)
                
        df_hist_no_lr['x_jitter'] = df_hist_no_lr.apply(get_jitter, axis=1)

        if len(df_hist_no_lr) >= 3:
            dbscan_arch = DBSCAN(eps=0.15, min_samples=2)
            df_hist_no_lr['Cluster_Arch'] = dbscan_arch.fit_predict(df_hist_no_lr[['x_jitter', 'f1_score']])
        else:
            df_hist_no_lr['Cluster_Arch'] = -1

        fig_f1 = go.Figure()

        # 1. Боксы (фоном) и Разделительные линии
        for family, x_num in family_to_num.items():
            family_data = df_hist_no_lr[df_hist_no_lr['Model_Family'] == family]['f1_score']
            fig_f1.add_trace(go.Box(
                y=family_data, x=[x_num]*len(family_data),
                marker_color='rgba(220,220,220,0.3)', line_color='rgba(150,150,150,0.5)', fillcolor='rgba(220,220,220,0.2)',
                showlegend=False, boxpoints=False, name=family
            ))
            # Линия разделитель
            fig_f1.add_vline(x=x_num, line_width=1, line_dash="dash", line_color="rgba(150,0,0,0.5)")
            fig_f1.add_annotation(x=x_num-0.15, y=df_hist_no_lr['f1_score'].min()-0.01, text="Конс.", font=dict(size=8, color="red"), showarrow=False)
            fig_f1.add_annotation(x=x_num+0.15, y=df_hist_no_lr['f1_score'].min()-0.01, text="Станд.", font=dict(size=8, color="blue"), showarrow=False)

        # 2. Кластеры (пунктирные контуры)
        cluster_colors_list = px.colors.qualitative.Plotly
        clusters = df_hist_no_lr[df_hist_no_lr['Cluster_Arch'] != -1]
        for c_id in clusters['Cluster_Arch'].unique():
            c_df = clusters[clusters['Cluster_Arch'] == c_id]
            if len(c_df) < 3: continue
            
            points = c_df[['x_jitter', 'f1_score']].values
            try:
                hull = ConvexHull(points)
                hull_points = points[hull.vertices].tolist()
                hull_points.append(hull_points[0])
                x_hull, y_hull = zip(*hull_points)
                
                color_idx = int(c_id) % len(cluster_colors_list)
                rgb = px.colors.hex_to_rgb(cluster_colors_list[color_idx])
                fillcolor = f'rgba({rgb[0]}, {rgb[1]}, {rgb[2]}, 0.05)'
                
                fig_f1.add_trace(go.Scatter(
                    x=x_hull, y=y_hull, fill='toself',
                    fillcolor=fillcolor,
                    line=dict(color=cluster_colors_list[color_idx], width=2, dash='dot'),
                    name=f'Кластер {c_id}', hoverinfo='skip', showlegend=False
                ))
            except: pass

        # 3. Точки (без текста)
        for i, profile_name in enumerate(profiles):
            group_df = df_hist_no_lr[df_hist_no_lr['Profile'] == profile_name]
            fig_f1.add_trace(go.Scatter(
                x=group_df['x_jitter'], y=group_df['f1_score'], mode='markers',
                marker=dict(
                    size=12, 
                    color=profile_colors[i % len(profile_colors)],
                    line=dict(width=1, color='black'),
                    symbol=group_df['Model_Family'].map(family_shapes).fillna('circle')
                ),
                name=f'Профиль: {profile_name}',
                hovertext=group_df['Model']
            ))

        fig_f1.update_xaxes(tickvals=list(family_to_num.values()), ticktext=list(family_to_num.keys()), title_text="Семейство алгоритмов (Форма)")
        fig_f1.update_yaxes(title_text="F1 Score")
        render_tall_chart(fig_f1, height=700, width=800)
        st.divider()

        # ========================================== #
        # ГРАФИК 2: Микроскоп P vs R
        # ========================================== #
        st.subheader("Матрица ошибок: Микроскоп Precision vs Recall")
        st.caption("Облака — кластеры поведения (DBSCAN). Форма точки — семейство алгоритмов. Масштаб 0.8–1.0.")

        if len(df_hist_no_lr) >= 3:
            dbscan_pr = DBSCAN(eps=0.035, min_samples=2)
            df_hist_no_lr['Cluster_PR'] = dbscan_pr.fit_predict(df_hist_no_lr[['precision', 'recall']])
            
            fig_pr = px.scatter(
                df_hist_no_lr, x="recall", y="precision", color="Model_Family", symbol="Model_Family",
                symbol_map=family_shapes,
                hover_data={"Model": True, "Run": True, "Cluster_PR": True}
            )
            
            for c_id in df_hist_no_lr['Cluster_PR'].unique():
                if c_id == -1: continue
                c_df = df_hist_no_lr[df_hist_no_lr['Cluster_PR'] == c_id]
                if len(c_df) < 3: continue
                points = c_df[['recall', 'precision']].values
                try:
                    hull = ConvexHull(points)
                    hull_points = points[hull.vertices].tolist()
                    hull_points.append(hull_points[0])
                    x_hull, y_hull = zip(*hull_points)
                    color_idx = int(c_id) % len(cluster_colors_list)
                    rgb = px.colors.hex_to_rgb(cluster_colors_list[color_idx])
                    fillcolor = f'rgba({rgb[0]}, {rgb[1]}, {rgb[2]}, 0.05)'
                    fig_pr.add_trace(go.Scatter(
                        x=x_hull, y=y_hull, fill='toself',
                        fillcolor=fillcolor,
                        line=dict(color=cluster_colors_list[color_idx], width=2, dash='dot'),
                        name=f'Кластер {c_id}', hoverinfo='skip', showlegend=False
                    ))
                except: pass

            # Убран текст, оставлены только маркеры
            fig_pr.update_traces(
                mode='markers', 
                marker=dict(size=10, line=dict(width=1, color='DarkSlateGrey'))
            )
        else:
            fig_pr = px.scatter(df_hist_no_lr, x="recall", y="precision", color="Model_Family", symbol="Model_Family")

        fig_pr.update_xaxes(range=[0.8, 1.0], dtick=0.05, constrain="domain")
        fig_pr.update_yaxes(range=[0.8, 1.0], dtick=0.05, scaleanchor="x", scaleratio=1, constrain="domain")
        fig_pr.add_shape(type="line", x0=0.8, y0=0.8, x1=1.0, y1=1.0, line=dict(color="gray", width=2, dash="dash"))
        fig_pr.update_layout(width=750, height=700, legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1))
        
        col1, col2, col3 = st.columns([1, 3, 1])
        with col2: st.plotly_chart(fig_pr, use_container_width=False)
        st.divider()

        # ========================================== #
        # ГРАФИК 3: Влияние профиля данных
        # ========================================== #
        st.subheader("Чувствительность к препроцессингу: Архитектура vs Профиль данных")
        st.caption("Если строка одного цвета — модель игнорирует шум в данных.")

        pivot_df = df_hist_no_lr.groupby(['Model_Family', 'Profile'])['f1_score'].mean().reset_index()
        heatmap_prof_data = pivot_df.pivot(index='Model_Family', columns='Profile', values='f1_score')
        heatmap_prof_data = heatmap_prof_data.fillna(heatmap_prof_data.mean(axis=1))
        
        fig_prof = px.imshow(
            heatmap_prof_data.values, labels=dict(x="Профиль данных", y="Семейство", color="F1 Score"),
            x=heatmap_prof_data.columns, y=heatmap_prof_data.index, text_auto=".3f", aspect="auto", 
            color_continuous_scale="RdYlGn", range_color=[heatmap_prof_data.min().min() - 0.01, heatmap_prof_data.max().max() + 0.01]
        )
        fig_prof.update_xaxes(side="top")
        render_tall_chart(fig_prof, height=500, width=800)
        st.divider()

        # ========================================== #
        # ГРАФИК 4: Архитектурная спарсификация (Кластеры исправлены)
        # ========================================== #
        st.subheader("Архитектурная спарсификация: F1 vs Количество признаков")
        st.caption("Тепловые волны (KDE). Форма и цвет — семейство алгоритмов. Консервативные модели сжимаются влево.")

        fig_sparse = go.Figure()

        if len(df_hist_no_lr) >= 3:
            try:
                from scipy.stats import gaussian_kde
                x_sp = df_hist_no_lr['Num_Features'].values
                y_sp = df_hist_no_lr['f1_score'].values
                xi_s = np.linspace(x_sp.min(), x_sp.max(), 100)
                yi_s = np.linspace(y_sp.min() - 0.01, y_sp.max() + 0.01, 100)
                xx_s, yy_s = np.meshgrid(xi_s, yi_s)
                values_s = np.vstack([x_sp, y_sp])
                kernel_s = gaussian_kde(values_s)
                zz_s = kernel_s(np.vstack([xx_s.ravel(), yy_s.ravel()])).reshape(xx_s.shape)
                
                fig_sparse.add_trace(go.Contour(
                    z=zz_s, x=xi_s, y=yi_s, showscale=False,
                    colorscale=[
                        [0, 'rgba(255,255,255,0.0)'], [0.15, 'rgba(255,240,200,0.15)'],
                        [0.4, 'rgba(255,200,100,0.25)'], [1, 'rgba(255,150,50,0.4)']
                    ],
                    line_width=0.5, line_color='rgba(200,150,100,0.2)', hoverinfo='skip'
                ))
            except Exception: pass

            # Кластеризация (исправлен баг пустых квадратов)
            scaler_sp = MinMaxScaler()
            scaled_sp = scaler_sp.fit_transform(df_hist_no_lr[['Num_Features', 'f1_score']])
            dbscan_sp = DBSCAN(eps=0.15, min_samples=2)
            df_hist_no_lr['Cluster_Sparse'] = dbscan_sp.fit_predict(scaled_sp)

            for c_id in df_hist_no_lr['Cluster_Sparse'].unique():
                if c_id == -1: continue
                c_df = df_hist_no_lr[df_hist_no_lr['Cluster_Sparse'] == c_id]
                if len(c_df) < 3: continue
                
                # Используем ОРИГИНАЛЬНЫЕ координаты из c_df
                points = c_df[['Num_Features', 'f1_score']].values
                try:
                    hull = ConvexHull(points)
                    hull_points = points[hull.vertices].tolist()
                    hull_points.append(hull_points[0])
                    x_hull, y_hull = zip(*hull_points)
                    color_idx = int(c_id) % len(cluster_colors_list)
                    rgb = px.colors.hex_to_rgb(cluster_colors_list[color_idx])
                    fillcolor = f'rgba({rgb[0]}, {rgb[1]}, {rgb[2]}, 0.05)'
                    fig_sparse.add_trace(go.Scatter(
                        x=x_hull, y=y_hull, fill='toself',
                        fillcolor=fillcolor,
                        line=dict(color=cluster_colors_list[color_idx], width=2, dash='dot'),
                        name=f'Кластер {c_id}', hoverinfo='skip', showlegend=False
                    ))
                except: pass

        # Группировка точек по Семействам
        for family_name, group_df in df_hist_no_lr.groupby('Model_Family'):
            fig_sparse.add_trace(go.Scatter(
                x=group_df['Num_Features'], y=group_df['f1_score'], mode='markers',
                marker=dict(
                    size=12, 
                    color=family_colors_sparse.get(family_name, '#999999'),
                    line=dict(width=1, color='black'),
                    symbol=family_shapes.get(family_name, 'circle')
                ),
                name=family_name, 
                hovertext=group_df['Model']
            ))

        fig_sparse.update_xaxes(title_text="Количество значимых признаков")
        fig_sparse.update_yaxes(title_text="F1 Score")
        render_tall_chart(fig_sparse, height=600, width=800)
        st.divider()

        # ========================================== #
        # ГРАФИК 5: Граница Парето (Исправлен баг цвета)
        # ========================================== #
        st.subheader("Идеальный компромисс: Граница Парето (Сложность vs Качество)")
        st.caption("Звёзды — оптимальные модели: лучший F1 при минимальном числе признаков.")

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
            
            for family_name, group_df in df_hist_no_lr.groupby('Model_Family'):
                # Безопасное создание rgba строки
                hex_color = family_colors_sparse.get(family_name, '#999999')
                rgb = px.colors.hex_to_rgb(hex_color)
                rgba_color = f'rgba({rgb[0]}, {rgb[1]}, {rgb[2]}, 0.4)'
                
                fig_pareto.add_trace(go.Scatter(
                    x=group_df['Num_Features'], y=group_df['f1_score'], mode='markers',
                    marker=dict(
                        size=8, 
                        color=rgba_color,
                        symbol=family_shapes.get(family_name, 'circle')
                    ),
                    name=family_name, hovertext=group_df['Run']
                ))
                
            fig_pareto.add_trace(go.Scatter(
                x=pareto_df_final['Num_Features'], y=pareto_df_final['f1_score'], mode='lines+markers',
                line=dict(color='#2ecc71', width=3, dash='dash'),
                marker=dict(size=14, color='#2ecc71', symbol='star', line=dict(width=2, color='DarkGreen')),
                name='Граница Парето', hovertext=pareto_df_final['Run']
            ))
            
            fig_pareto.update_xaxes(title_text="Количество значимых признаков")
            fig_pareto.update_yaxes(title_text="F1 Score")
            fig_pareto.update_layout(legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1))
            render_tall_chart(fig_pareto, height=600, width=800)
        st.divider()

        # ========================================== #
        # ГРАФИК 6: Время инференса vs F1
        # ========================================== #
        st.subheader("Production Ready: Время отклика vs Качество")
        st.caption("Замер времени на предсказание одной строки (100 прогонов). Размер точки — число признаков. Идеально для выбора модели в Real-Time.")

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
                            for _ in range(100):
                                model_data['model'].predict(X_bench)
                            elapsed_ms = (time.perf_counter() - start_time) / 100 * 1000
                            
                            hist_row = df_hist_no_lr[df_hist_no_lr['Run'] == model_name.replace('.pkl', '')]
                            f1_val = hist_row['f1_score'].iloc[0] if not hist_row.empty else 0
                            fam = hist_row['Model_Family'].iloc[0] if not hist_row.empty else get_model_family(model_name)
                            n_feat = hist_row['Num_Features'].iloc[0] if not hist_row.empty else len(model_data['features'])
                            
                            latency_results.append({
                                "Model": model_name.replace('.pkl', ''),
                                "Latency_ms": elapsed_ms,
                                "F1": f1_val,
                                "Family": fam,
                                "Num_Features": n_feat
                            })
                        except Exception as e: pass
                    
                    if latency_results:
                        df_lat = pd.DataFrame(latency_results)
                        fig_lat = px.scatter(
                            df_lat, x="Latency_ms", y="F1", color="Family", symbol="Family",
                            symbol_map=family_shapes,
                            size="Num_Features", hover_data=["Model"],
                            title="Диагностика 1 строки (мс) vs F1"
                        )
                        fig_lat.update_traces(marker=dict(line=dict(width=1, color='DarkSlateGrey')))
                        fig_lat.update_xaxes(title_text="Время инференса (мс) [Меньше = Быстрее]")
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
