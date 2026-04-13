# src/app.py
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
from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score

warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).parent.resolve()))
from src.core import load_model_pipeline

# --- LOAD CSS ---
def load_css(file_name):
    with open(file_name) as f:
        st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)

load_css("src/style.css")

# --- ARGS ---
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--champion', type=str, required=True, help='Path to Champion model')
    parser.add_argument('--challenger', type=str, required=True, help='Path to Challenger model')
    parser.add_argument('--data', type=str, required=True, help='Path to data')
    return parser.parse_args(sys.argv[1:])

try:
    args = parse_args()
except SystemExit:
    st.stop()

# --- LOADERS ---
@st.cache_resource
def get_model(_path):
    return load_model_pipeline(_path)

@st.cache_data
def get_data(_path):
    return pd.read_csv(_path).replace(-1, 0)

# --- LOAD RESOURCES ---
try:
    m_champ = get_model(args.champion)
    m_chall = get_model(args.challenger)
    df = get_data(args.data)
except Exception as e:
    st.error(f"Critical Error: {e}")
    st.stop()

df['label'] = df['traffic_type'].str.contains('VPN', na=False)
y_true = df['label']

# --- PREDICT ---
def get_pred_and_metrics(pipeline, X, y):
    pred = pipeline.predict(X)
    try:
        proba = pipeline.predict_proba(X)[:, 1]
    except:
        proba = np.array([0.5] * len(pred))
    
    return {
        "pred": pred,
        "proba": proba,
        "acc": accuracy_score(y, pred),
        "prec": precision_score(y, pred, zero_division=0),
        "rec": recall_score(y, pred, zero_division=0),
        "f1": f1_score(y, pred, zero_division=0)
    }

X_champ = df[m_champ['features']]
X_chall = df[m_chall['features']]

res_c = get_pred_and_metrics(m_champ['model'], X_champ, y_true)
res_h = get_pred_and_metrics(m_chall['model'], X_chall, y_true)

# --- SIDEBAR: CONFIG ---
with st.sidebar:
    st.header("Settings")
    win_metric_key = st.selectbox(
        "Select Winning Metric", 
        options=["f1", "accuracy", "precision", "recall"],
        format_func=lambda x: x.upper(),
        index=0
    )
    st.caption("The Champion is determined by the selected metric.")

# Determine Winner based on selection
val_c = res_c[win_metric_key]
val_h = res_h[win_metric_key]
winner = "CHAMPION" if val_c >= val_h else "CHALLENGER"

# --- UI ---
st.title("MODEL DUEL: CHAMPION vs CHALLENGER")
st.markdown(f"**Current Winner ({win_metric_key.upper()}):** <span style='color:#2ecc71; font-weight:bold'>{winner}</span>", unsafe_allow_html=True)

# 1. TOP: VISUAL COMPARISON
col_c, col_h = st.columns(2)

def render_card(title, metrics, is_winner, win_metric, delta_val):
    css_class = "model-card winner-card" if is_winner else "model-card loser-card"
    winner_badge = " 🏆" if is_winner else ""
    
    # Display delta only for the selected metric
    sign = "+" if delta_val > 0 else ""
    color = "delta-pos" if delta_val >= 0 else "delta-neg"
    delta_html = f"<div class='{color}'>{sign}{delta_val:.4f} ({win_metric.upper()})</div>"

    st.markdown(f"""
    <div class='{css_class}'>
        <h3 style='margin-top:0'>{title}{winner_badge}</h3>
        <div style='display:flex; justify-content:space-between; align-items:center; margin-bottom: 20px;'>
            <div>
                <div class='metric-label'>{win_metric.upper()}</div>
                <div class='metric-value'>{metrics[win_metric]:.4f}</div>
            </div>
            <div style='text-align:right'>
                {delta_html}
            </div>
        </div>
        <div style='display:grid; grid-template-columns: 1fr 1fr 1fr; gap: 10px; font-size: 0.85rem; color: #666; border-top: 1px solid #eee; padding-top: 10px;'>
            <div><b>Acc:</b> {metrics['acc']:.3f}</div>
            <div><b>Prec:</b> {metrics['prec']:.3f}</div>
            <div><b>Rec:</b> {metrics['rec']:.3f}</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

with col_c:
    delta = val_c - val_h
    render_card(m_champ['algo_name'], res_c, (winner == "CHAMPION"), win_metric_key, delta)

with col_h:
    delta = val_h - val_c
    render_card(m_chall['algo_name'], res_h, (winner == "CHALLENGER"), win_metric_key, delta)

st.divider()

# 2. MIDDLE: ANALYTICS
tab_prob, tab_disagree = st.tabs(["Probability Distributions", "Model Disagreements"])

with tab_prob:
    st.subheader("Confidence Analysis")
    df_plot = pd.DataFrame({
        'Champion': res_c['proba'],
        'Challenger': res_h['proba']
    }).melt(var_name='Model', value_name='Probability')
    
    fig = px.histogram(df_plot, x='Probability', color='Model', barmode='overlay', 
                       opacity=0.6, nbins=50, color_discrete_map={'Champion':'#3498db', 'Challenger':'#e74c3c'})
    fig.update_layout(layout_title_text='Distribution of Prediction Probabilities')
    st.plotly_chart(fig, use_container_width=True)

with tab_disagree:
    st.subheader("Where models disagree")
    diff_mask = res_c['pred'] != res_h['pred']
    disagree_df = df[diff_mask].copy()
    disagree_df['Champion_Pred'] = res_c['pred'][diff_mask]
    disagree_df['Challenger_Pred'] = res_h['pred'][diff_mask]
    disagree_df['Champion_Proba'] = res_c['proba'][diff_mask]
    disagree_df['Challenger_Proba'] = res_h['proba'][diff_mask]
    disagree_df['Real_Label'] = y_true[diff_mask]
    
    st.write(f"Found {len(disagree_df)} disagreements out of {len(df)} rows.")
    st.dataframe(disagree_df, use_container_width=True)

# 3. BASEMENT: FULL LEADERBOARD
st.markdown("<div class='basement-header'><h3>HISTORY LEADERBOARD (BASEMENT)</h3></div>", unsafe_allow_html=True)
st.caption("All past experiments loaded from /results directory.")

RESULTS_DIR = "results"
if os.path.exists(RESULTS_DIR):
    files = sorted([f for f in os.listdir(RESULTS_DIR) if f.endswith('.json')], reverse=True)
    history = []
    
    for f in files:
        try:
            with open(os.path.join(RESULTS_DIR, f), 'r') as file:
                data = json.load(file)
                m = data.get('metrics', {})
                history.append({
                    "Run Name": data.get('run_name', 'Unknown'),
                    "Model": data.get('model_name', 'Unknown'),
                    "Date": data.get('timestamp', ''),
                    "Accuracy": m.get('accuracy', 0),
                    "Precision": m.get('precision', 0),
                    "Recall": m.get('recall', 0),
                    "F1 Score": m.get('f1_score', 0)
                })
        except Exception:
            continue
            
    if history:
        hist_df = pd.DataFrame(history)
        for col in ["Accuracy", "Precision", "Recall", "F1 Score"]:
            hist_df[col] = hist_df[col].apply(lambda x: f"{x:.4f}")
        st.dataframe(hist_df, use_container_width=True, height=300)