# src/core.py
from datetime import datetime
import json
import pandas as pd
import numpy as np
import joblib
import os
from pathlib import Path
import warnings

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report, roc_auc_score, log_loss
from sklearn.inspection import permutation_importance
from sklearn.decomposition import PCA

from .config import MODEL_REGISTRY, PIPELINE_PROFILES

warnings.filterwarnings("ignore")

class DataPreprocessor(BaseEstimator, TransformerMixin):
    """
    Универсальный препроцессор.
    """
    def __init__(self, drop_patterns=None, drop_exact=None, new_features=None, generate_interactions=False):
        self.drop_patterns = drop_patterns or []
        self.drop_exact = drop_exact or []
        self.new_features = new_features or []
        self.generate_interactions = generate_interactions
        self.feature_names_in_ = None
        self.feature_names_out_ = None 

    def fit(self, X, y=None):
        if hasattr(X, 'columns'):
            self.feature_names_in_ = X.columns.tolist()
        return self

    def transform(self, X):
        X = X.copy()
        
        # 1. Удаление по паттернам
        cols_to_drop_pattern = set()
        for pattern in self.drop_patterns:
            cols_to_drop_pattern.update([c for c in X.columns if pattern in c])
        
        cols_to_drop = cols_to_drop_pattern.union(set(self.drop_exact))
        X.drop(columns=list(cols_to_drop), errors='ignore', inplace=True)

        # 2. Новые признаки
        for name, formula in self.new_features:
            try:
                X[name] = X.eval(formula)
            except Exception:
                X[name] = 0.0

        # 3. Взаимодействия
        if self.generate_interactions:
            numeric_cols = X.select_dtypes(include=[np.number]).columns.tolist()
            for i, col_a in enumerate(numeric_cols):
                for col_b in numeric_cols[i+1:]:
                    X[f'{col_a}_x_{col_b}'] = X[col_a] * X[col_b]
                    X[f'{col_a}_div_{col_b}'] = X[col_a] / (X[col_b] + 1e-5)

        X.replace([np.inf, -np.inf], np.nan, inplace=True)
        X.fillna(0, inplace=True)
        self.feature_names_out_ = X.columns.tolist()
        return X

def get_model_instance(algo_name):
    if algo_name not in MODEL_REGISTRY:
        raise ValueError(f"Модель '{algo_name}' не найдена.")
    return MODEL_REGISTRY[algo_name]

def prepare_target(df):
    if 'traffic_type' not in df.columns:
        raise ValueError("Нет колонки 'traffic_type'")
    return df['traffic_type'].str.contains('VPN', na=False)

def save_run_results(model_data, metrics, output_path):
    """Сохраняет результаты эксперимента в папку results"""
    results_dir = Path("results")
    results_dir.mkdir(parents=True, exist_ok=True)
    
    algo_name = model_data.get('algo_name', 'unknown')
    profile_name = model_data.get('profile_name', 'default')
    
    fi_list = []
    processed_features = model_data.get('processed_features', [])
    importances = model_data.get('importances', np.array([]))
    
    if len(importances) > 0 and len(importances) == len(processed_features):
        for f, imp in zip(processed_features, importances):
            fi_list.append({"feature": f, "importance": float(imp)})
            
    result_data = {
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "model_name": algo_name,
        "run_name": f"{algo_name}_{profile_name}",
        "train_dataset": Path(model_data.get('dataset_path', '')).name,
        "metrics": {
            "accuracy": float(metrics["accuracy"]),
            "precision": float(metrics["precision"]),
            "recall": float(metrics["recall"]),
            "f1_score": float(metrics["f1"]),
        },
        "feature_importance_full": fi_list
    }
    
    timestamp_str = datetime.now().strftime('%Y%m%d_%H%M%S')
    filename = results_dir / f"{timestamp_str}_{algo_name}_{profile_name}.json"
    
    with open(filename, "w", encoding='utf-8') as f:
        json.dump(result_data, f, ensure_ascii=False, indent=4)

def train_pipeline(algo_name, data_path, output_path, profile_name="default", use_pca=False):
    profile_config = PIPELINE_PROFILES.get(profile_name)
    if not profile_config:
        raise ValueError(f"Профиль '{profile_name}' не найден.")

    df = pd.read_csv(data_path).replace(-1, 0)
    y = prepare_target(df)
    X = df.drop(columns=['traffic_type', 'label'], errors='ignore')

    steps = [
        ('preprocessor', DataPreprocessor(
            drop_patterns=profile_config.get('drop_patterns', []),
            drop_exact=profile_config.get('drop_exact', []),
            new_features=profile_config.get('new_features', []),
            generate_interactions=profile_config.get('generate_interactions', False)
        )),
        ('scaler', StandardScaler())
    ]

    if use_pca:
        steps.append(('pca', PCA(n_components=0.95)))
    
    classifier = get_model_instance(algo_name)
    steps.append(('classifier', classifier))
    pipeline = Pipeline(steps)
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    
    pipeline.fit(X_train, y_train)
    y_pred = pipeline.predict(X_test)
    
    metrics = {
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred, zero_division=0),
        "recall": recall_score(y_test, y_pred, zero_division=0),
        "f1": f1_score(y_test, y_pred, zero_division=0),
    }
    
    importances = np.array([])
    processed_feature_names = []
    
    if 'preprocessor' in pipeline.named_steps:
        processed_feature_names = pipeline.named_steps['preprocessor'].feature_names_out_

    if use_pca:
        importances = np.array([])
    elif hasattr(classifier, 'feature_importances_'):
        importances = classifier.feature_importances_
    elif hasattr(classifier, 'coef_'):
        importances = np.abs(classifier.coef_[0])
    elif hasattr(classifier, 'estimators_'):
         valid_imps = [est.feature_importances_ for est in classifier.estimators_ if hasattr(est, 'feature_importances_')]
         if valid_imps:
             importances = np.mean(valid_imps, axis=0)
            
    if len(importances) == 0 and not use_pca:
        result = permutation_importance(pipeline, X_test.iloc[:300], y_test.iloc[:300], n_repeats=5, random_state=42, n_jobs=-1)
        importances = result.importances_mean
        processed_feature_names = X_test.columns.tolist()

    model_data = {
        "model": pipeline,
        "algo_name": algo_name.upper(),
        "profile_name": profile_name,
        "features": X.columns.tolist(),
        "processed_features": processed_feature_names,
        "importances": importances,
        "dataset_path": data_path
    }
    
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model_data, output_path)
    save_run_results(model_data, metrics, output_path)
    return metrics

def load_model_pipeline(model_path):
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Модель не найдена: {model_path}")
    return joblib.load(model_path)