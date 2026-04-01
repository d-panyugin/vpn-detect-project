import pandas as pd
import numpy as np
import joblib

import os
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report
# Импортируем возможные модели
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
import warnings
# УБИРАЕМ ВСЕ ПРЕДУПРЕЖДЕНИЯ
warnings.filterwarnings("ignore")
# --- РЕЕСТР МОДЕЛЕЙ ---
from run import MODEL_REGISTRY
MODELS_DIR = Path("models")
MODELS_DIR.mkdir(exist_ok=True)

def get_model_instance(algo_name):
    """Возвращает класс модели по имени из реестра."""
    if algo_name not in MODEL_REGISTRY:
        raise ValueError(f"Алгоритм '{algo_name}' не найден. Доступные: {list(MODEL_REGISTRY.keys())}")
    return MODEL_REGISTRY[algo_name]

def prepare_data(csv_path):
    """Универсальная подготовка данных."""
    df = pd.read_csv(csv_path)
    df = df.replace(-1, 0)
    if 'traffic_type' not in df.columns:
        raise ValueError("Dataset missing 'traffic_type'")
    df['label'] = df['traffic_type'].str.contains('VPN', na=False)
    
    features = [c for c in df.columns if c not in ['traffic_type', 'label']]
    X = df[features]
    y = df['label']
    return X, y, features

def train_pipeline(algo_name, data_path, output_path):
    """Универсальный пайплайн обучения."""
    print(f"[INFO] Инициализация алгоритма: {algo_name}")
    model = get_model_instance(algo_name)
    
    print(f"[INFO] Загрузка данных: {data_path}")
    X, y, features = prepare_data(data_path)
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    
    print("[INFO] Обучение модели...")
    model.fit(X_train, y_train)
    
    y_pred = model.predict(X_test)
    metrics = {
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred, zero_division=0),
        "recall": recall_score(y_test, y_pred, zero_division=0),
        "f1": f1_score(y_test, y_pred, zero_division=0),
        "report": classification_report(y_test, y_pred)
    }

    # Получение важности признаков (универсально для деревьев и линейных моделей)
    if hasattr(model, 'feature_importances_'):
        importances = model.feature_importances_
    elif hasattr(model, 'coef_'):
        importances = np.abs(model.coef_[0])
    else:
        importances = np.zeros(len(features))

    model_data = {
        "model": model,
        "algo_name": algo_name,
        "features": features,
        "importances": importances
    }
    
    joblib.dump(model_data, output_path)
    print(f"[SUCCESS] Модель сохранена в {output_path}")
    return metrics

def load_model_pipeline(model_path):
    """Загружает модель и метаданные."""
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Модель не найдена: {model_path}")
    return joblib.load(model_path)