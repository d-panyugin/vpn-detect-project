# src/config.py
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, BaggingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
import xgboost as xgb

MODEL_REGISTRY = {
    "rf": RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42, n_jobs=-1),
    "rf_deep": RandomForestClassifier(n_estimators=500, max_depth=15, random_state=42, n_jobs=-1),
    "gb": GradientBoostingClassifier(n_estimators=100, random_state=42),
    "lr": LogisticRegression(max_iter=1000, random_state=42, n_jobs=-1),
    "dt": DecisionTreeClassifier(random_state=42),
    "bag_dt": BaggingClassifier(DecisionTreeClassifier(), n_estimators=50, random_state=42, n_jobs=-1),
    "xgb" : xgb.XGBClassifier(
        n_estimators=300,       
        max_depth=8,            
        learning_rate=0.05,     
        subsample=0.8,          
        colsample_bytree=0.8,
        n_jobs=-1,
        random_state=42
    )
}

# --- PROFILE REGISTRY ---
PIPELINE_PROFILES = {
    "default": {
        "description": "Базовый профиль. Удаляем только duration и мусор.",
        "drop_patterns": [], 
        "drop_exact": ["duration"],
        "new_features": [] 
    },

    "drop_active_idle": {
        "description": "Удаляем признаки active/idle. Добавляем агрегации.",
        "drop_patterns": ["active", "idle"],
        "drop_exact": ["duration"],
        "new_features": [
            ("avg_packet_size", "flowBytesPerSecond / (flowPktsPerSecond + 1e-5)"),
            ("total_iat_ratio", "total_fiat / (total_biat + 1e-5)"),
            ("mean_iat_ratio", "mean_fiat / (mean_biat + 1e-5)")
        ]
    },

    "clean_max_biat": {
        "description": "Агрессивная очистка шумных признаков.",
        "drop_patterns": [], 
        "drop_exact": [
            "std_active", "mean_idle", "mean_active", "max_active", "min_active", 
            "min_idle", "std_idle", "max_idle", "total biat", "total fiat", 
            "duration", "max biat", "max fiat", "mean flowiat", "mean fiat", "std flowiat"
        ],
        "new_features": []
    },

    "feature_engineering": {
        "description": "Генерация полиномиальных признаков.",
        "drop_patterns": [],
        "drop_exact": [],
        "new_features": [],
        "generate_interactions": True
    }
}