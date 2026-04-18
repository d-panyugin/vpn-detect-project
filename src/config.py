# src/config.py
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier, GradientBoostingClassifier, BaggingClassifier, StackingClassifier
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
    ), 
    "stacking": StackingClassifier(
        estimators=[
            ('xgb', xgb.XGBClassifier(n_estimators=100, max_depth=6, learning_rate=0.1, n_jobs=-1, random_state=42)),
            ('rf', RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)),
            ('lr', LogisticRegression(max_iter=500, random_state=42))
        ],
        final_estimator=LogisticRegression(max_iter=1000, random_state=42),
        n_jobs=-1,
        passthrough=False
    ),
    "stacking_upgraded": StackingClassifier(
        estimators=[
            # 1. Разнообразие: разная природа алгоритмов
            ('xgb', xgb.XGBClassifier(
                n_estimators=300, max_depth=8, learning_rate=0.05, 
                subsample=0.8, colsample_bytree=0.8, n_jobs=-1, random_state=42
            )),
            ('et', ExtraTreesClassifier(n_estimators=200, max_depth=12, random_state=42, n_jobs=-1)), # ExtraTrees ищет сплиты рандомно, отличаясь от XGB
            ('lr', LogisticRegression(C=0.1, max_iter=1000, random_state=42)) # Линейная модель с регуляризацией
        ],
        final_estimator=xgb.XGBClassifier(
            n_estimators=50, max_depth=3, learning_rate=0.1, random_state=42, n_jobs=-1
        ),
        passthrough=True,
        n_jobs=-1
    ),
    "xgb_conservative": xgb.XGBClassifier(
        n_estimators=300,       
        max_depth=6,            # Мельче дерево -> меньше шансов переобучиться на шум
        learning_rate=0.05,     
        subsample=0.7,          
        colsample_bybytree=0.7,
        gamma=2.0,              # Жесткий штраф за создание нового листа (ищет только железные паттерны)
        min_child_weight=10,    # В листе должно быть много объектов, чтобы он расщепился
        scale_pos_weight=0.5,   # Штраф за False Positives (делаем модель параноиком по FP)
        n_jobs=-1,
        random_state=42
    ),
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
    },

    "quantile_profile": {
        "description": "Обрезка выбросов по квантилям для min/max признаков.",
        "drop_patterns": [], 
        "drop_exact": ["duration"],
        "new_features": [],
        "quantile_features": [
            ("min_active", [0.1, 0.25]),
            ("max_active", [0.75, 0.90]),
            ("min_idle", [0.1, 0.25]),
            ("max_idle", [0.75, 0.90]),
            ("max biat", [0.75, 0.90]),
            ("max fiat", [0.75, 0.90])
        ]
    }
}