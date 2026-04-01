import argparse
import os
import sys
import pandas as pd
import numpy as np
import joblib
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, BaggingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import VotingClassifier
import xgboost as xgb
import warnings
warnings.filterwarnings("ignore")

# --- MODEL REGISTRY ---
MODEL_REGISTRY = {
    "rf": RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42),
    "rf_deep": RandomForestClassifier(n_estimators=500, max_depth=15, random_state=42),
    "gb": GradientBoostingClassifier(n_estimators=100, random_state=42),
    "lr": LogisticRegression(max_iter=1000, random_state=42),
    "dt": DecisionTreeClassifier(random_state=42),
    "bag_dt": BaggingClassifier(DecisionTreeClassifier(), n_estimators=50, random_state=42),
    "xgb" : xgb.XGBClassifier(
        n_estimators=300,       # Больше деревьев
        max_depth=8,            # Ограничим глубину, чтобы не переобучиться
        learning_rate=0.05,     # Побольше шаг
        subsample=0.8,          # Регуляризация
        colsample_bytree=0.8,
        n_jobs=-1,
        random_state=42
    )
}

sys.path.insert(0, str(Path(__file__).parent.resolve()))

def get_feature_importance(classifier, feature_names):
    if hasattr(classifier, 'feature_importances_'):
        return classifier.feature_importances_
    elif hasattr(classifier, 'coef_'):
        return np.abs(classifier.coef_[0])
    elif hasattr(classifier, 'estimators_'): 
        try:
            importances = np.mean([est.feature_importances_ for est in classifier.estimators_], axis=0)
            return importances
        except:
            return np.array([])
    return np.array([])

def main():
    parser = argparse.ArgumentParser(description="Train VPN Detection Model")
    parser.add_argument('-t', '--train', action='store_true', help='Запустить обучение')
    parser.add_argument('-m', '--model', type=str, default='rf', 
                        choices=list(MODEL_REGISTRY.keys()),
                        help=f'Модель: {list(MODEL_REGISTRY.keys())}')
    parser.add_argument('-s', '--save', type=str, help='Путь для сохранения модели')
    parser.add_argument('-d', '--data', type=str, help='Путь к данным (уже очищенным)')
    parser.add_argument('-r', '--retrain_all', action='store_true', help='Дообучить на 100% данных')
    
    args = parser.parse_args()

    if not args.train:
        print("❌ Укажите -t")
        return

    data_path = args.data or os.environ.get("DATA_PATH")
    if not data_path or not os.path.exists(data_path):
        print(f"❌ Данные не найдены: {data_path}")
        return

    print(f"📂 Загрузка: {data_path}")
    df = pd.read_csv(data_path).replace(-1, 0)
    df['label'] = df['traffic_type'].str.contains('VPN', na=False)

    # 2. Разделение X/y
    if 'traffic_type' in df.columns:
        X = df.drop(columns=['traffic_type', 'label'])
    else:
        X = df.drop(columns=['label'])
        
    y = df['label']

    # 3. Определение типов колонок
    numeric_features = X.select_dtypes(include=[np.number]).columns.tolist()
    categorical_features = X.select_dtypes(exclude=[np.number]).columns.tolist()

    print(f"🔢 Числовые фичи: {len(numeric_features)}")
    print(f"🏷️  Категорийные фичи: {len(categorical_features)}")

    # 4. Preprocessor
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), numeric_features)
        ]
    )

    # 5. Split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    # 6. Pipeline
    classifier = MODEL_REGISTRY[args.model]
    pipeline = Pipeline([
        ('preprocessor', preprocessor),
        ('classifier', classifier)
    ])

    print(f"⚙️ Обучение {args.model}...")
    pipeline.fit(X_train, y_train)

    # 7. Оценка
    from sklearn.metrics import accuracy_score
    y_pred = pipeline.predict(X_test)
    print(f"✅ Test Accuracy: {accuracy_score(y_test, y_pred):.4f}")

    # 8. Retrain All
    if args.retrain_all:
        print("🔄 Retrain on 100% data...")
        pipeline.fit(X, y)

    # 9. Имена фичей
    try:
        ohe = pipeline.named_steps['preprocessor'].named_transformers_['cat']
        cat_feature_names = ohe.get_feature_names_out(categorical_features)
        all_features = numeric_features.tolist() + cat_feature_names.tolist()
    except:
        all_features = X.columns.tolist()

    importances = get_feature_importance(pipeline.named_steps['classifier'], all_features)

    # 10. Сохранение
    if not args.save:
        print("❌ Не указан путь сохранения (-s)")
        return

    save_dir = os.path.dirname(args.save)
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        
    # Сохраняем информацию о датасете
    model_data = {
        'model': pipeline,
        'features': all_features,
        'algo_name': args.model.upper(),
        'importances': importances,
        'dataset_path': data_path  # <-- ЗАПИСЫВАЕМ ПУТЬ, ИЗ КОТОРОГО БРАЛИ ДАННЫЕ
    }

    joblib.dump(model_data, args.save)
    print(f"💾 Модель сохранена: {args.save}")
    print(f"📊 Привязка к датасету: {data_path}")

if __name__ == "__main__":
    main()