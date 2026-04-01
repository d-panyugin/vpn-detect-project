import argparse
import os
import sys
import pandas as pd
import numpy as np
import joblib
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import VotingClassifier
from sklearn.ensemble import BaggingClassifier
import warnings
warnings.filterwarnings("ignore")

# --- MODEL REGISTRY ---
MODEL_REGISTRY = {
    "rf": RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42),
    "gb": GradientBoostingClassifier(n_estimators=100, random_state=42),
    "lr": LogisticRegression(max_iter=1000, random_state=42),
    "dt": DecisionTreeClassifier(random_state=42),
    "eh": VotingClassifier(
        estimators=[('rf', RandomForestClassifier(n_estimators=1000, max_depth=10)),
                    ('gb', GradientBoostingClassifier(n_estimators=100, random_state=42)),
                    ('dt', DecisionTreeClassifier(random_state=42))], 
        voting='hard'
    ),
    "bag_dt" : BaggingClassifier(
        estimator=DecisionTreeClassifier(), 
        n_estimators=100, 
        random_state=42
    ),
    
}

sys.path.insert(0, str(Path(__file__).parent.resolve()))

def get_feature_importance(classifier):
    if hasattr(classifier, 'feature_importances_'):
        return classifier.feature_importances_
    elif hasattr(classifier, 'coef_'):
        return np.abs(classifier.coef_[0])
    else:
        return np.array([])

def main():
    parser = argparse.ArgumentParser(description="Train VPN Detection Model")
    parser.add_argument('-t', '--train', action='store_true', help='Запустить обучение модели')
    parser.add_argument('-m', '--model', type=str, default='rf', 
                        choices=list(MODEL_REGISTRY.keys()),
                        help=f'Модель: {list(MODEL_REGISTRY.keys())}')
    parser.add_argument('-s', '--save', type=str, help='Куда сохранить модель (models/name.pkl)')
    # НОВЫЙ АРГУМЕНТ ДЛЯ ПУТИ К ДАННЫМ
    parser.add_argument('-d', '--data', type=str, help='Путь к файлу данных (CSV). Приоритет над DATA_PATH')
    
    args = parser.parse_args()

    if not args.train:
        print("❌ Флаг -t не указан.")
        return

    if not args.save:
        print("❌ Не указан путь для сохранения (-s).")
        return

    # Логика поиска пути к данным: сначала берем из аргумента (-d), потом из переменной среды
    data_path = args.data or os.environ.get("DATA_PATH")

    if not data_path:
        print("❌ Путь к данным не указан. Используйте флаг -d 'путь/к/файлу.csv'")
        return
    
    if not os.path.exists(data_path):
        print(f"❌ Файл не найден: {data_path}")
        return

    model_key = args.model
    print(f"🚀 Обучение: {model_key.upper()}...")
    print(f"📂 Данные: {data_path}")
    
    # 1. Загрузка
    df = pd.read_csv(data_path).replace(-1, 0)
    df['label'] = df['traffic_type'].str.contains('VPN', na=False)

    # 2. Подготовка
    X = df.select_dtypes(include=[np.number]).drop(columns=['label'], errors='ignore')
    y = df['label']
    features = X.columns.tolist()

    # 3. Разделение
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    # 4. Модель и Pipeline
    classifier = MODEL_REGISTRY[model_key]
    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('classifier', classifier)
    ])
    
    print(f"⚙️ Фиттинг {classifier.__class__.__name__}...")
    pipeline.fit(X_train, y_train)

    # 5. Импортансы
    importances = get_feature_importance(pipeline.named_steps['classifier'])

    # 6. Сохранение
    model_data = {
        'model': pipeline,
        'features': features,
        'algo_name': model_key.upper(),
        'importances': importances
    }

    save_dir = os.path.dirname(args.save)
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        
    joblib.dump(model_data, args.save)
    print(f"✅ Готово! Модель в: {args.save}")

if __name__ == "__main__":
    main()