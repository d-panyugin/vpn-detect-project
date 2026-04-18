# src/train.py
import streamlit as st
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent.resolve()))
from src.core import train_pipeline
from src.config import MODEL_REGISTRY, PIPELINE_PROFILES

st.set_page_config(page_title="Train Model", layout="centered")

st.title("Model Training Hub")
st.caption("Настройте параметры и запустите обучение прямо в браузере.")

# --- СЛОВАРИ С ОПИСАНИЕМ МОДЕЛЕЙ И ПАЙПЛАЙНОВ ---
MODEL_LABELS = {
    "rf": "Random Forest (быстрый)",
    "rf_deep": "Random Forest Deep (точный)",
    "gb": "Gradient Boosting",
    "lr": "Logistic Regression",
    "dt": "Decision Tree",
    "bag_dt": "Bagging + Decision Tree",
    "xgb": "XGBoost (Рекомендуемый)",
    "stacking": "Stacking (XGBoost + RandomForest + LogisticRegression -> LogisticRegression)",
    "stacking_upgraded" : "StackingClassifier(XGB + ExtraTreesClassifier + LogisticRegression -> XGB)",
    "xgb_conservative" : "XGB с большим штрафом за FP"
}

PROFILE_LABELS = {
    "default": "Базовый (без изменений)",
    "drop_active_idle": "Очистка Active/Idle",
    "clean_max_biat": "Агрессивная очистка шума",
    "feature_engineering": "Генерация фичей (медленно)"
}

# --- 1. ВЫБОР ДАННЫХ ---
st.subheader("1. Выбор данных")
data_dir = Path("data/processed")
data_files = [f.name for f in data_dir.glob("*.csv")] if data_dir.exists() else []

if not data_files:
    st.error("В папке data/processed не найдено CSV-файлов!")
    st.stop()
    
selected_data = st.selectbox("Датасет", data_files)

# --- 2. НАСТРОЙКА МОДЕЛИ ---
st.subheader("2. Настройка модели")

default_idx = list(MODEL_REGISTRY.keys()).index("xgb") if "xgb" in MODEL_REGISTRY else 0
model_key = st.selectbox(
    "Алгоритм машинного обучения", 
    list(MODEL_REGISTRY.keys()), 
    index=default_idx,
    format_func=lambda x: MODEL_LABELS.get(x, x)
)

# Динамическая подсказка по модели
with st.expander(f"Подробности: {MODEL_LABELS[model_key]}"):
    st.code(str(MODEL_REGISTRY[model_key].get_params()), language="python")

st.divider()

profile_key = st.selectbox(
    "Профиль предобработки данных", 
    list(PIPELINE_PROFILES.keys()),
    format_func=lambda x: PROFILE_LABELS.get(x, x)
)

# Динамическая подсказка по профилю
st.info(f"**Что делает этот профиль:** {PIPELINE_PROFILES[profile_key]['description']}")

use_pca = st.checkbox("Использовать PCA (сократить количество признаков, сохранив 95% информации)")

# --- 3. СОХРАНЕНИЕ ---
st.subheader("3. Сохранение результата")
default_name = f"{model_key}_{profile_key}{'_pca' if use_pca else ''}.pkl"
save_name = st.text_input("Имя файла (сохранится в папку /models)", value=default_name)

# --- КНОПКА ЗАПУСКА ---
st.divider()
submitted = st.button("Начать обучение", type="primary", use_container_width=True)

# --- 4. ВЫПОЛНЕНИЕ ---
if submitted:
    data_path = data_dir / selected_data
    save_path = Path("models") / save_name
    
    if save_path.exists():
        st.warning(f"Файл `{save_path}` уже существует и будет перезаписан.")
    
    with st.status("Обучение модели...", expanded=True) as status:
        try:
            metrics = train_pipeline(
                algo_name=model_key,
                data_path=str(data_path),
                output_path=str(save_path),
                profile_name=profile_key,
                use_pca=use_pca
            )
            
            status.update(label="Обучение успешно завершено!", state="complete", expanded=False)
            
            st.subheader("Результаты на тестовой выборке (30%)")
            col_m1, col_m2, col_m3, col_m4 = st.columns(4)
            col_m1.metric("Accuracy", f"{metrics['accuracy']:.4f}")
            col_m2.metric("Precision", f"{metrics['precision']:.4f}")
            col_m3.metric("Recall", f"{metrics['recall']:.4f}")
            col_m4.metric("F1 Score", f"{metrics['f1']:.4f}")
            
            st.info(f"Модель сохранена: `{save_path}`\n\nПерейдите в **Analyze Hub**, чтобы сравнить её с другими.")
            
        except Exception as e:
            status.update(label="Ошибка при обучении", state="error", expanded=False)
            st.error(f"Детали ошибки: {e}")