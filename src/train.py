# src/train.py
import streamlit as st
import pandas as pd
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent.resolve()))
from src.core import train_pipeline
from src.config import MODEL_REGISTRY, PIPELINE_PROFILES

st.set_page_config(page_title="Train Model", layout="centered")

st.title("Model Training Hub")
st.caption("Настройте параметры и запустите обучение прямо в браузере.")

# --- 1. КОНФИГУРАЦИЯ ---
with st.form("training_config"):
    st.subheader("1. Выбор данных")
    
    # Автоматический поиск датасетов в папке
    data_dir = Path("data/processed")
    data_files = [f.name for f in data_dir.glob("*.csv")] if data_dir.exists() else []
    
    if not data_files:
        st.error("В папке data/processed не найдено CSV-файлов!")
        st.stop()
        
    selected_data = st.selectbox("Датасет", data_files)
    
    st.subheader("2. Настройка модели")
    col1, col2 = st.columns(2)
    
    with col1:
        # Динамически ищем индекс xgb для дефолтного выбора, чтобы не сломалось при изменении config.py
        default_idx = list(MODEL_REGISTRY.keys()).index("xgb") if "xgb" in MODEL_REGISTRY else 0
        model_key = st.selectbox("Алгоритм", list(MODEL_REGISTRY.keys()), index=default_idx)
        
    with col2:
        profile_key = st.selectbox("Профиль предобработки", list(PIPELINE_PROFILES.keys()))
        # Отображаем описание профиля
        st.info(PIPELINE_PROFILES[profile_key]["description"])

    use_pca = st.checkbox("Использовать PCA (сохранять 95% дисперсии)")
    
    st.subheader("3. Сохранение")
    # Генерация имени по умолчанию на основе выбранных параметров
    default_name = f"{model_key}_{profile_key}{'_pca' if use_pca else ''}.pkl"
    save_name = st.text_input("Имя файла (сохранится в /models)", value=default_name)
    
    submitted = st.form_submit_button("🚀 Начать обучение", type="primary")

# --- 2. ВЫПОЛНЕНИЕ ---
if submitted:
    data_path = data_dir / selected_data
    save_path = Path("models") / save_name
    
    # Предупреждение о перезаписи существующей модели
    if save_path.exists():
        st.warning(f"Файл `{save_path}` уже существует и будет перезаписан.")
    
    # Используем st.status вместо spinner! 
    # При долгом обучении интерфейс не зависнет визуально, появится индикатор процесса.
    with st.status("Обучение модели...", expanded=True) as status:
        try:
            metrics = train_pipeline(
                algo_name=model_key,
                data_path=str(data_path),
                output_path=str(save_path),
                profile_name=profile_key,
                use_pca=use_pca
            )
            
            # Меняем статус на успешный после завершения
            status.update(label="Обучение успешно завершено!", state="complete", expanded=False)
            
            # Вывод метрик
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