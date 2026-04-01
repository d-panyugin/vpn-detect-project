# VPN Traffic Detector
📁 Структура проекта
.
- ├── src/
- │ ├── run.py # Скрипт для обучения моделей
- │ ├── app.py # Веб-интерфейс для инференса и оценки
- │ ├── analyze.py # Скрипт для визуализации сохраненной истории
- │ └── core.py # Вспомогательные функции (загрузка моделей)
- ├── data/ # Папка для датасетов (.csv)
- ├── models/ # Папка для сохраненных обученных моделей (.pkl)
- ├── results/ # Папка для сохраненных JSON-отчетов
- └── README.md # Этот файл
### Подготовка
```bash
pip install streamlit pandas numpy scikit-learn plotly joblib
```
## Запуск обучения
```
python src/run.py -t -m <model_key> -d <path_to_data> -s <path_to_save_model>
```
## Аргументы:

* -t : Флаг запуска обучения (обязательный).
* -m : Ключ модели из реестра (обязательный).
* -d : Путь к файлу с обучающими данными (обязательный).
* -s : Путь, куда сохранить модель (.pkl) (обязательный).
* -r : (Опционально) Дообучить модель на 100% данных после оценки.
## Доступные модели (-m):

* rf : Random Forest (n_estimators=100)
* rf_deep : Random Forest (n_estimators=500, depth=15)
* gb : Gradient Boosting
* lr : Logistic Regression
* dt : Decision Tree
* bag_dt : Bagging Decision Tree (Рекомендуемый)
### Примеры:

Обычное обучение (оценка на test split):

```bash
python src/run.py -t -m bag_dt -d data/my_dataset.csv -s models/bagging_model.pkl
```
## 4. Режим 2: Анализ и Инференс (App Mode)
```bash
streamlit run src/app.py -- --model <path_to_model> --data <path_to_test_data>
```
### Пример
```bash
streamlit run src/app.py -- --model models/bagging_model.pkl --data data/test_data.csv
```
## 5. Аналитика результатов (запускать из корня)
```bash
streamlit run src/analyze.py
```
