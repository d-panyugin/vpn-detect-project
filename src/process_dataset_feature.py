import argparse
import pandas as pd
import os
import numpy as np

def main():
    parser = argparse.ArgumentParser(description="Предварительная обработка датасета")
    parser.add_argument('-i', '--input', type=str, required=True, help='Путь к исходному файлу CSV')
    parser.add_argument('-o', '--output', type=str, required=True, help='Путь для сохранения обработанного файла CSV')
    
    args = parser.parse_args()

    print(f"📂 Чтение файла: {args.input}")
    try:
        df = pd.read_csv(args.input)
    except Exception as e:
        print(f"❌ Ошибка чтения: {e}")
        return

    initial_cols = len(df.columns)
    
    output_dir = os.path.dirname(args.output)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # 1. Выбираем только числовые колонки
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    
    # Исключаем целевую переменную (label), чтобы не умножать её на другие признаки
    # Если у вас другая колонка с меткой, измените 'label' на нужное имя
    if 'label' in numeric_cols:
        numeric_cols.remove('label')

    print(f"🔢 Обработка {len(numeric_cols)} числовых колонок...")

    # 2. Собираем новые признаки в список
    new_features = []

    for i, a in enumerate(numeric_cols):
        for b in numeric_cols[i+1:]:
            # Умножение
            new_features.append(pd.Series(df[a] * df[b], name=f'{a}*{b}'))
            
            # Деление A/B
            new_features.append(pd.Series(df[a] / df[b], name=f'{a}/{b}'))
            
            # Деление B/A
            new_features.append(pd.Series(df[b] / df[a], name=f'{b}/{a}'))

    print(f"🔌 Объединение колонок (concatenation)...")
    
    # 3. Объединяем все за один раз
    if new_features:
        df = pd.concat([df] + new_features, axis=1)

    # === ВАЖНО: Очистка данных ===
    print("🧹 Очистка данных (замена inf и nan на 0)...")
    # Деление на ноль создает inf, заменим их на NaN, а затем на 0
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    df.fillna(0, inplace=True)
    # =============================

    final_cols = len(df.columns)
    print(f"📊 Количество колонок: {initial_cols} -> {final_cols}")
    
    # Сохраняем
    df.to_csv(args.output, index=False)
    print(f"✅ Обработанный файл сохранен: {args.output}")

if __name__ == "__main__":
    main()