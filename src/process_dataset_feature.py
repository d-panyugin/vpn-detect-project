import argparse
import pandas as pd
import os

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
    
    """наименее важные признаки для лучшей модели(bagging decision tree),
        std_active
        mean_idle
        mean_active
        max_active
        min_active
        min_idle
        std_idle
        max_idle"""

    # Создаем папку, если её нет
    output_dir = os.path.dirname(args.output)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)
    df['avg_packet_size'] = df['flowBytesPerSecond'] / df['flowPktsPerSecond']
    df['total_volume'] = df['flowBytesPerSecond'] * df['duration']
    df['total_packets'] = df['flowPktsPerSecond'] * df['duration']
    df['active_duration_ratio'] = df['active'] / df['duration']
    df['std_idle_duration_ratio'] = df['std_idle'] / df['duration']
    df['mean_idle_duration_ratio'] = df['mean_idle'] / df['duration']
    df['std_active_duration_ratio'] = df['std_active'] / df['duration']

    final_cols = len(df.columns)
    print(f"📊 Количество колонок: {initial_cols} -> {final_cols}")
    # Сохраняем
    df.to_csv(args.output, index=False)
    print(f"✅ Обработанный файл сохранен: {args.output}")

if __name__ == "__main__":
    main()