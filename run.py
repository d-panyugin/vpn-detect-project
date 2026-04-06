# run.py
import argparse
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.resolve()))

from src.core import train_pipeline
from src.config import MODEL_REGISTRY, PIPELINE_PROFILES

def main():
    parser = argparse.ArgumentParser(description="Train VPN Detection Model")
    parser.add_argument('-t', '--train', action='store_true', help='Запустить обучение')
    
    parser.add_argument('-m', '--model', type=str, default='bag_dt', 
                        choices=list(MODEL_REGISTRY.keys()),
                        help=f'Модель: {list(MODEL_REGISTRY.keys())}')
    
    parser.add_argument('-p', '--profile', type=str, default='default',
                        choices=list(PIPELINE_PROFILES.keys()),
                        help=f'Профиль обработки данных: {list(PIPELINE_PROFILES.keys())}')
    
    parser.add_argument('-s', '--save', type=str, help='Путь для сохранения модели (.pkl)')
    parser.add_argument('-d', '--data', type=str, help='Путь к данным (.csv)')
    parser.add_argument('--pca', action='store_true', help='Использовать PCA')
    
    args = parser.parse_args()

    if not args.train:
        print("❌ Укажите флаг -t для запуска обучения.")
        return

    if not args.data or not os.path.exists(args.data):
        print(f"❌ Данные не найдены: {args.data}")
        return

    if not args.save:
        print("❌ Укажите путь сохранения через -s")
        return

    train_pipeline(
        algo_name=args.model, 
        data_path=args.data, 
        output_path=args.save,
        profile_name=args.profile, # Передаем выбранный профиль и поведение
        use_pca=args.pca
    )

if __name__ == "__main__":
    main()