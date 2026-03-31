import argparse
import os
import sys
import subprocess
from pathlib import Path

# --- НАСТРОЙКА ПУТЕЙ ---
# Получаем абсолютную папку, где лежит этот run.py (src/)
SRC_DIR = Path(__file__).parent.resolve()
# Получаем корень проекта (на уровень выше src/)
PROJECT_ROOT = SRC_DIR.parent.resolve()

APP_SCRIPT = (SRC_DIR / "app.py").resolve()
PATH_MODELS = PROJECT_ROOT / "models"
DEFAULT_DATA = PROJECT_ROOT / "data" / "processed" / "consolidated_traffic_data.csv"

def main():
    parser = argparse.ArgumentParser(description="VPN Detection CLI")

    parser.add_argument('--train', '-t', action='store_true', help="Запустить обучение")
    parser.add_argument('--algo', '-a', type=str, choices=['rf', 'gb', 'lr', 'dt', 'rf_deep'], 
                        default='rf', help="Алгоритм для обучения")
    # Принимаем строку и сразу превращаем в Path объект для надежности
    parser.add_argument('--data', type=lambda p: Path(p).resolve(), default=DEFAULT_DATA, help="Путь к данным")
    parser.add_argument('--output', type=str, default=None, help="Путь для сохранения модели")
    
    parser.add_argument('--visualize', '-v', action='store_true', help="Запустить Streamlit UI")
    parser.add_argument('--model', '-m', type=str, default=None, help="Путь к модели для загрузки в UI")

    args = parser.parse_args()

    # --- ВЕТКА 1: ОБУЧЕНИЕ ---
    if args.train:
        if not args.output:
            PATH_MODELS.mkdir(parents=True, exist_ok=True)
            args.output = str(PATH_MODELS / f"{args.algo}_vpn_model.pkl")
        else:
            # Если указан путь, тоже делаем абсолютным относительно корня, если он относительный
            args.output = str(Path(args.output).resolve())
            
        print(f"🚀 Запуск обучения: {args.algo.upper()}")
        print(f"📂 Данные: {args.data}")
        print(f"💾 Модель: {args.output}")
        
        try:
            # Добавляем src в sys.path, чтобы импорт core сработал
            sys.path.insert(0, str(SRC_DIR))
            from core import train_pipeline
            
            metrics = train_pipeline(args.algo, str(args.data), args.output)
            
            print("\n📊 Результаты:")
            print(f"   Accuracy: {metrics['accuracy']:.2%}")
            print(f"   F1 Score: {metrics['f1']:.2%}")
            print(f"\n💡 Запуск UI:")
            print(f"   python src/run.py --visualize --model {args.output}")
            
        except Exception as e:
            print(f"❌ Ошибка: {e}")
            import traceback
            traceback.print_exc()
            sys.exit(1)

    # --- ВЕТКА 2: ВИЗУАЛИЗАЦИЯ ---
    elif args.visualize:
        if not APP_SCRIPT.exists():
            print(f"❌ UI скрипт не найден: {APP_SCRIPT}")
            sys.exit(1)

        env = os.environ.copy()
        
        if args.model:
            model_path = Path(args.model).resolve()
            if not model_path.exists():
                print(f"❌ Модель не найдена: {model_path}")
                sys.exit(1)
            env["VPN_MODEL_PATH"] = str(model_path)
            print(f"👀 Модель: {model_path}")
        else:
            print("⚠️ Модель не указана.")

        env["DATA_PATH"] = str(args.data)
        print(f"📂 Данные: {args.data}")
        print(f"🚀 Запуск Streamlit: {APP_SCRIPT}")

        try:
            subprocess.run([sys.executable, "-m", "streamlit", "run", str(APP_SCRIPT)], env=env)
        except KeyboardInterrupt:
            print("\n🛑 Остановлено.")

    else:
        parser.print_help()

if __name__ == "__main__":
    main()