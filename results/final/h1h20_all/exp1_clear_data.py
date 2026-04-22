# coding: utf-8
import sys
import os
import numpy as np
from evaluation import research
from multiprocessing import Pool


def run_single_size(N, horizon):
    """
    Запускает предсказание для чистого обучающего набора размера N.
    """
    how_many_gaps = 1000
    target_r = 28.0

    # Для list_ts[0] (откуда берутся тесты) нужен ряд большего размера,
    # чтобы хватило истории для всех 1000 окон тестирования.
    test_series_size = N + how_many_gaps + 100

    try:
        # Передаем два раза target_r:
        # list_ts[0] пойдет на тесты (размер test_series_size)
        # list_ts[1] пойдет в fit() как чистая обучающая выборка (размер N)
        rmse1, np1, mape1, rmse2, np2, mape2 = research(
            r_values=[target_r, target_r],
            ts_size=np.array([test_series_size, N]),
            how_many_gaps=how_many_gaps,
            test_size_constant=horizon
        )
        return (N, rmse1, np1, mape1, rmse2, np2, mape2)
    except Exception as e:
        print(f"Ошибка при размере обучающей выборки N={N}: {e}", file=sys.stderr)
        return None


def main():
    if len(sys.argv) < 2:
        print("Использование: python exp1_clear_data.py <horizon>")
        sys.exit(1)

    horizon = int(sys.argv[1])

    # Директория и файл для сохранения результатов
    output_dir = "../assets/results"
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, f"exp1_clear_data_h{horizon}.csv")

    if not os.path.exists(output_path):
        with open(output_path, "w") as f:
            f.write("Size,rmse_01,np_01,mape_01,rmse_03,np_03,mape_03\n")

    # Генерируем ровно 24 интервала от 10000 до 40000 (как в статье)
    # astype(int) даст в точности те же размеры: 10000, 11304, 12608...
    sizes = np.linspace(10000, 40000, 24).astype(int)

    print(f"--- Запуск Эксперимента 1 (Clear Data Expansion) ---")
    print(f"Горизонт прогноза (h): {horizon}")
    print(f"Параллельный расчет для {len(sizes)} различных размеров (от {sizes[0]} до {sizes[-1]})...")

    # Запускаем пул процессов. Если вы на узле с 48 ядрами,
    # он посчитает все 24 размера одновременно!
    num_cores = min(24, os.cpu_count() or 1)
    with Pool(num_cores) as pool:
        args = [(N, horizon) for N in sizes]
        results = pool.starmap(run_single_size, args)

    # Убираем возможные ошибки и сортируем результаты по размеру N
    results = [r for r in results if r is not None]
    results.sort(key=lambda x: x[0])

    # Записываем всё в файл
    with open(output_path, "a") as f:
        for res in results:
            N, rmse1, np1, mape1, rmse2, np2, mape2 = res
            f.write(f"{N},{rmse1:.6f},{np1:.6f},{mape1:.6f},{rmse2:.6f},{np2:.6f},{mape2:.6f}\n")

    print(f"Эксперимент завершен! Результаты сохранены в {output_path}")


if __name__ == "__main__":
    main()