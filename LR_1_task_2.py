# LR_1_task_2.py — Варіант 28
import numpy as np
from sklearn import preprocessing

input_data = np.array([
    [-4.1, -5.5,  3.3],
    [ 6.9,  4.6,  3.9],
    [-4.2,  3.8,  2.3],
    [ 3.9,  3.4, -1.2],
], dtype=float)
threshold = 3.0

np.set_printoptions(precision=6, suppress=True)

print("=== Вхідні дані (Варіант 28) ===")
print(input_data)

# 1) Бінаризація
data_binarized = preprocessing.Binarizer(threshold=threshold).transform(input_data)
print(f"\n=== Бінаризація (поріг = {threshold}) ===")
print(data_binarized)

# 2) Виключення середнього
print("\n=== ДО стандартизації (середнє / стандартне відхилення) ===")
print("Середнє =", input_data.mean(axis=0))
print("Стандартне відхилення =", input_data.std(axis=0))

data_scaled = preprocessing.scale(input_data)
print("\n=== ПІСЛЯ стандартизації (середнє / стандартне відхилення) ===")
print("Середнє =", data_scaled.mean(axis=0))
print("Стандартне відхилення =", data_scaled.std(axis=0))

# 3) Масштабування MinMax
scaler_minmax = preprocessing.MinMaxScaler(feature_range=(0, 1))
data_scaled_minmax = scaler_minmax.fit_transform(input_data)
print("\n=== Масштабування MinMax до [0, 1] ===")
print(data_scaled_minmax)

# 4) Нормалізація (L1, L2)
data_normalized_l1 = preprocessing.normalize(input_data, norm='l1')
data_normalized_l2 = preprocessing.normalize(input_data, norm='l2')
print("\n=== Нормалізація L1 (пострічкова) ===")
print(data_normalized_l1)
print("\n=== Нормалізація L2 (пострічкова) ===")
print(data_normalized_l2)
