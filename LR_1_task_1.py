# LR_1_task_1.py

from sklearn import preprocessing

# Початкові текстові мітки
input_labels = ['red', 'black', 'red', 'green', 'black', 'yellow']

# Створення енкодера
encoder = preprocessing.LabelEncoder()
encoder.fit(input_labels)

# Виведення класів
print("Класи:", list(encoder.classes_))

# Кодування міток у числовий формат
encoded_values = encoder.transform(input_labels)
print("Закодовані значення:", list(encoded_values))

# Декодування чисел назад у текст
decoded_list = encoder.inverse_transform(encoded_values)
print("Декодовані значення:", list(decoded_list))