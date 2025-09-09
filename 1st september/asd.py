import pickle
import pandas as pd

# Исходный словарь
input_dict = {
    "type": "Konut",
    "sub_type": "Daire",
    "listing_type": 1,
    "tom": 30,
    "building_age": 5,
    "total_floor_count": 15.0,
    "floor_no": 3.0,
    "room_count": 3,
    "size": 100.0,
    "heating_type": "Fancoil",
    "city": "İstanbul",
    "district": "Kadıköy",
    "neighborhood": "Fenerbahçe"
}

# Проверка загрузки label_encoders
with open('label_encoders.pkl', 'rb') as f:
    loaded_encoders = pickle.load(f)

encoder_keys = loaded_encoders.keys()

# Создайте копию словаря для хранения закодированных значений
encoded_dict = input_dict.copy()

# Примените LabelEncoder к категориальным признакам
for key in encoder_keys:
    try:
        # Преобразуем значение в список (LabelEncoder ожидает массив)
        encoded_value = loaded_encoders[key].transform([encoded_dict[key]])[0]
        encoded_dict[key] = encoded_value
    except ValueError:
        # Обработка неизвестных категорий
        encoded_dict[key] = -1  # Или другое значение, например, 0 или значение для 'Unknown'
        print(f"Warning: Value '{encoded_dict[key]}' for key '{key}' not found in encoder. Assigned -1.")

# Выведите закодированный словарь
print("Encoded dictionary:", encoded_dict)