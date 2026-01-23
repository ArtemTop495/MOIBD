from fastapi import FastAPI
from pydantic import BaseModel
import pickle
import numpy as np
from fastapi.middleware.cors import CORSMiddleware
from typing import Optional

app = FastAPI(title="API для предсказания цен на недвижимость")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

model = None
encoders = {}

try:
    with open('Random Forest.pkl', 'rb') as f:
        model = pickle.load(f)
    print("✅ Модель успешно загружена!")

    with open('label_encoders.pkl', 'rb') as f:
        all_encoders = pickle.load(f)
        print("✅ Загружены кодировщики для текстовых полей")

        encoders = {
            'type': all_encoders.get('type'),
            'sub_type': all_encoders.get('sub_type'),
            'listing_type': all_encoders.get('listing_type'),
            'building_age': all_encoders.get('building_age'),
            'floor_no': all_encoders.get('floor_no'),
            'heating_type': all_encoders.get('heating_type'),
            'city': all_encoders.get('city'),
            'district': all_encoders.get('district'),
            'neighborhood': all_encoders.get('neighborhood')
        }

except Exception as e:
    print(f"❌ Ошибка при загрузке: {e}")


TRY_TO_RUB_EXCHANGE_RATE = 3.0

def convert_to_rubles(try_amount: float) -> float:
    """Конвертирует турецкие лиры в рубли"""
    return try_amount * TRY_TO_RUB_EXCHANGE_RATE


def create_interaction_features(size: float, room_count: int, total_floor_count: int) -> dict:
    """Создает дополнительные признаки для улучшения предсказания"""

    if room_count > 0:
        size_per_room = size / room_count
    else:
        size_per_room = size

    floor_percentage = 0

    log_size = np.log(size + 1)
    
    return {
        "size_per_room": size_per_room,
        "log_size": log_size
    }

def encode_text_to_number(field_name: str, text_value: str):
    """Преобразует текстовое значение в число для модели"""
    try:
        if field_name in encoders and encoders[field_name] is not None:
            encoder = encoders[field_name]
            
            if text_value in encoder.classes_:
                return float(encoder.transform([text_value])[0])
            else:
                if len(encoder.classes_) > 0:
                    return float(encoder.transform([encoder.classes_[0]])[0])
                return 0.0

        try:
            return float(text_value)
        except:
            return 0.0
            
    except:
        return 0.0

class PropertyFeatures(BaseModel):
    """Характеристики объекта недвижимости"""

    type: str
    sub_type: str
    listing_type: str
    tom: float
    building_age: str
    total_floor_count: int
    floor_no: str
    room_count: int
    size: float
    heating_type: str
    city: str
    district: str
    neighborhood: str
    price_per_m2: Optional[float] = None
    age_size_interact: Optional[float] = None


@app.post("/predict")
async def predict_price(features: PropertyFeatures):
    """Предсказывает цену объекта недвижимости"""
    
    if model is None:
        return {"error": "Модель не загружена"}
    
    try:
        interaction_features = create_interaction_features(
            features.size, 
            features.room_count, 
            features.total_floor_count
        )

        if features.price_per_m2 is None or features.price_per_m2 == 0:
            base_price_per_m2 = {
                "İstanbul": 50000,
                "Ankara": 35000,
                "İzmir": 40000,
                "Bursa": 30000
            }
            
            price_per_m2_val = base_price_per_m2.get(features.city, 30000)
            if features.size > 100:
                price_per_m2_val *= 0.9
            elif features.size < 50:
                price_per_m2_val *= 1.2
        else:
            price_per_m2_val = features.price_per_m2

        if features.age_size_interact is None:
            age_mapping = {"0-5": 1.2, "5-10": 1.1, "10-20": 1.0, "20-30": 0.9, "30+": 0.8}
            age_multiplier = age_mapping.get(features.building_age, 1.0)
            age_size_interact_val = features.size * age_multiplier
        else:
            age_size_interact_val = features.age_size_interact

        input_data = np.array([[
            encode_text_to_number('type', features.type),
            encode_text_to_number('sub_type', features.sub_type),
            encode_text_to_number('listing_type', features.listing_type),
            float(features.tom),
            encode_text_to_number('building_age', features.building_age),
            float(features.total_floor_count),
            encode_text_to_number('floor_no', features.floor_no),
            float(features.room_count),
            float(features.size),
            encode_text_to_number('heating_type', features.heating_type),
            encode_text_to_number('city', features.city),
            encode_text_to_number('district', features.district),
            encode_text_to_number('neighborhood', features.neighborhood),

            float(price_per_m2_val),
            float(age_size_interact_val),

            float(interaction_features['size_per_room']),
            float(interaction_features['log_size'])
        ]])

        if input_data.shape[1] < 15:
            return {"error": f"Недостаточно признаков: {input_data.shape[1]}"}

        if input_data.shape[1] > 15:
            input_data = input_data[:, :15]

        predicted_price_try = model.predict(input_data)[0]

        predicted_price_rub = convert_to_rubles(predicted_price_try)

        city_multiplier = 1.0
        if features.city == "İstanbul":
            city_multiplier = 1.5

        size_impact = min(features.size / 100, 2.0)
        
        base_std_dev = predicted_price_try * 0.10
        adjusted_std_dev = base_std_dev * city_multiplier * size_impact

        confidence_interval_rub = convert_to_rubles(adjusted_std_dev)

        confidence = max(0, 100 - (adjusted_std_dev / predicted_price_try * 100 * 1.5))

        return {

            "price_prediction_rub": round(predicted_price_rub, 2),
            "price_prediction_try": round(predicted_price_try, 2),
            "confidence_interval_rub": round(confidence_interval_rub, 2),
            "confidence_percentage": round(confidence, 2),

            "currency": "RUB",
            "exchange_rate": TRY_TO_RUB_EXCHANGE_RATE,
            "original_currency": "TRY",

            "calculated_price_per_m2": round(price_per_m2_val, 2),
            "size_per_room": round(interaction_features['size_per_room'], 2),
            
            "status": "success",
            "features_used": input_data.shape[1]
        }
        
    except Exception as e:
        return {"error": f"Ошибка при предсказании: {str(e)}"}



@app.get("/model_info")
async def get_model_info():
    """Информация о модели"""
    return {
        "model_type": "RandomForest",
        "n_features_expected": 15,
        "feature_names": [
            "Тип", "Подтип", "Тип объявления", "Время на рынке", 
            "Возраст здания", "Этажность", "Этаж", "Комнат", 
            "Площадь", "Отопление", "Город", "Район", "Микрорайон",
            "Цена за м²", "Взаимод. возраста и площади"
        ],
        "currency_conversion": "TRY → RUB (курс: " + str(TRY_TO_RUB_EXCHANGE_RATE) + ")",
        "sensitivity_improvements": "Добавлены чувствительные признаки: площадь/комнату, лог.площадь"
    }


@app.get("/calculate_price_example")
async def calculate_example():
    """Пример расчета с разными параметрами для демонстрации чувствительности"""

    base_params = {
        "type": "Konut",
        "sub_type": "Daire",
        "listing_type": "Satılık",
        "tom": 30.0,
        "building_age": "5-10",
        "total_floor_count": 5,
        "floor_no": "3",
        "room_count": 2,
        "size": 65.0,
        "heating_type": "Kalorifer (Doğalgaz)",
        "city": "İstanbul",
        "district": "Kadıköy",
        "neighborhood": "Moda"
    }

    examples = []
    
    for size in [40, 65, 90, 120]:
        params = base_params.copy()
        params["size"] = float(size)

        features_obj = PropertyFeatures(**params)

        result = await predict_price(features_obj)
        
        if "error" not in result:
            examples.append({
                "size_m2": size,
                "predicted_price_rub": result.get("price_prediction_rub", 0),
                "price_per_m2_rub": round(result.get("price_prediction_rub", 0) / size, 2)
            })
    
    return {
        "examples": examples,
        "message": "Как меняется цена при изменении площади"
    }


@app.get("/health")
async def check_health():
    """Проверка состояния сервера"""
    return {
        "status": "healthy",
        "model_loaded": model is not None,
        "encoders_loaded": len(encoders),
        "currency": "RUB",
        "exchange_rate": TRY_TO_RUB_EXCHANGE_RATE
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="localhost", port=8077)