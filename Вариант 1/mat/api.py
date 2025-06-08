from fastapi import FastAPI
from pydantic import BaseModel
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np

app = FastAPI()

# Загрузка модели и векторизатора
with open('model_lr.pkl', 'rb') as f:
    model = pickle.load(f)

with open('vectorizer_updated.pkl', 'rb') as f:
    tfidf = pickle.load(f)


class TextRequest(BaseModel):
    text: str


@app.post("/predict")
async def predict(request: TextRequest):
    # Преобразование текста
    text_vector = tfidf.transform([request.text])

    # Предсказание
    prediction = model.predict(text_vector)[0]
    proba = model.predict_proba(text_vector)[0]

    return {
        "sentiment": "positive" if prediction == 1 else "negative",
        "confidence": float(np.max(proba)),
        "probabilities": {
            "positive": float(proba[1]),
            "negative": float(proba[0])
        }
    }


@app.get("/model_info")
async def model_info():
    return {
        "model_type": type(model).__name__,
        "vectorizer_type": "TF-IDF",
        "features": tfidf.get_feature_names_out().shape[0]
    }


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)

# uvicorn api:app --reload
# streamlit run app.py