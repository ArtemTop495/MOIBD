import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
import pickle
import re

#Загрузка и подготовка данных
data = pd.read_csv('da.csv')  # Ваш датасет с колонками 'text' и 'label'

# Предобработка текста (упрощенный пример)
def preprocess(text):
    # Приводим к нижнему регистру
    text = str(text).lower()
    # Удаляем все символы, кроме русских букв и пробелов
    text = re.sub(r'[^а-яё\s]', ' ', text)
    # Заменяем множественные пробелы на один
    text = re.sub(r'\s+', ' ', text).strip()
    return text

data['processed'] = data['text'].apply(preprocess)

#  Векторизация текста
tfidf = TfidfVectorizer(max_features=5000)
X = tfidf.fit_transform(data['processed'])
y = data['type']

#  Обучение модели
model = LogisticRegression(max_iter=1000)
model.fit(X, y)

#  Сохранение модели и векторизатора
with open('model.pkl', 'wb') as f:
    pickle.dump(model, f)


with open('tfidf.pkl', 'wb') as f:
    pickle.dump(tfidf, f)