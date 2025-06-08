from fastapi import FastAPI
import pickle
from pydantic import BaseModel
import pandas as pd
import numpy as np
import string
import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import pymorphy3

app = FastAPI()
with open('model_knn_pdf_files.pkl', 'rb') as file:
    model = pickle.load(file)

with open('vectorizer_pdf_files.pkl', 'rb') as file:
    vectorizer = pickle.load(file)


def remove_english_words(text):
    # Удаляет слова, состоящие только из английских букв (включая сокращения)
    return re.sub(r'\b[a-zA-Z]+\b', '', text)


def remove_punctuation(text):
    return "".join([ch if ch not in string.punctuation else ' ' for ch in text])


def remove_numbers(text):
    return ''.join([i if not i.isdigit() else ' ' for i in text])


def remove_multiple_spaces(text):
    return re.sub(r'\s+', ' ', text, flags=re.I)


def fun_prepare(text):
    return remove_english_words(remove_multiple_spaces(remove_numbers(remove_punctuation(text.lower()))))


def fun_punctuation_text(text):
    text = text.lower()
    text = ''.join([ch for ch in text if ch not in string.punctuation])
    text = ''.join([i if not i.isdigit() else '' for i in text])
    text = ''.join([i if i.isalpha() else ' ' for i in text])
    text = re.sub(r'\s+', ' ', text, flags=re.I)
    text = re.sub('[a-z]', '', text, flags=re.I)
    st = '❯\xa0'
    text = ''.join([ch if ch not in st else ' ' for ch in text])
    return text


def fun_lemmatizing_text(text):
    tokens = word_tokenize(text)
    res = list()
    for word in tokens:
        p = pymorphy3.MorphAnalyzer(lang='ru').parse(word)[0]
        res.append(p.normal_form)
    text = " ".join(res)
    return text


def fun_tokenize(text):
    nltk.download('stopwords')
    russian_stopwords = nltk.corpus.stopwords.words('russian')
    russian_stopwords.extend(['и', 'в', 'во', 'не', 'что', 'он', 'на', 'я', 'с', 'со', 'как', 'а', 'то', 'все', 'она', 'так', 'его', 'но', 'да', 'ты', 
    'к', 'у', 'же', 'вы', 'за', 'бы', 'по', 'только', 'ее', 'мне', 'было', 'вот', 'от', 'меня', 'еще', 'нет', 'о', 'из', 'ему', 
    'теперь', 'когда', 'даже', 'ну', 'вдруг', 'ли', 'если', 'уже', 'или', 'ни', 'быть', 'был', 'него', 'до', 'вас', 'нибудь', 
    'опять', 'уж', 'вам', 'ведь', 'там', 'потом', 'себя', 'ничего', 'ей', 'может', 'они', 'тут', 'где', 'есть', 'надо', 'ней', 
    'для', 'мы', 'тебя', 'их', 'чем', 'была', 'сам', 'чтоб', 'без', 'будто', 'чего', 'раз', 'тоже', 'себе', 'под', 'жизнь', 
    'будет', 'ж', 'тогда', 'кто', 'этот', 'того', 'потому', 'этого', 'какой', 'совсем', 'ним', 'здесь', 'этом', 'один', 'почти', 
    'тем', 'чтобы', 'нее', 'сейчас', 'были', 'куда', 'зачем', 'всех', 'никогда', 'можно', 'при', 'наконец', 'два', 'другой', 
    'хоть', 'после', 'над', 'больше', 'тот', 'через', 'эти', 'нас', 'про', 'всего', 'них', 'какая', 'много', 'разве', 'три', 'эту', 
    'моя', 'свою', 'этой', 'перед', 'иногда', 'лучше', 'чуть', 'том', 'нельзя', 'такой', 'им', 'более', 'всегда', 'конечно', 'всю', 
    'между', 'это', 'пока', 'об', 'часто', 'теперь', 'слишком', 'либо','хабр', 'хабрахабр', 'habr', 'комментарий', 'комментарии', 'статья', 'статьи', 'автор', 'пользователь', 'пост', 'блог',
    'читать', 'прочитать', 'написать', 'написал', 'подробнее', 'рейтинг', 'плюс', 'минус', 'карма', 'хаб', 'хабы',
    'git', 'github', 'пример', 'код', 'кода', 'коде', 'коду', 'кодом', 'python', 'javascript', 'программа', 'проект', 
    'использовать', 'использование', 'файл', 'файлы', 'сделать', 'создать', 'решение', 'задача', 'задачи', 'работа', 
    'примеры', 'документация', 'класс', 'функция', 'метод', 'сервер', 'клиент', 'данные', 'данных', 'база', 'базы',
    'http', 'https', 'www', 'com', 'ru', 'ссылка', 'img', 'src', 'div', 'класса', 'alt', 'nbsp', 'quot', 'lt', 'gt', 
    'является', 'являются', 'можно', 'найти', 'стоит', 'хочу', 'хотел', 'кажется', 'получить', 'помощью',
    'сегодня', 'вчера', 'неделя', 'месяц', 'год', 'года', 'лет', 'время', 'раз', 'версия'])
    t = word_tokenize(text)
    tokens = [token for token in t if token not in russian_stopwords]
    text = " ".join(tokens)
    return text


def fun_pred_text(text):
    text = fun_prepare(text)
    text = fun_punctuation_text(text)
    text = fun_tokenize(text)
    text = fun_lemmatizing_text(text)
    text = fun_tokenize(text)
    return text


def predict_cluster(text, threshold=0.1):
    text_vectorized = vectorizer.transform([fun_pred_text(text)])
    probabilities = model.predict_proba(text_vectorized)[0]
    prediction = model.predict(text_vectorized)[0]
    max_prob = max(probabilities)
    selected_clusters = [
        f'{i}' for i, prob in enumerate(probabilities)
        if max_prob - prob <= 0
    ]
    rez1 = f"{' или '.join(selected_clusters)}"
    rez2 = f"Вероятности по кластерам: {probabilities}"
    return rez1, rez2


class Item(BaseModel):
    text: str


@app.post("/predict")
def post_pred_text(item: Item):
    return {'cluster': predict_cluster(item.text)}

# uvicorn api:app --reload
# streamlit run app.py