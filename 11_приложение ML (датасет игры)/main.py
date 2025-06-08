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
with open('model_knn_games.pkl', 'rb') as file:
    model = pickle.load(file)

with open('vectorizer_games.pkl', 'rb') as file:
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
    russian_stopwords.extend(
        ['издание', 'игра', 'игре', 'т.д.', 'т', 'д', 'это', 'который', 'с', 'своём', 'всем', 'наш', 'свой',
         'об этой игре', 'это', 'в', 'на', 'и', 'или', 'для', 'что', 'как', 'также', 'с', 'от', 'до', 'по', 'из', 'при',
         'во', 'без', 'о', 'об', 'у', 'а', 'но', 'же', 'ли', 'бы', 'быть', 'был', 'была', 'будет', 'есть', 'нет', 'все',
         'его', 'ее', 'их', 'мы', 'вы', 'они', 'наш', 'ваш', 'который', 'этот', 'такие', 'тоже', 'игра', 'игры',
         'игровой', 'игрок', 'игроки', 'режим', 'режима', 'режимы', 'поддержка', 'поддерживает', 'доступен', 'доступна',
         'доступны', 'поддерживает', 'уровень', 'уровни', 'миссия', 'миссии', 'задания', 'задание', 'команда',
         'команды', 'персонаж', 'персонажи', 'система', 'системы', 'возможность', 'возможности', 'новый', 'новая',
         'новые', 'мир', 'мира', 'миры', 'разные', 'различные', 'вид', 'виды', 'тип', 'типы', 'версия', 'версии',
         'контент', 'контента', 'обновление', 'обновления', 'дополнение', 'дополнения', 'доступ', 'платформа',
         'платформы', 'windows', 'steam', 'pc', 'ps4', 'ps5', 'xbox', 'онлайн', 'singleplayer', 'multiplayer', 'coop',
         'игровой', 'механика', 'жанр', 'можно', 'нужно', 'будет', 'более', 'менее', 'позволяет', 'таким', 'образом',
         'далее', 'после', 'кроме', 'ещё', 'включая', 'включает', 'основе', 'основе', 'впереди', 'перед', 'через',
         'время', 'только', 'особый', 'каждый', 'несколько', 'разный', 'возможный', 'невероятный', 'уникальный',
         'главный', 'разнообразие', 'бесконечный', 'необычный', 'множество', 'много', 'достаточно', 'большой',
         'огромный', 'маленький', 'весь', 'самый', 'новый', 'поздний', 'ранний', 'вторжение', 'второй', 'последний',
         'следующий', 'будущий', 'собственный', 'свой', 'ваш', 'наш', 'поэтому', 'почему', 'только', 'очень',
         'поистине', 'просто', 'вполне', 'возможно', 'наконец', 'всего', 'почти', 'даже', 'лишь', 'путешествие',
         'отправиться', 'отправляться', 'поездка', 'поезд', 'охота', 'охотник', 'персонажа', 'персонажейв',
         'персонажава', 'экспедиция', 'навык', 'мастерство', 'исследовать', 'изучать', 'отряд', 'миссия', 'задание',
         'сражение', 'поединок', 'план', 'поиск', 'приключение', 'битва', 'бой', 'поединок', 'путь', 'борьба', 'магия',
         'чудовище', 'призрак', 'город', 'герой', 'враг', 'союзник', 'история', 'сюжет', 'роль', 'предмет', 'объект',
         'вещь', 'мир', 'вселенная', 'пространство', 'время', 'реальность', 'облик', 'формировать', 'режим', 'игра',
         'игрок', 'миссия', 'интерфейс', 'настройка', 'поезд', 'уровень', 'система', 'режимосновать', 'персонаж',
         'предмет', 'обновление', 'поддержка', 'боевой', 'арена', 'кампания', 'редакция', 'особняк', 'станция',
         'платформенный', 'платформа', 'персонализация', 'выживание', 'снаряжение', 'возможность', 'объект', 'знание',
         'истина', 'настроить', 'подстроить', 'использовать', 'предлагать', 'раскрыть', 'создавать', 'создаваться',
         'включать', 'выйти', 'процесс', 'находить', 'получать', 'узнать', 'освоить', 'прийтись', 'пройти', 'стараться',
         'приходиться', 'представить', 'превзойти', 'позволять', 'предсказывать', 'адаптироваться', 'создавать',
         'разрабатывать', 'выполнять', 'происходить', 'отражать', 'направить', 'проникать', 'распоряжение', 'двигаться',
         'возникать', 'вмешиваться', 'жить', 'открывать', 'закрывать', 'ожидать', 'включать', 'перейти', 'позволить'])
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
    mapping = {
        0: "симулятор или вождение",
        1: "сражения и выживание",
        2: "кооператив, открытый мир, песочница, стратегия, рпг",
        3: "мморпг, рпг, головоломка, стратегия, хоррор",
        4: "шутер, открытый мир, стратегия, экшен, бой, строительство базы"
    }
    max_prob = max(probabilities)
    selected_clusters = [
        mapping[i] for i, prob in enumerate(probabilities)
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