import streamlit as st
st.set_page_config(page_title="Модератор токсичных комментариев (по Захаренко)", layout="wide")
import pickle
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
import sqlite3
from datetime import datetime
import pandas as pd
import re, nltk, pymorphy3
from nltk.corpus import stopwords

# Предобработка (идентична обучению)
nltk.download('stopwords', quiet=True)
nltk.download('punkt', quiet=True)
morph = pymorphy3.MorphAnalyzer()
stop_words = set(stopwords.words('russian'))


def preprocess_text(text):
    text = text.lower().strip()
    text = re.sub(r'http\S+|www\S+|https\S+', '', text)
    text = re.sub(r'\W', ' ', text)
    text = re.sub(r'\s+', ' ', text)
    tokens = nltk.word_tokenize(text)
    tokens = [morph.parse(word)[0].normal_form for word in tokens if word not in stop_words]
    return ' '.join(tokens)


@st.cache_resource
def load_resources():
    model = load_model('multi_toxic_ru.h5')
    with open('tokenizer_multi.pkl', 'rb') as f:
        tokenizer = pickle.load(f)
    return model, tokenizer


model, tokenizer = load_resources()
MAX_LEN = 120
THRESHOLD = 0.5


# БД
def get_db():
    conn = sqlite3.connect('comments.db')
    conn.row_factory = sqlite3.Row
    return conn


def save_comment(comment, probs, main_class, is_toxic):
    conn = get_db()
    c = conn.cursor()
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    c.execute('''INSERT INTO comments 
                 (comment_text, normal, insult, threat, obscenity, 
                  main_class, is_toxic, timestamp)
                 VALUES (?,?,?,?,?,?,?,?)''',
              (comment, *probs, main_class, int(is_toxic), ts))
    conn.commit()
    conn.close()


def predict(comment):
    cleaned = preprocess_text(comment)
    seq = tokenizer.texts_to_sequences([cleaned])
    pad = pad_sequences(seq, maxlen=MAX_LEN, padding='post')
    probs = model.predict(pad, verbose=0)[0]
    classes = ['normal', 'insult', 'threat', 'obscenity']
    res = {cls: float(probs[i]) for i, cls in enumerate(classes)}
    main = max(res, key=res.get)
    is_toxic = any(p > THRESHOLD for p in probs[1:])
    return probs.tolist(), main, is_toxic, res


# ====================== ИНТЕРФЕЙС ======================
st.title("🔍 Фильтр токсичных комментариев (мульти-лейбл)")

comment = st.text_area("Введите комментарий:", height=150)

if st.button("Проверить и сохранить в БД"):
    if comment.strip():
        with st.spinner("Классификация..."):
            probs, main, is_toxic, res_dict = predict(comment)
            save_comment(comment, probs, main, is_toxic)

        st.subheader("Результат (4 метки)")
        cols = st.columns(4)
        colors = ["green", "orange", "red", "red"]
        for i, (cls, p) in enumerate(res_dict.items()):
            cols[i].metric(cls.upper(), f"{p:.3f}",
                           delta="ТОКСИЧНЫЙ" if p > THRESHOLD and cls != "normal" else None)

        verdict = "✅ НОРМАЛЬНЫЙ" if not is_toxic else "🚫 ТОКСИЧНЫЙ"
        st.success(f"**Вердикт:** {verdict} | Основной класс: **{main}**")

        if is_toxic:
            st.error("Комментарий отсеян!")
    else:
        st.warning("Введите текст")

# Просмотр БД
st.subheader("📋 Последние токсичные комментарии (из БД)")
conn = get_db()
df_log = pd.read_sql_query("""
    SELECT id, comment_text, main_class, insult, threat, obscenity, timestamp 
    FROM comments 
    WHERE is_toxic = 1 
    ORDER BY id DESC LIMIT 20
""", conn)
conn.close()

if not df_log.empty:
    st.dataframe(df_log.style.format({
        'insult': '{:.3f}', 'threat': '{:.3f}', 'obscenity': '{:.3f}'
    }).highlight_max(subset=['insult', 'threat', 'obscenity'], color='#ffcccc'),
                 use_container_width=True)
else:
    st.info("Пока нет токсичных записей")

st.caption("Модель BiLSTM • Датасет ~248k примеров • Предобработка как в статье Захаренко 2023")