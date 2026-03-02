import re
import nltk
import pandas as pd
import pymorphy3 as pymorphy2
from nltk.corpus import stopwords
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, Bidirectional, LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
import pickle
import tensorflow as tf
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

print("=" * 70)
print("TensorFlow версия:", tf.__version__)
print("Доступные устройства:")
for dev in tf.config.list_physical_devices():
    print("   →", dev)
print("GPU обнаружено:", len(tf.config.list_physical_devices('GPU')) > 0)
print("=" * 70)

# Скачиваем ресурсы NLTK
nltk.download('stopwords', quiet=True)
nltk.download('punkt', quiet=True)
nltk.download('punkt_tab', quiet=True)   # ← КРИТИЧНО!

morph = pymorphy2.MorphAnalyzer()
stop_words = set(stopwords.words('russian'))

def preprocess_text(text):
    text = text.lower().strip()
    text = re.sub(r'http\S+|www\S+|https\S+', '', text)
    text = re.sub(r'\W', ' ', text)
    text = re.sub(r'\s+', ' ', text)
    tokens = nltk.word_tokenize(text)
    tokens = [morph.parse(word)[0].normal_form for word in tokens if word not in stop_words]
    return ' '.join(tokens)

# Загрузка и предобработка (только dataset.txt, labeled.csv игнорируем)
data = []
with open('dataset.txt', 'r', encoding='utf-8') as f:
    for line in f:
        if not line.strip(): continue
        parts = line.strip().split(' ', 1)
        labels = parts[0].split(',')
        text = parts[1] if len(parts) > 1 else ""
        row = {
            'comment': preprocess_text(text),
            'normal':    1 if '__label__NORMAL' in labels else 0,
            'insult':    1 if '__label__INSULT' in labels else 0,
            'threat':    1 if '__label__THREAT' in labels else 0,
            'obscenity': 1 if '__label__OBSCENITY' in labels else 0
        }
        data.append(row)

df = pd.DataFrame(data)
df = df[df['comment'].str.strip() != '']

MAX_WORDS = 35000
MAX_LEN = 120

tokenizer = Tokenizer(num_words=MAX_WORDS, oov_token="<OOV>")
tokenizer.fit_on_texts(df['comment'])
sequences = tokenizer.texts_to_sequences(df['comment'])
padded = pad_sequences(sequences, maxlen=MAX_LEN, padding='post')

labels = df[['normal','insult','threat','obscenity']].values

X_train, X_test, y_train, y_test = train_test_split(padded, labels, test_size=0.12, stratify=labels.argmax(axis=1))

model = Sequential([
    Embedding(MAX_WORDS, 128, input_length=MAX_LEN),
    Bidirectional(LSTM(64, return_sequences=True)),
    Dropout(0.35),
    Bidirectional(LSTM(32)),
    Dropout(0.35),
    Dense(64, activation='relu'),
    Dropout(0.25),
    Dense(4, activation='sigmoid')
])

model.compile(optimizer='adam', loss='binary_crossentropy',
              metrics=['accuracy', 'Precision', 'Recall'])
early_stop = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)

model.fit(X_train, y_train, validation_data=(X_test, y_test),
          epochs=12, batch_size=96, callbacks=[early_stop])

model.save('multi_toxic_ru.h5')
with open('tokenizer_multi.pkl', 'wb') as f:
    pickle.dump(tokenizer, f)
print("Модель сохранена!")