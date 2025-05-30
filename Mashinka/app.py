import streamlit as st
import pandas as pd
import requests
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
from wordcloud import WordCloud

# Конфигурация страницы
st.set_page_config(
    page_title="Анализатор соц. медиа",
    page_icon="📊",
    layout="wide"
)

# Стили
st.markdown("""
<style>
    .positive { color: green; font-weight: bold; }
    .negative { color: red; font-weight: bold; }
    .stTextArea textarea { min-height: 150px; }
    .metric-box { padding: 15px; border-radius: 10px; background: #f0f2f6; }
</style>
""", unsafe_allow_html=True)


# Загрузка данных
@st.cache_data
def load_data():
    try:
        data = pd.read_csv('lala.csv')
        # Проверка необходимых колонок
        required_cols = ['text', 'type', 'date', 'name']
        for col in required_cols:
            if col not in data.columns:
                st.error(f"Отсутствует обязательная колонка: {col}")
                return pd.DataFrame()

        # Преобразование даты
        if 'date' in data.columns:
            data['date'] = pd.to_datetime(data['date'], errors='coerce')

        # Добавление длины текста
        data['text_length'] = data['text'].str.len()

        return data

    except Exception as e:
        st.error(f"Ошибка загрузки: {str(e)}")
        return pd.DataFrame()


data = load_data()

# Создаем вкладки
tab1, tab2, tab3 = st.tabs(["📝 Анализ текста", "📊 Полная статистика", "ℹ️ Руководство "])

with tab1:
    # Основной интерфейс анализа текста
    st.title("📊 Анализатор тональности текста")
    st.markdown("Введите текст для определения его эмоциональной окраски")

    text_input = st.text_area("Текст для анализа:", placeholder="Напишите ваш текст здесь...",
                            key="text_input", height=150)

    if st.button("Анализировать", type="primary", key="analyze_btn"):
        if not text_input.strip():
            st.warning("Пожалуйста, введите текст для анализа")
        else:
            try:
                # Отправка запроса к API
                response = requests.post(
                    "http://localhost:8000/predict",
                    json={"text": text_input},
                    timeout=5
                )

                if response.status_code == 200:
                    result = response.json()

                    # Отображение результата
                    sentiment_class = "positive" if result["sentiment"] == "positive" else "negative"
                    st.markdown(
                        f"Результат: <span class='{sentiment_class}'>{result['sentiment'].upper()}</span> "
                        f"(уверенность: {result['confidence']:.1%})",
                        unsafe_allow_html=True
                    )

                    # Визуализация вероятностей
                    fig, ax = plt.subplots(figsize=(8, 4))
                    ax.bar(["Негативный", "Позитивный"],
                           [result["probabilities"]["negative"], result["probabilities"]["positive"]],
                           color=["red", "green"])
                    ax.set_ylim(0, 1)
                    ax.set_ylabel("Вероятность")


                else:
                    st.error(f"Ошибка сервера: {response.status_code}")

            except requests.exceptions.RequestException:
                st.error("Не удалось подключиться к серверу анализа. Убедитесь, что API сервер запущен.")
                st.info("""
                Для запуска API выполните в терминале:
                ```bash
                uvicorn api:app --reload
                ```
                """)
with tab2:
    st.title("📊 Полная статистика датасета")

    if data.empty:
        st.warning("Данные не загружены")
    else:
        # Основные метрики
        st.subheader("Ключевые показатели")
        cols = st.columns(4)
        with cols[0]:
            st.metric("Всего записей", len(data))
        with cols[1]:
            st.metric("Уникальных авторов", data['name'].nunique())
        with cols[2]:
            avg_len = data['text_length'].mean()
            st.metric("Ср. длина текста", f"{avg_len:.1f} симв.")
        with cols[3]:
            pos_pct = (data['type'] == 1).mean() * 100
            st.metric("Доля позитивных", f"{pos_pct:.1f}%")





        # Распределение длины текста
        st.subheader("Длина текста сообщений")
        fig3, ax3 = plt.subplots(figsize=(10, 4))
        data['text_length'].hist(bins=30, ax=ax3)
        ax3.set_xlabel("Длина текста (символы)")
        ax3.set_ylabel("Количество сообщений")
        st.pyplot(fig3)

        # Топ авторов
        st.subheader("Топ-10 активных авторов")
        top_authors = data['name'].value_counts().head(10)
        st.bar_chart(top_authors)

        # Облако слов
        st.subheader("Облако слов из сообщений")
        all_text = " ".join(data['text'].dropna())
        wordcloud = WordCloud(width=800, height=400, background_color='white').generate(all_text)
        fig4, ax4 = plt.subplots(figsize=(12, 6))
        ax4.imshow(wordcloud, interpolation='bilinear')
        ax4.axis('off')
        st.pyplot(fig4)

        # Показать сырые данные
        st.subheader("Просмотр данных")
        st.dataframe(data.head(100))

with tab3:
    st.title("📖 Руководство пользователя")

    st.header("1. Как анализировать текст")
    st.markdown("""
    1. Перейдите на вкладку **📝 Анализ текста**
    2. Введите или вставьте ваш текст в поле
    3. Нажмите кнопку **Анализировать**
    4. Получите результат с оценкой тональности
    """)


    st.subheader("Пример:")
    try:
        guide_img1 = Image.open("images/img.png")
        st.image(guide_img1, caption="Пример анализа текста", width=600)
    except:
        st.warning("Здесь будет скриншот интерфейса анализа (guide1.png)")

    st.header("2. Как смотреть статистику")
    st.markdown("""
    1. Перейдите на вкладку **📊 Статистика**
    2. Изучите графики распределения данных
    3. Используйте фильтры при необходимости
    """)


    st.subheader("Пример:")
    try:
        guide_img2 = Image.open("images/img_1.png")
        st.image(guide_img2, caption="Пример статистики", width=600)
    except:
        st.warning("Здесь будет скриншот раздела статистики (guide2.png)")

    st.header("3. Технические требования")
    st.markdown("""
    - Поддерживаемые браузеры: Chrome, Firefox, Edge
    - Минимальное разрешение экрана: 1280×720
    - Для загрузки больших файлов требуется стабильное интернет-соединение
    """)

    st.header("4. Частые вопросы")
    with st.expander("Как интерпретировать результаты?"):
        st.write("""
        - **Позитивный (>70% уверенности)**: Текст содержит преимущественно положительные высказывания
        - **Негативный (>70% уверенности)**: Преобладает негативная лексика
        - **Нейтральный (30-70%)**: Нельзя однозначно определить тональность
        """)

    with st.expander("Где взять примеры текстов для анализа?"):
        st.write("""
        Вы можете использовать:
        - Отзывы с сайтов
        - Комментарии из соцсетей
        - Собственные тексты
        """)


    st.subheader("Пример интерпретации:")
    try:
        guide_img3 = Image.open("images/img_2.png")
        st.image(guide_img3, caption="Пример интерпретации результатов", width=600)
    except:
        st.warning("Здесь будет скриншот с примерами результатов (guide3.png)")
# Сайдбар с информацией
with st.sidebar:
    st.header("Информация")
    if not data.empty:

        st.write(f"Всего записей: {len(data):,}")
        st.write(f"Уникальных авторов: {data['name'].nunique()}")

    st.markdown("""
    ### Навигация:
    - **Анализ текста**: Проверка тональности
    - **Статистика**: Визуализация данных
    - **О данных**: Описание и метаинформация
    """)