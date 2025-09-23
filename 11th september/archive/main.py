import streamlit as st
import requests
import os

# Обнуление статусов прокси для корректного подключения
os.environ['HTTP_PROXY'] = ''
os.environ['HTTPS_PROXY'] = ''

st.title("Movie Recommender System")

# Справка в сайдбаре
with st.sidebar:
    st.markdown("""
    ### Справка по командам
    - **Top 10 Popular Movies**: Кнопка без параметров. Выводит топ-10 фильмов по взвешенной оценке.
    - **Recommendations by Genre**: Выберите жанр из комбо-бокса (параметр: genre). Доступные жанры: Action, Adventure, Animation, Comedy, Crime, Documentary, Drama, Family, Fantasy, History, Horror, Music, Mystery, Romance, Science Fiction, Thriller, War, Western.
    - **Content-Based Recommendations**: Введите название фильма (параметр: title). Рекомендации на основе контента (TF-IDF и NearestNeighbors).
    - **Collaborative Filtering Recommendations**: Введите название фильма (параметр: title). Рекомендации на основе коллаборативной фильтрации (NearestNeighbors на рейтингах).

    API URL: http://127.0.0.1:8000
    """)

# Кнопка для топ-10
if st.button("Get Top 10 Popular Movies"):
    try:
        response = requests.get("http://127.0.0.1:8000/top10")
        response.raise_for_status()
        top_movies = response.json()
        st.markdown("### Top 10 Popular Movies")
        for i, movie in enumerate(top_movies, 1):
            st.write(f"{i}. {movie}")
    except Exception as e:
        st.error(f"Error: {str(e)}")

# Комбо-бокс для жанра
genres = [
    "Action", "Adventure", "Animation", "Comedy", "Crime", "Documentary", "Drama",
    "Family", "Fantasy", "History", "Horror", "Music", "Mystery", "Romance",
    "Science Fiction", "Thriller", "War", "Western"
]
selected_genre = st.selectbox("Select Genre for Recommendations", genres)
if st.button("Get Recommendations by Genre"):
    try:
        response = requests.get(f"http://127.0.0.1:8000/recommend/genre?genre={selected_genre}")
        response.raise_for_status()
        rec_movies = response.json()
        st.markdown(f"### Recommendations for Genre: {selected_genre}")
        for i, movie in enumerate(rec_movies, 1):
            st.write(f"{i}. {movie}")
    except Exception as e:
        st.error(f"Error: {str(e)}")

# Поле ввода для контент-based рекомендаций
content_title = st.text_input("Enter Movie Title for Content-Based Recommendations")
if st.button("Get Content-Based Recommendations"):
    if content_title:
        try:
            response = requests.get(f"http://127.0.0.1:8000/recommend/title?title={content_title}")
            response.raise_for_status()
            rec_movies = response.json()
            st.markdown(f"### Content-Based Recommendations for '{content_title}'")
            for i, movie in enumerate(rec_movies, 1):
                st.write(f"{i}. {movie}")
        except Exception as e:
            st.error(f"Error: {str(e)}")
    else:
        st.warning("Please enter a movie title.")

# Поле ввода для коллаборативной фильтрации
collab_title = st.text_input("Enter Movie Title for Collaborative Filtering Recommendations")
if st.button("Get Collaborative Filtering Recommendations"):
    if collab_title:
        try:
            response = requests.get(f"http://127.0.0.1:8000/recommend/collaborative?title={collab_title}")
            response.raise_for_status()
            rec_movies = response.json()
            st.markdown(f"### Collaborative Filtering Recommendations for '{collab_title}'")
            for i, movie in enumerate(rec_movies, 1):
                st.write(f"{i}. {movie}")
        except Exception as e:
            st.error(f"Error: {str(e)}")
    else:
        st.warning("Please enter a movie title.")