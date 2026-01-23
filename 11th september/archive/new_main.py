# app.py

import streamlit as st
import requests
import uuid

API_URL = "http://127.0.0.1:8002"  # Или URL развернутого API


def main():
    st.title("Рекомендатель фильмов")

    if 'user_id' not in st.session_state:
        st.session_state.user_id = str(uuid.uuid4())

    # user_id = st.session_state.user_id
    user_id = "37cb45ae-efca-4685-9cab-2abac3d0739a"

    # Сайдбар для навигации
    menu = st.sidebar.selectbox("Выберите опцию", [
        "Топ-10 популярных",
        "Рекомендации по жанру",
        "Рекомендации по названию",
        "Рекомендации для пользователя",
        "Поставить оценку",
        "Список всех фильмов",
        "Мои оценки",  # New menu option
        "Справка"
    ])

    if menu == "Топ-10 популярных":
        st.header("Топ-10 популярных фильмов")
        response = requests.get(f"{API_URL}/top10_popular")
        if response.status_code == 200:
            recs = response.json()["recommendations"]
            for rec in recs:
                st.write(rec)
        else:
            st.error("Ошибка при получении данных")

    elif menu == "Рекомендации по жанру":
        st.header("Рекомендации по жанру")
        genres = [
            "Action", "Adventure", "Animation", "Comedy", "Crime", "Documentary", "Drama",
            "Family", "Fantasy", "History", "Horror", "Music", "Mystery", "Romance",
            "Science Fiction", "Thriller", "War", "Western"
        ]
        genre = st.selectbox("Select Genre for Recommendations", genres)
        if st.button("Получить рекомендации"):
            if genre:
                response = requests.get(f"{API_URL}/recommend_by_genre", params={"genre": genre})
                if response.status_code == 200:
                    recs = response.json()["recommendations"]
                    if recs:
                        for rec in recs:
                            st.write(rec)
                    else:
                        st.write("Нет рекомендаций для этого жанра")
                else:
                    st.error("Ошибка при получении данных")
            else:
                st.write("Введите жанр")

    elif menu == "Рекомендации по названию":
        st.header("Рекомендации по названию (по контенту)")
        title_query = st.text_input("Введите название фильма")
        if 'movies_found_title' not in st.session_state:
            st.session_state.movies_found_title = []
        if st.button("Найти фильмы"):
            if title_query:
                response = requests.get(f"{API_URL}/search_movies", params={"query": title_query})
                if response.status_code == 200:
                    st.session_state.movies_found_title = response.json()["movies"]
                    if not st.session_state.movies_found_title:
                        st.write("Фильм не найден")
                else:
                    st.error("Ошибка при получении данных")
            else:
                st.write("Введите название")
        if st.session_state.movies_found_title:
            options = [f"{m['original_title']} ({m.get('release_date', 'N/A')})" for m in
                       st.session_state.movies_found_title]
            selected = st.selectbox("Выберите фильм", options)
            if selected:
                selected_index = options.index(selected)
                selected_movie = st.session_state.movies_found_title[selected_index]
                movie_id = selected_movie['id']
                if st.button("Получить рекомендации"):
                    response_rec = requests.get(f"{API_URL}/recommend_by_title", params={"movie_id": movie_id})
                    if response.status_code == 200:
                        recs = response_rec.json()["recommendations"]
                        if recs:
                            for rec in recs:
                                st.write(rec)
                        else:
                            st.write("Нет рекомендаций")
                    else:
                        st.error(response_rec.json()["detail"])

    elif menu == "Рекомендации для пользователя":
        st.header("Рекомендации для пользователя")
        if st.button("Получить рекомендации"):
            response = requests.get(f"{API_URL}/recommend_for_user", params={"user_id": user_id})
            if response.status_code == 200:
                recs = response.json()["recommendations"]
                if recs:
                    for rec in recs:
                        st.write(rec)
                else:
                    st.write("Нет рекомендаций")
            else:
                st.error(response.json()["detail"])

    elif menu == "Поставить оценку":
        st.header("Поставить оценку")
        title_query = st.text_input("Введите название фильма")
        if 'movies_found_rating' not in st.session_state:
            st.session_state.movies_found_rating = []
        if st.button("Найти фильмы"):
            if title_query:
                response = requests.get(f"{API_URL}/search_movies", params={"query": title_query})
                if response.status_code == 200:
                    st.session_state.movies_found_rating = response.json()["movies"]
                    if not st.session_state.movies_found_rating:
                        st.write("Фильм не найден")
                else:
                    st.error("Ошибка при получении данных")
            else:
                st.write("Введите название")
        if st.session_state.movies_found_rating:
            options = [f"{m['original_title']} ({m.get('release_date', 'N/A')})" for m in
                       st.session_state.movies_found_rating]
            selected = st.selectbox("Выберите фильм", options)
            if selected:
                selected_index = options.index(selected)
                selected_movie = st.session_state.movies_found_rating[selected_index]
                movie_id = selected_movie['id']
                rating = st.number_input("Оценка (1-5)", min_value=1.0, max_value=5.0, step=0.5)
                if st.button("Сохранить оценку"):
                    response = requests.post(f"{API_URL}/set_rating",
                                             params={"user_id": user_id, "movie_id": movie_id, "rating": rating})
                    if response.status_code == 200:
                        st.success("Оценка сохранена")
                    else:
                        st.error(response.json()["detail"])

    elif menu == "Список всех фильмов":
        st.header("Список всех фильмов")
        response = requests.get(f"{API_URL}/list_all_movies")
        if response.status_code == 200:
            movies_list = response.json()["movies"]
            for movie in movies_list:
                st.write(f"{movie['original_title']} ({movie.get('release_date', 'N/A')})")
        else:
            st.error("Ошибка при получении данных")

    elif menu == "Мои оценки":
        st.header("Мои оценки")
        response = requests.get(f"{API_URL}/get_user_ratings", params={"user_id": user_id})
        if response.status_code == 200:
            ratings = response.json()["ratings"]
            if ratings:
                for rating in ratings:
                    st.write(f"Фильм: {rating['title']} ({rating.get('release_date', 'N/A')})")
                    st.write(f"Жанры: {rating['genres'] if rating['genres'] else 'N/A'}")
                    st.write(f"Оценка: {rating['rating']}")
                    st.write("---")
            else:
                st.write("Вы еще не оценили ни одного фильма")
        else:
            st.error(response.json()["detail"])

    elif menu == "Справка":
        st.header("Справка")
        st.write("""
Справка по командам:

- Топ-10 популярных: Показывает топ-10 фильмов по взвешенным рейтингам. Без параметров.

- Рекомендации по жанру: Введите жанр в поле. Показывает 10 рекомендаций.

- Рекомендации по названию: Введите название в поле 'Название фильма'. Рекомендации по контенту.

- Коллаборативные рекомендации: Введите название в поле 'Название фильма'. Item-based CF.

- Рекомендации для пользователя: Рекомендации на основе оценок пользователя.

- Поставить оценку: Введите название фильма и оценку (1-5).

- Список всех фильмов: Показывает все фильмы с датами релиза.

- Мои оценки: Показывает все ваши оценки с названиями фильмов, датами релиза и жанрами.

- Справка: Показывает это сообщение.
        """)


if __name__ == "__main__":
    main()
