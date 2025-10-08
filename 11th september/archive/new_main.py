import streamlit as st
import requests
import os

# Clear proxy settings for correct connection
os.environ['HTTP_PROXY'] = ''
os.environ['HTTPS_PROXY'] = ''

st.title("Movie Recommender System")

# Sidebar help
with st.sidebar:
    st.markdown("""
    ### Help
    - **Top 10 Popular Movies**: Click the button to get the top 10 movies by weighted rating. No parameters needed.
    - **Recommendations by Genre**: Select a genre from the dropdown (parameter: genre). Available genres: Action, Adventure, Animation, Comedy, Crime, Documentary, Drama, Family, Fantasy, History, Horror, Music, Mystery, Romance, Science Fiction, Thriller, War, Western.
    - **Content-Based Recommendations**: Enter a movie title (parameter: title). Optionally, specify release_date (e.g., '1987-09-01') or tmdb_id (e.g., '14815') to disambiguate duplicate titles. Returns recommendations based on content (TF-IDF and NearestNeighbors).
    - **Collaborative Filtering Recommendations**: Enter a movie title (parameter: title). Optionally, specify release_date or tmdb_id. Returns recommendations based on collaborative filtering (NearestNeighbors on ratings).
    - **Browse All Movies**: Click to load the list of all movies with titles, release dates, and TMDb IDs.
    - **Rate a Movie**: Enter movie title (and optional date/ID), select rating, and submit. Ratings are saved and used for personal recommendations.
    - **Personal Recommendations**: Get collaborative recommendations based on your saved ratings.

    API URL: http://127.0.0.1:8000
    """)

# Button for top 10
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

# Combo-box for genre
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

# Input fields for content-based recommendations
st.markdown("### Content-Based Recommendations")
content_title = st.text_input("Enter Movie Title for Content-Based Recommendations", key="content_title")
content_release_date = st.text_input("Enter Release Date (YYYY-MM-DD, optional)", key="content_release_date")
content_tmdb_id = st.text_input("Enter TMDb ID (optional)", key="content_tmdb_id")
if st.button("Get Content-Based Recommendations"):
    if content_title:
        try:
            params = {"title": content_title}
            if content_release_date:
                params["release_date"] = content_release_date
            if content_tmdb_id:
                params["tmdb_id"] = content_tmdb_id
            response = requests.get("http://127.0.0.1:8000/recommend/title", params=params)
            response.raise_for_status()
            rec_movies = response.json()
            st.markdown(f"### Content-Based Recommendations for '{content_title}'")
            if isinstance(rec_movies, list):
                st.warning("No recommendations found.")
                for i, movie in enumerate(rec_movies, 1):
                    st.write(f"{i}. {movie}")
            else:
                for movie_key, recommendations in rec_movies.items():
                    st.markdown(f"#### {movie_key}")
                    if not recommendations:
                        st.write("No recommendations available for this movie.")
                    else:
                        for i, movie in enumerate(recommendations, 1):
                            st.write(f"{i}. {movie}")
        except requests.exceptions.HTTPError as e:
            if e.response.status_code == 404:
                error_detail = e.response.json().get("detail", str(e))
                st.warning(error_detail)
            else:
                st.error(f"Error: {str(e)}")
        except Exception as e:
            st.error(f"Error: {str(e)}")
    else:
        st.warning("Please enter a movie title.")

# Input fields for collaborative filtering recommendations
st.markdown("### Collaborative Filtering Recommendations")
collab_title = st.text_input("Enter Movie Title for Collaborative Filtering Recommendations", key="collab_title")
collab_release_date = st.text_input("Enter Release Date (YYYY-MM-DD, optional)", key="collab_release_date")
collab_tmdb_id = st.text_input("Enter TMDb ID (optional)", key="collab_tmdb_id")
if st.button("Get Collaborative Filtering Recommendations"):
    if collab_title:
        try:
            params = {"title": collab_title}
            if collab_release_date:
                params["release_date"] = collab_release_date
            if collab_tmdb_id:
                params["tmdb_id"] = collab_tmdb_id
            response = requests.get("http://127.0.0.1:8000/recommend/collaborative", params=params)
            response.raise_for_status()
            rec_movies = response.json()
            st.markdown(f"### Collaborative Filtering Recommendations for '{collab_title}'")
            if isinstance(rec_movies, list):
                st.warning("No recommendations found.")
                for i, movie in enumerate(rec_movies, 1):
                    st.write(f"{i}. {movie}")
            else:
                for movie_key, recommendations in rec_movies.items():
                    st.markdown(f"#### {movie_key}")
                    if not recommendations:
                        st.write("No recommendations available for this movie.")
                    else:
                        for i, movie in enumerate(recommendations, 1):
                            st.write(f"{i}. {movie}")
        except requests.exceptions.HTTPError as e:
            if e.response.status_code == 404:
                error_detail = e.response.json().get("detail", str(e))
                st.warning(error_detail)
            else:
                st.error(f"Error: {str(e)}")
        except Exception as e:
            st.error(f"Error: {str(e)}")
    else:
        st.warning("Please enter a movie title.")

# Browse all movies
st.markdown("### Browse All Movies")
if st.button("Load All Movies"):
    try:
        response = requests.get("http://127.0.0.1:8000/movies")
        response.raise_for_status()
        movies_list = response.json()
        import pandas as pd
        df = pd.DataFrame(movies_list)
        df.rename(columns={'original_title': 'Title', 'release_date': 'Release Date', 'id': 'TMDb ID'}, inplace=True)
        st.dataframe(df)
    except Exception as e:
        st.error(f"Error: {str(e)}")

# Rate a movie
st.markdown("### Rate a Movie")
rate_title = st.text_input("Enter Movie Title to Rate", key="rate_title")
rate_release_date = st.text_input("Enter Release Date (YYYY-MM-DD, optional)", key="rate_release_date")
rate_tmdb_id = st.text_input("Enter TMDb ID (optional)", key="rate_tmdb_id")
rate_value = st.slider("Select Rating", min_value=0.5, max_value=5.0, step=0.5, value=3.0)
if st.button("Submit Rating"):
    if rate_title:
        try:
            body = {"title": rate_title, "release_date": rate_release_date or None, "tmdb_id": rate_tmdb_id or None, "rating": rate_value}
            response = requests.post("http://127.0.0.1:8000/rate", json=body)
            response.raise_for_status()
            st.success(response.json()["message"])
        except requests.exceptions.HTTPError as e:
            if e.response.status_code == 404:
                error_detail = e.response.json().get("detail", str(e))
                st.warning(error_detail)
            else:
                st.error(f"Error: {str(e)}")
        except Exception as e:
            st.error(f"Error: {str(e)}")
    else:
        st.warning("Please enter a movie title.")

# Personal recommendations
st.markdown("### Personal Recommendations")
if st.button("Get Personal Recommendations"):
    try:
        response = requests.get("http://127.0.0.1:8000/recommend/personal")
        response.raise_for_status()
        rec_movies = response.json()
        st.markdown("### Your Personalized Recommendations")
        if not rec_movies:
            st.warning("No recommendations found. Rate some movies first.")
        else:
            for i, movie in enumerate(rec_movies, 1):
                st.write(f"{i}. {movie}")
    except requests.exceptions.HTTPError as e:
        if e.response.status_code == 404:
            error_detail = e.response.json().get("detail", str(e))
            st.warning(error_detail)
        else:
            st.error(f"Error: {str(e)}")
    except Exception as e:
        st.error(f"Error: {str(e)}")