# recommender_api.py

from fastapi import FastAPI, HTTPException
import pandas as pd
import numpy as np
import pickle

from sklearn.metrics.pairwise import cosine_similarity
from sklearn.neighbors import NearestNeighbors
from sklearn.feature_extraction.text import TfidfVectorizer
from ast import literal_eval
import json
import os

app = FastAPI()

# Пути к файлам
MOVIES_FILE = 'movies_metadata.csv'
RATINGS_FILE = 'ratings_small.csv'
KEYWORDS_FILE = 'keywords.csv'
LINKS_FILE = 'links_small.csv'
USER_RATINGS_FILE = 'user_ratings.json'
TFIDF_MODEL_FILE = 'tfidf_vectorizer.pkl'
NN_MODEL_FILE = 'nn_model.pkl'

# Глобальные переменные
movies: pd.DataFrame = None
tfidf = None
tfidf_matrix = []
nn_model: NearestNeighbors = None
item_sim_df: pd.DataFrame = None
title_to_index = None
user_ratings = {}
user_item_matrix: pd.DataFrame = None


def load_user_ratings():
    global user_ratings
    if os.path.exists(USER_RATINGS_FILE):
        with open(USER_RATINGS_FILE, 'r') as f:
            user_ratings = json.load(f)
    else:
        user_ratings = {}


def save_user_ratings():
    with open(USER_RATINGS_FILE, 'w') as f:
        json.dump(user_ratings, f)


def load_data():
    global movies, tfidf, tfidf_matrix, nn_model, item_sim_df, title_to_index, user_item_matrix

    movies = pd.read_csv(MOVIES_FILE, low_memory=False)
    ratings = pd.read_csv(RATINGS_FILE, low_memory=False)
    keywords = pd.read_csv(KEYWORDS_FILE, low_memory=False)
    links = pd.read_csv(LINKS_FILE, low_memory=False)

    movies['id'] = pd.to_numeric(movies['id'], errors='coerce')
    movies = movies.dropna(subset=['id'])
    movies['id'] = movies['id'].astype('int')

    ratings = ratings.merge(links[['movieId', 'tmdbId']], left_on='movieId', right_on='movieId', how='inner')
    ratings['movieId'] = pd.to_numeric(ratings['tmdbId'], errors='coerce')
    ratings = ratings.merge(
        movies[['id', 'original_title', 'genres', 'overview', 'vote_average', 'vote_count', 'release_date']],
        left_on='movieId', right_on='id', how='inner')

    def get_genres(x):
        try:
            return ' '.join([d['name'] for d in literal_eval(x)]) if pd.notnull(x) else ''
        except:
            return ''

    movies['genres_str'] = movies['genres'].apply(get_genres)

    def get_keywords(x):
        try:
            return ' '.join([d['name'] for d in literal_eval(x)]) if pd.notnull(x) else ''
        except:
            return ''

    keywords['keywords_str'] = keywords['keywords'].apply(get_keywords)
    movies = movies.merge(keywords[['id', 'keywords_str']], on='id', how='left')

    movies['soup'] = movies['overview'].fillna('') + ' ' + movies['genres_str'] + ' ' + movies['keywords_str'].fillna(
        '')

    tfidf = TfidfVectorizer(stop_words='english')
    tfidf_matrix = tfidf.fit_transform(movies['soup'])

    nn_model = NearestNeighbors(n_neighbors=11, metric='cosine', algorithm='brute')
    nn_model.fit(tfidf_matrix)

    def weighted_rating(x, m=movies['vote_count'].quantile(0.8), c=movies['vote_average'].mean()):
        v = x['vote_count']
        R = x['vote_average']
        return (v / (v + m) * R) + (m / (m + v) * c)

    movies['weighted_rating'] = movies.apply(weighted_rating, axis=1)

    title_to_index = pd.Series(movies.index, index=movies['id'])

    user_item_matrix = ratings.pivot_table(index='userId', columns='movieId', values='rating').fillna(0)
    item_sim = cosine_similarity(user_item_matrix.T)
    item_sim_df = pd.DataFrame(item_sim, index=user_item_matrix.columns, columns=user_item_matrix.columns)

    with open(TFIDF_MODEL_FILE, 'wb') as f:
        pickle.dump(tfidf, f)
    with open(NN_MODEL_FILE, 'wb') as f:
        pickle.dump(nn_model, f)


# Загрузка данных при старте
load_user_ratings()
load_data()


@app.get("/top10_popular")
def api_get_top10_popular():
    top10 = movies.sort_values('weighted_rating', ascending=False)['original_title'].head(10).tolist()
    return {"recommendations": top10}


@app.get("/recommend_by_genre")
def api_recommend_by_genre(genre: str, n: int = 10):
    genre_movies = movies[movies['genres_str'].str.contains(genre, case=False, na=False)]
    genre_movies = genre_movies.sort_values('weighted_rating', ascending=False)
    recs = genre_movies['original_title'].head(n).tolist()
    return {"recommendations": recs}


@app.get("/search_movies")
def api_search_movies(query: str):
    matches = movies[movies['original_title'].str.lower() == query.lower()]
    return {"movies": matches[['id', 'original_title', 'release_date']].replace(np.nan, None).to_dict('records')}


@app.get("/recommend_by_title")
def api_recommend_by_title(movie_id: int, n: int = 10):
    if movie_id not in movies['id'].values:
        raise HTTPException(status_code=404, detail="Movie not found")
    idx = movies[movies['id'] == movie_id].index[0]
    movie_vector = tfidf_matrix[idx]
    distances, indices = nn_model.kneighbors(movie_vector, n_neighbors=n + 1)
    movie_indices = indices[0][1:]
    recs = movies['original_title'].iloc[movie_indices].tolist()
    return {"recommendations": recs}


@app.get("/list_all_movies")
def api_list_all_movies():
    movie_list = movies[['original_title', 'release_date']].replace(np.nan, None).to_dict('records')
    return {"movies": movie_list}


@app.post("/set_rating")
def api_set_rating(user_id: str, movie_id: int, rating: float):
    if movie_id not in movies['id'].values:
        raise HTTPException(status_code=404, detail="Movie not found")
    if user_id not in user_ratings:
        user_ratings[user_id] = {}
    user_ratings[user_id][movie_id] = rating
    save_user_ratings()
    return {"status": "success"}


@app.get("/get_user_ratings")
def api_get_user_ratings(user_id: str):
    if user_id not in user_ratings or not user_ratings[user_id]:
        raise HTTPException(status_code=404, detail="No ratings for user")
    user_ratings_list = []
    for movie_id, rating in user_ratings[user_id].items():
        movie = (movies[movies['id'] == int(movie_id)][['original_title', 'release_date', 'genres_str']]
                 .replace(np.nan, None).to_dict('records'))
        if movie:
            user_ratings_list.append({
                "movie_id": movie_id,
                "title": movie[0]['original_title'],
                "release_date": movie[0]['release_date'],
                "genres": movie[0]['genres_str'],
                "rating": rating
            })
    return {"ratings": user_ratings_list}


@app.get("/recommend_for_user")
def api_recommend_for_user(user_id: str, n: int = 10):
    if user_id not in user_ratings or not user_ratings[user_id]:
        raise HTTPException(status_code=404, detail="No ratings for user")
    temp_matrix = user_item_matrix.copy()
    if user_id not in temp_matrix.index:
        temp_matrix.loc[user_id] = 0
    for movie_id, rating in user_ratings.get(user_id, {}).items():
        if movie_id in temp_matrix.columns:
            temp_matrix.loc[user_id, movie_id] = rating
    user_sim = cosine_similarity(temp_matrix)
    user_sim_df = pd.DataFrame(user_sim, index=temp_matrix.index, columns=temp_matrix.index)
    similar_users = user_sim_df.loc[user_id].sort_values(ascending=False).iloc[1:11].index
    user_rated = set(user_ratings[user_id].keys())
    recommendations = {}
    for sim_user in similar_users:
        sim_user_ratings = temp_matrix.loc[sim_user]
        for movie_id, rating in sim_user_ratings[sim_user_ratings > 0].items():
            if movie_id not in user_rated:
                if movie_id not in recommendations:
                    recommendations[movie_id] = 0
                recommendations[movie_id] += rating * user_sim_df.loc[user_id, sim_user]
    if not recommendations:
        raise HTTPException(status_code=404, detail="No recommendations")
    sorted_recs = sorted(recommendations.items(), key=lambda x: x[1], reverse=True)[:n]
    rec_movie_ids = [m[0] for m in sorted_recs]
    recs = movies[movies['id'].isin(rec_movie_ids)]['original_title'].tolist()
    return {"recommendations": recs}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
