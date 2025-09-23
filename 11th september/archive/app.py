import pandas as pd
import pickle
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

# Initialize FastAPI app
app = FastAPI(title="Movie Recommender API")

# Load pickled models and data
with open('tfidf_vectorizer.pkl', 'rb') as f:
    tfidf = pickle.load(f)

with open('nn_model.pkl', 'rb') as f:
    nn_content = pickle.load(f)

movies = pd.read_csv('movies_metadata.csv', low_memory=False)
ratings = pd.read_csv('ratings.csv', low_memory=False)
keywords = pd.read_csv('keywords.csv', low_memory=False)
links = pd.read_csv('links.csv', low_memory=False)
title_to_index = pd.Series(movies.index, index=movies['original_title'])
user_ratings = ratings.pivot_table(index='userId', columns='movieId', values='rating').fillna(0)

# Recommend by genre
def recommend_by_genre(genre: str, n: int = 10):
    genre_movies = movies[movies['genres_str'].str.contains(genre, case=False, na=False)]
    genre_movies = genre_movies.sort_values('weighted_rating', ascending=False)
    return genre_movies['original_title'].head(n).tolist()

# Recommend by content (title) using NearestNeighbors
def recommend_by_title(title: str, n: int = 10):
    if title not in title_to_index:
        return []
    idx = title_to_index[title]
    # Transform the movie's soup using the loaded TfidfVectorizer
    movie_soup = movies.loc[idx, 'soup']
    movie_vector = tfidf.transform([movie_soup])  # Transform single movie
    distances, indices = nn_content.kneighbors(movie_vector, n_neighbors=n+1)
    movie_indices = indices[0][1:]  # Exclude the movie itself
    return movies['original_title'].iloc[movie_indices].tolist()

# Recommend by collaborative filtering using NearestNeighbors
def recommend_collaborative(movie_title: str, n: int = 10):
    if movie_title not in title_to_index:
        return []
    movie_id = movies[movies['original_title'] == movie_title]['id'].values
    if len(movie_id) == 0:
        return []
    movie_id = int(movie_id[0])
    if movie_id not in user_ratings.columns:
        return []
    similar_movies = user_ratings.loc[movie_id].sort_values(ascending=False).iloc[1:n+1].index
    similar_titles = movies[movies['id'].isin(similar_movies)]['original_title'].head(n).tolist()
    return similar_titles

# Pydantic model for request validation
class TitleRequest(BaseModel):
    title: str

class GenreRequest(BaseModel):
    genre: str

# API Endpoints
@app.get("/top10")
async def get_top10():
    return movies.sort_values('weighted_rating', ascending=False)['original_title'].head(10).tolist()

@app.get("/recommend/genre")
async def get_by_genre(genre: str = "Comedy"):
    try:
        recommendations = recommend_by_genre(genre)
        if not recommendations:
            raise HTTPException(status_code=404, detail=f"No movies found for genre: {genre}")
        return recommendations
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/recommend/title")
async def get_by_title(title: str = "Toy Story"):
    try:
        recommendations = recommend_by_title(title)
        if not recommendations:
            raise HTTPException(status_code=404, detail=f"No recommendations found for title: {title}")
        return recommendations
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/recommend/collaborative")
async def get_collaborative(title: str = "Toy Story"):
    try:
        recommendations = recommend_collaborative(title)
        if not recommendations:
            raise HTTPException(status_code=404, detail=f"No collaborative recommendations found for title: {title}")
        return recommendations
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Run with: uvicorn app:app --reload