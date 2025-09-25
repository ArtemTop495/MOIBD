import pandas as pd
import pickle
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from scipy.sparse import csr_matrix
from sklearn.metrics.pairwise import cosine_similarity
import json
import ast

app = FastAPI(title="Movie Recommender API")

# Load data
with open('tfidf_vectorizer.pkl', 'rb') as f:
    tfidf = pickle.load(f)
with open('nn_model.pkl', 'rb') as f:
    nn_content = pickle.load(f)
movies = pd.read_csv('movies_metadata.csv', low_memory=False)
ratings = pd.read_csv('ratings.csv', low_memory=False)
keywords = pd.read_csv('keywords.csv', low_memory=False)
links = pd.read_csv('links.csv', low_memory=False)

# Map TMDb IDs to MovieLens IDs
links['tmdbId'] = links['tmdbId'].astype(str).str.strip().replace('', pd.NA)
links = links.dropna(subset=['tmdbId'])
movies['id'] = movies['id'].astype(str).str.strip()
movies = movies.merge(links[['movieId', 'tmdbId']], left_on='id', right_on='tmdbId', how='left')
movies['movieId'] = movies['movieId'].astype(str).replace('nan', '')
title_to_index = pd.Series(movies.index, index=movies['original_title'])

# Debug data
print("Movies columns:", movies.columns)
print("Toy Story check:", movies[movies['original_title'].str.contains("Toy Story", case=False, na=False)][['original_title', 'id', 'movieId']])
print("Ratings movieId dtype:", ratings['movieId'].dtype)
print("Movies movieId dtype:", movies['movieId'].dtype)
print("Links sample:", links[['movieId', 'tmdbId']].head())

# Calculate weighted rating
def calculate_weighted_rating(df, m=None, C=None):
    if m is None:
        m = df['vote_count'].quantile(0.80)
    if C is None:
        C = df['vote_average'].mean()
    v = df['vote_count']
    R = df['vote_average']
    return (v / (v + m) * R) + (m / (v + m) * C)

# Clean and compute weighted rating
movies = movies[movies['vote_count'].notna() & movies['vote_average'].notna()]
movies['weighted_rating'] = calculate_weighted_rating(movies)

# Parse genres
def parse_genres(genres):
    try:
        genres_list = ast.literal_eval(genres) if pd.notna(genres) else []
        return ' '.join([g['name'] for g in genres_list if isinstance(g, dict) and 'name' in g])
    except (ValueError, SyntaxError):
        return ''
movies['genres_str'] = movies['genres'].apply(parse_genres)

# Create soup
def create_soup(row):
    genres = row['genres_str'] if pd.notna(row['genres_str']) else ''
    overview = row['overview'] if pd.notna(row['overview']) else ''
    return f"{genres} {overview}".strip()
movies['soup'] = movies.apply(create_soup, axis=1)

# Create sparse matrix
def create_sparse_matrix(ratings):
    user_ids = ratings['userId'].astype('category')
    movie_ids = ratings['movieId'].astype(str)
    sparse_matrix = csr_matrix(
        (ratings['rating'], (user_ids.cat.codes, movie_ids.cat.codes)),
        shape=(len(user_ids.cat.categories), len(movie_ids.cat.categories))
    )
    return sparse_matrix, user_ids.cat.categories, movie_ids.cat.categories

# Subsample ratings
top_users = ratings['userId'].value_counts().head(500).index
top_movies = ratings['movieId'].value_counts().head(2000).index
ratings = ratings[ratings['userId'].isin(top_users) & ratings['movieId'].isin(top_movies)]
user_ratings, user_index, movie_index = create_sparse_matrix(ratings)

# Recommend by genre
def recommend_by_genre(genre: str, n: int = 10):
    genre_movies = movies[movies['genres_str'].str.contains(genre, case=False, na=False)]
    genre_movies = genre_movies.sort_values('weighted_rating', ascending=False)
    return genre_movies['original_title'].head(n).tolist()

# Recommend by content (title)
def recommend_by_title(title: str, n: int = 10):
    print(f"Checking title: {title}")
    if title not in title_to_index:
        print(f"Title {title} not in title_to_index")
        return []
    idx = title_to_index[title]
    movie_soup = movies.loc[idx, 'soup']
    movie_vector = tfidf.transform([movie_soup])
    distances, indices = nn_content.kneighbors(movie_vector, n_neighbors=n+1)
    movie_indices = indices[0][1:]  # Exclude the movie itself
    return movies['original_title'].iloc[movie_indices].tolist()

# Recommend by collaborative filtering
def recommend_collaborative(movie_title: str, n: int = 10):
    print(f"Checking title: {movie_title}")
    if movie_title not in title_to_index:
        print(f"Title {movie_title} not in title_to_index")
        return []
    movie_id = movies[movies['original_title'] == movie_title]['movieId'].values
    print(f"Movie ID: {movie_id}")
    if len(movie_id) == 0 or movie_id[0] == '':
        print(f"No movieId found for {movie_title}")
        return []
    movie_id = str(movie_id[0])
    movie_idx = pd.Series(movie_index, index=movie_index).get(movie_id)
    print(f"Movie index: {movie_idx}")
    if movie_idx is None:
        print(f"No movie_idx found for movieId {movie_id}")
        return []
    movie_vector = user_ratings[:, movie_idx]
    similarities = cosine_similarity(movie_vector.T, user_ratings.T)[0]
    similar_indices = similarities.argsort()[-n-1:-1][::-1]
    similar_movie_ids = movie_index[similar_indices]
    similar_titles = movies[movies['movieId'].isin(similar_movie_ids)]['original_title'].head(n).tolist()
    print(f"Recommendations: {similar_titles}")
    return similar_titles

# Pydantic models
class TitleRequest(BaseModel):
    title: str

class GenreRequest(BaseModel):
    genre: str

# API Endpoints
@app.get("/top10")
async def get_top10():
    try:
        return movies.sort_values('weighted_rating', ascending=False)['original_title'].head(10).tolist()
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

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
            similar_titles = movies[movies['original_title'].str.contains(title, case=False, na=False)]['original_title'].tolist()
            error_detail = f"No recommendations found for title: {title}"
            if similar_titles:
                error_detail += f". Did you mean one of these? {similar_titles[:5]}"
            raise HTTPException(status_code=404, detail=error_detail)
        return recommendations
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/recommend/collaborative")
async def get_collaborative(title: str = "Toy Story"):
    try:
        recommendations = recommend_collaborative(title)
        if not recommendations:
            similar_titles = movies[movies['original_title'].str.contains(title, case=False, na=False)]['original_title'].tolist()
            error_detail = f"No collaborative recommendations found for title: {title}"
            if similar_titles:
                error_detail += f". Did you mean one of these? {similar_titles[:5]}"
            raise HTTPException(status_code=404, detail=error_detail)
        return recommendations
    except HTTPException as e:
        raise e
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")