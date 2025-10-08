import pandas as pd
import pickle
from fastapi import FastAPI, HTTPException, Body
from pydantic import BaseModel
from scipy.sparse import csr_matrix, vstack
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import ast
import time
import os

app = FastAPI(title="Movie Recommender API")

# Load data
with open('tfidf_vectorizer.pkl', 'rb') as f:
    tfidf = pickle.load(f)
with open('nn_model.pkl', 'rb') as f:
    nn_content = pickle.load(f)
movies = pd.read_csv('movies_metadata.csv', low_memory=False)
ratings = pd.read_csv('ratings.csv', low_memory=False)
links = pd.read_csv('links.csv', low_memory=False)

# Clean and align IDs
movies['id'] = pd.to_numeric(movies['id'], errors='coerce').astype('Int64').astype(str)
links['tmdbId'] = pd.to_numeric(links['tmdbId'], errors='coerce').astype('Int64').astype(str)
links = links.dropna(subset=['tmdbId'])
movies = movies.dropna(subset=['id'])
movies = movies.merge(links[['movieId', 'tmdbId']], left_on='id', right_on='tmdbId', how='inner')
movies['movieId'] = movies['movieId'].astype(str)
ratings['movieId'] = ratings['movieId'].astype(str)

# Filter movies to those in ratings
movies = movies[movies['movieId'].isin(ratings['movieId'].unique())]
title_to_index = pd.Series(movies.index, index=movies['original_title'])

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
    return f"{genres} {overview}".strip() or "unknown"

movies['soup'] = movies.apply(create_soup, axis=1)
print("Missing soup values:", movies['soup'].isna().sum())
print("Empty soup values:", (movies['soup'] == '').sum())

# Create sparse matrix
def create_sparse_matrix(ratings):
    from scipy.sparse import csr_matrix
    # Convert both user_ids and movie_ids to categorical
    user_ids = ratings['userId'].astype('category')
    movie_ids = ratings['movieId'].astype('category')
    # Create the sparse matrix
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

# Compute similarity matrix
sim_matrix = cosine_similarity(user_ratings.T)

# New user ID
new_user_id = ratings['userId'].max() + 1 if not ratings.empty else 1

# Load user ratings if exists and add to matrix
if os.path.exists('user_ratings.csv'):
    ur = pd.read_csv('user_ratings.csv')
    if not ur.empty:
        ur['movieId'] = ur['movieId'].astype(str)
        # Add new user to index and matrix
        user_index = user_index.append(pd.Index([new_user_id]))
        empty_row = csr_matrix((1, user_ratings.shape[1]))
        user_ratings = vstack([user_ratings, empty_row])
        # Set user ratings
        user_row_idx = user_ratings.shape[0] - 1
        for _, row in ur.iterrows():
            m_id = row['movieId']
            if m_id in movie_index:
                col = movie_index.get_loc(m_id)
                user_ratings[user_row_idx, col] = row['rating']
        # Recompute sim_matrix
        sim_matrix = cosine_similarity(user_ratings.T)

# Recommend by genre
def recommend_by_genre(genre: str, n: int = 10):
    genre_movies = movies[movies['genres_str'].str.contains(genre, case=False, na=False)]
    genre_movies = genre_movies.sort_values('weighted_rating', ascending=False)
    return genre_movies['original_title'].head(n).tolist()

# Recommend by content (title)
def recommend_by_title(title: str, release_date: str = None, tmdb_id: str = None, n: int = 10):
    print(f"Checking title: {title}, release_date: {release_date}, tmdb_id: {tmdb_id}")
    if title not in title_to_index:
        print(f"Title {title} not in title_to_index")
        return []
    indices = title_to_index[title]
    if isinstance(indices, pd.Series):
        indices = indices.values
    else:
        indices = [indices]
    # Filter by release_date or tmdb_id if provided
    if release_date or tmdb_id:
        filtered_indices = []
        for idx in indices:
            movie = movies.loc[idx]
            if (release_date and movie['release_date'] == release_date) or (tmdb_id and movie['id'] == tmdb_id):
                filtered_indices.append(idx)
        indices = filtered_indices
        if not indices:
            print(f"No movie found for {title} with release_date {release_date} or tmdb_id {tmdb_id}")
            return []
    recommendations = {}
    for idx in indices:
        movie_data = movies.loc[idx, ['original_title', 'release_date', 'id', 'movieId']].to_dict()
        movie_key = f"{movie_data['original_title']} ({movie_data['release_date']}) [ID: {movie_data['id']}]"
        movie_soup = movies.loc[idx, 'soup']
        movie_vector = tfidf.transform([movie_soup])
        distances, nn_indices = nn_content.kneighbors(movie_vector, n_neighbors=n + 1)
        movie_indices = nn_indices[0][1:]  # Exclude the movie itself
        rec_titles = movies['original_title'].iloc[movie_indices].tolist()
        recommendations[movie_key] = rec_titles
        print(f"Recommendations for {movie_key}: {rec_titles}")
    if not recommendations:
        print(f"No recommendations generated for {title}")
    return recommendations

# Recommend by collaborative filtering
def recommend_collaborative(title: str, release_date: str = None, tmdb_id: str = None, n: int = 10):
    print(f"Checking title: {title}, release_date: {release_date}, tmdb_id: {tmdb_id}")
    if title not in title_to_index:
        print(f"Title {title} not in title_to_index")
        return []
    indices = title_to_index[title]
    if isinstance(indices, pd.Series):
        indices = indices.values
    else:
        indices = [indices]
    # Filter by release_date or tmdb_id if provided
    if release_date or tmdb_id:
        filtered_indices = []
        for idx in indices:
            movie = movies.loc[idx]
            if (release_date and movie['release_date'] == release_date) or (tmdb_id and movie['id'] == tmdb_id):
                filtered_indices.append(idx)
        indices = filtered_indices
        if not indices:
            print(f"No movie found for {title} with release_date {release_date} or tmdb_id {tmdb_id}")
            return []
    recommendations = {}
    for idx in indices:
        movie_data = movies.loc[idx, ['original_title', 'release_date', 'id', 'movieId']].to_dict()
        movie_key = f"{movie_data['original_title']} ({movie_data['release_date']}) [ID: {movie_data['id']}]"
        movie_id = movie_data['movieId']
        if movie_id not in movie_index:
            print(f"No movie_idx found for movieId {movie_id}")
            recommendations[movie_key] = []
            continue
        movie_idx = movie_index.get_loc(movie_id)
        print(f"Movie index: {movie_idx}")
        movie_vector = user_ratings[:, movie_idx]
        similarities = cosine_similarity(movie_vector.T, user_ratings.T)[0]
        similar_indices = similarities.argsort()[-n - 1:-1][::-1]
        similar_movie_ids = movie_index[similar_indices]
        similar_titles = movies[movies['movieId'].isin(similar_movie_ids)]['original_title'].head(n).tolist()
        recommendations[movie_key] = similar_titles
        print(f"Recommendations for {movie_key}: {similar_titles}")
    return recommendations

# Pydantic models
class TitleRequest(BaseModel):
    title: str
    release_date: str | None = None
    tmdb_id: str | None = None

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
async def get_by_title(title: str = "Toy Story", release_date: str | None = None, tmdb_id: str | None = None):
    try:
        recommendations = recommend_by_title(title, release_date, tmdb_id)
        if not recommendations:
            matching_movies = movies[movies['original_title'].str.contains(title, case=False, na=False)][
                ['original_title', 'release_date', 'id']].to_dict('records')
            error_detail = f"No recommendations found for title: {title}"
            if matching_movies:
                error_detail += f". Multiple movies found. Please specify release_date or tmdb_id. Matching movies: {matching_movies[:5]}"
            raise HTTPException(status_code=404, detail=error_detail)
        return recommendations
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")

@app.get("/recommend/collaborative")
async def get_collaborative(title: str = "Toy Story", release_date: str | None = None, tmdb_id: str | None = None):
    try:
        recommendations = recommend_collaborative(title, release_date, tmdb_id)
        if not recommendations:
            matching_movies = movies[movies['original_title'].str.contains(title, case=False, na=False)][
                ['original_title', 'release_date', 'id']].to_dict('records')
            error_detail = f"No collaborative recommendations found for title: {title}"
            if matching_movies:
                error_detail += f". Multiple movies found. Please specify release_date or tmdb_id. Matching movies: {matching_movies[:5]}"
            raise HTTPException(status_code=404, detail=error_detail)
        return recommendations
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")

@app.get("/movies")
async def get_movies():
    try:
        return movies[['original_title', 'release_date', 'id']].sort_values('original_title').to_dict('records')
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/rate")
async def rate_movie(request: TitleRequest, rating: float = Body(...)):
    global user_ratings, user_index, sim_matrix
    title = request.title
    release_date = request.release_date
    tmdb_id = request.tmdb_id
    if title not in title_to_index:
        raise HTTPException(status_code=404, detail=f"Movie not found: {title}")
    indices = title_to_index[title]
    if not isinstance(indices, np.ndarray):
        indices = [indices]
    filtered_indices = []
    for idx in indices:
        movie = movies.loc[idx]
        if (release_date and movie['release_date'] == release_date) or (tmdb_id and str(movie['id']) == str(tmdb_id)):
            filtered_indices.append(idx)
    if not filtered_indices:
        if len(indices) > 1:
            raise HTTPException(status_code=404, detail="Multiple movies found, please specify release_date or tmdb_id")
        filtered_indices = indices
    if len(filtered_indices) > 1:
        raise HTTPException(status_code=404, detail="Multiple matches found, please be more specific")
    idx = filtered_indices[0]
    movie_id = movies.loc[idx, 'movieId']
    timestamp = time.time()
    new_rating = pd.DataFrame({'movieId': [movie_id], 'rating': [rating], 'timestamp': [timestamp]})
    if os.path.exists('user_ratings.csv'):
        ur = pd.read_csv('user_ratings.csv')
        ur['movieId'] = ur['movieId'].astype(str)
        if movie_id in ur['movieId'].values:
            ur.loc[ur['movieId'] == movie_id, 'rating'] = rating
            ur.loc[ur['movieId'] == movie_id, 'timestamp'] = timestamp
        else:
            ur = pd.concat([ur, new_rating], ignore_index=True)
    else:
        ur = new_rating
    ur.to_csv('user_ratings.csv', index=False)
    # Update matrix
    if new_user_id not in user_index:
        user_index = user_index.append(pd.Index([new_user_id]))
        empty_row = csr_matrix((1, user_ratings.shape[1]))
        user_ratings = vstack([user_ratings, empty_row])
    user_row_idx = user_ratings.shape[0] - 1
    user_ratings[user_row_idx, :] = 0  # Reset row
    for _, row in ur.iterrows():
        m_id = str(row['movieId'])
        if m_id in movie_index:
            col = movie_index.get_loc(m_id)
            user_ratings[user_row_idx, col] = row['rating']
    sim_matrix = cosine_similarity(user_ratings.T)
    return {"message": "Rating added/updated"}

@app.get("/recommend/personal")
async def get_personal(n: int = 10):
    global user_ratings, sim_matrix
    if not os.path.exists('user_ratings.csv') or pd.read_csv('user_ratings.csv').empty:
        raise HTTPException(status_code=404, detail="No user ratings yet. Rate some movies first.")
    if new_user_id not in user_index:
        return []
    user_row_idx = user_ratings.shape[0] - 1
    user_row = user_ratings[user_row_idx, :]
    rated_cols = user_row.indices  # nonzero columns (sparse)
    if len(rated_cols) == 0:
        return []
    preds = []
    for j in range(user_ratings.shape[1]):
        if j in rated_cols:
            continue
        sims = sim_matrix[j, rated_cols]
        user_ratings_rated = user_row[0, rated_cols].toarray().flatten()
        sum_abs_sims = np.sum(np.abs(sims))
        if sum_abs_sims == 0:
            continue
        pred = np.dot(sims, user_ratings_rated) / sum_abs_sims
        preds.append((j, pred))
    if not preds:
        return []
    top_preds = sorted(preds, key=lambda x: x[1], reverse=True)[:n]
    top_movie_ids = [movie_index[j] for j, _ in top_preds]
    top_titles = movies[movies['movieId'].isin(top_movie_ids)]['original_title'].tolist()
    return top_titles