import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# ---------------------------------------------------------
# Page Configuration
# ---------------------------------------------------------
st.set_page_config(
    page_title="Movie Recommender System",
    page_icon="🎬",
    layout="wide"
)

st.title("🎬 Movie Recommendation System")
st.write("A Content-Based Recommender powered by `CountVectorizer` and `Cosine Similarity`.")

# ---------------------------------------------------------
# 1. Load Data & Cache for Performance
# ---------------------------------------------------------
@st.cache_data
def load_data():
    # Replace 'movies.csv' with your dataset path if different
    movies = pd.read_csv('movies.csv')
    movies['genres'] = movies['genres'].fillna('')
    return movies

@st.cache_resource
def compute_similarity(movies):
    # Vectorize genres
    cv = CountVectorizer(token_pattern=r'(?u)\b[\w-]+\b')
    genre_matrix = cv.fit_transform(movies['genres'])
    
    # Compute Cosine Similarity Matrix
    cosine_sim = cosine_similarity(genre_matrix, genre_matrix)
    return cosine_sim

# Load dataset and calculate matrix
movies = load_data()
cosine_sim = compute_similarity(movies)

# ---------------------------------------------------------
# 2. Recommendation Logic Function
# ---------------------------------------------------------
def recommend_movies(title, top_n=5):
    # Find row index of chosen title
    idx = movies[movies['title'] == title].index[0]
    
    # Pair scores with movie indices
    sim_scores = list(enumerate(cosine_sim[idx]))
    
    # Sort by similarity score descending, skip the movie itself
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)[1:top_n+1]
    
    # Extract indices and similarity scores for display
    movie_indices = [i[0] for i in sim_scores]
    scores = [round(i[1], 2) for i in sim_scores]
    
    # Create a result DataFrame
    results = movies[['title', 'genres']].iloc[movie_indices].copy()
    results['Similarity Score'] = scores
    return results

# ---------------------------------------------------------
# 3. Streamlit UI Controls
# ---------------------------------------------------------
st.sidebar.header("⚙️ User Controls")

# Dropdown for selecting a movie
selected_movie = st.sidebar.selectbox(
    "Select or type a movie title:",
    options=movies['title'].values
)

# Slider for choosing number of recommendations
top_n = st.sidebar.slider(
    "Number of recommendations:",
    min_value=1,
    max_value=10,
    value=5
)

# Recommendation Trigger Button
if st.sidebar.button("Get Recommendations 🚀"):
    st.subheader(f"Recommendations for: **{selected_movie}**")
    
    # Fetch recommendations
    recommendations = recommend_movies(selected_movie, top_n=top_n)
    
    # Display selected movie details
    selected_genres = movies[movies['title'] == selected_movie]['genres'].values[0]
    st.info(f"**Genres for {selected_movie}:** {selected_genres}")
    
    st.write("---")
    
    # Display recommendations table nicely
    st.dataframe(
        recommendations,
        column_config={
            "title": "Movie Title",
            "genres": "Genres",
            "Similarity Score": st.column_config.ProgressColumn(
                "Match Score",
                format="%.2f",
                min_value=0.0,
                max_value=1.0,
            )
        },
        use_container_width=True,
        hide_index=True
    )