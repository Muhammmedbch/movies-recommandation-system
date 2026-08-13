
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

movies = pd.read_csv("dataset/movies.csv")
ratings = pd.read_csv("dataset/ratings.csv")

df = pd.merge(movies, ratings, on='movieId')

stats = df.groupby('title')['rating'].agg(['mean', 'count'])

# Top 10 most highly rated movies (minimum 50 ratings)
popular_movies = stats[stats['count'] > 50].sort_values(by='count', ascending=False)

# clean genre column so can the vectorizer d'ont crash when finding empty cells and NaN values
movies['genres'] = movies['genres'].fillna('')

# create the vectorizer object and we specify the pattern so he cannot split words like Sci-Fic ...
tfid = TfidfVectorizer(token_pattern = r'(?u) [\w-]+ ')

# we tranform the genre text values into numerical values -> each movies now has a vector with values based on the dataset

genre_matrix = tfid.fit_transform(movies['genres'])
# Calculate Cosine Similarity Matrix across all movies
cosine_sim = cosine_similarity(genre_matrix, genre_matrix)

# Recommendation function
def recommend_content(title, top_n=5):
 idx = movies[movies['title'] == title].index[0]
 sim_scores = list(enumerate(cosine_sim[idx]))
 sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)[1:top_n+1]
 movie_indices = [i[0] for i in sim_scores]
 return movies['title'].iloc[movie_indices]

# Example Usage
print(recommend_content('Toy Story (1995)'))

