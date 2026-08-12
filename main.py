# init code

import pandas as pd

movies = pd.read_csv("dataset/movies.csv")
ratings = pd.read_csv("dataset/ratings.csv")

df = pd.merge(movies, ratings, on='movieId')

stats = df.groupby('title')['rating'].agg(['mean', 'count'])

# Top 10 most highly rated movies (minimum 50 ratings)
popular_movies = stats[stats['count'] > 50].sort_values(by='count', ascending=False)
print(popular_movies.head(10))