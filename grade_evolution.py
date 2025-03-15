#TODO checker si pour avoir meilleur multiplicateur vaut mieux buzzer des la sortie
#aussi peut check evolution de la grade des films en fonction du temps
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from scipy.stats import kurtosis, skew, ttest_ind

# Load high budget movies with high multiplier
df_high_budget = pd.read_csv("archive/low_budget_movies.csv")

# Load ratings dataset
ratings_file = "archive/ratings.csv"  # Adjust the path
ratings_df = pd.read_csv(ratings_file)

# Load movies metadata to get release dates
movies_metadata = pd.read_csv("archive/movies_metadata.csv", low_memory=False)
movies_metadata = movies_metadata[['id', 'release_date']]
movies_metadata.dropna(subset=['release_date'], inplace=True)
movies_metadata['release_date'] = pd.to_datetime(movies_metadata['release_date'], errors='coerce')

# Ensure id and movieId are the same type
movies_metadata['id'] = movies_metadata['id'].astype(str)
ratings_df['movieId'] = ratings_df['movieId'].astype(str)

# Convert timestamps to datetime
ratings_df['timestamp'] = pd.to_datetime(ratings_df['timestamp'], unit='s')

# Merge ratings with movies metadata
ratings_df = ratings_df.merge(movies_metadata, left_on='movieId', right_on='id', how='inner')

# Define the initial rating period (e.g., first 3 months after release)
time_window = pd.Timedelta(days=90)
ratings_df['within_initial_period'] = ratings_df['timestamp'] <= (ratings_df['release_date'] + time_window)

# Filter movies that have ratings within the initial release period
initial_ratings = ratings_df[ratings_df['within_initial_period']]

# Compute initial and final ratings per movie
movie_ratings = ratings_df.groupby("movieId").agg(
    final_mean_rating=("rating", "mean")
).reset_index()

initial_movie_ratings = initial_ratings.groupby("movieId").agg(
    initial_mean_rating=("rating", "mean"),
    num_initial_reviews=("rating", "count")
).reset_index()

# Merge with high-budget movies
df_high_budget['id'] = df_high_budget['id'].astype(str)
df_high_budget = df_high_budget.merge(movie_ratings, left_on="id", right_on="movieId", how="inner")
df_high_budget = df_high_budget.merge(initial_movie_ratings, left_on="id", right_on="movieId", how="inner")

# Define rating bins
bins = [2,2.5, 3, 3.5, 4, 4.5, 5]
bin_labels = ["2-2.5", "2.5-3", "3-3.5", "3.5-4", "4-4.5", "4.5-5"]
df_high_budget["rating_bin"] = pd.cut(df_high_budget["final_mean_rating"], bins=bins, labels=bin_labels, include_lowest=True)

# Initialize results list
results = []

# Process each rating bin separately
for label in bin_labels:
    subset = df_high_budget[df_high_budget["rating_bin"] == label].copy()  # Ensure a copy to avoid modifying original DataFrame
    if len(subset) > 1:
        median_initial = subset["initial_mean_rating"].median()
        subset["high_initial_rating"] = subset["initial_mean_rating"] > median_initial  # Assign within subset
        
        high_initial = subset[subset["high_initial_rating"]]["multiplier"]
        low_initial = subset[~subset["high_initial_rating"]]["multiplier"]
        t_stat, p_value = ttest_ind(high_initial, low_initial, nan_policy='omit')
        results.append([label, high_initial.mean(), low_initial.mean(), p_value])

# Convert to DataFrame
results_df = pd.DataFrame(results, columns=["Rating Bin", "High Initial Mean Multiplier", "Low Initial Mean Multiplier", "p-value"])
print("Comparison of Multipliers Based on Initial Ratings:")
print(results_df)