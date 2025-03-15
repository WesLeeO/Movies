import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from scipy.stats import kurtosis, skew

#TODO scraping pour comparer professional reviews et random reviews
# Load high budget movies with high multiplier
df_high_budget = pd.read_csv("archive/high_budget_high_multiplier.csv")

# Load ratings dataset
ratings_file = "archive/ratings.csv"  # Adjust the path
ratings_df = pd.read_csv(ratings_file)

# Ensure correct column names
print(ratings_df.head())

# Filter the movies that have at least 10 reviews
movie_counts = ratings_df["movieId"].value_counts()
valid_movies = movie_counts[movie_counts >= 10].index
ratings_df = ratings_df[ratings_df["movieId"].isin(valid_movies)]

# Aggregate rating statistics per movie
movie_ratings = ratings_df.groupby("movieId").agg(
    mean_rating=("rating", "mean"),
    std_rating=("rating", "std"),
    harsh_reviews=("rating", lambda x: (x <= 2).sum() / len(x)),
    very_good_reviews=("rating", lambda x: (x >= 4).sum() / len(x)),
    skewness=("rating", lambda x: skew(x, bias=False)),
    kurtosis=("rating", lambda x: kurtosis(x, bias=False))
).reset_index()

# Fill NaN values in std_rating (happens when only one review exists)
movie_ratings["std_rating"].fillna(0, inplace=True)

# Merge ratings with high-budget financially successful movies
df_high_budget = df_high_budget.merge(movie_ratings, left_on="id", right_on="movieId", how="inner")

# Select features for clustering
features = ["mean_rating", "std_rating", "harsh_reviews", "very_good_reviews", "skewness", "kurtosis"]
X = df_high_budget[features]

# Standardize the features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Apply K-Means Clustering
kmeans = KMeans(n_clusters=2, random_state=40, n_init=10)
df_high_budget["cluster"] = kmeans.fit_predict(X_scaled)

# Print the average multiplier for each cluster
cluster_multipliers = df_high_budget.groupby("cluster")["multiplier"].mean()
print("Average Multiplier per Cluster:")
print(cluster_multipliers)

#print the std of multiplier for each cluster
cluster_multipliers_std = df_high_budget.groupby("cluster")["multiplier"].std()
print("Std Multiplier per Cluster:")
print(cluster_multipliers_std)


# Print the cluster centers
cluster_centers = pd.DataFrame(scaler.inverse_transform(kmeans.cluster_centers_), columns=features)
print("Cluster Centers:")
print(cluster_centers)


