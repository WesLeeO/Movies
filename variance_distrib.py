import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import pearsonr

# Load high and low budget movie datasets
df_high_budget = pd.read_csv("archive/high_budget_high_multiplier.csv")
df_low_budget = pd.read_csv("archive/low_budget_high_multiplier.csv")

# Load ratings dataset
ratings_file = "archive/ratings.csv"  # Adjust the path
ratings_df = pd.read_csv(ratings_file)

# Filter movies with at least 10 reviews
movie_counts = ratings_df["movieId"].value_counts()
valid_movies = movie_counts[movie_counts >= 10].index
ratings_df = ratings_df[ratings_df["movieId"].isin(valid_movies)]

# Compute mean rating and variance per movie
movie_ratings = ratings_df.groupby("movieId").agg(
    mean_rating=("rating", "mean"),
    variance_rating=("rating", "var")
).reset_index()

# Merge ratings with high and low budget movies
df_high_budget = df_high_budget.merge(movie_ratings, left_on="id", right_on="movieId", how="inner")
df_low_budget = df_low_budget.merge(movie_ratings, left_on="id", right_on="movieId", how="inner")


# Plot histograms of rating variance
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)

sns.histplot(df_low_budget["variance_rating"], bins=30, kde=True, color='blue', alpha=0.7)
plt.xlabel("Rating Variance" , fontsize=14)
plt.ylabel("Frequency"  , fontsize=14)
plt.title("Variance of Ratings - Low Budget Movies" , fontsize=16)
#xticks in 14
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)

plt.subplot(1, 2, 2)


sns.histplot(df_high_budget["variance_rating"], bins=30, kde=True, color='orange', alpha=0.7)
plt.xlabel("Rating Variance", fontsize=14)
plt.ylabel("Frequency", fontsize=14)
plt.title("Variance of Ratings - High Budget Movies" , fontsize=16)
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)

plt.tight_layout()

plt.savefig("part2_plots/variance_distrib.png")
plt.show()

# Compute correlation between variance and multiplier
correlation_high_var, p_high_var = pearsonr(df_high_budget["variance_rating"], df_high_budget["multiplier"])
correlation_low_var, p_low_var = pearsonr(df_low_budget["variance_rating"], df_low_budget["multiplier"])

print("Correlation between Rating Variance and Multiplier:")
print(f"High Budget Movies: {correlation_high_var:.2f} (p-value: {p_high_var:.4f})")
print(f"Low Budget Movies: {correlation_low_var:.2f} (p-value: {p_low_var:.4f})")
