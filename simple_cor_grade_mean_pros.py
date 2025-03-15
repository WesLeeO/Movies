import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import pearsonr

# Load high and low budget movie datasets
df_high_budget = pd.read_csv("archive/high_budget_movies.csv")
df_low_budget = pd.read_csv("archive/low_budget_movies.csv")

# Load ratings dataset
ratings_file = "archive/ratings.csv"  # Adjust the path
ratings_df = pd.read_csv(ratings_file)

# Filter movies with at least 10 reviews
movie_counts = ratings_df["movieId"].value_counts()
valid_movies = movie_counts[movie_counts >= 5].index
ratings_df = ratings_df[ratings_df["movieId"].isin(valid_movies)]

#filter when the log of multiplier is greater tahn 10
df_high_budget = df_high_budget[np.log1p(df_high_budget["multiplier"]) < 7]
df_low_budget = df_low_budget[np.log1p(df_low_budget["multiplier"]) < 7]

# Compute mean rating per movie
movie_ratings = ratings_df.groupby("movieId").agg(mean_rating=("rating", "mean")).reset_index()

# Merge mean ratings with high and low budget movies
df_high_budget = df_high_budget.merge(movie_ratings, left_on="id", right_on="movieId", how="inner")
df_low_budget = df_low_budget.merge(movie_ratings, left_on="id", right_on="movieId", how="inner")

# Compute correlation between mean rating and multiplier
correlation_high, p_high = pearsonr(df_high_budget["mean_rating"], df_high_budget["multiplier"])
correlation_low, p_low = pearsonr(df_low_budget["mean_rating"], df_low_budget["multiplier"])

print("Correlation between Mean Rating and Multiplier:")
print(f"High Budget Movies: {correlation_high:.2f} (p-value: {p_high:.4f})")
print(f"Low Budget Movies: {correlation_low:.2f} (p-value: {p_low:.4f})")

# Assign categories for visualization
df_high_budget["category"] = "high"
df_low_budget["category"] = "low"

# Combine both datasets
df_combined = pd.concat([df_high_budget, df_low_budget], ignore_index=True)

# Apply log transformation to multiplier to improve visualization
df_combined["log_multiplier"] = np.log1p(df_combined["multiplier"])

# Plot scatter plot using Matplotlib
plt.figure(figsize=(10, 6))
sns.scatterplot(
    data=df_combined, x="mean_rating", y="log_multiplier", hue="category", alpha=0.7
)
plt.xlabel("Mean Rating")
plt.ylabel("Log Multiplier")
plt.title("Log Multiplier by Mean Rating")
plt.legend(title="Category")
plt.show()

# Compute and print average multiplier for high- and low-budget movies
high_budget_avg = df_high_budget["multiplier"].mean()
low_budget_avg = df_low_budget["multiplier"].mean()

print("\nAverage Multiplier:")
print(f"High Budget Movies: {high_budget_avg:.2f}")
print(f"Low Budget Movies: {low_budget_avg:.2f}")