import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import pearsonr
import re

# Load high and low budget movie datasets
df_high_budget = pd.read_csv("archive/high_budget_movies.csv")
df_low_budget = pd.read_csv("archive/low_budget_movies.csv")

# Load matched movies dataset
matched_movies = pd.read_csv("archive/matched_movies.csv")

# Load Rotten Tomatoes ratings dataset
rt_ratings = pd.read_csv("perso/rotten_tomatoes_critic_reviews.csv")  # Adjust with actual file path

# Merge movies with Rotten Tomatoes links
df_high_budget = df_high_budget.merge(matched_movies, left_on="id", right_on="movie_id", how="inner")
df_low_budget = df_low_budget.merge(matched_movies, left_on="id", right_on="movie_id", how="inner")


# Function to normalize review scores
def normalize_review_score(score):
    match = re.match(r"(\d+)/(\d+)", str(score))
    if match:
        num, denom = map(int, match.groups())
        return (num / denom) * 5 if denom > 0 else np.nan
    return np.nan

# Apply normalization to the review scores
rt_ratings["normalized_review_score"] = rt_ratings["review_score"].apply(normalize_review_score)

# Remove rows with NaN scores (invalid or missing review scores)
rt_ratings.dropna(subset=["normalized_review_score"], inplace=True)

# Merge with Rotten Tomatoes reviews for both high and low budget movies
rt_ratings_high = rt_ratings.merge(df_high_budget, left_on="rotten_tomatoes_link", right_on="rotten_tomatoes_link", how="inner")
rt_ratings_low = rt_ratings.merge(df_low_budget, left_on="rotten_tomatoes_link", right_on="rotten_tomatoes_link", how="inner")

# Compute mean normalized rating per movie
movie_ratings_high = rt_ratings_high.groupby("movie_id").agg(mean_rating=("normalized_review_score", "mean")).reset_index()
movie_ratings_low = rt_ratings_low.groupby("movie_id").agg(mean_rating=("normalized_review_score", "mean")).reset_index()

# Merge mean ratings with high and low budget movies
df_high_budget = df_high_budget.merge(movie_ratings_high, left_on="id", right_on="movie_id", how="inner")
df_low_budget = df_low_budget.merge(movie_ratings_low, left_on="id", right_on="movie_id", how="inner")

# Filter out extreme multiplier values to prevent outliers affecting analysis
df_high_budget = df_high_budget[np.log1p(df_high_budget["multiplier"]) < 7]
df_low_budget = df_low_budget[np.log1p(df_low_budget["multiplier"]) < 7]

# Compute correlation between normalized mean rating and multiplier
correlation_high, p_high = pearsonr(df_high_budget["mean_rating"], df_high_budget["multiplier"])
correlation_low, p_low = pearsonr(df_low_budget["mean_rating"], df_low_budget["multiplier"])

print("Correlation between Normalized Mean Rating and Multiplier:")
print(f"High Budget Movies: {correlation_high:.2f} (p-value: {p_high:.4f})")
print(f"Low Budget Movies: {correlation_low:.2f} (p-value: {p_low:.4f})")

# Assign categories for visualization
df_high_budget["category"] = "high"
df_low_budget["category"] = "low"

# Combine both datasets
df_combined = pd.concat([df_high_budget, df_low_budget], ignore_index=True)

# Apply log transformation to multiplier for better visualization
df_combined["log_multiplier"] = np.log1p(df_combined["multiplier"])

# Plot scatter plot using Matplotlib
plt.figure(figsize=(10, 6))
sns.scatterplot(
    data=df_combined, x="mean_rating", y="log_multiplier", hue="category", alpha=0.7
)
plt.xlabel("Normalized Mean Rating ")
plt.ylabel("Log Multiplier")
plt.title("Log Multiplier by Normalized Mean Rating")
plt.legend(title="Category")
plt.show()

# Compute and print average multiplier for high- and low-budget movies
high_budget_avg = df_high_budget["multiplier"].mean()
low_budget_avg = df_low_budget["multiplier"].mean()

print("\nAverage Multiplier:")
print(f"High Budget Movies: {high_budget_avg:.2f}")
print(f"Low Budget Movies: {low_budget_avg:.2f}")