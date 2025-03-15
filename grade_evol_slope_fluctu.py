import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from scipy.stats import ttest_ind

# Load high budget movies with high multiplier
df_high_budget = pd.read_csv("archive/low_budget_movies.csv")

# Load ratings dataset
ratings_file = "archive/ratings.csv"  # Adjust the path
ratings_df = pd.read_csv(ratings_file)

# Ensure id and movieId are the same type
df_high_budget['id'] = df_high_budget['id'].astype(str)
ratings_df['movieId'] = ratings_df['movieId'].astype(str)

# Convert timestamps to datetime
ratings_df['timestamp'] = pd.to_datetime(ratings_df['timestamp'], unit='s')

# Filter ratings to only include movies in the high-budget dataset
ratings_df = ratings_df[ratings_df["movieId"].isin(df_high_budget["id"])]

# Filter movies with at least 20 reviews
movie_review_counts = ratings_df.groupby("movieId")["rating"].count()
valid_movies = movie_review_counts[movie_review_counts >= 20].index
ratings_df = ratings_df[ratings_df["movieId"].isin(valid_movies)]

# Compute overall rating trend (slope) for each movie
def compute_slope(group):
    if len(group) < 2:
        return np.nan  # Not enough data for a trend
    group = group.sort_values("timestamp")
    time_numeric = (group["timestamp"] - group["timestamp"].min()).dt.total_seconds()
    model = LinearRegression()
    model.fit(time_numeric.values.reshape(-1, 1), group["rating"].values)
    return model.coef_[0]  # Extract slope

rating_trends = ratings_df.groupby("movieId").apply(compute_slope).reset_index()
rating_trends.columns = ["movieId", "rating_slope"]

# Compute segmented rating trend fluctuation
segment_length = pd.Timedelta(days=30)

def compute_slope_fluctuation(group):
    if len(group) < 5:
        return np.nan  # Not enough data in a segment
    group = group.sort_values("timestamp")
    min_time = group["timestamp"].min()
    slopes = []
    for start_time in pd.date_range(min_time, group["timestamp"].max(), freq=segment_length):
        segment = group[(group["timestamp"] >= start_time) & (group["timestamp"] < start_time + segment_length)]
        if len(segment) >= 5:  # Only consider segments with at least 5 reviews
            time_numeric = (segment["timestamp"] - segment["timestamp"].min()).dt.total_seconds()
            model = LinearRegression()
            model.fit(time_numeric.values.reshape(-1, 1), segment["rating"].values)
            slopes.append(model.coef_[0])
    return np.std(slopes) if len(slopes) > 1 else np.nan

rating_fluctuations = ratings_df.groupby("movieId").apply(compute_slope_fluctuation).reset_index()
rating_fluctuations.columns = ["movieId", "rating_slope_fluctuation"]

# Merge with high-budget movies
df_high_budget = df_high_budget.merge(rating_trends, left_on="id", right_on="movieId", how="inner")
df_high_budget = df_high_budget.merge(rating_fluctuations, left_on="id", right_on="movieId", how="inner")

# Define extreme slope and fluctuation groups
low_slope_threshold = df_high_budget["rating_slope"].quantile(0.25)
high_slope_threshold = df_high_budget["rating_slope"].quantile(0.75)
low_fluctuation_threshold = df_high_budget["rating_slope_fluctuation"].quantile(0.25)
high_fluctuation_threshold = df_high_budget["rating_slope_fluctuation"].quantile(0.75)

df_high_budget["extreme_high_slope"] = df_high_budget["rating_slope"] >= high_slope_threshold
df_high_budget["extreme_low_slope"] = df_high_budget["rating_slope"] <= low_slope_threshold
df_high_budget["extreme_high_fluctuation"] = df_high_budget["rating_slope_fluctuation"] >= high_fluctuation_threshold
df_high_budget["extreme_low_fluctuation"] = df_high_budget["rating_slope_fluctuation"] <= low_fluctuation_threshold

# Compare multipliers based on rating trend and fluctuation
trend_results = ttest_ind(
    df_high_budget[df_high_budget["extreme_high_slope"]]["multiplier"],
    df_high_budget[df_high_budget["extreme_low_slope"]]["multiplier"],
    nan_policy='omit'
)

fluctuation_results = ttest_ind(
    df_high_budget[df_high_budget["extreme_high_fluctuation"]]["multiplier"],
    df_high_budget[df_high_budget["extreme_low_fluctuation"]]["multiplier"],
    nan_policy='omit'
)

# Print results
print("Comparison of Multipliers Based on Overall Rating Trends (Slope):")
print(f"High Slope Mean Multiplier: {df_high_budget[df_high_budget['extreme_high_slope']]['multiplier'].mean()}")
print(f"Low Slope Mean Multiplier: {df_high_budget[df_high_budget['extreme_low_slope']]['multiplier'].mean()}")
print(f"p-value: {trend_results.pvalue}")

print("\nComparison of Multipliers Based on Rating Trend Fluctuation:")
print(f"High Fluctuation Mean Multiplier: {df_high_budget[df_high_budget['extreme_high_fluctuation']]['multiplier'].mean()}")
print(f"Low Fluctuation Mean Multiplier: {df_high_budget[df_high_budget['extreme_low_fluctuation']]['multiplier'].mean()}")
print(f"p-value: {fluctuation_results.pvalue}")
