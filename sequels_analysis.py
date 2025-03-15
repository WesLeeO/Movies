import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import ttest_ind, pearsonr
from ast import literal_eval

# Load filtered movies dataset
movies_df = pd.read_csv("archive/low_budget_movies.csv", low_memory=False)
movies_df = movies_df[['id', 'title', 'budget', 'revenue', 'belongs_to_collection', 'release_date']]

# Convert release_date to datetime
movies_df['release_date'] = pd.to_datetime(movies_df['release_date'], errors='coerce')

# Extract collection name safely
movies_df['collection'] = movies_df['belongs_to_collection'].apply(lambda x: literal_eval(x)['name'] if pd.notna(x) and x != 'None' else None)

# Compute multiplier and log multiplier
movies_df['multiplier'] = movies_df['revenue'] / movies_df['budget']
movies_df['log_multiplier'] = np.log1p(movies_df['multiplier'])  # Log transformation
movies_df = movies_df.dropna(subset=['multiplier'])  # Remove movies with missing multipliers
#remove the movies that have a multiplier greater than 10
movies_df = movies_df[np.log1p(movies_df["multiplier"]) < 7]

# Separate standalone and franchise movies
standalone_movies = movies_df[movies_df['collection'].isna()]
franchise_movies = movies_df[movies_df['collection'].notna()]

# Compare standalone vs franchise multipliers
t_stat, p_value = ttest_ind(franchise_movies['multiplier'], standalone_movies['multiplier'], nan_policy='omit')
print("Standalone vs Franchise Multiplier Comparison:")
print(f"Standalone Mean Multiplier: {standalone_movies['multiplier'].mean()}")
print(f"Franchise Mean Multiplier: {franchise_movies['multiplier'].mean()}")
print(f"p-value: {p_value}\n")

# Analyze financial trends within franchises
franchise_trends = []
for collection, group in franchise_movies.groupby('collection'):
    group = group.sort_values('release_date')
    multipliers = group['multiplier'].values
    if len(multipliers) > 1:
        correlation, _ = pearsonr(multipliers[:-1], multipliers[1:]) if len(multipliers) > 2 else (np.nan, np.nan)
        franchise_trends.append([collection, correlation, len(multipliers)])

franchise_trends_df = pd.DataFrame(franchise_trends, columns=['collection', 'correlation_prev_next', 'num_movies'])

# Filter out small franchises (at least 3 movies)
franchise_trends_df = franchise_trends_df[franchise_trends_df['num_movies'] >= 3]
print("Correlation between Previous Movie Multiplier and Next Movie Multiplier:")
print(franchise_trends_df[['collection', 'correlation_prev_next']].sort_values(by='correlation_prev_next'))

# Analyze whether the last movie in a franchise is a financial failure
last_movie_analysis = []
first_movie_analysis = []
next_movie_correlation_analysis = []
for collection, group in franchise_movies.groupby('collection'):
    group = group.sort_values('release_date')
    if len(group) > 1:
        last_movie = group.iloc[-1]
        first_movie = group.iloc[0]
        avg_earlier_multiplier = group.iloc[:-1]['multiplier'].mean()
        last_movie_analysis.append([collection, last_movie['multiplier'], avg_earlier_multiplier])
        first_movie_analysis.append([collection, first_movie['multiplier']])
        
        for i in range(1, len(group)):
            prev_avg_multiplier = group.iloc[:i]['multiplier'].mean()
            next_multiplier = group.iloc[i]['multiplier']
            next_movie_correlation_analysis.append([collection, prev_avg_multiplier, next_multiplier])

last_movie_df = pd.DataFrame(last_movie_analysis, columns=['collection', 'last_movie_multiplier', 'avg_earlier_multiplier'])
first_movie_df = pd.DataFrame(first_movie_analysis, columns=['collection', 'first_movie_multiplier'])
next_movie_df = pd.DataFrame(next_movie_correlation_analysis, columns=['collection', 'avg_prev_multiplier', 'next_multiplier'])

# Compare last movie multiplier vs previous movies
last_movie_df = last_movie_df.dropna()
t_stat_last, p_value_last = ttest_ind(last_movie_df['last_movie_multiplier'], last_movie_df['avg_earlier_multiplier'], nan_policy='omit')
print("\nLast Movie Financial Analysis:")
print(f"Average Last Movie Multiplier: {last_movie_df['last_movie_multiplier'].mean()}")
print(f"Average Earlier Movies Multiplier: {last_movie_df['avg_earlier_multiplier'].mean()}")
print(f"p-value: {p_value_last}")

# Compare first movie in franchise vs standalone movies
t_stat_first, p_value_first = ttest_ind(first_movie_df['first_movie_multiplier'], standalone_movies['multiplier'], nan_policy='omit')
print("\nFirst Movie in Franchise vs Standalone Movie Multiplier Comparison:")
print(f"Average First Movie Multiplier: {first_movie_df['first_movie_multiplier'].mean()}")
print(f"Average Standalone Movie Multiplier: {standalone_movies['multiplier'].mean()}")
print(f"p-value: {p_value_first}")

# Compute correlation between previous movie average multiplier and next movie multiplier
correlation_next_movie, p_value_next = pearsonr(next_movie_df['avg_prev_multiplier'], next_movie_df['next_multiplier'])
print("\nCorrelation Between Previous Movies' Average Multiplier and Next Movie Multiplier:")
print(f"Correlation Coefficient: {correlation_next_movie}")
print(f"p-value: {p_value_next}")

# Plot franchise vs standalone log multiplier distribution
plt.figure(figsize=(10, 6))
sns.boxplot(x=movies_df['collection'].notna(), y=movies_df['log_multiplier'])
plt.xticks([0, 1], ['Standalone', 'Franchise'])
plt.ylabel("Log Multiplier")
plt.title("Standalone vs Franchise Movie Log Multipliers")
plt.show()

# Plot financial decline in franchises using log multipliers
plt.figure(figsize=(12, 6))
sns.histplot(np.log1p(last_movie_df['last_movie_multiplier']) - np.log1p(last_movie_df['avg_earlier_multiplier']), bins=30, kde=True)
plt.xlabel("Log Last Movie Multiplier - Log Previous Average")
plt.ylabel("Count")
plt.title("Financial Performance of Last Movie in Franchise (Log Scale)")
plt.show()