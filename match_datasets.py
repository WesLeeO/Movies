import pandas as pd
import numpy as np
from fuzzywuzzy import process

# Load high budget movies with high multiplier
filtered_movies = pd.read_csv("archive/movies_metadata_filtered.csv")
filtered_movies = filtered_movies[['id', 'title']]

# Load Rotten Tomatoes dataset
movies_metadata = pd.read_csv("perso/rotten_tomatoes_movies.csv", low_memory=False)
movies_metadata = movies_metadata[['movie_title', 'rotten_tomatoes_link']]
movies_metadata.dropna(subset=['movie_title'], inplace=True)

# Normalize titles for case-insensitive matching
def normalize_title(title):
    return str(title).strip().lower()

filtered_movies['normalized_title'] = filtered_movies['title'].apply(normalize_title)
movies_metadata['normalized_title'] = movies_metadata['movie_title'].apply(normalize_title)

# Perform exact matching based on normalized titles
matched_movies = filtered_movies.merge(movies_metadata, on='normalized_title', how='inner', suffixes=('_filtered', '_metadata'))

# Optional: Fuzzy Matching for better results (only on unmatched titles)
unmatched_titles = set(filtered_movies['normalized_title']) - set(matched_movies['normalized_title'])
metadata_titles = movies_metadata[['normalized_title', 'movie_title']].drop_duplicates()

print("mdr")
print(len(unmatched_titles))
i = 0
fuzzy_matches = []
for title in unmatched_titles:
    print(i)
    i += 1
    best_match, score = process.extractOne(title, metadata_titles['normalized_title'].values)
    if score > 90:  # Adjust threshold as needed
        fuzzy_matches.append((title, best_match))

# Convert fuzzy matches into a DataFrame and merge again
fuzzy_df = pd.DataFrame(fuzzy_matches, columns=['normalized_title_filtered', 'normalized_title_metadata'])

# Merge fuzzy matches with filtered_movies and movies_metadata
matched_fuzzy_movies = fuzzy_df.merge(filtered_movies, left_on='normalized_title_filtered', right_on='normalized_title', how='inner')

# Change suffixes to avoid duplicates
matched_fuzzy_movies = matched_fuzzy_movies.merge(movies_metadata, left_on='normalized_title_metadata', right_on='normalized_title', how='inner', suffixes=('_fuzzy_filtered', '_fuzzy_metadata'))

# Combine exact and fuzzy matches
final_matched_movies = pd.concat([matched_movies, matched_fuzzy_movies], ignore_index=True)

# Create final dataset with only movie ID and Rotten Tomatoes link
final_matched_movies = final_matched_movies[['id', 'rotten_tomatoes_link']]
final_matched_movies.columns = ['movie_id', 'rotten_tomatoes_link']

# Save matched movies to a new CSV
final_matched_movies.to_csv("archive/matched_movies.csv", index=False)

print(f"Matched {len(final_matched_movies)} movies between datasets. Saved to matched_movies.csv")
