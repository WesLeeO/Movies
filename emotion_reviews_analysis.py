import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import pearsonr
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from tqdm import tqdm  # For progress tracking

# Load filtered movies dataset (movies_metadata_filtered.csv)
filtered_movies = pd.read_csv("movies_metadata_filtered.csv")
filtered_movies = filtered_movies[['id', 'budget', 'revenue']]  # Ensure relevant columns
filtered_movies = filtered_movies.dropna(subset=['budget', 'revenue'])  # Remove invalid data
filtered_movies = filtered_movies[filtered_movies['budget'] > 0]  # Avoid division by zero
filtered_movies['multiplier'] = filtered_movies['revenue'] / filtered_movies['budget']
filtered_movies = filtered_movies.dropna(subset=['multiplier'])  # Remove bad multipliers

# Load matched movies dataset
matched_movies = pd.read_csv("matched_movies.csv")

# Load Rotten Tomatoes ratings dataset
rt_ratings = pd.read_csv("rotten_tomatoes_critic_reviews.csv")  # Adjust with actual file path
rt_ratings = rt_ratings.head(2000)

# Merge movies with Rotten Tomatoes links
filtered_movies = filtered_movies.merge(matched_movies, left_on="id", right_on="movie_id", how="inner")
rt_ratings = rt_ratings.merge(matched_movies, left_on="rotten_tomatoes_link", right_on="rotten_tomatoes_link", how="inner")

# Filter out empty reviews
rt_ratings = rt_ratings[rt_ratings["review_content"].notna() & (rt_ratings["review_content"].str.strip() != "")]

# Sentiment Analysis using paraphrase-multilingual-MiniLM-L12-v2
model = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")

# Define relevant film critique emotions
emotions = ["joy", "sadness", "fear", "disgust", "surprise", "excitement", "confusion"]

# Compute emotion embeddings
emotion_embeddings = model.encode(emotions)

# Compute embeddings for reviews with progress tracking
review_embeddings = []
for review in tqdm(rt_ratings["review_content"], desc="Encoding reviews"):
    review_embeddings.append(model.encode(review))
rt_ratings["review_embedding"] = review_embeddings

# Compute similarity scores between reviews and each emotion
emotion_scores = []
for embedding in tqdm(rt_ratings["review_embedding"], desc="Computing similarity scores"):
    scores = cosine_similarity([embedding], emotion_embeddings)[0]
    emotion_scores.append(scores)

# Convert to DataFrame
emotion_scores_df = pd.DataFrame(emotion_scores, columns=emotions)
rt_ratings = pd.concat([rt_ratings.reset_index(drop=True), emotion_scores_df], axis=1)

# Aggregate emotion scores per movie
movie_emotions = rt_ratings.groupby("movie_id")[emotions].mean().reset_index()

# Merge emotion scores with filtered movies
filtered_movies = filtered_movies.merge(movie_emotions, left_on="id", right_on="movie_id", how="inner")

# Ensure there are at least two values for correlation computation
valid_movies = filtered_movies.dropna(subset=emotions + ["multiplier"])
if len(valid_movies) > 1:
    emotion_correlations = {emotion: pearsonr(valid_movies[emotion], valid_movies["multiplier"])[0] for emotion in emotions}
else:
    emotion_correlations = {emotion: np.nan for emotion in emotions}

# Convert to DataFrame for visualization
emotion_correlation_df = pd.DataFrame(list(emotion_correlations.items()), columns=["Emotion", "Correlation"])

# Plot correlation between emotions and financial success
plt.figure(figsize=(10, 6))
sns.barplot(x="Emotion", y="Correlation", data=emotion_correlation_df, palette="coolwarm")
plt.xlabel("Emotion")
plt.ylabel("Correlation Coefficient")
plt.title("Correlation Between Emotion in Reviews and Financial Success")
plt.show()

# Print correlation values
print("Emotion Correlations with Financial Success:")
print(emotion_correlation_df)