import pandas as pd
import nltk
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from gensim import corpora
from gensim.models import LdaModel
from tqdm import tqdm
from gensim.models import Phrases
from gensim.models.phrases import Phraser
from nltk.corpus import names  
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from matplotlib.lines import Line2D
import os

nltk.download('stopwords')
nltk.download('punkt_tab')
nltk.download('wordnet')
nltk.download('names')


custom_stopwords = set([
    "film", "movie", "story", "documentary", "director", "character", "scene", 
    "cinema", "screen", "series", "episode", "cast", "actor", "actress", 
    "...", "life", "two", "one", "get", "find", "new"
]).union(set([name.lower() for name in names.words()]))

#remove film, director, punctuation
stop_words = set(stopwords.words('english')).union(custom_stopwords)

def process(summary):
    summary = summary.lower()
    tokens = word_tokenize(summary)
    tokens = [token for token in tokens if token not in stop_words and len(token) >= 3]
    lemmatizer = WordNetLemmatizer()
    tokens = [lemmatizer.lemmatize(word) for word in tokens]  # Lemmatization
    return tokens

df = pd.read_csv("./dataset/7/movies_metadata.csv")
df = df.dropna(subset=['overview'])

processed_summaries = []
movie_bow = {}

if os.path.exists("processed_summaries.pkl") and os.path.exists("movie_bow.pkl"):
    print("Loading preprocessed summaries and movie BoW from disk...")
    processed_summaries = pd.read_pickle("processed_summaries.pkl")
    movie_bow = pd.read_pickle("movie_bow.pkl")
else:
    print("Processing summaries...")

    for index, row in tqdm(df.iterrows(), total=len(df), desc="Processing Summaries"):
        processed_summary = process(row["overview"])
        processed_summaries.append(processed_summary)
        movie_bow[(row['id'], row['title'])] = processed_summary

    pd.to_pickle(processed_summaries, "processed_summaries.pkl")
    pd.to_pickle(movie_bow, "movie_bow.pkl")


#Add bigrams
bigram_model = Phrases(processed_summaries, min_count=5, threshold=10)
bigram_phraser = Phraser(bigram_model)

processed_summaries = [bigram_phraser[summary] for summary in processed_summaries]

dictionary = corpora.Dictionary(processed_summaries)
dictionary.filter_extremes(no_below=5, no_above=0.5)
corpus = [dictionary.doc2bow(text) for text in processed_summaries]

lda_model_path = "lda_model.model"
lda_model = None
num_topics = 8 

if os.path.exists(lda_model_path):
    print("Loading LDA model from disk...")
    lda_model = LdaModel.load(lda_model_path)
else:
    print("Training new LDA model...")
    lda_model = LdaModel(corpus, num_topics=num_topics, id2word=dictionary, passes=10, iterations=100, random_state=42)
    lda_model.save(lda_model_path)

# Show topics
for topic in lda_model.print_topics():
    print(topic)

movie_topics = []  # This will store the topic assignments for each movie
topic_to_movies = {i: [] for i in range(num_topics)}
movies_to_topics = {}
for i, doc_bow in enumerate(corpus):
    topic_probs = lda_model.get_document_topics(doc_bow)
    # Get the topic with the highest probability (or keep multiple if needed)
    dominant_topic = sorted(topic_probs, key=lambda x: x[1], reverse=True)[0][0]
    topic_to_movies[dominant_topic].append(df.iloc[i]['title'])
    movies_to_topics[df.iloc[i]['title']] = dominant_topic
    movie_topics.append(dominant_topic)

for i in range(num_topics):
    print(topic_to_movies[i][:10])

transformed_df = df.copy(deep=True)
transformed_df['dominant_topic'] = movie_topics
transformed_df['budget'] = pd.to_numeric(transformed_df['budget'], errors='coerce')
transformed_df['revenue'] = pd.to_numeric(transformed_df['revenue'], errors='coerce')
transformed_df['ROI'] = transformed_df['revenue'] - transformed_df['budget']
transformed_df = transformed_df[transformed_df['budget'] > 0]
transformed_df = transformed_df[transformed_df['revenue'] > 0]

# Group by dominant_topic and calculate the average ROI
average_roi_by_topic = transformed_df.groupby('dominant_topic')['ROI'].mean().reset_index()

# Plot the average ROI by dominant topic
plt.figure(figsize=(10, 6))
sns.barplot(x='dominant_topic', y='ROI', data=average_roi_by_topic, palette='viridis')
plt.title('Average ROI by Dominant Topic')
plt.xlabel('Dominant Topic')
plt.ylabel('Average ROI')

topic_labeling = {
    0: "People, History & Journeys",
    1: "Battle & War",
    2: "Adaptations From Books or Novels",
    3: "Spy & Gang",
    4: "Families & Relationships",
    5: "Youth Adventures",
    6: "School & Friends",
    7: "Crime & Investigation"
}

handles = [Line2D([0], [0], color=sns.color_palette("viridis", n_colors= num_topics)[i], lw=2) 
           for i in range(num_topics)]
labels = [f"{topic_labeling[i]}" for i in range(num_topics)]
plt.legend(handles=handles, labels=labels, title="Themes", loc="center left", bbox_to_anchor=(1, 0.5))
plt.tight_layout()
plt.show()


"""
topic_counts = np.array([movie_topics.count(i) for i in range(lda_model.num_topics)])
# Plotting
plt.figure(figsize=(10, 6))
plt.bar(range(lda_model.num_topics), topic_counts, color='skyblue')
plt.xlabel('Topic Number')
plt.ylabel('Frequency')
plt.title('Frequency of Topics in Movies')
plt.xticks(range(lda_model.num_topics))
plt.show()
"""
