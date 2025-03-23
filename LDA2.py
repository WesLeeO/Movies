import pandas as pd
import os
from tqdm import tqdm
import gensim
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from gensim.models import Phrases
from gensim.models.phrases import Phraser
from gensim import corpora
from gensim.models import LdaModel
from gensim.models import CoherenceModel
import matplotlib.pyplot as plt
import seaborn as sns


#nltk.download('stopwords')
#nltk.download('punkt_tab')
#nltk.download('wordnet')


def LDA():

    df1 = pd.read_csv("./archive/rotten_tomatoes_critic_reviews.csv")
    reviews = df1[['rotten_tomatoes_link', 'review_content']]
    df2 = pd.read_csv("./dataset/7/movies_metadata.csv")
    movies = df2[['id', 'budget', 'revenue']]
    df3 = pd.read_csv("./archive/matched_movies.csv")

    custom_stopwords = set([
        "film", "movie", "...", "n\'t", "one", "make", "get", "life", "seen", "--", "\'\'",
        "``", "nan", "full_review_spanish", "spanish", "even", "lot", "fact", "review", "full_review", "also",
        "ever", "way", "story", "full", "much", "like", "doe", "since", "tell", "stone", "less", "sometimes",
        "enough", "year", "thing", "car", "really", "made", "would", "might", "could", "see", "thanks", "time",
        "good", "best", "feel", "better", "something", "choice", "rarely", "part", "prof", "star_war", "never",
        "give", "want", "take", "little", "still", "nothing", "may", "ultimately", "first", "need", "people",
        "say", "know", "seems", "manages", "yet", "mostly", "rogue", "oliver", "moment", "genre", "find", "end",
        "come", "back", "new", "minute"

    ])

    stop_words = set(stopwords.words('english')).union(custom_stopwords)

    movies['id'] = movies['id'].astype(str)
    movies = movies[movies['id'].str.isnumeric()]
    movies['id'] = movies['id'].astype(int)

    df3['movie_id'] = df3['movie_id'].astype(int)

    m1 = pd.merge(movies, df3, left_on='id', right_on='movie_id')
    m2 = pd.merge(m1, reviews, left_on='rotten_tomatoes_link', right_on='rotten_tomatoes_link')

    m2 = m2.dropna(subset=['review_content'])

    
    processed_reviews = []

    if os.path.exists("processed_reviews.pkl"):
        print("Loading preprocessed reviews...")
        processed_reviews = pd.read_pickle("processed_reviews.pkl")

    else:
        print("Processing reviews...")
        for index, row in tqdm(m2.iterrows(), total=len(m2), desc="Processing Summaries"):
            processed_review = process(str(row["review_content"]), stop_words)
            processed_reviews.append(processed_review)
        pd.to_pickle(processed_reviews, "processed_reviews.pkl")
        
    # Create bigram model
    bigram_model = Phrases(processed_reviews, min_count=5, threshold=10)
    bigram_phraser = Phraser(bigram_model)

    # Create trigram model
    trigram_model = Phrases(bigram_phraser[processed_reviews], min_count=5, threshold=10)
    trigram_phraser = Phraser(trigram_model)

    # Apply bigrams and trigrams
    processed_reviews = [
        trigram_phraser[bigram_phraser[review]] for review in processed_reviews
    ]

    dictionary = corpora.Dictionary(processed_reviews)
    dictionary.filter_extremes(no_below=5, no_above=0.2)
    corpus = [dictionary.doc2bow(review) for review in processed_reviews]

    lda_model_path = "lda_model_reviews.model"
    lda_model = None
    num_topics = 4

    if os.path.exists(lda_model_path):
        print("Loading LDA model from disk...")
        lda_model = LdaModel.load(lda_model_path)
    else:
        print("Training new LDA model...")
        lda_model = LdaModel(corpus, num_topics=num_topics, id2word=dictionary, passes=10, iterations=100, random_state=42)
        lda_model.save(lda_model_path)

    for topic in lda_model.print_topics():
        print(topic)
    
    coherence_model = CoherenceModel(model=lda_model, texts=processed_reviews, dictionary=dictionary, coherence='c_v')
    coherence_score = coherence_model.get_coherence()
    print(f"Coherence Score: {coherence_score}")

    review_labeling = {
        0:  'Thriller & Suspense',
        1:  'Franchise Movies & Audience Reaction',
        2:  'Acting & Performance',
        3:  'Funny & Entertaining'
    }

    review_colors = ['blue', 'green', 'orange', 'red']

    review_types = get_reviews_type(lda_model, corpus)
    transformed_df = m2.copy(deep=True)
    transformed_df['dominant_type'] = review_types
    transformed_df['budget'] = pd.to_numeric(transformed_df['budget'], errors='coerce')
    transformed_df['revenue'] = pd.to_numeric(transformed_df['revenue'], errors='coerce')
    transformed_df['ROI'] = transformed_df['revenue'] - transformed_df['budget']
    transformed_df = transformed_df[transformed_df['budget'] > 0]
    transformed_df = transformed_df[transformed_df['revenue'] > 0]

    type_proportions = transformed_df.groupby('id')['dominant_type'].value_counts(normalize=True).unstack(fill_value=0)
    final_df = type_proportions.join(transformed_df[['id', 'ROI']].drop_duplicates().set_index('id'))

    fig, axes = plt.subplots(2, 2, figsize=(20, 3 * 2))

    axes = axes.flatten()

    for i, review_type in enumerate(final_df.columns[:-1]):  # Exclude ROI column
        sns.scatterplot(x=final_df['ROI'], y=final_df[review_type], ax=axes[i], color = review_colors[i], label=f"Review Type: {review_labeling[review_type]}")
        axes[i].set_title(f"{review_labeling[review_type]} - Proportion vs ROI")
        axes[i].set_xlabel('ROI')
        axes[i].set_ylabel(f'Proportion of {review_labeling[review_type]}')
        axes[i].legend()

    plt.tight_layout()
    plt.suptitle("Proportions of Review Types vs ROI", fontsize=16, y=1.03) 
    plt.show()

    




def get_reviews_type(lda_model, corpus):
    review_types = [] 
    for i, doc_bow in enumerate(corpus):
        review_probs = lda_model.get_document_topics(doc_bow)
        dominant_type = sorted(review_probs, key=lambda x: x[1], reverse=True)[0][0]
        review_types.append(dominant_type)
    return review_types


def process(review_content, stop_words):
    review_content = review_content.lower()
    tokens = word_tokenize(review_content)
    lemmatizer = WordNetLemmatizer()
    tokens = [lemmatizer.lemmatize(word) for word in tokens]
    tokens = [token for token in tokens if token.isalpha() and token not in stop_words and len(token) >= 3]
    return tokens









if __name__ == '__main__':
    LDA()