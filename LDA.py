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
import pyLDAvis.gensim_models

nltk.download('stopwords')
nltk.download('punkt_tab')
nltk.download('wordnet')
nltk.download('names')


def LDA():

    custom_stopwords = set([
        "film", "movie", "story", "documentary", "director", "character", "scene", 
        "cinema", "screen", "series", "episode", "cast", "actor", "actress", 
        "...", "life", "two", "one", "get", "find", "new"
    ]).union(set([name.lower() for name in names.words()]))

    stop_words = set(stopwords.words('english')).union(custom_stopwords)
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
            processed_summary = process(row["overview"], stop_words)
            processed_summaries.append(processed_summary)
            movie_bow[(row['id'], row['title'])] = processed_summary

        pd.to_pickle(processed_summaries, "processed_summaries.pkl")
        pd.to_pickle(movie_bow, "movie_bow.pkl")

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
    
   # ROI_per_dominant_topic(lda_model, corpus, df, num_topics)
   # topic_evolution(lda_model, corpus, df, num_topics)
   # display_topics(lda_model)
   # visualization_of_topics(lda_model, corpus, dictionary)
   # topic_evolution(lda_model, corpus, df, num_topics)
   # vote_average_per_topic(lda_model, corpus, df, num_topics)
    vote_count_per_topic(lda_model, corpus, df, num_topics)


def vote_count_per_topic(lda_model, corpus, df, num_topics):
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

    movie_topics = get_movie_topics(lda_model, corpus)  

    transformed_df = df.copy(deep=True)
    transformed_df['dominant_topic'] = movie_topics
    transformed_df = transformed_df[transformed_df['vote_count'] > 0]
    

    # Group by dominant_topic and calculate the average ROI
    average_count_by_topic = transformed_df.groupby('dominant_topic')['vote_count'].mean().reset_index()

    # Plot the average ROI by dominant topic
    plt.figure(figsize=(10, 6))
    sns.barplot(x='dominant_topic', y='vote_count', data=average_count_by_topic, palette='viridis')
    plt.title('Vote Count Average by Dominant Topic')
    plt.xlabel('Dominant Topic')
    plt.ylabel('Vote Count Average')

    handles = [Line2D([0], [0], color=sns.color_palette("viridis", n_colors= num_topics)[i], lw=2) 
            for i in range(num_topics)]
    labels = [f"{topic_labeling[i]}" for i in range(num_topics)]
    plt.legend(handles=handles, labels=labels, title="Themes", loc="center left", bbox_to_anchor=(1, 0.5))
    plt.tight_layout()
    plt.show()

def vote_average_per_topic(lda_model, corpus, df, num_topics):

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

    movie_topics = get_movie_topics(lda_model, corpus)  

    transformed_df = df.copy(deep=True)
    transformed_df['dominant_topic'] = movie_topics
    transformed_df = transformed_df[transformed_df['vote_average'] > 0]
    

    # Group by dominant_topic and calculate the average ROI
    average_vote_by_topic = transformed_df.groupby('dominant_topic')['vote_average'].mean().reset_index()

    # Plot the average ROI by dominant topic
    plt.figure(figsize=(10, 6))
    sns.barplot(x='dominant_topic', y='vote_average', data=average_vote_by_topic, palette='viridis')
    plt.title('Vote Average by Dominant Topic')
    plt.xlabel('Dominant Topic')
    plt.ylabel('Vote Average')

    handles = [Line2D([0], [0], color=sns.color_palette("viridis", n_colors= num_topics)[i], lw=2) 
            for i in range(num_topics)]
    labels = [f"{topic_labeling[i]}" for i in range(num_topics)]
    plt.legend(handles=handles, labels=labels, title="Themes", loc="center left", bbox_to_anchor=(1, 0.5))
    plt.tight_layout()
    plt.show()



def topic_evolution(model, corpus, df, num_topics):
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

    movie_topics = get_movie_topics(model, corpus)
   
    transformed_df = df.copy(deep=True)
    transformed_df['dominant_topic'] = movie_topics
    transformed_df['release_year'] = pd.to_datetime(transformed_df['release_date'], 'coerce').dt.year
    transformed_df = transformed_df.dropna(subset=['release_year'])
    counts_by_yt = transformed_df.groupby(['release_year', 'dominant_topic']).size().reset_index(name='count')
    counts_filtered = counts_by_yt[counts_by_yt['release_year'].between(2012, 2017)]
    counts_filtered['topic'] = counts_filtered['dominant_topic'].map(topic_labeling)
    years = counts_filtered['release_year'].unique()
    fig, axes = plt.subplots(nrows=3, ncols=2, figsize=(15, 10))
    axes = axes.flatten()  

    for i, year in enumerate(years):
        year_data = counts_filtered[counts_filtered['release_year'] == year]
        # Create pie chart data (topic distribution)
        topic_counts = year_data['count']
        topic_labels = year_data['topic']
        # Pie chart for each year
        axes[i].pie(topic_counts, radius=1.5, labels=topic_labels, autopct='%1.1f%%', startangle=90)
        axes[i].set_title(f"Topic Distribution in {int(year)}", pad=50)

    # Adjust the layout and show the plot
    plt.tight_layout()
    plt.show()

def ROI_per_dominant_topic(model, corpus, df, num_topics):

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

    movie_topics = get_movie_topics(model, corpus)  

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

    handles = [Line2D([0], [0], color=sns.color_palette("viridis", n_colors= num_topics)[i], lw=2) 
            for i in range(num_topics)]
    labels = [f"{topic_labeling[i]}" for i in range(num_topics)]
    plt.legend(handles=handles, labels=labels, title="Themes", loc="center left", bbox_to_anchor=(1, 0.5))
    plt.tight_layout()
    plt.show()


def display_topics(model):
    for topic in model.print_topics():
        print(topic)

def visualization_of_topics(lda_model, corpus, dictionary):
    vis = pyLDAvis.gensim_models.prepare(lda_model, corpus, dictionary)
    pyLDAvis.save_html(vis, 'lda_visualization.html')

def process(summary, stop_words):
    summary = summary.lower()
    tokens = word_tokenize(summary)
    tokens = [token for token in tokens if token not in stop_words and len(token) >= 3]
    lemmatizer = WordNetLemmatizer()
    tokens = [lemmatizer.lemmatize(word) for word in tokens]  # Lemmatization
    return tokens

def get_movie_topics(lda_model, corpus):
    movie_topics = [] 
    for i, doc_bow in enumerate(corpus):
        topic_probs = lda_model.get_document_topics(doc_bow)
        dominant_topic = sorted(topic_probs, key=lambda x: x[1], reverse=True)[0][0]
        movie_topics.append(dominant_topic)
    return movie_topics


if __name__ == '__main__':
    LDA()