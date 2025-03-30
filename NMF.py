
import pandas as pd
import numpy as np
import nltk
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from matplotlib.lines import Line2D
from tqdm import tqdm
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.corpus import names  
import operator
from wordcloud import WordCloud
import os
from sklearn import decomposition
import scipy.stats as stats


def NMF():
    custom_stopwords = set([
        "film", "movie", "story", "documentary", "director", "character", "scene", 
        "cinema", "screen", "series", "episode", "cast", "actor", "actress", 
        "...", "life", "two", "one", "get", "find", "new", "filmmaker", "people",
        "based", "dr.", "day", "make", "time", "today", "look", "begin", "take", "local",
        "first", "work", "feature", "many", "set", "japanese", "moon", "must", "show", "year",
        "man", "world", "group", "last", "tell", "back", "want", "called", "like",
        "event", "animal", "directed", "want", "short", "early", "comedy", "person", "interview",
        "small", "space", "former", "special", "company", "place", "best", "made", "also", "men",
        "part", "follows", "audience", "play", "help", "stage", "american", "fall",
        "game", "christmas", "santa", "everyday", "computer", "including", "image", "sucess", "video",
        "german", "russian", "seven", "woman", "followed", "genre", "writer", "girl", "comedian", "city",
        "street", "italian", "polish", "examines", "india", "indian", "named", "plant", "book", "daily", "stranded",
        "soviet", "russia", "main", "become", "great", "light", "mind", "hong_kong", "moscow",
        "united_state", "three", "comic", "online", "going", "human", "china", "chinese",
        "novel", "london", "stop", "nazi", "becomes", "around", "change", "original", "white", "dog",
        "along", "hollywood", "footage", "met", "idea", "come", "reality", "true", "america",
        "state", "amp", "different", "famous", "tale", "modern", "four", "try", "start", "bos", "working", "completely", 
        "featuring", "unique", "and", "turn", "return", "young", "friend", "family", "mother", "child", "boy",
        "father", "daughter", "old", "together", "york", "soon", "since", "high", "put", "paris", "brother",
        "sister", "british", "vietnam", "los", "los angeles", "california", "overview", "yet", "live", "living",
        "thing", "little", "however", "even"
    ]).union(set([name.lower() for name in names.words()]))

    stop_words = set(stopwords.words('english')).union(custom_stopwords)
    df = pd.read_csv("./dataset/7/movies_metadata.csv")
    df = df.dropna(subset=['overview'])


    processed_summaries = []


    if os.path.exists("processed_summaries.pkl"):
        print("Loading preprocessed summaries and movie BoW from disk...")
        processed_summaries = pd.read_pickle("processed_summaries.pkl")

    else:
        print("Processing summaries...")
        for index, row in tqdm(df.iterrows(), total=len(df), desc="Processing Summaries"):
            processed_summary = process(row["overview"], stop_words)
            processed_summaries.append(processed_summary)
        pd.to_pickle(processed_summaries, "processed_summaries.pkl")
    

    vectorizer = TfidfVectorizer(max_df=0.4, min_df = 20, ngram_range=(1,1))
    matrix = vectorizer.fit_transform(processed_summaries)
    terms = list(vectorizer.get_feature_names_out())
    print(matrix.shape)
    ranks = rank_terms(matrix, terms)
    for i, pair in enumerate(ranks[:20]):
        print(f"{(i + 1)}: Term -> {pair[0]}, Weight -> {pair[1]}")

    #display_cloud_words(matrix, terms)
   
    W = None
    H = None
    num_topics = 8

    if os.path.exists('W_matrix.npy') and os.path.exists('H_matrix.npy'):
        W = np.load('W_matrix.npy')
        H = np.load('H_matrix.npy')
    else:
        model = decomposition.NMF(init= "nndsvd", n_components = num_topics)
        W = model.fit_transform(matrix)
        H = model.components_
        np.save('W_matrix.npy', W)  
        np.save('H_matrix.npy', H)

    for topic in range(num_topics):
        words = get_topic_top_k_words(terms, H, topic, k=15)
        movies = get_topic_top_k_movies(df['title'].to_list(), W, topic, k=30)
        print(f'Topic {topic + 1}: {words}')
        print(f'Topic {topic + 1}: {movies}')

    df = tranform(W, df)
    #ROI_per_dominant_topic(df, num_topics)
    #ANOVA(df)
    #vote_average_per_topic(df, num_topics)
    #vote_count_per_topic(df, num_topics)

def tranform(W, df):
    topics = np.argmax(W, axis = 1)
    transformed_df = df.copy(deep=True)
    transformed_df['topic'] = topics
    transformed_df['budget'] = pd.to_numeric(transformed_df['budget'], errors='coerce')
    transformed_df['revenue'] = pd.to_numeric(transformed_df['revenue'], errors='coerce')
    transformed_df = transformed_df[transformed_df['budget'] > 0]
    transformed_df = transformed_df[transformed_df['revenue'] > 0]
    transformed_df['ROI'] = (transformed_df['revenue'] - transformed_df['budget']) / transformed_df['budget']
    lower_quantile = transformed_df['ROI'].quantile(0.01)
    higher_quantile = transformed_df['ROI'].quantile(0.99)
    transformed_df = transformed_df[(transformed_df['ROI'] >= lower_quantile) & (transformed_df['ROI'] <= higher_quantile)]
    return transformed_df

def ROI_per_dominant_topic(W, df, num_topics):

    topic_labeling = {
        0: "Supernatural Adventures & Superheroes",
        1: "Crime & Investigation",
        2: "School & College Life",
        3: "War & Battle",
        4: "Criminal Activities & Gang",
        5: "Family Drama & Marital Issues",
        6: "Romance & Relationships",
        7: "Strange Supernatural Mysteries"
    }

    average_roi_by_topic = df.groupby('topic')['ROI'].mean().reset_index()


    # Plot the average ROI by dominant topic
    plt.figure(figsize=(10, 6))

    sns.barplot(x='topic', y='ROI', data=average_roi_by_topic, palette='viridis')
    plt.title('Average ROI by Dominant Topic')
    plt.xlabel('Dominant Topic')
    plt.ylabel('Average ROI')

    handles = [Line2D([0], [0], color=sns.color_palette("viridis", n_colors= num_topics)[i], lw=2) 
            for i in range(num_topics)]
    labels = [f"{topic_labeling[i]}" for i in range(num_topics)]
    plt.legend(handles=handles, labels=labels, title="Themes", loc="center left", bbox_to_anchor=(1, 0.5))
    plt.tight_layout()
    plt.show()



def ANOVA(df):

    f_statistic, p_value = stats.f_oneway(
        df[df['topic'] == 0]['ROI'], 
        df[df['topic'] == 1]['ROI'], 
        df[df['topic'] == 2]['ROI'], 
        df[df['topic'] == 3]['ROI'], 
        df[df['topic'] == 4]['ROI'], 
        df[df['topic'] == 5]['ROI'], 
        df[df['topic'] == 6]['ROI'], 
        df[df['topic'] == 7]['ROI']
    )

    print(f"ANOVA F-statistic: {f_statistic}, p-value: {p_value}")

    # If p-value < 0.05, reject null hypothesis and conclude that the means are significantly different
    if p_value < 0.05:
        print("There is a significant difference in ROI between topics.")
    else:
        print("There is no significant difference in ROI between topics.")

def vote_count_per_topic(W, df, num_topics):
    
    topic_labeling = {
        0: "Supernatural Adventures & Superheroes",
        1: "Crime & Investigation",
        2: "School & College Life",
        3: "War & Battle",
        4: "Criminal Activities & Gang",
        5: "Family Drama & Marital Issues",
        6: "Romance & Relationships",
        7: "Strange Supernatural Mysteries"
    }



    df = df[df['vote_count'] > 0]
    

    # Group by dominant_topic and calculate the average ROI
    average_count_by_topic = df.groupby('topic')['vote_count'].mean().reset_index()

    # Plot the average ROI by dominant topic
    plt.figure(figsize=(10, 6))
    sns.barplot(x='topic', y='vote_count', data=average_count_by_topic, palette='viridis')
    plt.title('Vote Count Average by Dominant Topic')
    plt.xlabel('Dominant Topic')
    plt.ylabel('Vote Count Average')

    handles = [Line2D([0], [0], color=sns.color_palette("viridis", n_colors= num_topics)[i], lw=2) 
            for i in range(num_topics)]
    labels = [f"{topic_labeling[i]}" for i in range(num_topics)]
    plt.legend(handles=handles, labels=labels, title="Themes", loc="center left", bbox_to_anchor=(1, 0.5))
    plt.tight_layout()
    plt.show()


def vote_average_per_topic(W, df, num_topics):
    
    topic_labeling = {
        0: "Supernatural Adventures & Superheroes",
        1: "Crime & Investigation",
        2: "School & College Life",
        3: "War & Battle",
        4: "Criminal Activities & Gang",
        5: "Family Drama & Marital Issues",
        6: "Romance & Relationships",
        7: "Strange Supernatural Mysteries"
    }



    df = df[df['vote_average'] > 0]
    

    # Group by dominant_topic and calculate the average ROI
    average_vote_by_topic = df.groupby('topic')['vote_average'].mean().reset_index()

    # Plot the average ROI by dominant topic
    plt.figure(figsize=(10, 6))
    sns.barplot(x='topic', y='vote_average', data=average_vote_by_topic, palette='viridis')
    plt.title('Vote Average by Dominant Topic')
    plt.xlabel('Dominant Topic')
    plt.ylabel('Vote Average')

    handles = [Line2D([0], [0], color=sns.color_palette("viridis", n_colors= num_topics)[i], lw=2) 
            for i in range(num_topics)]
    labels = [f"{topic_labeling[i]}" for i in range(num_topics)]
    plt.legend(handles=handles, labels=labels, title="Themes", loc="center left", bbox_to_anchor=(1, 0.5))
    plt.tight_layout()
    plt.show()




def get_topic_top_k_words(terms, H, topic_index, k):
    top_indices = np.argsort(H[topic_index, :])[::-1]
    top_words = []
    for term_index in top_indices[0:k]:
        top_words.append(terms[term_index])
    return top_words

def get_topic_top_k_movies(movies, W, topic_index, k):
    top_indices = np.argsort(W[:, topic_index])[::-1]
    top_movies = []
    for movie_index in top_indices[0:k]:
        top_movies.append(movies[movie_index])
    return top_movies
    
def display_cloud_words(matrix, terms):
    tfidf_scores = np.array(matrix.sum(axis=0)).flatten()
    word_freq = dict(zip(terms, tfidf_scores))
    wordcloud = WordCloud(width=800, height=400, background_color="white").generate_from_frequencies(word_freq)
    plt.figure(figsize=(10, 5))
    plt.imshow(wordcloud, interpolation="bilinear")
    plt.axis("off")
    plt.show()


def rank_terms(matrix, terms):
    sums = matrix.sum(axis = 0)
    weights = {}
    for col, term in enumerate(terms):
        weights[term] = sums[0, col]
    return sorted(weights.items(), key=operator.itemgetter(1), reverse=True)

def process(summary, stop_words):
    summary = summary.lower()
    tokens = word_tokenize(summary)
    lemmatizer = WordNetLemmatizer()
    tokens = [lemmatizer.lemmatize(word) for word in tokens]  # Lemmatization
    tokens = [token for token in tokens if token.isalpha() and token not in stop_words and len(token) >= 3]
    return " ".join(tokens)



if __name__ == '__main__':
    NMF()