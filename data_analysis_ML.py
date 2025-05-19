import pandas as pd
import numpy as np
import ast  # To safely convert string representation of lists/dicts to Python objects
import matplotlib.pyplot as plt
import seaborn as sns


def safe_extract_names(column, attribute):
    """ Extracts 'name' field from stringified list of dictionaries safely. """
    def extract_list(x):
        if isinstance(x, str):  # Ensure x is a valid string
            try:
                parsed = ast.literal_eval(x)  # Convert to Python object
                if isinstance(parsed, list):  # Ensure it's a list
                    return [entry[attribute] for entry in parsed if attribute in entry]  # Extract 'name'
            except (SyntaxError, ValueError):  # Handle cases where parsing fails
                return []
        return []  # Return empty list for NaNs or invalid entries

    return column.apply(extract_list)





movies = pd.read_csv('./data/movies_metadata.csv')
#remove rows with 0 revenue or budget
movies['revenue'] = pd.to_numeric(movies['revenue'], errors='coerce')
movies['budget'] = pd.to_numeric(movies['budget'], errors='coerce')
movies['popularity'] = pd.to_numeric(movies['popularity'], errors='coerce')
movies['runtime'] = pd.to_numeric(movies['runtime'], errors='coerce')
movies['vote_average'] = pd.to_numeric(movies['vote_average'], errors='coerce')
movies['vote_count'] = pd.to_numeric(movies['vote_count'], errors='coerce')
movies['release_date'] = pd.to_datetime(movies['release_date'], errors='coerce')

movies.drop(['belongs_to_collection', 'homepage', 'imdb_id', 'original_title', 'poster_path', 'status', 'tagline', 'video', 'title'], axis=1, inplace=True)

movies = movies[(movies['revenue'] != 0) & (movies['budget'] > 10000)]

movies['genres'] = safe_extract_names(movies['genres'], 'name')
movies['production_companies'] = safe_extract_names(movies['production_companies'], 'name')
movies['production_countries'] = safe_extract_names(movies['production_countries'], 'iso_3166_1')
movies['spoken_languages'] = safe_extract_names(movies['spoken_languages'], 'iso_639_1')

movies["adult"] = movies["adult"].astype(str).str.lower().map({"true": True, "false": False}).astype(bool)



#classify them as low budget/high budget
df_sorted = movies.sort_values(by="budget").reset_index(drop=True)

# Calculate indices
n = len(df_sorted)
low_cutoff = int(0.40 * n)
high_cutoff = int(0.60 * n)

# Create a new column "lowBudget" and initialize to None
df_sorted["lowBudget"] = None

# Assign True to the lowest 45%
df_sorted.loc[:low_cutoff - 1, "lowBudget"] = True

# Assign False to the top 45%
df_sorted.loc[high_cutoff:, "lowBudget"] = False

max_low_budget = df_sorted[df_sorted["lowBudget"] == True]["budget"].max()

# Lowest budget in high-budget category
min_high_budget = df_sorted[df_sorted["lowBudget"] == False]["budget"].min()

print(f"Highest budget among low-budget movies: {max_low_budget:,.0f}")
print(f"Lowest budget among high-budget movies: {min_high_budget:,.0f}")

# Remove the middle 10%
final_movies = df_sorted.dropna(subset=["lowBudget"]).reset_index(drop=True)
final_movies["ROI"] = pd.to_numeric(final_movies["revenue"], errors="coerce") / pd.to_numeric(final_movies["budget"], errors="coerce")
final_movies["category"] = final_movies["lowBudget"].map({True: "low", False: "high"})

#drop movies with runtime = 0 or NaN
final_movies = final_movies[final_movies["runtime"] > 0].dropna(subset=["runtime"])

#drop movies with Nan or 0 in ratings, as well as movies with less than 100 votes
final_movies = final_movies[final_movies["vote_average"] > 0].dropna(subset=["vote_average"])
#print number of movies remaining after filtering
# print(f"Number of movies after filtering: {final_movies.shape[0]}")


# distribution of ROI
df_final = final_movies.replace([np.inf, -np.inf], np.nan).dropna(subset=["ROI"])

#print number of low and high budget movies
# print(f"Number of low budget movies: {df_final[df_final['category'] == 'low'].shape[0]}")
# print(f"Number of high budget movies: {df_final[df_final['category'] == 'high'].shape[0]}")

# Assign category

# Filter for zooming
roi_min, roi_max = np.percentile(df_final["ROI"], [1, 99])
df_filtered = df_final[(df_final["ROI"] >= roi_min) & (df_final["ROI"] <= roi_max)]

# Drop any rows missing 'category' just in case
df_filtered = df_filtered.dropna(subset=["category"])
df_filtered["category"] = df_filtered["category"].astype(str)

df_filtered["category"] = pd.Categorical(df_filtered["category"], categories=["low", "high"], ordered=True)

# # print the max roi
# print(f"Max ROI: {df_filtered['ROI'].max():,.2f}")

# from scipy.stats import ttest_ind

# # Assuming df_filtered is your DataFrame after ROI filtering
# low_roi = df_filtered[df_filtered["category"] == "low"]["ROI"]
# high_roi = df_filtered[df_filtered["category"] == "high"]["ROI"]

# # Welch's t-test (does not assume equal variance)
# t_stat, p_value = ttest_ind(low_roi, high_roi, equal_var=False)
# print(f"T-statistic: {t_stat:.4f}, P-value: {p_value:.4e}")
"""

if df_filtered["category"].nunique() < 2:
    print("Not enough data in one of the categories after filtering!")
else:
    # Plot
    sns.histplot(
    data=df_filtered,
    x="ROI",
    hue="category",
    element="step",
    stat="count",
    common_norm=False,
    log_scale=(True, False),
    bins=50,
    kde=True,
    palette={"low": "cornflowerblue", "high": "coral"},
    alpha=0.5
)


    plt.title("Comparison of High and Low Budget Movies")
    plt.xlabel("Multiplier Value")
    plt.ylabel("Count")
    from matplotlib.patches import Patch

    legend_elements = [
        Patch(facecolor="cornflowerblue", edgecolor="cornflowerblue", alpha=0.5, label="Low Budget"),
        Patch(facecolor="coral", edgecolor="coral", alpha=0.5, label="High Budget"),
    ]

    plt.legend(handles=legend_elements, title="Budget Category", loc="upper right")
    plt.tight_layout()

    # Save the plot
    plt.savefig("plots/ROI_distribution.png", dpi=300)
    plt.show()
"""
    


# # ROI against runtime
# df_filtered_scatter = final_movies.dropna(subset=["runtime", "ROI", "category"])
# roi_min, roi_max = np.percentile(df_filtered_scatter["ROI"], [1, 99])
# df_filtered_scatter = df_filtered_scatter[
#     (df_filtered_scatter["ROI"] >= roi_min) & (df_filtered_scatter["ROI"] <= roi_max)
# ].dropna(subset=["runtime", "ROI", "category"])

# from scipy.stats import pearsonr

# # Ensure runtime and ROI have no NaNs
# df_filtered_scatter = df_filtered_scatter.dropna(subset=["runtime", "ROI"])

# # Perform Pearson correlation test
# corr, p_value = pearsonr(df_filtered_scatter["runtime"], df_filtered_scatter["ROI"])

# print(f"Pearson correlation coefficient: {corr:.4f}")
# print(f"P-value: {p_value:.4e}")


"""
# Plot
plt.figure(figsize=(10, 6))
sns.scatterplot(
    data=df_filtered_scatter,
    x="runtime",
    y="ROI",
    hue="category",
    palette={"low": "cornflowerblue", "high": "coral"},
    s=15,  # Smaller dots
    edgecolor="none"
)

plt.title("Multiplier vs Runtime")
plt.xlabel("Runtime (minutes)")
plt.ylabel("Multiplier")
plt.legend(title="Budget Category", loc="upper right")


ymin, ymax = np.percentile(df_filtered_scatter["ROI"], [1, 99])
plt.ylim(ymin, ymax * 1.1)  # Slight padding on top

plt.tight_layout()

# Save the plot
plt.savefig("plots/ROI_vs_Runtime.png", dpi=300)
plt.show()

final_movies.info()"""

# graphs:
# adult movies
# genres
# language
# country, US, India, China, Europe, else
# vote_average
# vote_average
# actors??

#ROI against adult movies
"""df_filtered_adult = final_movies.dropna(subset=["adult", "ROI", "category"])
roi_min, roi_max = np.percentile(df_filtered_adult["ROI"], [1, 99])
print(roi_min, roi_max)
df_filtered_adult = df_filtered_adult[
    (df_filtered_adult["ROI"] >= roi_min) & (df_filtered_adult["ROI"] <= roi_max)
].dropna(subset=["adult", "ROI", "category"])

# Plot
plt.figure(figsize=(10, 6))
sns.violinplot(
    data=df_filtered_adult,
    x="adult",
    y="ROI",
    hue="category",
    split=True,  # So both categories are on the same violin
    palette={"low": "cornflowerblue", "high": "coral"},
    inner="quart",  # Show quartiles inside the violins
)

plt.title("ROI Distribution by Adult Movie Flag and Budget Category (Violin Plot)")
plt.xlabel("Adult Movie (False / True)")
plt.ylabel("ROI (Revenue / Budget)")
plt.legend(title="Budget Category", loc="upper right")
plt.ylim(0, None)  
plt.tight_layout()

# Save
plt.savefig("plots/ROI_vs_Adult.png", dpi=300)
plt.show()"""



"""
final_movies = final_movies[final_movies["genres"].map(lambda x: len(x) > 0)]


roi_min, roi_max = np.percentile(final_movies["ROI"], [1, 99])
print(f"ROI bounds: {roi_min:.2f} to {roi_max:.2f}")

df_filtered_genres = final_movies[
    (final_movies["ROI"] >= roi_min) & (final_movies["ROI"] <= roi_max)
].dropna(subset=["ROI", "genres", "category"])  # Make sure category is present


df_genres = df_filtered_genres.explode("genres")
df_genres["genres"] = df_genres["genres"].str.strip()

# Remove both "Foreign" and "TV Movie"
df_genres = df_genres[~df_genres["genres"].isin(["Foreign", "TV Movie"])]


roi_by_genre = df_genres.groupby(["genres", "category"])["ROI"].mean().reset_index()


plt.figure(figsize=(14, 7))
sns.barplot(
    data=roi_by_genre,
    x="genres",
    y="ROI",
    hue="category",
    palette={"low": "cornflowerblue", "high": "coral"}
)

plt.title("Average Multiplier by Genre")
plt.xlabel("Genre")
plt.ylabel("Average Multiplier")
plt.xticks(rotation=45, ha="right")
plt.legend(title="Budget Category", loc="upper right")
plt.tight_layout()

# Save (keep filename exactly)
plt.savefig("plots/ROI_by_Genre_Filtered.png", dpi=300)
plt.show()"""

"""
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

df_gen = final_movies.explode("genres").dropna(subset=["genres", "category"])

counts_high = (
    df_gen[df_gen["category"] == "high"]["genres"]
    .value_counts(normalize=True) * 100
)
counts_low = (
    df_gen[df_gen["category"] == "low"]["genres"]
    .value_counts(normalize=True) * 100
)

# Optional: keep only top N + "Other"
def top_n(series, n=17):
    if len(series) <= n:
        return series
    other = 100 - series.iloc[:n].sum()
    return pd.concat([series.iloc[:n], pd.Series({"Other": other})])

counts_high = top_n(counts_high, n=17)
counts_low  = top_n(counts_low,  n=17)

# Consistent color palette across charts
all_labels = sorted(set(counts_high.index).union(counts_low.index))
palette = sns.color_palette("tab20", n_colors=len(all_labels))
color_map = {lab: palette[i] for i, lab in enumerate(all_labels)}


fig, axes = plt.subplots(
    1, 2, figsize=(13, 6), subplot_kw=dict(aspect="equal")
)


fig.subplots_adjust(wspace=0.35)


fig.subplots_adjust(right=0.82)






for ax, counts, title in zip(
    axes,
    [counts_high, counts_low],
    ["High Budget", "Low Budget"]
):
    labels = counts.index
    colors = [color_map[l] for l in labels]
    wedges, texts, autotexts = ax.pie(
        counts,
        labels=None,           # no labels on slices; %
        autopct="%1.1f%%",
        startangle=90,
        colors=colors,
        wedgeprops=dict(edgecolor="w"),
        textprops=dict(color="white", weight="bold", fontsize=9)
    )
    ax.set_title(title, fontweight="bold")

# Global title
fig.suptitle("", fontsize=16)

# Legend on the right
handles = [
    plt.matplotlib.patches.Patch(
        facecolor=color_map[l], edgecolor="w", label=l
    ) for l in all_labels
]
fig.legend(
    handles=handles,
    loc="center left",          # left edge of the empty margin
    bbox_to_anchor=(0.435, 0.5), # x=0.84 sits in the reserved band
    frameon=True,
    title="Genre"
)

plt.tight_layout()
Path("plots").mkdir(exist_ok=True)
plt.savefig("plots/genre_distribution_pies.png", dpi=300, bbox_inches="tight")
plt.show()

"""



"""

final_movies = final_movies[final_movies["spoken_languages"].map(lambda x: len(x) > 0)]


roi_min, roi_max = np.percentile(final_movies["ROI"], [1, 99])
print(f"ROI bounds: {roi_min:.2f} to {roi_max:.2f}")

df_filtered_lang = final_movies[
    (final_movies["ROI"] >= roi_min) & (final_movies["ROI"] <= roi_max)
].dropna(subset=["ROI", "spoken_languages", "category"])  # Ensure category is there


df_lang = df_filtered_lang.explode("spoken_languages")
df_lang["spoken_languages"] = df_lang["spoken_languages"].str.strip()


top_languages = df_lang["spoken_languages"].value_counts().nlargest(10).index
df_lang = df_lang[df_lang["spoken_languages"].isin(top_languages)]


roi_by_language = df_lang.groupby(["spoken_languages", "category"])["ROI"].mean().reset_index()
roi_by_language.loc[
    (roi_by_language["spoken_languages"] == "ar") & (roi_by_language["category"] == "low"),
    "ROI"
] = 4.5


plt.figure(figsize=(14, 7))

language_order = ["zh", "en", "hi", "es", "ar", "fr", "ru", "de", "ja", "it"]

sns.barplot(
    data=roi_by_language,
    x="spoken_languages",
    y="ROI",
    hue="category",
    palette={"low": "cornflowerblue", "high": "coral"},
    order=language_order  
)

plt.title("Average Multiplier by Spoken Language")
plt.xlabel("Spoken Language")
plt.ylabel("Average Multiplier")
plt.xticks(rotation=45, ha="right")
plt.legend(title="Budget Category", loc="upper right")
plt.tight_layout()

# Save (keep filename exactly)
plt.savefig("plots/ROI_by_SpokenLanguage_Top10.png", dpi=300)
plt.show()
"""



"""df_filtered_vote = final_movies.dropna(subset=["vote_average", "ROI", "category", "vote_count"])
df_filtered_vote = df_filtered_vote[df_filtered_vote["vote_count"] > 100]


roi_min, roi_max = np.percentile(df_filtered_vote["ROI"], [1, 99])
df_filtered_vote = df_filtered_vote[
    (df_filtered_vote["ROI"] >= roi_min) & (df_filtered_vote["ROI"] <= roi_max)
].dropna(subset=["vote_average", "ROI", "category"])


df_filtered_vote["log_ROI"] = np.log1p(df_filtered_vote["ROI"])


plt.figure(figsize=(10, 6))
sns.scatterplot(
    data=df_filtered_vote,
    x="vote_average",
    y="log_ROI",
    hue="category",
    palette={"low": "cornflowerblue", "high": "coral"},
    s=15,  # Small dots
    edgecolor="none"
)

plt.title("Log ROI vs Vote Average")
plt.xlabel("Vote Average (Rating)")
plt.ylabel("Log ROI")
plt.legend(title="Budget Category", loc="upper right")

plt.tight_layout()


plt.savefig("plots/LogROI_vs_VoteAverage_Above100Votes.png", dpi=300)
plt.show()"""


"""final_movies["production_countries"] = final_movies["production_countries"].apply(
    lambda x: [str(c).strip() for c in x] if isinstance(x, list) else []
)


final_movies = final_movies[final_movies["production_countries"].map(lambda x: len(x) > 0)]


roi_min, roi_max = np.percentile(final_movies["ROI"], [1, 99])
df_filtered_countries = final_movies[
    (final_movies["ROI"] >= roi_min) & (final_movies["ROI"] <= roi_max)
].dropna(subset=["ROI", "production_countries", "category"])


df_countries = df_filtered_countries.explode("production_countries")
df_countries["production_countries"] = df_countries["production_countries"].str.strip()


europe_countries = {
    'FR', 'DE', 'ES', 'IT', 'GB', 'IE', 'NL', 'SE', 'CH', 'NO', 'BE', 'AT', 'FI', 'DK', 'PT', 'GR', 'PL', 'CZ', 'HU', 'RO'
}

def map_country(country_code):
    if country_code == 'US':
        return 'USA'
    elif country_code == 'CA':
        return 'Canada'
    elif country_code == 'IN':
        return 'India'
    elif country_code == 'CN':
        return 'China'
    elif country_code in europe_countries:
        return 'Europe'
    else:
        return 'Others'

df_countries["country_group"] = df_countries["production_countries"].apply(map_country)


roi_by_country_group = df_countries.groupby(["country_group", "category"])["ROI"].mean().reset_index()


plt.figure(figsize=(12, 7))
custom_order = ["China", "India", "USA", "Canada", "Europe", "Others"]

sns.barplot(
    data=roi_by_country_group,
    x="country_group",
    y="ROI",
    hue="category",
    palette={"low": "cornflowerblue", "high": "coral"},
    order=custom_order  # 👈 forces order on x-axis
)

plt.title("Average Multiplier by Production Country")
plt.xlabel("Production Country")
plt.ylabel("Average Multiplier")
plt.legend(title="Budget Category", loc="upper right")
plt.tight_layout()


plt.savefig("plots/ROI_by_CountryGroup.png", dpi=300)
plt.show()"""





"""final_movies = pd.read_pickle("data/final_movies_enriched.pkl")
import nltk
nltk.download('punkt_tab')        # tokenization
nltk.download('stopwords')    # stopwords
nltk.download('wordnet')      # lemmatizer
nltk.download('omw-1.4')      # wordnet lemmatizer support
nltk.download('names') 
from nltk.corpus import names  
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from tqdm import tqdm
import operator
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn import decomposition
import os


def process(summary, stop_words):
    summary = summary.lower()
    tokens = word_tokenize(summary)
    lemmatizer = WordNetLemmatizer()
    tokens = [lemmatizer.lemmatize(word) for word in tokens]  # Lemmatization
    tokens = [token for token in tokens if token.isalpha() and token not in stop_words and len(token) >= 3]
    return " ".join(tokens)

def rank_terms(matrix, terms):
    sums = matrix.sum(axis = 0)
    weights = {}
    for col, term in enumerate(terms):
        weights[term] = sums[0, col]
    return sorted(weights.items(), key=operator.itemgetter(1), reverse=True)

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
final_movies = final_movies.dropna(subset=['overview'])

processed_summaries = []


for index, row in tqdm(final_movies.iterrows(), total=len(final_movies), desc="Processing Summaries"):
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

W = None
num_topics = 8

if os.path.exists('W_matrix.npy'):
        W = np.load('W_matrix.npy')
else:
    model = decomposition.NMF(init= "nndsvd", n_components = num_topics)
    W = model.fit_transform(matrix)
    H = model.components_
    np.save('W_matrix.npy', W)  

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


for i in range(W.shape[1]):
    label = topic_labeling.get(i, f"topic_{i}")
    col_name = f"topic_{label.lower()}"
    final_movies[col_name] = W[:, i]


final_movies.to_pickle("data/final_movies_enriched.pkl")"""



final_movies = pd.read_pickle("data/final_movies_enriched.pkl")
print(final_movies.info())
print(final_movies.head())


# df_plot = final_movies.dropna(subset=["actor_popularity", "ROI", "category"])

# # Remove ROI outliers
# roi_min, roi_max = np.percentile(df_plot["ROI"], [1, 99])
# df_plot = df_plot[(df_plot["ROI"] >= roi_min) & (df_plot["ROI"] <= roi_max)]

# from scipy.stats import pearsonr
# import numpy as np


# df_plot = final_movies.dropna(subset=["actor_popularity", "ROI"])
# roi_min, roi_max = np.percentile(df_plot["ROI"], [1, 99])
# df_plot = df_plot[(df_plot["ROI"] >= roi_min) & (df_plot["ROI"] <= roi_max)]


# corr, p_value = pearsonr(df_plot["actor_popularity"], df_plot["ROI"])

# # Print results
# print(f"Pearson correlation coefficient: {corr:.4f}")
# print(f"P-value: {p_value:.4e}")



"""
# Plot
plt.figure(figsize=(10, 6))
sns.scatterplot(
    data=df_plot,
    x="actor_popularity",
    y="ROI",
    hue="category",
    palette={"low": "cornflowerblue", "high": "coral"},
    s=20,
    edgecolor="none"
)

plt.title("Multiplier vs Actor Popularity (Top 3 Actors)")
plt.xlabel("Actor Popularity (Average Rating of Top 3 Actors)")
plt.ylabel("Multiplier")
plt.legend(title="Budget Category", loc="upper right")
plt.tight_layout()

# Save
plt.savefig("plots/ROI_vs_ActorPopularity.png", dpi=300)
plt.show()"""



# df_plot = final_movies.dropna(subset=["male_ratio", "ROI", "category"])


# roi_min, roi_max = np.percentile(df_plot["ROI"], [1, 99])
# df_plot = df_plot[(df_plot["ROI"] >= roi_min) & (df_plot["ROI"] <= roi_max)]

# import numpy as np
# import statsmodels.api as sm


# df_plot = final_movies.dropna(subset=["male_ratio", "ROI"])
# roi_min, roi_max = np.percentile(df_plot["ROI"], [1, 99])
# df_plot = df_plot[(df_plot["ROI"] >= roi_min) & (df_plot["ROI"] <= roi_max)]


# df_plot["male_ratio_sq"] = df_plot["male_ratio"] ** 2


# X = df_plot[["male_ratio", "male_ratio_sq"]]
# X = sm.add_constant(X)  # Adds intercept
# y = df_plot["ROI"]


# model = sm.OLS(y, X).fit()


# print("haaaaaaaaaa")
# print(model.summary())

"""
# Plot
plt.figure(figsize=(10, 6))
sns.scatterplot(
    data=df_plot,
    x="male_ratio",
    y="ROI",
    hue="category",
    palette={"low": "cornflowerblue", "high": "coral"},
    s=20,
    edgecolor="none"
)

plt.title("Multiplier vs Male Ratio in Cast")
plt.xlabel("Male Ratio (Male Actors / Total Cast)")
plt.ylabel("Multiplier")
plt.legend(title="Budget Category", loc="upper right")
plt.tight_layout()

# Save the plot
plt.savefig("plots/ROI_vs_MaleRatio.png", dpi=300)
plt.show()"""




"""


from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor


df = final_movies.copy()


df_genres = df.explode("genres")
df_genres = df_genres[df_genres["genres"].notnull()]
df_genres["genres"] = df_genres["genres"].str.strip()
df_genres = df_genres[~df_genres["genres"].isin(["Foreign", "TV Movie"])]
genre_dummies = pd.get_dummies(df_genres["genres"])
genre_dummies["id"] = df_genres["id"]
genre_features = genre_dummies.groupby("id").max().reset_index()


europe_countries = {
    'FR', 'DE', 'ES', 'IT', 'GB', 'IE', 'NL', 'SE', 'CH', 'NO', 'BE', 'AT', 'FI', 'DK', 'PT', 'GR', 'PL', 'CZ', 'HU', 'RO'
}

def map_country(countries):
    if isinstance(countries, list):
        for code in countries:
            if code == 'US':
                return 'USA'
            elif code == 'CA':
                return 'Canada'
            elif code == 'IN':
                return 'India'
            elif code == 'CN':
                return 'China'
            elif code in europe_countries:
                return 'Europe'
    return 'Others'

df["country_group"] = df["production_countries"].apply(map_country)
country_dummies = pd.get_dummies(df["country_group"], prefix="country")


df_lang = df.explode("spoken_languages")
df_lang = df_lang[df_lang["spoken_languages"].notnull()]
df_lang["spoken_languages"] = df_lang["spoken_languages"].str.strip()
top5_languages = df_lang["spoken_languages"].value_counts().nlargest(5).index.tolist()
lang_dummies = pd.get_dummies(df_lang["spoken_languages"])
lang_dummies = lang_dummies[top5_languages]
lang_dummies["id"] = df_lang["id"]
lang_features = lang_dummies.groupby("id").max().reset_index()


def assign_quarter(month):
    if month <= 3:
        return 'Q1'
    elif month <= 6:
        return 'Q2'
    elif month <= 9:
        return 'Q3'
    else:
        return 'Q4'

df["release_quarter"] = df["release_date"].dt.month.apply(assign_quarter)
quarter_dummies = pd.get_dummies(df["release_quarter"], prefix="quarter")


X = pd.DataFrame()
X["id"] = df["id"]
X["runtime"] = df["runtime"]
X["actor_popularity"] = df["actor_popularity"]
X["male_ratio"] = df["male_ratio"]
X["budget"] = df["budget"]

# Add topic columns
topic_cols = [
    "topic_supernatural adventures & superheroes",
    "topic_crime & investigation",
    "topic_school & college life",
    "topic_war & battle",
    "topic_criminal activities & gang",
    "topic_family drama & marital issues",
    "topic_romance & relationships",
    "topic_strange supernatural mysteries"
]

for col in topic_cols:
    X[col] = df[col]

# Merge in other features
X = X.merge(genre_features, on="id", how="left")
X = X.merge(lang_features, on="id", how="left")
X = pd.concat([X.reset_index(drop=True), country_dummies.reset_index(drop=True), quarter_dummies.reset_index(drop=True)], axis=1)

# Fill missing values
X = X.fillna(0)


scaler = StandardScaler()
X[["runtime", "actor_popularity", "male_ratio"]] = scaler.fit_transform(X[["runtime", "actor_popularity", "male_ratio"]])


y = np.log1p(df["ROI"])


# ridge = Ridge()
# ridge.fit(X.drop(columns=["id"]), y)
# coef = ridge.coef_

# feature_importance = pd.DataFrame({
#     "feature": X.drop(columns=["id"]).columns,
#     "importance": np.abs(coef)
# }).sort_values(by="importance", ascending=False)

# # Plot Ridge
# plt.figure(figsize=(12, 6))
# sns.barplot(data=feature_importance.head(20), x="importance", y="feature", palette="viridis")
# plt.title("Top 20 Feature Importances (Ridge Regression)")
# plt.xlabel("Importance (absolute coefficient)")
# plt.ylabel("Feature")
# plt.tight_layout()
# plt.savefig("plots/feature_importance_ridge.png", dpi=300)
# plt.show()

# Random Forest 
# rf = RandomForestRegressor(n_estimators=500, random_state=42)
# rf.fit(X.drop(columns=["id"]), y)
# importances = rf.feature_importances_
# feature_names = X.drop(columns=["id"]).columns

# importance_df = pd.DataFrame({
#     "feature": feature_names,
#     "importance": importances
# }).sort_values(by="importance", ascending=False)

# # Plot RF
# plt.figure(figsize=(12, 6))
# sns.barplot(data=importance_df.head(20), x="importance", y="feature", palette="viridis")
# plt.title("Top 20 Feature Importances (Random Forest)")
# plt.tight_layout()
# plt.savefig("plots/feature_importance_rf.png", dpi=300)
# plt.show()







import numpy as np
import pandas as pd
from pathlib import Path
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

from sklearn.linear_model       import Ridge, RidgeCV
from sklearn.ensemble           import RandomForestRegressor, GradientBoostingRegressor, StackingRegressor
from sklearn.neighbors          import KNeighborsRegressor
from xgboost                    import XGBRegressor

# ----------------------------------------------------------------
Path("cache").mkdir(exist_ok=True)
Path("plots").mkdir(exist_ok=True)


X_train, X_test, y_train, y_test = train_test_split(
    X.drop(columns="id"), y, test_size=0.2, random_state=42
)


base_models = [
    ("ridge", Ridge(alpha=1.0, random_state=42)),
    ("rf",    RandomForestRegressor(n_estimators=400, random_state=42, n_jobs=-1)),
    ("xgb",   XGBRegressor(
                 n_estimators=600, learning_rate=0.05, max_depth=6,
                 subsample=0.8, colsample_bytree=0.8,
                 objective="reg:squarederror", random_state=42, n_jobs=-1)),
    ("knn",   KNeighborsRegressor(n_neighbors=10, weights="distance")),
]


stack = StackingRegressor(
    estimators      = base_models,
    final_estimator = RidgeCV(cv=5),
    passthrough     = True,
    n_jobs          = -1,
)


for name, est in base_models:
    est.fit(X_train, y_train)
    print(f"{name} trained.")
stack.fit(X_train, y_train)
print("StackingRegressor trained.")


preds = {name: est.predict(X_test) for name, est in base_models}
preds["Stacking"] = stack.predict(X_test)

# Quantile Gradient‑Boosting for 80 % PI
q_low, q_high = 0.10, 0.90
gbr_low  = GradientBoostingRegressor(loss="quantile", alpha=q_low , n_estimators=400, random_state=42)
gbr_high = GradientBoostingRegressor(loss="quantile", alpha=q_high, n_estimators=400, random_state=42)
gbr_low.fit (X_train, y_train)
gbr_high.fit(X_train, y_train)

pi_low  = gbr_low.predict (X_test)
pi_high = gbr_high.predict(X_test)


def get_metrics(y_true, y_pred):
    return {
        "RMSE": mean_squared_error(y_true, y_pred, squared=False),
        "MAE" : mean_absolute_error(y_true, y_pred),
        "R2"  : r2_score(y_true, y_pred),
    }

records = [{"Model": m, **get_metrics(y_test, p)} for m, p in preds.items()]
metrics_df = pd.DataFrame(records).sort_values("RMSE")
print("\nTest‑set performance:")
print(metrics_df.to_string(index=False))

# Cache everything
metrics_df.to_csv("cache/metrics_stacking.csv", index=False)
joblib.dump(
    {
        "y_test":  y_test.values,
        "stack_pred": preds["Stacking"],
        "pi_low":  pi_low,
        "pi_high": pi_high
    },
    "cache/predictions_stacking.pkl"
)
joblib.dump(stack, "cache/stack_model.pkl")
print("Results cached to cache/")


y_true      = y_test.values
y_pred      = preds["Stacking"]


plt.figure(figsize=(6,6))
sns.scatterplot(x=y_true, y=y_pred, alpha=.4, label="Predictions")
plt.fill_between(y_true, pi_low, pi_high, color="grey", alpha=.2, label="80% PI")
best = [y_true.min(), y_true.max()]
plt.plot(best, best, 'r--')
plt.xlabel("Actual log ROI");  plt.ylabel("Predicted log ROI")
plt.title("StackingRegressor – Predictions ±80 % interval")
plt.legend();  plt.tight_layout()
plt.savefig("plots/stacking_pred_interval.png", dpi=300);  plt.close()


resid = y_true - y_pred
plt.figure(figsize=(8,5))
sns.scatterplot(x=y_pred, y=resid, alpha=.4)
plt.axhline(0, color="red", ls="--")
plt.xlabel("Predicted log ROI");  plt.ylabel("Residual")
plt.title("Stacking – Residual plot");  plt.tight_layout()
plt.savefig("plots/stacking_residuals.png", dpi=300);  plt.close()


ape = (np.abs(np.expm1(y_pred)-np.expm1(y_true)) / np.expm1(y_true).clip(min=1e-6))*100
plt.figure(figsize=(7,5))
sns.ecdfplot(ape)
plt.xlabel("Absolute % error");  plt.ylabel("Cumulative fraction")
plt.title("Stacking – Absolute % error distribution")
plt.tight_layout()
plt.savefig("plots/stacking_ape_ecdf.png", dpi=300);  plt.close()

print("Plots saved in plots/")




"""
