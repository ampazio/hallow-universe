import os
import re
import nltk
from collections import Counter
from nltk import pos_tag
from nltk.util import ngrams
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords, wordnet
from nltk.stem import WordNetLemmatizer
from gensim import corpora
from gensim.models import LdaModel
import pyLDAvis.gensim_models as gensimvis
import pyLDAvis
from wordcloud import WordCloud
import matplotlib.pyplot as plt

# Download NLTK resources (first run may take some time)
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('omw-1.4')
nltk.download('averaged_perceptron_tagger')

# === PATH CONFIGURATION ===
# You need to specify where your .txt files are located.
# Option 1: Set environment variable CORPUS_PATH, e.g.:
#   export CORPUS_PATH="/path/to/your/folder"
# Option 2: Put your .txt files in a local folder called "data"
#   (default if CORPUS_PATH is not set).
base_path = os.getenv("CORPUS_PATH", "./data")

# Make sure the path exists
if not os.path.exists(base_path):
    raise FileNotFoundError(
        f"Corpus path '{base_path}' does not exist. "
        "Please set CORPUS_PATH environment variable or create ./data folder with .txt files."
    )

# NLTK setup
stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()
NUM_TOPICS = 5

# Text cleaning and lemmatization
def clean_text(text):
    text = text.lower()
    tokens = word_tokenize(re.sub(r"[^\w\s]", " ", text))
    return [lemmatizer.lemmatize(token) for token in tokens if token not in stop_words and token.isalpha()]

# Load and process files
all_texts = []
flat_tokens = []
file_count = 0
raw_token_count = 0
raw_word_count = 0
pos_all = []

for root, dirs, files in os.walk(base_path):
    for filename in files:
        if filename.endswith(".txt"):
            file_path = os.path.join(root, filename)
            with open(file_path, "r", encoding="utf-8") as f:
                raw_text = f.read()
                tokens = word_tokenize(re.sub(r"[^\w\s]", " ", raw_text.lower()))
                raw_token_count += len(tokens)
                raw_word_count += len([t for t in tokens if t.isalpha()])
                cleaned = clean_text(raw_text)
                if cleaned:
                    all_texts.append(cleaned)
                    flat_tokens.extend(cleaned)
                    file_count += 1
                    pos_all.extend(pos_tag(cleaned))

if not all_texts:
    print("No processed documents found in the corpus.")
    exit()

# Corpus statistics
print("\nCorpus statistics:")
print(f"Number of documents: {file_count}")
print(f"Total number of tokens: {raw_token_count}")
print(f"Number of alphabetic words: {raw_word_count}")

# Categories and their word patterns
semantic_groups = {
    # Individualism / fitting / autonomy
    "personal": [r"^personal"],
    "individual": [r"^individual"],
    "customization": [r"^custom", r"^tailor", r"^bespoke", r"^specific", r"^unique"],
    "self_related": [r"^self", r"^own", r"^myself", r"^yourself", r"^oneself"],
    "choice_autonomy": [r"^choose", r"^select", r"^prefer", r"^option", r"^freedom", r"^decision"],
    "personalization": [r"^personaliz", r"^individualiz", r"^customiz"],

    # Community / relational / togetherness
    "community": [r"^community", r"^together", r"^collective", r"^shared", r"^communal", r"^unity", r"^team", r"^group", r"^we$", r"^us$"]
}

# Counting matches
group_counts = {key: 0 for key in semantic_groups}

for token in flat_tokens:
    for key, patterns in semantic_groups.items():
        if any(re.match(p, token) for p in patterns):
            group_counts[key] += 1

total_tokens = len(flat_tokens)

print("\nNumber of occurrences of words related to individualism and community:")
for key, count in group_counts.items():
    percent = (count / total_tokens) * 100 if total_tokens else 0
    print(f"{key}: {count} ({percent:.2f}%)")

# Sum for individualistic vs. community categories
individualism_keys = ["personal", "individual", "customization", "self_related", "choice_autonomy", "personalization"]
community_keys = ["community"]

individualism_total = sum(group_counts[k] for k in individualism_keys)
community_total = sum(group_counts[k] for k in community_keys)

print("\nTrend comparison:")
print(f"Individualistic words: {individualism_total} ({(individualism_total/total_tokens)*100:.2f}%)")
print(f"Community-related words: {community_total} ({(community_total/total_tokens)*100:.2f}%)")


# Create dictionary and corpus
dictionary = corpora.Dictionary(all_texts)
corpus = [dictionary.doc2bow(text) for text in all_texts]

# Save dictionary and corpus
dictionary.save(os.path.join(base_path, "hallow_dict_all.dict"))
corpora.MmCorpus.serialize(os.path.join(base_path, "hallow_corpus_all.mm"), corpus)

# Train LDA model
lda_model = LdaModel(corpus=corpus,
                     id2word=dictionary,
                     num_topics=NUM_TOPICS,
                     random_state=42,
                     passes=10,
                     alpha='auto',
                     per_word_topics=True)

# Display topics
print("\nTopics and keywords:")
for idx, topic in lda_model.print_topics(-1):
    print(f"\nTopic {idx + 1}:")
    print(topic)

# pyLDAvis visualization
vis = gensimvis.prepare(lda_model, corpus, dictionary)
vis_path = os.path.join(base_path, "lda_hallow_topics_all.html")
pyLDAvis.save_html(vis, vis_path)
print(f"\nVisualization saved to {vis_path}")

# Most frequent keywords
keyword_freq = Counter(flat_tokens)
print("\nTop 20 keywords:")
for word, freq in keyword_freq.most_common(20):
    print(f"{word}: {freq}")

# Most frequent 2-grams and 3-grams
bigrams = Counter(ngrams(flat_tokens, 2))
trigrams = Counter(ngrams(flat_tokens, 3))

print("\nTop 10 2-grams:")
for bg, freq in bigrams.most_common(10):
    print(" ".join(bg), ":", freq)

print("\nTop 10 3-grams:")
for tg, freq in trigrams.most_common(10):
    print(" ".join(tg), ":", freq)

# Word clouds – verbs and adjectives with lemmatization
def get_wordnet_pos(treebank_tag):
    if treebank_tag.startswith('J'):
        return wordnet.ADJ
    elif treebank_tag.startswith('V'):
        return wordnet.VERB
    else:
        return None

excluded_words = {'mary', 'god', 'jesus', 'christ', 'lord', 'amen', 'hallelujah', 'father', 'holy'}

def lemmatize_and_filter(pos_list, pos_tag_prefix):
    lemmatized = []
    for word, tag in pos_list:
        wn_tag = get_wordnet_pos(tag)
        if wn_tag and tag.startswith(pos_tag_prefix):
            lemma = lemmatizer.lemmatize(word, wn_tag)
            if (lemma not in stop_words and 
                lemma not in excluded_words and 
                lemma.isalpha() and 
                len(lemma) > 2):
                lemmatized.append(lemma)
    return lemmatized

verbs = lemmatize_and_filter(pos_all, 'VB')
adjectives = lemmatize_and_filter(pos_all, 'JJ')

def show_wordcloud(words, title, filename):
    wc = WordCloud(width=800, height=400, background_color='white').generate(" ".join(words))
    plt.figure(figsize=(10, 5))
    plt.imshow(wc, interpolation='bilinear')
    plt.axis('off')
    plt.title(title, fontsize=16)
    plt.tight_layout()
    plt.savefig(os.path.join(base_path, filename))
    plt.show()

print("\nGenerating word cloud for verbs (lemmatized)...")
show_wordcloud(verbs, "Word Cloud – Verbs (lemmas)", "wordcloud_verbs.png")

print("Generating word cloud for adjectives (lemmatized)...")
show_wordcloud(adjectives, "Word Cloud – Adjectives (lemmas)", "wordcloud_adjectives.png")
