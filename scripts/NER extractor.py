"""
NER Extractor for Text Corpus
-----------------------------
This script reads a text file, cleans the text, extracts named entities (NER) 
of selected types using spaCy, and saves them into a CSV file.
"""

import re
import spacy
import pandas as pd

# --- Function to clean text ---
def clean_text(text):
    text = text.lower()
    text = re.sub(r'http\S+|www\.\S+', '', text)
    text = re.sub(r'\b(cookies?|accept|privacy policy|terms and conditions|sign up|login|my account|open web ap)\b',
                  '', text, flags=re.IGNORECASE)
    text = re.sub(r'\b(doc\s?id\S*|filename\S*|file\s?id\S*|free\s?login|sign\s?up\s?for\s?free)\b',
                  '', text, flags=re.IGNORECASE)
    text = re.sub(r'\b\d+(\.\d+)?\b', '', text)
    time_patterns = r"\b(one|two|three|four|five|six|seven|eight|nine|ten|a few|several|many)\s+" \
                    r"(seconds?|minutes?|hours?|days?|weeks?|months?|years?)\b"
    text = re.sub(time_patterns, '', text, flags=re.IGNORECASE)
    text = re.sub(r'\b\w{1,2}\b', '', text)
    text = re.sub(r'[^\w\s,.?!]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

# --- Load text file ---
file_path = "hallow_strona.txt"  # <-- replace with your file path
with open(file_path, "r", encoding="utf-8") as file:
    corpus = file.read()

cleaned_corpus = clean_text(corpus)

# --- Load spaCy model ---
nlp = spacy.load("en_core_web_sm")
doc = nlp(cleaned_corpus)

# --- Select entity types of interest ---
allowed_types = {"PERSON", "ORG", "GPE", "EVENT"}

# --- Extract named entities ---
entities = [(ent.text, ent.label_) for ent in doc.ents if ent.label_ in allowed_types]

# --- Save to CSV ---
df_entities = pd.DataFrame(entities, columns=["Entity", "Type"])
df_entities.to_csv("entities.csv", index=False)

print("CSV file 'entities.csv' created with the selected named entities.")
print("Sample of extracted entities:")
print(df_entities.head(10))
