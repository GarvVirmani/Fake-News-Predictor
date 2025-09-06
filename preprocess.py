# preprocess_optimized.py

import pandas as pd
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

nltk.download('stopwords')

# load only necessary columns
df = pd.read_csv('news.csv', usecols=['title', 'text', 'label'])

# drop empty rows
df = df.dropna(subset=['title', 'text'])

# combine title and text
df['content'] = df['title'] + ' ' + df['text']

# initialize stemmer and stopwords set once
port_stem = PorterStemmer()
stop_words = set(stopwords.words('english'))

# optimized preprocessing function
def clean_text(text):
    text = re.sub('[^a-zA-Z]', ' ', text)
    text = text.lower()
    words = text.split()
    # list comprehension with preloaded stop_words
    words = [port_stem.stem(word) for word in words if word not in stop_words]
    return ' '.join(words)

# apply preprocessing
df['content'] = df['content'].apply(clean_text)

# map labels
df['label_num'] = df['label'].map({'REAL': 0, 'FAKE': 1})

# save cleaned CSV
df.to_csv('news_clean.csv', index=False)
print("Preprocessing done! Saved as news_clean.csv")