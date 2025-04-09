
import pandas as pd
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Load dataset
df = pd.read_csv('dataset.csv')

# Preprocessing
df['overview'] = df['overview'].fillna('')
df['genre'] = df['genre'].fillna('')
df['combined_features'] = df['overview'] + ' ' + df['genre']

# TF-IDF Vectorization
tfidf = TfidfVectorizer(stop_words='english')
tfidf_matrix = tfidf.fit_transform(df['combined_features'])

# Cosine Similarity
similarity = cosine_similarity(tfidf_matrix)

# Save similarity matrix
with open('similarity.pkl', 'wb') as f:
    pickle.dump(similarity, f)

print("similarity.pkl has been saved successfully.")
