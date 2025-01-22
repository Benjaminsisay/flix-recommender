Code to train the model

import pandas as pd
import faiss
import requests
import numpy as np

# Load the data
df = pd.read_csv('netflix_titles.csv')

# Function to create textual representation
def create_textual_representation(row):
    textual_representation = f'''Type: {row['type']},
Title: {row['title']},
Director: {row['director']},
Cast: {row['cast']},
Released: {row['release_year']},
Genres: {row['listed_in']},
    
Description: {row['description']}'''
    return textual_representation

df['textual_representation'] = df.apply(create_textual_representation, axis=1)

# Dimension of the embedding
dim = 3072

# Initialize FAISS index
index = faiss.IndexFlatL2(dim)
X = np.zeros((len(df['textual_representation']), dim), dtype='float32')

# Process embeddings and add to index
for i, representation in enumerate(df['textual_representation']):
    if i % 30 == 0:
        print('Processed', str(i), 'instances')
    res = requests.post('http://localhost:11434/api/embeddings',
                        json={'model': 'llama3.2', 'prompt': representation})
    embedding = res.json().get('embedding', None)
    if embedding is not None:
        X[i] = np.array(embedding)
index.add(X)

# Save the index
faiss.write_index(index, 'faiss_index.index')

# Later, load the index
index = faiss.read_index('faiss_index.index')

# Example usage
fav_movie = df.iloc[28]
res = requests.post('http://localhost:11434/api/embeddings',
                    json={'model': 'llama3.2', 'prompt': fav_movie['textual_representation']})
embedding = np.array([res.json()['embedding']], dtype='float32')

# Search the index
D, I = index.search(embedding, 5)

# Find best matches
best_matches = np.array(df['textual_representation'])[I.flatten()]
for match in best_matches:
    print('NEXT MOVIE')
    print(match)
    print()