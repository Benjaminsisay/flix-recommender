import pandas as pd
import faiss
import requests
import numpy as np
from flask import Flask, render_template, request

app = Flask(__name__)

df = pd.read_csv('netflix_titles.csv')

def create_textual_representation(row):
    textual_representation = f'''Type: {row['type']},
Title: {row['title']},
Director: {row['director']},
Cast: {row['cast']},
Released: {row['release_year']},
Genres: {row['listed_in']},    
Description: {row['description']}'''
    return textual_representation

df['textual_representation'] = df.apply(create_textual_representation, axis = 1)

dim = 3072
index = faiss.read_index('index')

def get_recommendations(fav_film_title):
    fav_film  = df[df['title'].str.contains(fav_film_title, case=False, na=False)].iloc[0]
    textual_rep = fav_film['textual_representation']
    res = requests.post('http://localhost:11434/api/embeddings',
                        json= {
                            'model': 'llama3.2',
                            'prompt': textual_rep
                        })
    embedding = np.array([res.json()['embedding']], dtype='float32')
    D, I = index.search(embedding, 6)
    best_matches = df.iloc[I.flatten()]
    best_matches = best_matches[best_matches['title'].str.lower() != fav_film_title.lower()]
    if best_matches.empty:
        best_matches = df.sample(5) 
    recommendations = best_matches[['title', 'director', 'cast', 'listed_in', 'release_year', 'description', ]].to_dict(orient='records')
    return recommendations
    
@app.route('/')
def home():
    return render_template('index.html')

@app.route('/recommend', methods=['POST'])
def recommend():
    fav_film_title = request.form['fav_film']
    recommendations = get_recommendations(fav_film_title)
    return render_template('recommendations.html', recommendations=recommendations, fav_film=fav_film_title)

if __name__ == '__main__':
    app.run(debug=True)
# ollama run llama3.2 &
