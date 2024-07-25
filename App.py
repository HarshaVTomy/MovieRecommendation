import streamlit as st
import pandas as pd
import pickle
import numpy as np

# Load the pre-trained model and encoder
model_file_path = 'best_model.pkl'
encoder_file_path = 'encoder.pkl'
with open(model_file_path, 'rb') as model_file:
    best_algo = pickle.load(model_file)
with open(encoder_file_path, 'rb') as encoder_file:
    ohe = pickle.load(encoder_file)

# Load the dataset for recommendation lookup
file_path = 'imdb_top_movies.csv'
movies_df = pd.read_csv(file_path)

# Streamlit app
st.title("Movie Recommendation System")

st.sidebar.header("User Input")

# Input fields
imdb_rating = st.sidebar.slider("IMDB Rating", min_value=0.0, max_value=10.0, step=0.1)
genre = st.sidebar.text_input("Genre", value='Crime, Drama')
director = st.sidebar.text_input("Director", value='Francis Ford Coppola')
cast = st.sidebar.text_input("Cast", value='Al Pacino')

def recommend_movies(imdb_rating, genre, director, cast, top_n=10):
    input_data = pd.DataFrame({
        'IMDB Rating': [imdb_rating],
        'Genre': [genre],
        'Director': [director],
        'Cast': [cast]
    })
    input_encoded = ohe.transform(input_data[['Genre', 'Director', 'Cast']]).toarray()
    input_combined = pd.concat([pd.DataFrame(input_data['IMDB Rating']), pd.DataFrame(input_encoded)], axis=1)
    input_combined.columns = input_combined.columns.astype(str)  # Ensure all column names are strings

    # Predict movie names
    movie_predictions = best_algo.predict(input_combined)
    movie_scores = best_algo.predict_proba(input_combined)[0]

    # Create a DataFrame with movies and their predicted scores
    movie_score_df = pd.DataFrame({
        'Movie Name': best_algo.classes_,
        'Score': movie_scores
    })

    # Sort movies by score and get the top N
    recommended_movies = movie_score_df.sort_values(by='Score', ascending=False).head(top_n)

    # Retrieve movie details for the recommended movies
    recommendations = []
    for movie in recommended_movies['Movie Name']:
        movie_details = movies_df[movies_df['Movie Name'] == movie].iloc[0]
        recommendations.append({
            'Movie Name': movie,
            'IMDB Rating': movie_details['IMDB Rating'],
            'Genre': movie_details['Genre'],
            'Cast': movie_details['Cast'],
            'Overview': movie_details['Overview']
        })
    
    return recommendations

if st.sidebar.button("Get Recommendations"):
    results = recommend_movies(imdb_rating, genre, director, cast)
    st.subheader("Recommended Movies")
    for result in results:
        st.write(f"**Movie Name:** {result['Movie Name']}")
        st.write(f"**IMDB Rating:** {result['IMDB Rating']}")
        st.write(f"**Genre:** {result['Genre']}")
        st.write(f"**Cast:** {result['Cast']}")
        st.write(f"**Overview:** {result['Overview']}")
        st.write("---")
