import streamlit as st
import random
import pandas as pd
import numpy as np
import pickle

# Load the pickled files
with open('average_similarity.pkl', 'rb') as f:
    average_similarity = pickle.load(f)

with open('tfidf_vectorizer.pkl', 'rb') as f:
    vector = pickle.load(f)

combined_data = pd.read_pickle('combined_data.pkl')

with open('indices.pkl', 'rb') as f:
    indices = pickle.load(f)

# Function to recommend movies
def recommend_from_combined_similarity(title, all_data, indices, average_similarity):
    if title not in indices:
        return None

    idx = indices[title]
    sim_scores = list(enumerate(average_similarity[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    sim_scores = sim_scores[1:4]  # Get top 3 recommendations
    movie_indices = [i[0] for i in sim_scores]
    return all_data[['title', 'overview', 'genre_names', 'poster_path']].iloc[movie_indices]

# Streamlit UI
st.title("Movie Recommendation System")
st.write("Enter a movie title to get recommendations.")

movie_title = st.text_input("Movie Title")

if st.button("Get Recommendations"):
    if movie_title:
        recommendations = recommend_from_combined_similarity(movie_title, combined_data, indices, average_similarity)
        if recommendations is not None:
            for i in range(len(recommendations)):
                st.subheader(recommendations.iloc[i]['title'])
                st.text(f"Genres: {', '.join(recommendations.iloc[i]['genre_names'])}")
                
                # Create two columns for layout
                col1, col2 = st.columns(2)
                
                # Left column: Overview
                with col1:
                    st.text_area("Overview", recommendations.iloc[i]['overview'], height=200)
                
                # Right column: Poster Image
                with col2:
                    a = random.randint(0,6)
                    a = np.round(a)
                    if a==1:
                        st.image('https://i.ytimg.com/vi/DzfpyUB60YY/maxresdefault.jpg',use_column_width=True)
                    elif a==2:
                        st.image('https://st2.depositphotos.com/1105977/9877/i/450/depositphotos_98775856-stock-photo-retro-film-production-accessories-still.jpg',use_column_width=True)
                    elif a==3:
                        st.image('https://images.saymedia-content.com/.image/ar_1:1%2Cc_fill%2Ccs_srgb%2Cfl_progressive%2Cq_auto:eco%2Cw_1200/MTc0NDI1MDExOTk2NTk5OTQy/top-10-greatest-johnny-depp-movies-of-all-time.jpg',use_column_width=True)
                    elif a==4:
                        st.image('https://img.freepik.com/free-vector/cinema-realistic-poster-with-illuminated-bucket-popcorn-drink-3d-glasses-reel-tickets-blue-background-with-tapes-vector-illustration_1284-77070.jpg',use_column_width=True)
                    elif a==5:
                        st.image('https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcTSjj7qf8c8Dj6QYlDOXdN1-0xlJCjzEiWC7w&s',use_column_width=True)
                    elif a==6:
                        st.image('https://cdn.pixabay.com/photo/2016/09/16/00/16/movie-1673021_640.jpg',use_column_width=True)
                    else:
                        st.image('https://media.istockphoto.com/id/1355176914/photo/movie-theater-during-the-screening-of-an-animated-movie.jpg?s=612x612&w=0&k=20&c=IMnAa8LT6Da6is7QMo3wJWJFdYFEyHGMaB2XAquIWwY= ',use_column_width=True)

                st.markdown("---")
        else:
            st.error("Movie not found in the dataset.")
    else:
        st.warning("Please enter a movie title.")
