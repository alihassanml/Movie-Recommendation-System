import streamlit as st
import pandas as pd
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
    sim_scores = sim_scores[1:4]  # Get top 10 recommendations
    movie_indices = [i[0] for i in sim_scores]
    return all_data[['title', 'overview', 'genre_names']].iloc[movie_indices]

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
                st.text_area("Overview", recommendations.iloc[i]['overview'], height=150)
                st.markdown("---")
        else:
            st.error("Movie not found in the dataset.")
    else:
        st.warning("Please enter a movie title.")

