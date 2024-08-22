# Movie Recommendation System

This project implements a movie recommendation system using content-based filtering. It processes a dataset of movies, computes similarity scores using TF-IDF vectorization and cosine similarity, and provides recommendations based on a given movie title.

## Features

- **Data Processing**: Cleans and preprocesses movie data, including titles, overviews, and genres.
- **TF-IDF Vectorization**: Converts text data into numerical features.
- **Cosine Similarity**: Computes similarity scores between movies.
- **Recommendation System**: Provides movie recommendations based on similarity scores.
- **Streamlit App**: Interactive web app where users can input a movie title and receive recommendations.

## Installation

### Requirements

- Python 3.x
- Pandas
- NumPy
- Scikit-learn
- NLTK
- Streamlit

Install the required packages using:

```bash
pip install -r requirements.txt
```

## Usage

### Running the Streamlit App

To run the Streamlit app:

```bash
streamlit run app.py
```

The app will launch in your default web browser. You can enter a movie title to get recommendations.

### Example

To use the recommendation function directly:

```python
import pickle
import pandas as pd

# Load pickled files
with open('average_similarity.pkl', 'rb') as f:
    average_similarity = pickle.load(f)

with open('indices.pkl', 'rb') as f:
    indices = pickle.load(f)

combined_data = pd.read_pickle('combined_data.pkl')

# Get recommendations
recommendations = recommend_from_combined_similarity('The Matrix', combined_data, indices, average_similarity)
print(recommendations)
```

## Data

The dataset used is `movies_metadata.csv`, which contains information about movies, including titles, overviews, and genres.

## Pickling

The model, vectorizer, and similarity matrices are saved as pickle files for efficient reuse.

## Repository Structure

```plaintext
Movie-Recommendation-System/
│
├── app.py                   # Streamlit app
├── requirements.txt         # Python dependencies
├── average_similarity.pkl   # Pickled similarity matrix
├── indices.pkl              # Pickled indices dictionary
├── combined_data.pkl        # Pickled combined data DataFrame
└── README.md                # Project documentation
```

## Contributing

Contributions are welcome! Please fork the repository and submit a pull request.

## License

This project is licensed under the MIT License.

## Contact

For any questions, please contact [alihassanml](https://github.com/alihassanml).
```

This `README.md` provides an overview of the project, instructions on setting up the environment, and how to use the app or the recommendation system directly.
