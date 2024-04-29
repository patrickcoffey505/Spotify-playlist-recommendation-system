# spotify-playlist-recommendation-system

# Spotify Recommender

## Data

### Million Playlist Dataset
- basically an item-user(playlist)-interaction matrix
- also has basic info about songs
- uses songs' Spotify uri

## Files

- playlist-data-extraction.ipynb: builds the playlist-track matrices (*.npz) for collaborative filtering. Also creates track_map.npy to get a mapping from the matrices to the songs.
- collaborative-filtering/: hosts collaborative filtering candidate generation sources
    - cf-knn.ipynb: a knn cf model that surfaces 100 candidates for each user

## Running Locally

- `pip install -r requirements.txt`
- `jupyter nbconvert --execute playlist-data-extraction.ipynb` to download playlist-interaction matrices
- `jupyter nbconvert --execute collaborative-filtering/cf-knn-ipynb` to generate cf candidates