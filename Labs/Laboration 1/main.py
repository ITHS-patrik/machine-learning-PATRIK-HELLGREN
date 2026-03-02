import pandas as pd
import numpy as np
import requests
from dotenv import load_dotenv 
import os

class MovieRecommender:

    def __init__(self):
        load_dotenv()
        self.tmdb_api_key = os.getenv("TMDB_API_KEY")
        self.omdb_api_key = os.getenv("OMDB_API_KEY")
        self.genre_cache = {}

    def fetch_genres(self, row):

        tmdb_id = row.get("tmdbId")
        imdb_id = row.get("imdbId")

        cache_key = f"tmdb:{tmdb_id}" if pd.notna(tmdb_id) else f"imdb:{imdb_id}"
        if cache_key in self.genre_cache:
            return self.genre_cache[cache_key]

        if pd.notna(tmdb_id):
            try:
                tmdb_id_int = int(tmdb_id)
                url = f"https://api.themoviedb.org/3/movie/{tmdb_id_int}?api_key={self.tmdb_api_key}"
                data = requests.get(url).json()

                if "genres" in data and data["genres"]:
                    result = "|".join(g["name"] for g in data["genres"])
                    self.genre_cache[cache_key] = result
                    return result
                
                if "imdb_id" in data and data["imdb_id"]:
                    imdb_id = data["imdb_id"]
            except Exception:
                data = {}

        if pd.notna(imdb_id):
            try:
                if isinstance(imdb_id, str) and imdb_id.startswith("tt"):
                    imdb_id_str = imdb_id
                else:
                    imdb_id_str = "tt" + str(int(imdb_id)).zfill(7)

                url = f"http://www.omdbapi.com/?i={imdb_id_str}&apikey={self.omdb_api_key}"
                data = requests.get(url).json()

                if "Genre" in data and data["Genre"] not in (None, "N/A"):
                    result = data["Genre"].replace(", ", "|")
                    self.genre_cache[cache_key] = result
                    return result
            except Exception:
                data = {}
        
        result = np.nan
        self.genre_cache[cache_key] = result
        return result

    def impute_missing_genres(self, movie_data: pd.DataFrame):
        movie_data = movie_data.copy()

        imax_only_mask = movie_data["genres"] == "IMAX"
        movie_data.loc[imax_only_mask, "genres"] = movie_data.loc[imax_only_mask].apply(self.fetch_genres, axis=1)

        movie_data["genres"] = movie_data["genres"].replace("(no genres listed)", np.nan)

        mask_nan = movie_data["genres"].isna()
        movie_data.loc[mask_nan, "genres"] = movie_data.loc[mask_nan].apply(self.fetch_genres, axis=1)

        return movie_data