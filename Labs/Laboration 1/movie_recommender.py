import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import MinMaxScaler
from sklearn.cluster import KMeans
from scipy.sparse import coo_matrix
from rapidfuzz import process

class MovieRecommender:
    def __init__(self, 
                 min_df=13, 
                 ngram_range=(1,2), 
                 max_features=115_000, 
                 n_components=115, 
                 top_n=10, 
                 alpha=0.04, 
                 diversify=True, 
                 n_clusters=2):
        
        self.movies_df = None
        self.ratings_df = None
        self.tags_df = None
        self.links_df = None

        self.min_df = min_df
        self.ngram_range = ngram_range
        self.max_features = max_features
        self.n_components = n_components
        self.top_n = top_n
        self.alpha = alpha
        self.diversify = diversify
        self.n_clusters = n_clusters        

        self.movie_profiles_df = None
        self.tfidf_matrix = None
        self.vectorizer = None
        self.lsa_matrix = None
        self.svd = None

        self.user_item_matrix = None
        self.user_map = None
        self.item_map = None

    def load_data(self, movies_df, ratings_df, tags_df, links_df):
        self.movies_df = movies_df.copy()
        self.ratings_df = ratings_df.copy()
        self.tags_df = tags_df.copy()
        self.links_df = links_df.copy()
        return self

    def build_movie_profile(self):
        tags_per_movie = (
            self.tags_df.groupby("movieId")["tag"]
            .apply(lambda tags: " ".join(str(tag).lower() for tag in tags))
            .to_dict())

        profiles = []

        for _, row in self.movies_df.iterrows():
            movieId = int(row["movieId"])

            tags_text = tags_per_movie.get(movieId, "")
            overview_text = str(row.get("overview") or "").lower()
            genres_text = str(row.get("genres_full") or "").replace("|", " ").lower()
            keywords_text = str(row.get("keywords") or "").replace("|", " ").lower()
            cast_text = str(row.get("cast") or "").replace("|", " ").lower()
            director_text = str(row.get("director") or "").lower()

            movie_profile = " ".join([
                genres_text,
                tags_text,
                overview_text,
                keywords_text,
                cast_text,
                director_text])

            profiles.append({
                "movieId": movieId,
                "movie_profile": movie_profile})

        self.movie_profiles_df = pd.DataFrame(profiles)
        return self.movie_profiles_df

    def build_tfidf_matrix(self):
        vectorizer = TfidfVectorizer(
            stop_words="english",
            min_df=self.min_df,
            max_df=0.8,
            ngram_range=self.ngram_range,
            max_features=self.max_features,
            lowercase=True,
            sublinear_tf=True)

        self.tfidf_matrix = vectorizer.fit_transform(self.movie_profiles_df["movie_profile"])
        self.vectorizer = vectorizer
        return self.tfidf_matrix

    def build_lsa_matrix(self):
        svd = TruncatedSVD(n_components=self.n_components)
        self.lsa_matrix = svd.fit_transform(self.tfidf_matrix)
        self.svd = svd
        return self.lsa_matrix

    def build_user_item_matrix(self):
        self.item_map = {movieId: idx for idx, movieId in enumerate(self.movies_df["movieId"])}
        self.ratings_df = self.ratings_df[self.ratings_df["movieId"].isin(self.item_map.keys())].copy()

        user_codes = self.ratings_df["userId"].astype("category").cat.codes
        user_categories = self.ratings_df["userId"].astype("category").cat.categories
        self.user_map = dict(zip(user_categories, range(len(user_categories))))

        item_codes = self.ratings_df["movieId"].map(self.item_map).values
        rating_values = self.ratings_df["rating"].values

        self.user_item_matrix = coo_matrix((rating_values, (user_codes, item_codes))).tocsr()
        return self.user_item_matrix
    
    def get_conf_recommendations(self, movie_index, top_n=50):
        target_vector = self.lsa_matrix[movie_index]
        similarities = cosine_similarity(target_vector.reshape(1, -1), self.lsa_matrix).flatten()
        similarities[movie_index] = -np.inf

        top_indices = np.argpartition(similarities, -top_n)[-top_n:]
        top_indices = top_indices[np.argsort(similarities[top_indices])[::-1]]

        movie_ids = self.movies_df.iloc[top_indices]["movieId"].values
        titles = self.movies_df.iloc[top_indices]["title"].values

        return pd.DataFrame({
            "movieId": movie_ids,
            "title": titles,
            "score": similarities[top_indices]})

    def get_colf_recommendations(self, movieId, top_n=50):
        movie_index = self.item_map[movieId]

        item_vectors = self.user_item_matrix.T
        similarities = cosine_similarity(item_vectors[movie_index].reshape(1, -1), item_vectors).flatten()
        similarities[movie_index] = -1

        top_indices = np.argpartition(similarities, -top_n)[-top_n:]
        top_indices = top_indices[np.argsort(similarities[top_indices])[::-1]]

        movie_ids = self.movies_df.iloc[top_indices]["movieId"].values
        titles = self.movies_df.iloc[top_indices]["title"].values

        return pd.DataFrame({
            "movieId": movie_ids,
            "title": titles,
            "score": similarities[top_indices]})

    def hybrid_recommendations(self, movieId):
        movie_index = self.item_map[movieId]

        conf_df = self.get_conf_recommendations(movie_index, top_n=50)
        conf_df = conf_df.rename(columns={"score": "conf_score"})

        colf_df = self.get_colf_recommendations(movieId, top_n=50)
        colf_df = colf_df.rename(columns={"score": "colf_score"})

        merged = pd.merge(conf_df, colf_df, on=["movieId", "title"], how="outer")
        merged["conf_score"] = merged["conf_score"].fillna(0)
        merged["colf_score"] = merged["colf_score"].fillna(0)

        scaler = MinMaxScaler()
        merged["conf_norm"] = scaler.fit_transform(merged["conf_score"].values.reshape(-1, 1)).flatten()
        merged["colf_norm"] = scaler.fit_transform(merged["colf_score"].values.reshape(-1, 1)).flatten()

        merged["hybrid_score"] = self.alpha * merged["conf_norm"] + (1 - self.alpha) * merged["colf_norm"]

        if self.diversify:
            return self.diversify_recommendations(merged)

        merged = merged.sort_values("hybrid_score", ascending=False).head(self.top_n)
        return merged[["movieId", "title", "hybrid_score"]]

    def recommend_by_title(self, title):
        matches = self.movies_df[self.movies_df["title"].str.lower() == title.lower()]

        if matches.empty:
            best_match = self.find_best_title_match(title)
            matches = self.movies_df[self.movies_df["title"] == best_match]

        movieId = int(matches.iloc[0]["movieId"])
        return self.hybrid_recommendations(movieId)

    def find_best_title_match(self, user_input):
        titles = self.movies_df["title"].tolist()
        best_match, _, _ = process.extractOne(user_input, titles)
        return best_match

    def search_titles(self, query):
        query = query.lower()
        mask = self.movies_df["title"].str.lower().str.contains(query)
        return self.movies_df[mask][["movieId", "title"]]

    def diversify_recommendations(self, merged_df):
        vectors = self.lsa_matrix[
            [self.movies_df.index[self.movies_df["movieId"] == mid][0]
             for mid in merged_df["movieId"]]]

        kmeans = KMeans(n_clusters=self.n_clusters, random_state=42)
        labels = kmeans.fit_predict(vectors)

        diversified_rows = []

        for cluster in range(self.n_clusters):
            cluster_movies = merged_df[labels == cluster]
            if cluster_movies.empty:
                continue

            best_row = cluster_movies.sort_values("hybrid_score", ascending=False).iloc[0]
            diversified_rows.append(best_row)

        return pd.DataFrame(diversified_rows).head(self.top_n)
