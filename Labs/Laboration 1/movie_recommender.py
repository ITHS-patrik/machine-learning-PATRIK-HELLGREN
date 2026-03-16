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
                 ngram_range=(1,3), 
                 n_components=115, 
                 top_n=5, 
                 alpha=0.2, 
                 diversify=True, 
                 n_clusters=4):
        
        self.movies_df = None
        self.ratings_df = None
        self.tags_df = None
        self.links_df = None

        self.min_df = min_df
        self.ngram_range = ngram_range
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

    def normalize_title(self, title):
        if ", The" in title:
            return "The " + title.replace(", The", "")
        if ", A" in title:
            return "A " + title.replace(", A", "")
        if ", An" in title:
            return "An " + title.replace(", An", "")
        return title

    def load_data(self, movies_df, ratings_df, tags_df, links_df):
        self.movies_df = movies_df.copy().reset_index(drop=True)
        self.ratings_df = ratings_df.copy()
        self.tags_df = tags_df.copy()
        self.links_df = links_df.copy()

        self.movies_df["title"] = (self.movies_df["title"].apply(self.normalize_title))
        return self

    def build_movie_profile(self):
        tags_per_movie = (
            self.tags_df.groupby("movieId")["tag"]
            .apply(lambda tags: " ".join(str(tag).lower() for tag in tags)).to_dict())

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
            stop_words="english", # risk att ta bort semantik (rapport)
            min_df=self.min_df,
            max_df=0.8,
            ngram_range=self.ngram_range,
            lowercase=True, # risk pga olika teckenuppsättningar (rapport)
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

        user_ids = self.ratings_df["userId"].astype("category").cat.codes
        user_categories = self.ratings_df["userId"].astype("category").cat.categories
        self.user_map = dict(zip(user_categories, range(len(user_categories))))

        item_ids = self.ratings_df["movieId"].map(self.item_map).values
        rating_values = self.ratings_df["rating"].values
        self.user_item_matrix = coo_matrix((rating_values, (user_ids, item_ids))).tocsr()
        return self.user_item_matrix
    
    def get_conf_recommendations(self, movie_index, n_candidates=200):
        target_vector = self.lsa_matrix[movie_index]
        similarities = cosine_similarity(target_vector.reshape(1, -1), self.lsa_matrix).flatten()
        similarities[movie_index] = -np.inf

        top_indices = np.argpartition(similarities, -n_candidates)[-n_candidates:]
        top_indices = top_indices[np.argsort(similarities[top_indices])[::-1]]

        movie_ids = self.movies_df.iloc[top_indices]["movieId"].values
        titles = self.movies_df.iloc[top_indices]["title"].values

        return pd.DataFrame({
            "movieId": movie_ids,
            "title": titles,
            "score": similarities[top_indices]})

    def get_colf_recommendations(self, movieId, n_candidates=200):
        movie_index = self.item_map[movieId]

        item_vectors = self.user_item_matrix.T
        similarities = cosine_similarity(item_vectors[movie_index].reshape(1, -1), item_vectors).flatten()
        similarities[movie_index] = -1

        top_indices = np.argpartition(similarities, -n_candidates)[-n_candidates:]
        top_indices = top_indices[np.argsort(similarities[top_indices])[::-1]]

        movie_ids = self.movies_df.iloc[top_indices]["movieId"].values
        titles = self.movies_df.iloc[top_indices]["title"].values

        return pd.DataFrame({
            "movieId": movie_ids,
            "title": titles,
            "score": similarities[top_indices]})

    def hybrid_recommendations(self, movieId):
        movie_index = self.item_map[movieId]

        conf_df = self.get_conf_recommendations(movie_index).rename(columns={"score": "conf_score"})
        colf_df = self.get_colf_recommendations(movieId).rename(columns={"score": "colf_score"})

        merged = pd.merge(conf_df, colf_df, on=["movieId", "title"], how="outer")
        merged["conf_score"] = merged["conf_score"].fillna(0)
        merged["colf_score"] = merged["colf_score"].fillna(0)

        scaler = MinMaxScaler()
        merged["conf_norm"] = scaler.fit_transform(merged["conf_score"].values.reshape(-1, 1)).flatten()
        merged["colf_norm"] = scaler.fit_transform(merged["colf_score"].values.reshape(-1, 1)).flatten()

        merged["hybrid_score"] = (self.alpha * merged["conf_norm"] +(1 - self.alpha) * merged["colf_norm"])
        merged = merged[merged["conf_score"] > 0].sort_values("hybrid_score", ascending=False).reset_index(drop=True)
        
        if self.diversify == True:
            return self.diversify_recommendations(merged)

        merged.index += 1
        return merged[["movieId", "title", "conf_score", "colf_score", "hybrid_score"]].head(self.top_n)

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

    def diversify_recommendations(self, merged_df):
        lsa_positions = []
        for id in merged_df["movieId"].tolist():
            matches = self.movies_df.index[self.movies_df["movieId"] == id].tolist()
            lsa_positions.append(int(matches[0]))

        vectors = self.lsa_matrix[lsa_positions]
        n_candidates = vectors.shape[0]
        n_clusters_eff = min(self.n_clusters, max(1, n_candidates))

        kmeans = KMeans(n_clusters=n_clusters_eff)
        labels = kmeans.fit_predict(vectors)
        merged_df["cluster_label"] = labels

        diversified_rows = []
        grouped = merged_df.groupby("cluster_label")
        for _, group in grouped:
            best = group.sort_values("hybrid_score", ascending=False).iloc[0]
            diversified_rows.append(best)

        selected_ids = set([int(row["movieId"]) for row in diversified_rows])
        remaining = merged_df[~merged_df["movieId"].isin(selected_ids)].sort_values("hybrid_score", ascending=False)

        for _, row in remaining.iterrows():
            if len(diversified_rows) >= self.top_n:
                break
            diversified_rows.append(row)

        result = pd.DataFrame(diversified_rows).sort_values("hybrid_score", ascending=False).reset_index(drop=True)
        result.index += 1
        return result[["movieId", "title", "cluster_label", "conf_score", "colf_score", "hybrid_score"]].head(self.top_n)
