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
                 alpha=0.8, 
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
        self.lsa_matrix = None
        self.user_item_matrix = None
        self.item_map = None

    def load_data(self, movies_df, ratings_df, tags_df, links_df):
        """ Load and store the MovieLens dataframes into the recommender. Also normalizes movie titles for improved search behavior. """

        self.movies_df = movies_df.copy().reset_index(drop=True)
        self.ratings_df = ratings_df.copy()
        self.tags_df = tags_df.copy()
        self.links_df = links_df.copy()
        return self
    
    def preprocess_data(self):
        """ Apply title normalization and clean placeholder metadata fields. """

        self.movies_df["title"] = (self.movies_df["title"].apply(self.normalize_title))

        placeholders = ["Unknown", "No overview available."]
        cols_to_clean = ["director", "overview"]
        for column in cols_to_clean:
            for placeholder in placeholders:
                self.movies_df[column] = self.movies_df[column].str.replace(placeholder, "", regex=False)
            self.movies_df[column] = self.movies_df[column].str.strip(" ,")

        self.movies_df["genres_full"] = (
            self.movies_df["genres_full"].fillna("").astype(str)
            .str.replace("(no genres listed)", "", regex=False)
            .str.replace(r"\|+", "|", regex=True)
            .str.strip("|")
            .str.strip(" ,"))

        return self

    def normalize_title(self, title):
        """ Normalize movie titles by moving "The", "A" and "An" to the front of the title for consistent matching. """

        if ", The" in title:
            return "The " + title.replace(", The", "")
        if ", A" in title:
            return "A " + title.replace(", A", "")
        if ", An" in title:
            return "An " + title.replace(", An", "")
        return title

    def build_movie_profile(self):
        """ Construct a text-based profile for each movie by combining genres, tags, overview, keywords, cast, and director into a single string. """

        tags_per_movie = (
            self.tags_df.groupby("movieId")["tag"]
            .apply(lambda tags: " ".join(str(tag) for tag in tags)).to_dict())

        profiles = []
        for _, row in self.movies_df.iterrows():
            movieId = int(row["movieId"])

            tags_text = tags_per_movie.get(movieId, "")
            overview_text = str(row.get("overview") or "")
            genres_text = str(row.get("genres_full") or "")
            keywords_text = str(row.get("keywords") or "")
            cast_text = str(row.get("cast") or "")
            director_text = str(row.get("director") or "")

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
        """ Create a TF-IDF matrix from the movie profiles, turning each film's text data into a numerical representation. """

        vectorizer = TfidfVectorizer(
            stop_words="english", 
            min_df=self.min_df,
            max_df=0.8,
            ngram_range=self.ngram_range,
            lowercase=True, 
            sublinear_tf=True)

        self.tfidf_matrix = vectorizer.fit_transform(self.movie_profiles_df["movie_profile"])
        return self.tfidf_matrix

    def build_lsa_matrix(self):
        """ Apply Truncated SVD (LSA) to reduce the dimensionality of the TF-IDF matrix, capturing deeper semantic relationships between movies. """

        svd = TruncatedSVD(n_components=self.n_components)
        self.lsa_matrix = svd.fit_transform(self.tfidf_matrix)
        return self.lsa_matrix
    
    def get_conf_recommendations(self, movie_index, n_candidates=200):
        """ Generate content-based recommendations using cosine similarity between LSA movie vectors. """

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

    def build_user_item_matrix(self):
        """ Build a sparse user-item matrix where rows represent users, columns represent movies, and values represent ratings. Convert to CSR for fast similarity computations. """

        self.item_map = {movieId: idx for idx, movieId in enumerate(self.movies_df["movieId"])}
        user_ids = self.ratings_df["userId"].astype("category").cat.codes
        item_ids = self.ratings_df["movieId"].map(self.item_map).values
        rating_values = self.ratings_df["rating"].values
        
        self.user_item_matrix = coo_matrix((rating_values, (user_ids, item_ids))).tocsr()
        return self.user_item_matrix

    def get_colf_recommendations(self, movieId, n_candidates=200):
        """ Generate collaborative filtering-based recommendations by comparing rating patterns between movies using the user-item matrix. """

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
        """ Combine content-based and collaborative filtering scores into a hybrid score. Optionally applies diversification to avoid redundant recommendations. """

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
        if self.alpha == 0.0:
            merged = merged[merged["colf_score"] > 0].sort_values("hybrid_score", ascending=False).reset_index(drop=True)
        elif self.alpha == 1.0:
            merged = merged[merged["conf_score"] > 0].sort_values("hybrid_score", ascending=False).reset_index(drop=True)
        else:
            merged = merged.sort_values("hybrid_score", ascending=False).reset_index(drop=True)
        
        if self.diversify == True:
            return self.diversify_recommendations(merged)

        merged.index += 1
        return merged[["movieId", "title", "conf_score", "colf_score", "hybrid_score"]].head(self.top_n)

    def recommend_by_title(self, title):
        """ Recommend movies based on user-input. Uses exact matching first, then fuzzy matching if needed. """

        matches = self.movies_df[self.movies_df["title"].str.lower() == title.lower()]
        if matches.empty:
            best_match = self.find_best_title_match(title)
            matches = self.movies_df[self.movies_df["title"] == best_match]

        movieId = int(matches.iloc[0]["movieId"])
        return self.hybrid_recommendations(movieId)

    def find_best_title_match(self, user_input):
        """ Find the closest matching movie title using fuzzy string matching. """

        titles = self.movies_df["title"].tolist()
        best_match, _, _ = process.extractOne(user_input, titles)
        return best_match

    def diversify_recommendations(self, merged_df):
        """ Improve diversity by clustering the candidate movies, selecting the top movie from each cluster, and then adding the highest-scoring remaining movies until the list is full. """

        lsa_positions = []
        for id in merged_df["movieId"].tolist():
            matches = self.movies_df.index[self.movies_df["movieId"] == id].tolist()
            lsa_positions.append(int(matches[0]))

        vectors = self.lsa_matrix[lsa_positions]
        n_candidates = vectors.shape[0]
        n_clusters_safe = min(self.n_clusters, n_candidates)

        kmeans = KMeans(n_clusters=n_clusters_safe)
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
