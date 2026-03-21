import streamlit as st
import pandas as pd
from rapidfuzz import process, fuzz
from movie_recommender import MovieRecommender
import requests

if "input_disabled" not in st.session_state:
    st.session_state.input_disabled = True
if "button_disabled" not in st.session_state:
    st.session_state.button_disabled = True
if "recommender" not in st.session_state:
    st.session_state.recommender = None
if "model_version" not in st.session_state:
    st.session_state.model_version = 0
if "last_selected_title" not in st.session_state:
    st.session_state.last_selected_title = None
if "last_model_version" not in st.session_state:
    st.session_state.last_model_version = 0
if "last_recs" not in st.session_state:
    st.session_state.last_recs = None

@st.cache_resource
def load_recommender(movies_enriched, ratings, tags, links, top_n, alpha, n_components, n_clusters=None, diversify=False):
    """ Build and initialize a MovieRecommender instance using uploaded CSV files and run the full training pipeline. """

    recommender = MovieRecommender(
        top_n=top_n,
        alpha=alpha,
        n_components=n_components,
        diversify=bool(diversify),
        n_clusters=(n_clusters if diversify else None))

    progress.update(label="Loading data...")
    recommender.load_data(movies_enriched, ratings, tags, links)
    progress.update(label="Preprocessing data...")
    recommender.preprocess_data()
    progress.update(label="Building movie profiles...")
    recommender.build_movie_profile()
    progress.update(label="Building TF-IDF matrix...")
    recommender.build_tfidf_matrix()
    progress.update(label="Building LSA matrix...")
    recommender.build_lsa_matrix()
    progress.update(label="Building user-item matrix...")
    recommender.build_user_item_matrix()
    progress.update(label="Training successful!", state="complete")
    return recommender

def build_recommender():
    """ Read and validate uploaded CSV files and train a new recommender model based on the selected hyperparameters. Then store the trained model in session state. """

    diversify_choice = st.session_state["diversify"]
    diversify_bool = True if diversify_choice == "Yes" else False

    movies_and_media = st.session_state.get("movies_and_media")
    media_df, movies_enriched = (pd.read_csv(file) for file in movies_and_media)
    original_csvs = st.session_state.get("original_csvs")
    links, ratings, tags = (pd.read_csv(file) for file in original_csvs)

    if media_df is not None:
        media_map, missing = load_media(media_df)
        if missing:
            st.warning(f"Missing columns in media.csv: {', '.join(missing)}")
        st.session_state["media_map"] = media_map

    top_n = st.session_state["top_n"]
    alpha = st.session_state["alpha"]
    n_components = st.session_state["n_components"]
    n_clusters = st.session_state.get("n_clusters", None)

    st.cache_resource.clear()
    recommender = load_recommender(movies_enriched=movies_enriched,
                                   ratings=ratings, 
                                   tags=tags, 
                                   links=links, 
                                   top_n=top_n, 
                                   alpha=alpha, 
                                   n_components=n_components,
                                   n_clusters=n_clusters if diversify_bool else None, 
                                   diversify=diversify_bool)
    st.session_state.recommender = recommender
    st.session_state.model_version = st.session_state.get("model_version", 0) + 1
    st.session_state.input_disabled = False

def get_combined_matches(query, all_titles):
    """ Return a combined list of matching movie titles using normalization, prefix matching, substring matching, and fuzzy matching. """

    query_normalized = query.lower().replace("&", "and")
    if not query_normalized.strip():
        return []
    normalized_titles = [(title, title.lower().replace("&", "and")) for title in all_titles]

    prefix_matches = [title for title, title_normalized in normalized_titles if title_normalized.startswith(query_normalized)]
    substring_matches = [title for title, title_normalized in normalized_titles if query_normalized in title_normalized][:9]
    fuzzy_raw = process.extract(query_normalized, [title_normalized for _, title_normalized in normalized_titles], scorer=fuzz.WRatio, limit=10)
    fuzzy_matches = [all_titles[id] for (_, score, id) in fuzzy_raw if score >= 60]

    combined = list(dict.fromkeys(prefix_matches + substring_matches + fuzzy_matches))
    return combined

def load_media(df):
    """ Convert the media.csv file into a dictionary mapping movieId to metadata such as poster and YouTube trailer URL:s. """

    media_map = {int(row["movieId"]): row.to_dict() for _, row in df.iterrows()}
    return media_map, set()

def show_media_for_recommendations(recs, media_map, per_row=3):
    """ Display posters, trailers, and metadata for each recommended movie. """

    recommender = st.session_state.get("recommender")
    movies_df = getattr(recommender, "movies_df", None)
    movie_profile = {}
    if movies_df is not None and "movieId" in movies_df.columns:
        movie_profile = movies_df.set_index("movieId").to_dict(orient="index")

    n = len(recs)
    rows = (n + per_row - 1) // per_row
    idx = 0

    with st.container():
        for _ in range(rows):
            cols = st.columns(per_row, border=True)
            for col in cols:
                if idx >= n:
                    continue

                row = recs.iloc[idx]
                movie_id = int(row["movieId"])
                title = row.get("title", "Unknown title")
                meta = media_map.get(movie_id, {})
                metadata = movie_profile.get(movie_id, {})

                genres = metadata.get("genres_full").replace("|", ", ").title()
                plot = metadata.get("overview")
                director = metadata.get("director")
                cast = metadata.get("cast").replace("|", ", ")

                col.markdown(f"**{title}**")
                with col.expander("Movie info", icon="ℹ️"):
                    movie_info_df = pd.DataFrame([genres, plot, director, cast])
                    movie_info_df.index = [":blue[**GENRES**]", ":blue[**PLOT**]", ":blue[**DIRECTOR**]", ":blue[**CAST**]"]

                    for id, row in movie_info_df.iterrows():
                        left, right = st.columns([0.2, 0.8])
                        left.markdown(id)
                        right.write(row.iloc[0])

                left, right = col.columns([1, 2.57])
                poster_url = meta.get("poster_url")
                if pd.isna(poster_url):
                    poster_url = None
                if poster_url:
                    left.image(poster_url)
                else:
                    left.badge("**No poster available**", icon="❌", color="gray")

                yt_url = meta.get("youtube_url")
                if pd.isna(yt_url):
                    yt_url = None
                if yt_url:
                    right.video(yt_url)
                else:
                    right.badge("**No trailer available**", icon="❌", color="gray")

                idx += 1

def download_button_movies_enriched():
    """ Download the movies_enriched.csv file from a predefined Dropbox link. Contains original movies.csv metadata + plot/overview, keywords, director, cast and some of the missing genres. """

    movies_enriched_csv = requests.get("https://www.dropbox.com/scl/fi/vegn87jr1l9z3rhnck1sb/movies_enriched.csv?rlkey=wrci5hsqv08kkcctgaarqh54r&st=8u317zoi&dl=1")
    return movies_enriched_csv.content

def download_button_media():
    """ Download the media.csv file from a predefined Dropbox link. Contains poster URLs and YouTube trailer links for movies. """

    media_csv = requests.get("https://www.dropbox.com/scl/fi/fj74kjcllpax1s6yyh7ke/media.csv?rlkey=d0hzhmfxszyyk5l7xd462avwe&st=m9y3tccr&dl=1")
    return media_csv.content

st.set_page_config(layout="wide")
st.markdown("""<style>.block-container {padding-top: 2rem;padding-botton: 0rem;padding-left: 1rem; padding-right: 1rem;}</style>""", unsafe_allow_html=True)

header = st.container()
with header:
    st.title("🎬 Movie Recommender")
    st.info("Search for a movie and recieve recommendations including posters & trailers.")

col1, col2, col3 = st.columns([0.7,0.8,1.2])

with col1:
    st.subheader("Step 1: Upload csv-files", divider="blue")
    download_col1, download_col2, download_col3 = st.columns([0.20,0.30,0.50], gap="xxsmall", vertical_alignment="center")
    with download_col1:
        st.caption("Download: ", width="content")
    with download_col2:
        st.download_button("media", width="stretch", data=download_button_media, file_name="media.csv", mime="text/csv", on_click="ignore", type="secondary", icon="💾")
    with download_col3:
        st.download_button("movies_enriched", width="stretch", data=download_button_movies_enriched, file_name="movies_enriched.csv", mime="text/csv", on_click="ignore", type="secondary", icon="💾")

    movies_and_media = st.file_uploader("Upload enriched/extras: :primary[**media.csv**], :primary[**movies_enriched.csv**]:", type="csv", accept_multiple_files=True, key="movies_and_media")
    warning_placeholder1 = st.empty()
    original_csvs = st.file_uploader("Upload original files: :blue[**links.csv**], :blue[**ratings.csv**], :blue[**tags.csv**]:", type="csv", accept_multiple_files=True, key="original_csvs")
    warning_placeholder2 = st.empty()

    media_df = None
    movies_and_media_is_ok, original_is_ok = False, False

    valid_media_names = ["media.csv", "movies_enriched.csv"]
    if movies_and_media:
        movies_and_media = sorted(movies_and_media, key=lambda file: file.name.lower())
        if len(movies_and_media) == 2 and all(file.name in valid_media_names for file in movies_and_media):
            movies_and_media_is_ok = True
        elif len(movies_and_media) == 1 and movies_and_media[0].name not in valid_media_names:
            warning_placeholder1.error(f"Error: names must match 'media.csv' and 'movies_enriched.csv'.")
        elif len(movies_and_media) == 2 and any(file.name not in valid_media_names for file in movies_and_media):
            warning_placeholder1.error(f"Error: names must match 'media.csv' and 'movies_enriched.csv'.")
        elif len(movies_and_media) > 2:
            warning_placeholder1.error(f"Error: you have uploaded too many files ({len(movies_and_media)}).")

    valid_original_names = ["links.csv", "ratings.csv", "tags.csv"]
    if original_csvs:
        original_csvs = sorted(original_csvs, key=lambda file: file.name.lower())
        if len(original_csvs) == 3 and all(f.name in valid_original_names for f in original_csvs):
            original_is_ok = True
        elif len(original_csvs) == 1 and original_csvs[0].name not in valid_original_names:
            warning_placeholder2.error("Error: names must match 'links.csv', 'ratings.csv' and 'tags.csv'.")
        elif len(original_csvs) == 2 and any(f.name not in valid_original_names for f in original_csvs):
            warning_placeholder2.error("Error: names must match 'links.csv', 'ratings.csv' and 'tags.csv'.")
        elif len(original_csvs) == 3 and any(f.name not in valid_original_names for f in original_csvs):
            warning_placeholder2.error("Error: names must match 'links.csv', 'ratings.csv' and 'tags.csv'.")
        elif len(original_csvs) > 3:
            warning_placeholder2.error(f"Error: you have uploaded too many files ({len(original_csvs)}).")

    csv_table = pd.DataFrame({
        "File": ["**media.csv**", "**movies_enriched.csv**", "**links.csv**", "**ratings.csv**", "**tags.csv**"], 
        "Comment": [
            "Contains poster and YouTube URL:s.", 
            "The movies.csv enriched with movie plot, director, cast & keywords + some of the missing genres.",  
            "The original links.csv.", 
            "The original ratings.csv.", 
            "The original tags.csv."]}, 
            index=[1, 2, 3, 4, 5])
    
    if len(movies_and_media+original_csvs) < 2:
        st.table(csv_table)

if st.session_state.input_disabled and movies_and_media_is_ok and original_is_ok:
    st.session_state.button_disabled = False

with col2:
    default_hp = {
        "diversify": "Yes",
        "n_clusters": 4,
        "top_n": 5,
        "n_components": 115,
        "alpha": 0.8}

    st.subheader("Step 2: Set hyperparameters", divider="blue")
    st.warning("Re-train the model if you change the hyperparameters.")
    diversify = st.radio(":rainbow[Diversify] recommendations?", ["Yes", "No"], key="diversify")
    if st.session_state.diversify == "Yes":
        n_clusters = st.slider("🫧 How many clusters do you want to get recommendations from?", 2, 30, default_hp["n_clusters"], key="n_clusters", help="Clusters the data with :green[KMeans] and selects at least one movie from each cluster (up to # of recommendations).")

    top_n = st.slider("🍿 How many recommendations do you want?", 1, 10, default_hp["top_n"], key="top_n")
    n_components = st.slider("⚙️ Set the number of features for the LSA matrix.", 50, 1000, default_hp["n_components"], step=5, key="n_components", help=":green[TruncatedSVD] reduces the dimensions from the TF-IDF matrix to this value.")
    if 200 < n_components < 500:
        st.warning("200+ features: grab a coffee, this might take a while.", icon="🧋")
    elif n_components >= 500:
        st.error("500+ features: your CPU will need a vacation after this.", icon="⛱️")
    else:
        pass
    alpha = st.slider("⚖️ Set the balance between collaborative & content-based filtering.", 0.0, 1.0, default_hp["alpha"], step=0.01, key="alpha", help=":green[alpha = 0.0]: collaborative filtering, :green[alpha = 1.0]: content filtering")
    
    button_container = st.container(horizontal=True, horizontal_alignment="distribute")
    with button_container:
        st.session_state.setdefault("training_requested", False)

        def request_training():
            st.session_state.training_requested = True
        st.button("Train model", type="secondary", icon="🚂", disabled=st.session_state.button_disabled, on_click=request_training)

        if st.session_state.training_requested:
            progress = st.status("Building model...")
            with progress:
                st.session_state.progress = progress
                build_recommender()
    
            st.session_state.training_requested = False
            st.session_state.button_disabled = False

        def reset_defaults():
            for key, value in default_hp.items():
                st.session_state[key] = value
        st.button("Reset values", type="secondary", icon="🔄️", key="reset_defaults", on_click=reset_defaults)

with col3:
    st.subheader("Step 3: Recieve recommendations", divider="blue")
    query = st.text_input("Input the title of a movie you like:", icon="🎞️", disabled=st.session_state.input_disabled)

    if st.session_state.recommender and query:
        all_titles = st.session_state.recommender.movies_df["title"].tolist()
        titles = get_combined_matches(query, all_titles)

        if titles:
            selected_title = st.selectbox("Choose movie:", titles)
        else:
            selected_title = None
            st.warning("No movies matched your search, try again.")
    else:
        selected_title = None

    if selected_title:
        st.write(f"Recommendations based on: **{selected_title}**")

        last_title = st.session_state.get("last_selected_title")
        last_model_ver = st.session_state.get("last_model_version", 0)
        current_model_ver = st.session_state.get("model_version", 0)

        need_recompute = (last_title != selected_title) or (current_model_ver != last_model_ver)

        if need_recompute:
            recommender = st.session_state.get("recommender")
            with st.spinner("Calculating recommendations..."):
                recs = recommender.recommend_by_title(selected_title)
            st.session_state.last_recs = recs
            st.session_state.last_selected_title = selected_title
            st.session_state.last_model_version = current_model_ver
        else:
            recs = st.session_state.get("last_recs")

        if st.session_state.get("diversify") == "Yes":
            st.dataframe(recs, column_config={
                "movieId": "Movie ID", 
                "title": "Title", 
                "cluster_label": "Cluster #", 
                "conf_score": "Content score", 
                "colf_score": "Collab. score", 
                "hybrid_score": "𝗛𝘆𝗯𝗿𝗶𝗱 𝘀𝗰𝗼𝗿𝗲"})
        else:
            st.dataframe(recs, column_config={
                "movieId": "Movie ID", 
                "title": "Title", 
                "conf_score": "Content score", 
                "colf_score": "Collab. score", 
                "hybrid_score": "𝗛𝘆𝗯𝗿𝗶𝗱 𝘀𝗰𝗼𝗿𝗲"})

recs = st.session_state.get("last_recs", pd.DataFrame())
media_map = st.session_state.get("media_map", {})

st.subheader("Step 4: Trailers, posters & metadata for recommended movies", divider="blue")
if recs is not None:
    show_media_for_recommendations(recs, media_map)
