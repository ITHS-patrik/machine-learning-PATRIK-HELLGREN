import streamlit as st
import pandas as pd
from rapidfuzz import process, fuzz
from movie_recommender import MovieRecommender

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
    recommender = MovieRecommender(
        top_n=top_n,
        alpha=alpha,
        n_components=n_components,
        diversify=bool(diversify),
        n_clusters=(n_clusters if diversify else None))

    progress.update(label="Loading data...")
    recommender.load_data(movies_enriched, ratings, tags, links)
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
    diversify_choice = st.session_state["diversify"]
    diversify_bool = True if diversify_choice == "Yes" else False

    movies_and_media = st.session_state.get("movies_and_media")
    original_csvs = st.session_state.get("original_csvs")

    if movies_and_media:
        movies_and_media = sorted(movies_and_media, key=lambda file: file.name.lower())
        if len(movies_and_media) == 2:
            try:
                media_df, movies_enriched = (pd.read_csv(file) for file in movies_and_media)
            except Exception as e:
                st.warning(f"Error reading movies_enriched.csv and/or media.csv: {e}")
                media_df = None

    if original_csvs:
        original_csvs = sorted(original_csvs, key=lambda file: file.name.lower())
        if len(original_csvs) == 3:
            try:
                links, ratings, tags = (pd.read_csv(file) for file in original_csvs) 
            except Exception as e:
                st.warning(f"Error reading links.csv, ratings.csv and/or tags.csv: {e}")

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
    query_lower = query.lower()
    substring_matches = [title for title in all_titles if query_lower in title.lower()][:10]

    fuzzy_raw = process.extract(query, all_titles, scorer=fuzz.WRatio, limit=7)
    fuzzy_matches = [match[0] for match in fuzzy_raw if match[1] >= 60]

    combined = list(dict.fromkeys(substring_matches + fuzzy_matches))
    return combined

@st.cache_data
def load_media(df):
    media_map = {int(row["movieId"]): row.to_dict() for _, row in df.iterrows()}
    return media_map, set()

def show_media_for_recommendations(recs, media_map, per_row=3):
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

st.set_page_config(layout="wide")
st.markdown("""<style>.block-container {padding-top: 2rem;padding-botton: 0rem;padding-left: 1rem; padding-right: 1rem;}</style>""", unsafe_allow_html=True)

header = st.container()
with header:
    st.title("🎬 Movie Recommender")
    st.info("Search for a movie and recieve recommendations.")

col1, col2, col3 = st.columns([0.7,0.8,1.2])

with col1:
    st.subheader("Step 1: Upload csv-files", divider="blue")

    movies_and_media = st.file_uploader("Upload enriched/extras: :primary[**movies_enriched.csv**], :primary[**media.csv**]:", type="csv", accept_multiple_files=True, key="movies_and_media")
    original_csvs = st.file_uploader("Upload original files: :blue[**ratings.csv**], :blue[**tags.csv**], :blue[**links.csv**]:", type="csv", accept_multiple_files=True, key="original_csvs")

    csv_table = pd.DataFrame({
        "File": ["**movies_enriched.csv**", "**ratings.csv**", "**tags.csv**", "**links.csv**", "**media.csv**"], 
        "Comment": [
            "The movies.csv enriched with movie plot, director, cast & keywords + some of the missing genres.", 
            "The original ratings.csv.", 
            "The original tags.csv.", 
            "The original links.csv.", 
            "Contains poster and YouTube URL:s."]}, 
            index=[1, 2, 3, 4, 5])
    
    if len(movies_and_media) < 2:
        st.table(csv_table)

if st.session_state.input_disabled and movies_and_media and original_csvs:
    st.session_state.button_disabled = False

with col2:
    default_hp = {
        "diversify": "Yes",
        "n_clusters": 4,
        "top_n": 5,
        "n_components": 115,
        "alpha": 0.2}

    st.subheader("Step 2: Set hyperparameters", divider="blue")
    st.warning("Re-train the model if you change the hyperparameters.")
    diversify = st.radio(":rainbow[Diversify] recommendations?", ["Yes", "No"], key="diversify")
    if st.session_state.diversify == "Yes":
        n_clusters = st.slider("🫧 How many clusters do you want to get recommendations from?", 1, 10, 4, key="n_clusters", help="Clusters the data with :green[KMeans] and chooses one movie from each cluster.")

    top_n = st.slider("🍿 How many recommendations do you want?", 1, 10, 5, key="top_n")
    n_components = st.slider("⚙️ Set the number of features for the LSA matrix.", 50, 1000, 115, step=5, key="n_components", help=":green[TruncatedSVD] reduces the dimensions from the TF-IDF matrix to this value.")
    if 200 < n_components < 500:
        st.warning("200+ features: grab a coffee, this might take a while.", icon="🧋")
    elif n_components >= 500:
        st.error("500+ features: your CPU will need a vacation after this.", icon="⛱️")
    else:
        pass
    alpha = st.slider("⚖️ Set the balance between content & collaborative filtering.", 0.0, 1.0, 0.4, step=0.01, key="alpha", help=":green[alpha = 0.0]: collaborative filtering, :green[alpha = 1.0]: content filtering")
    
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
                "cluster_label": "Cluster no.", 
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

if recs is not None:
    st.subheader("Step 4: Trailers & posters for recommended movies", divider="blue")
    show_media_for_recommendations(recs, media_map)
