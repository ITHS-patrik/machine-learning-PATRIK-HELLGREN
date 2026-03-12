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
def load_recommender(ratings, tags, links, top_n, alpha, n_components, n_clusters=None, diversify=False):
    movies = pd.read_csv("data/movies_enriched.csv")

    recommender = MovieRecommender(
        top_n=top_n,
        alpha=alpha,
        n_components=n_components,
        diversify=bool(diversify),
        n_clusters=(n_clusters if diversify else None))

    recommender.load_data(movies, ratings, tags, links)
    recommender.build_movie_profile()
    recommender.build_tfidf_matrix()
    recommender.build_lsa_matrix()
    recommender.build_user_item_matrix()
    return recommender

def build_recommender():
    diversify_choice = st.session_state["diversify"]
    diversify_bool = True if diversify_choice == "Yes" else False

    ratings = pd.read_csv(st.session_state["ratings"])
    tags = pd.read_csv(st.session_state["tags"])
    links = pd.read_csv(st.session_state["links"])
    top_n = st.session_state["top_n"]
    alpha = st.session_state["alpha"]
    n_components = st.session_state["n_components"]
    n_clusters = st.session_state.get("n_clusters", None)

    st.cache_resource.clear()
    recommender = load_recommender(ratings=ratings, 
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

st.set_page_config(layout="wide")
st.markdown("""<style>.block-container {padding-top: 2rem;padding-botton: 0rem;}</style>""", unsafe_allow_html=True)

header = st.container()
with header:
    st.title("🎬 Movie Recommender")
    st.info("Search for a movie and recieve recommendations.")

col1, col2, col3 = st.columns([0.65,0.75,1.2])

with col1:
    st.subheader("Step 1: Upload csv-files", divider="blue")
    movies = st.file_uploader("Already uploaded: :red[**movies_enriched.csv**]:", type="csv", disabled=True)
    ratings = st.file_uploader("Upload :blue[**ratings.csv**]:", type="csv", key="ratings")
    tags = st.file_uploader("Upload :blue[**tags.csv**]:", type="csv", key="tags")
    links = st.file_uploader("Upload :blue[**links.csv**]:", type="csv", key="links")
    
if st.session_state.input_disabled and ratings and tags and links:
    st.session_state.button_disabled = False

with col2:
    default_hp = {
        "diversify": "Yes",
        "n_clusters": 4,
        "top_n": 5,
        "n_components": 115,
        "alpha": 0.4}

    st.subheader("Step 2: Set hyperparameters", divider="blue")
    st.warning("Remember to re-train the model if you change the hyperparameters")
    diversify = st.radio(":rainbow[Diversify] recommendations?", ["Yes", "No"], key="diversify")
    if st.session_state.diversify == "Yes":
        n_clusters = st.slider("🫧 How many different clusters do you want to get recommendations from?", 1, 10, 4, key="n_clusters", help=":green[KMeans] clusters the data and the model chooses one movie from each cluster.")

    top_n = st.slider("🍿 How many recommendations do you want?", 1, 10, 5, key="top_n")
    n_components = st.slider("⚙️ Set the number of components/features for the LSA matrix.", 100, 1000, 115, step=5, key="n_components", help=":green[TruncatedSVD] reduces the dimensions from the TF-IDF matrix to this value.")
    if n_components >= 300:
        st.warning("Components > 300: grab a coffee, this might take a while.", icon="☕")
    else:
        pass
    alpha = st.slider("⚖️ Set the balance (alpha) between content & collaborative filtering.", 0.0, 1.0, 0.4, step=0.01, key="alpha", help=":green[alpha=0.0]: 100% collaborative filtering, :green[alpha=1.0]: 100% content filtering")
    sub_col1, sub_col2 = st.columns(2, width=790)
    with sub_col1:
        if st.button("Train model", type="secondary", icon="🚂", disabled=st.session_state.button_disabled):
            with st.spinner("Training the model...", show_time=True):
                build_recommender()
            st.success("Training successful! You can now move on to step 3.", icon="🥳")
    with sub_col2:
        def reset_defaults():
            for key, value in default_hp.items():
                st.session_state[key] = value
        if st.button("Reset values", type="secondary", icon="🔄️", key="reset_defaults", on_click=reset_defaults):
            pass

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
            
# on_select (st.dataframe)? -> url YouTube till trailer med hjälp av titel + official trailer?
# poster? URL IMDB?