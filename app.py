import streamlit as st
import pandas as pd
import numpy as np

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel

# 1. Data loading and preprocessing (cached)
@st.cache_data
def load_and_preprocess_data(csv_path: str = "sample_steam.csv"):
    df = pd.read_csv(csv_path)

    # Keep only English games (1 == English) if that column exists
    if "english" in df.columns:
        df = df[df["english"] == 1].copy()

    # Drop duplicate appids
    if "appid" in df.columns:
        df = df.drop_duplicates(subset="appid").reset_index(drop=True)

    # Ensure price is numeric if present
    if "price" in df.columns:
        df["price"] = pd.to_numeric(df["price"], errors="coerce").fillna(0.0)
    else:
        df["price"] = 0.0

    # Fill missing text fields (create if missing)
    text_cols = ["genres", "steamspy_tags", "categories", "developer", "publisher"]
    for col in text_cols:
        if col in df.columns:
            df[col] = df[col].fillna("")
        else:
            df[col] = ""

    # Create a combined text feature for content-based similarity
    def combine_features(row):
        return "; ".join([
            str(row["genres"]),
            str(row["steamspy_tags"]),
            str(row["categories"]),
            str(row["developer"]),
            str(row["publisher"])
        ])

    df["content"] = df.apply(combine_features, axis=1).str.lower()

    # Cleaned game name (optional convenience)
    df["name_clean"] = df["name"].astype(str).str.strip().str.lower()

    # Map appid -> index (row number)
    if "appid" in df.columns:
        appid_to_index = pd.Series(df.index, index=df["appid"]).to_dict()
    else:
        raise ValueError("Column 'appid' is required in the dataset for this system.")

    return df, appid_to_index


@st.cache_resource
def build_tfidf_matrix(df: pd.DataFrame):
    tfidf = TfidfVectorizer(stop_words="english")
    tfidf_matrix = tfidf.fit_transform(df["content"])
    return tfidf_matrix


# 2. Similarity and filter helpers
def get_similarity_series(idx, tfidf_matrix, df):
    """
    Returns a pandas Series of cosine similarities between the game at index idx
    and all other games.
    """
    cosine_similarities = linear_kernel(tfidf_matrix[idx], tfidf_matrix).flatten()
    return pd.Series(cosine_similarities, index=df.index)


def apply_filters(base_df, genre=None, max_price=None, min_positive_ratings=0):
    """
    Filter the given DataFrame by optional genre, max_price, and minimum positive ratings.
    Works on ANY DataFrame that has at least 'genres', 'price', 'positive_ratings'.
    """
    filtered = base_df.copy()

    # Filter by price
    if max_price is not None and "price" in filtered.columns:
        filtered = filtered[filtered["price"] <= max_price]

    # Filter by genre (substring match in 'genres' column, case-insensitive)
    if genre and "genres" in filtered.columns:
        genre_lower = str(genre).lower()
        filtered = filtered[
            filtered["genres"].astype(str).str.lower().str.contains(genre_lower, na=False)
        ]

    # Filter by popularity (positive_ratings)
    if min_positive_ratings > 0 and "positive_ratings" in filtered.columns:
        filtered = filtered[filtered["positive_ratings"] >= min_positive_ratings]

    return filtered


def recommend_similar_by_appid(
    appid,
    df,
    tfidf_matrix,
    appid_to_index,
    n_recommendations=10,
    genre=None,
    max_price=None,
    min_positive_ratings=0
):
    """
    Recommend games similar to the one with the given appid.
    You can optionally filter by genre, max_price, and minimum positive_ratings.
    Returns a DataFrame with ALL original columns + 'similarity'.
    """
    if appid not in appid_to_index:
        raise ValueError(f"AppID {appid} not found in the dataset.")

    idx = appid_to_index[appid]

    # Compute similarity to all games
    sim_series = get_similarity_series(idx, tfidf_matrix, df)

    # Create a DataFrame with similarity and join with full df
    recs = pd.DataFrame({"similarity": sim_series})

    # Remove the game itself
    recs = recs.drop(index=idx, errors="ignore")

    # Join with ALL original columns (index-based)
    recs = recs.join(df, how="inner")

    # Apply optional filters
    recs = apply_filters(
        recs,
        genre=genre,
        max_price=max_price,
        min_positive_ratings=min_positive_ratings
    )

    # Sort by similarity and popularity if available
    if "positive_ratings" in recs.columns:
        recs = recs.sort_values(
            by=["similarity", "positive_ratings"],
            ascending=[False, False]
        )
    else:
        recs = recs.sort_values(by="similarity", ascending=False)

    # Limit to top N
    recs = recs.head(n_recommendations).reset_index(drop=True)

    # Put similarity as the first column, then all original df columns
    cols = ["similarity"] + [c for c in df.columns if c != "similarity"]
    recs = recs[[c for c in cols if c in recs.columns]]

    return recs


def filter_games(
    df,
    genre=None,
    max_price=None,
    min_positive_ratings=0,
    n_results=20
):
    """
    Return games that match the given filters, without using similarity.
    Results are sorted by positive_ratings (popularity) if available.
    Returns ALL original dataset columns.
    """
    # Start with full dataset
    filtered = df.copy()

    # Apply filters
    filtered = apply_filters(
        filtered,
        genre=genre,
        max_price=max_price,
        min_positive_ratings=min_positive_ratings
    )

    # Sort by popularity if possible
    if "positive_ratings" in filtered.columns:
        filtered = filtered.sort_values(by="positive_ratings", ascending=False)

    filtered = filtered.head(n_results).reset_index(drop=True)
    return filtered


def recommend_with_filters(
    df,
    tfidf_matrix,
    appid_to_index,
    appid=None,
    genre=None,
    max_price=None,
    min_positive_ratings=0,
    n_recommendations=10
):
    """
    If appid is given: content-based recommendations starting from that game,
    then filters are applied (genre, max_price, min_positive_ratings).

    If appid is None: purely filter-based recommendations (no similarity),
    using genre and/or max_price.

    All outputs contain ALL original dataset columns (plus similarity if appid is used).
    """
    if appid is not None:
        return recommend_similar_by_appid(
            appid=appid,
            df=df,
            tfidf_matrix=tfidf_matrix,
            appid_to_index=appid_to_index,
            n_recommendations=n_recommendations,
            genre=genre,
            max_price=max_price,
            min_positive_ratings=min_positive_ratings
        )
    else:
        return filter_games(
            df=df,
            genre=genre,
            max_price=max_price,
            min_positive_ratings=min_positive_ratings,
            n_results=n_recommendations
        )


# 3. Streamlit UI
def main():
    st.title("🎮 Game Recommendation Platform")
    st.write(
        "A content-based recommendation system that leverages game genres, tags, categories, developers, and publishers. It supports searching by AppID and offers filtering options based on genre, price, and popularity. "
    )

    # Load data and model
    df, appid_to_index = load_and_preprocess_data("sample_steam.csv")
    tfidf_matrix = build_tfidf_matrix(df)

    # ----- Build genre dropdown options -----
    genre_set = set()
    for s in df["genres"].dropna():
        for g in str(s).split(";"):
            g = g.strip()
            if g:
                genre_set.add(g)

    genre_options = ["Any"] + sorted(genre_set)

    # ----- Dataset explorer -----
    st.subheader("📂 Browse Steam dataset")

    with st.expander("Show / filter full dataset", expanded=False):
        search_name = st.text_input("Search by game name (optional)", "")

        browse_genre = st.selectbox(
            "Filter by genre (optional)",
            ["Any"] + sorted(genre_set),
            index=0,
            key="browse_genre"
        )

        df_view = df.copy()

        # Filter by name search
        if search_name.strip():
            df_view = df_view[
                df_view["name"].astype(str).str.contains(search_name.strip(), case=False, na=False)
            ]

        # Filter by genre selection
        if browse_genre != "Any":
            df_view = df_view[
                df_view["genres"].astype(str).str.contains(browse_genre, case=False, na=False)
            ]

        st.write(f"Showing {len(df_view)} matching games.")

        # Show a useful subset of columns (you can change this)
        cols_to_show = ["appid", "name", "genres", "price", "positive_ratings", "negative_ratings"]
        cols_to_show = [c for c in cols_to_show if c in df_view.columns]

        st.dataframe(df_view[cols_to_show])


    # ----- Inputs (no sidebar, all in main area) -----
    st.subheader("ℹ️ Add Filters/Details")

    col1, col2 = st.columns(2)
    with col1:
        appid_input = st.text_input("AppID (optional)", "")
        max_price_input = st.text_input("Maximum price (optional)", "")

    with col2:
        selected_genre = st.selectbox("Genre", genre_options, index=0)
        min_pos_input = st.text_input("Minimum positive ratings (optional)", "")

    n_results = st.number_input("Number of results", min_value=1, max_value=50, value=10)

    if st.button("Get recommendations"):
        # Convert inputs
        appid_val = None
        if appid_input.strip():
            try:
                appid_val = int(appid_input.strip())
            except ValueError:
                st.error("AppID must be an integer.")
                return

        genre_val = None if selected_genre == "Any" else selected_genre

        max_price_val = None
        if max_price_input.strip():
            try:
                max_price_val = float(max_price_input.strip())
            except ValueError:
                st.error("Maximum price must be a number.")
                return

        min_pos_val = 0
        if min_pos_input.strip():
            try:
                min_pos_val = int(min_pos_input.strip())
            except ValueError:
                st.error("Minimum positive ratings must be an integer.")
                return

        # Run recommendations
        try:
            results = recommend_with_filters(
                df=df,
                tfidf_matrix=tfidf_matrix,
                appid_to_index=appid_to_index,
                appid=appid_val,
                genre=genre_val,
                max_price=max_price_val,
                min_positive_ratings=min_pos_val,
                n_recommendations=int(n_results)
            )

            # Show selected seed game details
            if appid_val is not None:
                if appid_val in appid_to_index:
                    st.subheader("Seed game")
                    seed_idx = appid_to_index[appid_val]
                    seed_row = df.loc[seed_idx]
                    st.write(f"**{seed_row['name']}**")
                    st.write(f"Genres: {seed_row['genres']}")
                    st.write(f"Price: {seed_row['price']}")
                    if "positive_ratings" in df.columns:
                        st.write(f"Positive ratings: {seed_row['positive_ratings']}")
                else:
                    st.warning(f"AppID {appid_val} was not found in the dataset.")

            st.subheader("Recommended games")

            if results.empty:
                st.warning("No games found matching your criteria.")
            else:
                st.dataframe(results)

        except Exception as e:
            st.error(f"Error: {e}")


if __name__ == "__main__":
    main()



