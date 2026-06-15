import streamlit as st
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from huggingface_hub import hf_hub_download
import warnings

warnings.filterwarnings('ignore')

# -------------------------------------------------------------------------
# DATA LOADING & SIMILARITY MATRIX (Explicit ratings only for IBCF)
# -------------------------------------------------------------------------
@st.cache_data
def load_and_prepare_data():
    # Load your final filtered dataframe from Hugging Face
    final_filtered_df_path = hf_hub_download(repo_id="IamPradeep/BRS_DATA", filename="final_filtered_df.csv", repo_type="dataset")
    final_filtered_df = pd.read_csv(final_filtered_df_path)

    # Load the dataframe containing book URLs from Hugging Face
    book_urls_df_path = hf_hub_download(repo_id="IamPradeep/BRS_DATA", filename="Books.csv", repo_type="dataset")
    book_urls_df = pd.read_csv(book_urls_df_path)
    book_urls_df.rename(columns={'Book-Title': 'title'}, inplace=True)

    # Merge the dataframes on the title
    final_filtered_df = final_filtered_df.merge(book_urls_df, on='title', how='left')

    # Fix some image URLs
    replacements = {
        'Jacob Have I Loved': 'http://images.amazon.com/images/P/0690040784.01.LZZZZZZZ.jpg',
        'Needful Things': 'http://images.amazon.com/images/P/0451172817.01.LZZZZZZZ.jpg',
        'All Creatures Great and Small': 'http://images.amazon.com/images/P/0312084986.01.LZZZZZZZ.jpg',
        "The Kitchen God's Wife": 'http://images.amazon.com/images/P/1590400356.01.LZZZZZZZ.jpg'
    }
    for title, url in replacements.items():
        final_filtered_df.loc[final_filtered_df['title'] == title, 'Image-URL-L'] = url

    # Build similarity matrix using ONLY explicit ratings (>0)
    explicit_ratings_df = final_filtered_df[final_filtered_df['rating'] > 0]
    book_user_mat = explicit_ratings_df.pivot_table(index='title', columns='userId', values='rating').fillna(0)

    cosine_sim = cosine_similarity(book_user_mat)
    cosine_sim_df = pd.DataFrame(cosine_sim, index=book_user_mat.index, columns=book_user_mat.index)

    return final_filtered_df, cosine_sim_df

final_filtered_df, cosine_sim_df = load_and_prepare_data()

# -------------------------------------------------------------------------
# ITEM-TO-ITEM SIMILARITY (for the "similar books" feature)
# -------------------------------------------------------------------------
def get_top_similar_books(book_title, n=10):
    if book_title not in cosine_sim_df.index:
        return "⚠️ Book not found in the database."
    
    similar_scores = cosine_sim_df[book_title]
    similar_books = similar_scores.sort_values(ascending=False)[1:n+1]
    return similar_books

# -------------------------------------------------------------------------
# USER-BASED RECOMMENDATIONS (IBCF logic with explicit/implicit handling)
# -------------------------------------------------------------------------
def get_recommendations_for_user(user_id, df, sim_matrix, k=10):
    """
    Generate top-K recommendations for a user.
    Uses all interactions (explicit & implicit) to seed candidates, but
    excludes already interacted items from final list.
    """
    # 1. All items the user interacted with (both rated and zero‑rated)
    user_history_all = df[df['userId'] == user_id]['title'].tolist()
    if not user_history_all:
        return [], None

    # 2. Get user's explicit ratings for display
    user_history_rated = df[(df['userId'] == user_id) & (df['rating'] > 0)][['title', 'rating']] \
                         .sort_values(by='rating', ascending=False)

    # 3. Generate candidate scores from similar items
    scores = {}
    for item in user_history_all:
        if item in sim_matrix.index:
            # Top 50 similar items (excluding the item itself)
            similar_items = sim_matrix[item].sort_values(ascending=False)[1:51]
            for sim_item, score in similar_items.items():
                if sim_item not in user_history_all:
                    scores[sim_item] = scores.get(sim_item, 0) + score

    # 4. Sort and pick top K
    sorted_scores = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    top_recommendations = [x[0] for x in sorted_scores[:k]]

    return top_recommendations, user_history_rated

# -------------------------------------------------------------------------
# STREAMLIT UI
# -------------------------------------------------------------------------
st.markdown("""
    <h1 style='font-size: 40px; text-align: center; margin-bottom: 5px; padding-bottom: 0px;'>
        Book Recommendation System
    </h1>
    <p class='subheader'>Let Us Help You Choose Your Next Book!</p>
""", unsafe_allow_html=True)

st.image('https://img.freepik.com/premium-vector/bookcase-with-books_182089-197.jpg', use_container_width=True)

# CSS (same as original)
st.markdown("""
    <style>
    html, body, [class*="css"], [class*="st-"], h1, h2, h3, h4, h5, h6, p, div, span, label, input, button, select, option, textarea {
        font-family: 'Tiempos', 'Tiempos Text', Georgia, 'Times New Roman', serif !important;
    }
    .subheader {
        font-size: 22px;
        font-weight: bold;
        margin-bottom: 10px;
        color: #1a73e8;
        text-align: center;
    }
    .stButton > button {
        font-family: 'Tiempos', 'Tiempos Text', Georgia, 'Times New Roman', serif !important;
        font-size: 16px;
        background: linear-gradient(90deg, #ff8a00, #e52e71);
        color: white !important;
        border: none;
        border-radius: 25px;
        padding: 10px 20px;
        font-weight: bold;
        cursor: pointer;
        transition: transform 0.2s ease, box-shadow 0.2s ease;
        display: inline-flex;
        align-items: center;
        justify-content: center;
        margin: 4px 2px;
        width: auto;
        min-width: 100px;
    }
    .stButton > button:hover {
        transform: scale(1.05);
        box-shadow: 0px 5px 15px rgba(0, 0, 0, 0.3);
        color: white !important;
    }
    .stButton > button:active {
        transform: scale(0.98);
    }
    
    .book-info {
        background: #1e1e1e;
        padding: 20px 15px;
        border-radius: 0 0 10px 10px;
        border-top: 3px solid #e52e71;
        text-align: center;
        display: flex;
        flex-direction: column;
        justify-content: flex-start;
        align-items: center;
        min-height: 150px;
        height: 150px;
        box-sizing: border-box;
    }
    
    .premium-title {
        font-size: 16px;
        font-weight: bold;
        color: #F7E7A1;
        margin-bottom: 8px;
        line-height: 1.4;
        width: 100%;
        white-space: nowrap;
        overflow-x: auto;
        overflow-y: hidden;
        display: block;
        padding-bottom: 5px;
        height: 38px;
        box-sizing: border-box;
    }

    .premium-title::-webkit-scrollbar {
        height: 6px;
    }

    .premium-title::-webkit-scrollbar-thumb {
        background: #ccc;
        border-radius: 10px;
    }

    .premium-divider {
        width: 35px;
        height: 3px;
        background: linear-gradient(90deg, #ff8a00, #e52e71);
        margin: 6px 0 12px 0;
        border-radius: 5px;
    }

    .premium-author {
        font-size: 13.5px;
        color: #c4c4c4;
        font-style: italic;
        margin-bottom: 6px;
        width: 100%;
        white-space: nowrap;
        overflow: hidden;
        text-overflow: ellipsis;
    }

    .premium-year {
        font-size: 11.5px;
        color: #888888;
        text-transform: uppercase;
        letter-spacing: 1.2px;
        font-weight: 600;
    }
    
    img {
        object-fit: contain;
        max-height: 300px;
        width: auto;
        display: block;
        margin: 0 auto;
    }
    hr {
        border: none !important;
        border-top: 10px solid #B2BEB5 !important;
        margin-top: 25px !important;
        margin-bottom: 25px !important;
        opacity: 1 !important;
        border-radius: 999px !important;
    }
    .book-column {
        position: relative;
        padding: 0;
        border: 2px solid #2b2b2b;
        border-radius: 12px;
        background-color: rgba(128, 128, 128, 0.05);
        box-shadow: 0 4px 8px rgba(0, 0, 0, 0.1);
        margin-top: 28px;
        margin-bottom: 15px;
        transition: transform 0.2s ease, box-shadow 0.2s ease;
        overflow: visible;
    }
    .book-column:hover {
        transform: translateY(-5px);
        box-shadow: 0 8px 16px rgba(0, 0, 0, 0.2);
    }
    .book-image-area {
        padding: 35px 20px 20px 20px;
    }
    .recommendation-badge {
        position: absolute;
        top: -22px;
        left: 50%;
        transform: translateX(-50%);
        width: 48px;
        height: 48px;
        border-radius: 50%;
        background: #28a745;
        color: white;
        border: 2px solid #2b2b2b;
        display: flex;
        align-items: center;
        justify-content: center;
        font-size: 18px;
        font-weight: bold;
        z-index: 10;
        box-shadow: 0 4px 10px rgba(0, 0, 0, 0.25);
    }
    .extra-space {
        margin-top: 50px;
    }
    .recommendation-header {
        font-size: 15px;
        border-left: 5px solid #B2BEB5;
        padding-left: 12px;
        margin-left: 5px;
    }
    </style>
""", unsafe_allow_html=True)

# -------------------------------------------------------------------------
# CREATE TABS
# -------------------------------------------------------------------------
tab1, tab2 = st.tabs(["📖 Find Similar Books", "👤 Personalised Recommendations"])

# ================== TAB 1: ITEM-ITEM SIMILARITY ==========================
with tab1:
    st.subheader("Discover books similar to a title you love")

    all_books = sorted(final_filtered_df['title'].unique().tolist())
    book_title = st.selectbox(
        'Enter a book title:',
        all_books,
        index=None,
        placeholder="Choose or enter a book title...",
        key='book_title'
    )
    num_recommendations = st.number_input(
        'Number of similar books:',
        min_value=1, max_value=50, value=10,
        key='num_similar'
    )

    if 'similar_books' not in st.session_state:
        st.session_state.similar_books = None
        st.session_state.similar_book_query = None

    if st.button('Find Similar Books', key='similar_btn'):
        if book_title:
            similar = get_top_similar_books(book_title, num_recommendations)
            st.session_state.similar_books = similar
            st.session_state.similar_book_query = book_title
        else:
            st.warning("⚠️ Please select or enter a book title.")
            st.session_state.similar_books = None

    if st.session_state.similar_books is not None:
        similar_books = st.session_state.similar_books
        query_title = st.session_state.similar_book_query

        if isinstance(similar_books, str):
            st.write(similar_books)
        else:
            st.markdown(
                f"<div class='recommendation-header'>Top {len(similar_books)} books similar to '<strong>{query_title}</strong>':</div>",
                unsafe_allow_html=True
            )
            st.write("")

            # Display in rows of 3
            for i in range(0, len(similar_books), 3):
                cols = st.columns(3)
                for j in range(3):
                    if i + j < len(similar_books):
                        book = similar_books.index[i + j]
                        book_info = final_filtered_df[final_filtered_df['title'] == book].iloc[0]
                        safe_title = str(book).replace('"', '&quot;').replace("'", "&#39;")
                        safe_author = str(book_info['Book-Author']).replace('"', '&quot;').replace("'", "&#39;")
                        with cols[j]:
                            st.markdown(f"""
                            <div class='book-column'>
                                <div class='recommendation-badge'>{i + j + 1}</div>
                                <div class='book-image-area'>
                                    <img src='{book_info['Image-URL-L']}' style='height:290px; width:auto; display:block;'>
                                </div>
                                <div class='book-info'>
                                    <div class='premium-title' title="{safe_title}">{book}</div>
                                    <div class='premium-divider'></div>
                                    <div class='premium-author' title="{safe_author}">{book_info['Book-Author']}</div>
                                    <div class='premium-year'>{book_info['Year-Of-Publication']}</div>
                                </div>
                            </div>
                            """, unsafe_allow_html=True)
                if i < len(similar_books) - 3:
                    st.markdown("<br><hr><br>", unsafe_allow_html=True)

# ================== TAB 2: PERSONALISED RECOMMENDATIONS ==================
with tab2:
    st.subheader("Get personalised recommendations based on your reading history")

    user_id_input = st.text_input(
        "Enter your User ID:",
        placeholder="e.g., 277427",
        key='user_id_input'
    )
    top_k = st.number_input(
        "Number of recommendations:",
        min_value=1, max_value=50, value=10,
        key='user_k'
    )

    if st.button('Get Recommendations', key='user_rec_btn'):
        if user_id_input:
            try:
                uid = int(user_id_input)
            except ValueError:
                st.error("❌ User ID must be a numeric value.")
                st.session_state.user_recs = None
                st.session_state.user_history = None
            else:
                recs, history = get_recommendations_for_user(uid, final_filtered_df, cosine_sim_df, k=top_k)
                st.session_state.user_recs = recs
                st.session_state.user_history = history
        else:
            st.warning("⚠️ Please enter a User ID.")
            st.session_state.user_recs = None
            st.session_state.user_history = None

    # Display results if available
    if 'user_recs' in st.session_state and st.session_state.user_recs is not None:
        recs = st.session_state.user_recs
        history = st.session_state.user_history

        # Show user's history
        st.markdown("---")
        st.markdown("#### 📚 Your Reading History (Top Rated)")
        if history is not None and len(history) > 0:
            for _, row in history.head(5).iterrows():
                st.write(f"- **{row['title']}** (Rating: {row['rating']})")
            if len(history) > 5:
                st.caption(f"... and {len(history) - 5} more books.")
        else:
            st.write("No explicit ratings found (only implicit interactions).")

        # Show recommendations
        st.markdown("---")
        if len(recs) == 0:
            st.info("No recommendations could be generated. You may have already explored all similar books, or your history is too limited.")
        else:
            st.markdown(f"<div class='recommendation-header'>✨ Top {len(recs)} Recommendations for You</div>", unsafe_allow_html=True)
            st.write("")

            for i in range(0, len(recs), 3):
                cols = st.columns(3)
                for j in range(3):
                    idx = i + j
                    if idx < len(recs):
                        book = recs[idx]
                        book_info = final_filtered_df[final_filtered_df['title'] == book].iloc[0]
                        safe_title = str(book).replace('"', '&quot;').replace("'", "&#39;")
                        safe_author = str(book_info['Book-Author']).replace('"', '&quot;').replace("'", "&#39;")
                        with cols[j]:
                            st.markdown(f"""
                            <div class='book-column'>
                                <div class='recommendation-badge'>{idx + 1}</div>
                                <div class='book-image-area'>
                                    <img src='{book_info['Image-URL-L']}' style='height:290px; width:auto; display:block;'>
                                </div>
                                <div class='book-info'>
                                    <div class='premium-title' title="{safe_title}">{book}</div>
                                    <div class='premium-divider'></div>
                                    <div class='premium-author' title="{safe_author}">{book_info['Book-Author']}</div>
                                    <div class='premium-year'>{book_info['Year-Of-Publication']}</div>
                                </div>
                            </div>
                            """, unsafe_allow_html=True)
                if i < len(recs) - 3:
                    st.markdown("<br><hr><br>", unsafe_allow_html=True)

    # Reset button to clear session state (optional)
    if st.button("Clear User Recommendations", key='clear_user'):
        st.session_state.user_recs = None
        st.session_state.user_history = None
        st.rerun()

st.markdown("<div class='extra-space'></div><div class='extra-space'></div>", unsafe_allow_html=True)
st.image('https://github.com/MarpakaPradeepSai/Employee-Churn-Prediction/blob/main/Data/Images%20&%20GIFs/thank-you-33.gif?raw=true', use_container_width=True)
