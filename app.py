import streamlit as st
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from huggingface_hub import hf_hub_download
import warnings

warnings.filterwarnings('ignore')

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

    # URL to replace
    url1 = 'http://images.amazon.com/images/P/0690040784.01.LZZZZZZZ.jpg'
    url2 = 'http://images.amazon.com/images/P/0451172817.01.LZZZZZZZ.jpg'
    url3 = 'http://images.amazon.com/images/P/0312084986.01.LZZZZZZZ.jpg'
    url4 = 'http://images.amazon.com/images/P/1590400356.01.LZZZZZZZ.jpg'

    # Replace URL based on condition
    final_filtered_df.loc[final_filtered_df['title'] == 'Jacob Have I Loved', 'Image-URL-L'] = url1
    final_filtered_df.loc[final_filtered_df['title'] == 'Needful Things', 'Image-URL-L'] = url2
    final_filtered_df.loc[final_filtered_df['title'] == 'All Creatures Great and Small', 'Image-URL-L'] = url3
    final_filtered_df.loc[final_filtered_df['title'] == "The Kitchen God's Wife", 'Image-URL-L'] = url4

    # -------------------------------------------------------------------------
    #  BUILD SIMILARITY MATRIX USING ONLY EXPLICIT RATINGS (>0)
    # -------------------------------------------------------------------------
    explicit_ratings_df = final_filtered_df[final_filtered_df['rating'] > 0]
    book_user_mat = explicit_ratings_df.pivot_table(index='title', columns='userId', values='rating').fillna(0)

    # Calculate the cosine similarity matrix
    cosine_sim = cosine_similarity(book_user_mat)
    cosine_sim_df = pd.DataFrame(cosine_sim, index=book_user_mat.index, columns=book_user_mat.index)

    # Get all valid user IDs (users who have at least one interaction)
    all_user_ids = sorted(final_filtered_df['userId'].unique().tolist())

    return final_filtered_df, cosine_sim_df, all_user_ids


# -------------------------------------------------------------------------
# USER-BASED RECOMMENDATION FUNCTION
# -------------------------------------------------------------------------
def get_user_recommendations(user_id, df, sim_matrix, k=10):
    """
    Generates book recommendations for a specific user based on
    item-item collaborative filtering.
    """
    # Get all books the user has interacted with (rated or implicit)
    user_history_all = df[df['userId'] == user_id]['title'].tolist()

    # Get explicitly rated books sorted by rating descending
    user_history_rated = (
        df[(df['userId'] == user_id) & (df['rating'] > 0)][['title', 'rating']]
        .drop_duplicates(subset='title')
        .sort_values(by='rating', ascending=False)
    )

    if len(user_history_all) == 0:
        return None, None, "no_history"

    # Build candidate scores using item-item similarity
    scores = {}
    for item in user_history_all:
        if item in sim_matrix.index:
            # Get top-50 similar items (excluding the item itself)
            similar_items = sim_matrix[item].sort_values(ascending=False)[1:50]
            for sim_item, score in similar_items.items():
                if sim_item not in user_history_all:
                    scores[sim_item] = scores.get(sim_item, 0) + score

    if not scores:
        return None, user_history_rated, "no_recommendations"

    # Sort scores and pick top-k
    sorted_scores = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    top_recommendations = [x[0] for x in sorted_scores[:k]]

    return top_recommendations, user_history_rated, "success"


# -------------------------------------------------------------------------
# LOAD DATA
# -------------------------------------------------------------------------
final_filtered_df, cosine_sim_df, all_user_ids = load_and_prepare_data()

# -------------------------------------------------------------------------
# PAGE CONFIG & GLOBAL STYLES
# -------------------------------------------------------------------------
st.markdown("""
    <h1 style='font-size: 40px; text-align: center; margin-bottom: 5px; padding-bottom: 0px;'>
        Book Recommendation System
    </h1>
    <p class='subheader'>Let Us Help You Choose Your Next Book!</p>
""", unsafe_allow_html=True)

st.image(
    'https://img.freepik.com/premium-vector/bookcase-with-books_182089-197.jpg',
    use_container_width=True
)

st.markdown("""
    <style>
    html, body, [class*="css"], [class*="st-"], h1, h2, h3, h4, h5, h6,
    p, div, span, label, input, button, select, option, textarea {
        font-family: 'Tiempos', 'Tiempos Text', Georgia, 'Times New Roman', serif !important;
    }
    .subheader {
        font-size: 22px;
        font-weight: bold;
        margin-bottom: 10px;
        color: #1a73e8;
        text-align: center;
    }
    /* ---- Buttons ---- */
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
    /* ---- Book Cards ---- */
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
    .premium-title::-webkit-scrollbar { height: 6px; }
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
    .extra-space { margin-top: 50px; }
    .recommendation-header {
        font-size: 15px;
        border-left: 5px solid #B2BEB5;
        padding-left: 12px;
        margin-left: 5px;
    }
    /* ---- History Table ---- */
    .history-table {
        width: 100%;
        border-collapse: collapse;
        margin-top: 10px;
        margin-bottom: 20px;
        font-size: 14px;
    }
    .history-table th {
        background: linear-gradient(90deg, #ff8a00, #e52e71);
        color: white;
        padding: 10px 14px;
        text-align: left;
        font-weight: bold;
        letter-spacing: 0.5px;
    }
    .history-table td {
        padding: 9px 14px;
        border-bottom: 1px solid #2b2b2b;
        color: #e0e0e0;
        background: #1a1a1a;
    }
    .history-table tr:hover td {
        background: #252525;
    }
    .history-table .rating-star {
        color: #FFD700;
        font-weight: bold;
    }
    /* ---- User Info Box ---- */
    .user-info-box {
        background: linear-gradient(135deg, #1e1e1e, #2a2a2a);
        border: 1px solid #333;
        border-left: 5px solid #1a73e8;
        border-radius: 10px;
        padding: 16px 20px;
        margin-bottom: 20px;
    }
    .user-info-box h4 {
        color: #1a73e8;
        margin: 0 0 8px 0;
        font-size: 16px;
    }
    .user-info-box p {
        color: #c4c4c4;
        margin: 3px 0;
        font-size: 14px;
    }
    .user-info-box span {
        color: #F7E7A1;
        font-weight: bold;
    }
    /* ---- Tab styling ---- */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
    }
    .stTabs [data-baseweb="tab"] {
        height: 50px;
        padding: 0 24px;
        background-color: #1e1e1e;
        border-radius: 10px 10px 0 0;
        border: 1px solid #333;
        color: #c4c4c4 !important;
        font-size: 15px;
        font-weight: bold;
    }
    .stTabs [aria-selected="true"] {
        background: linear-gradient(90deg, #ff8a00, #e52e71) !important;
        color: white !important;
        border: none !important;
    }
    </style>
""", unsafe_allow_html=True)


# -------------------------------------------------------------------------
# HELPER: Render a grid of book cards
# -------------------------------------------------------------------------
def render_book_cards(book_list, df, header_text):
    """
    Renders a labelled grid of book cards (3 per row).
    book_list : list of book titles (ordered, e.g. top-k recommendations)
    df        : dataframe with book metadata
    header_text: markdown string shown above the grid
    """
    st.markdown(f"<div class='recommendation-header'>{header_text}</div>", unsafe_allow_html=True)
    st.write("")

    for i in range(0, len(book_list), 3):
        cols = st.columns(3)
        for j in range(3):
            if i + j < len(book_list):
                book = book_list[i + j]

                # Safely retrieve metadata (use first matching row)
                rows = df[df['title'] == book]
                if rows.empty:
                    continue
                book_info = rows.iloc[0]

                safe_title  = str(book).replace('"', '&quot;').replace("'", "&#39;")
                safe_author = str(book_info['Book-Author']).replace('"', '&quot;').replace("'", "&#39;")

                with cols[j]:
                    st.markdown(f"""
                    <div class='book-column'>
                        <div class='recommendation-badge'>{i + j + 1}</div>
                        <div class='book-image-area'>
                            <img src='{book_info['Image-URL-L']}'
                                 style='height:290px; width:auto; display:block;'>
                        </div>
                        <div class='book-info'>
                            <div class='premium-title' title="{safe_title}">{book}</div>
                            <div class='premium-divider'></div>
                            <div class='premium-author' title="{safe_author}">{book_info['Book-Author']}</div>
                            <div class='premium-year'>{book_info['Year-Of-Publication']}</div>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)

        if i < len(book_list) - 3:
            st.markdown("<br><hr><br>", unsafe_allow_html=True)

    st.markdown("<div class='extra-space'></div><div class='extra-space'></div>", unsafe_allow_html=True)
    st.image(
        'https://github.com/MarpakaPradeepSai/Employee-Churn-Prediction/blob/main/Data/Images%20&%20GIFs/thank-you-33.gif?raw=true',
        use_container_width=True
    )


# =========================================================================
# TABS
# =========================================================================
tab1, tab2 = st.tabs(["📖  Book-to-Book Recommendations", "👤  User-Based Recommendations"])


# =========================================================================
# TAB 1 — Book-to-Book (original behaviour)
# =========================================================================
with tab1:

    def get_top_similar_books(book_title, n=10):
        if book_title not in cosine_sim_df.index:
            return "⚠️ Book not found in the database."
        similar_scores = cosine_sim_df[book_title]
        similar_books  = similar_scores.sort_values(ascending=False)[1:n + 1]
        return similar_books

    all_books  = sorted(final_filtered_df['title'].unique().tolist())
    book_title = st.selectbox(
        'Enter a book title:',
        all_books,
        index=None,
        placeholder="Choose or enter a book title...",
        key='book_title'
    )
    num_recommendations = st.number_input(
        'Enter the number of recommendations:',
        min_value=1, max_value=50, value=10,
        key='num_recs_tab1'
    )

    # Session state for tab1
    if 'tab1_recommendations' not in st.session_state:
        st.session_state.tab1_recommendations = None
    if 'tab1_recommended_book' not in st.session_state:
        st.session_state.tab1_recommended_book = None
    if 'tab1_recommended_num' not in st.session_state:
        st.session_state.tab1_recommended_num = None

    if st.button('Recommend Books', key='btn_tab1'):
        if book_title:
            similar_books = get_top_similar_books(book_title, num_recommendations)
            st.session_state.tab1_recommendations   = similar_books
            st.session_state.tab1_recommended_book  = book_title
            st.session_state.tab1_recommended_num   = num_recommendations
        else:
            st.session_state.tab1_recommendations = None
            st.warning("⚠️ Please select or enter a book title.")

    if st.session_state.tab1_recommendations is not None:
        similar_books = st.session_state.tab1_recommendations
        rec_book      = st.session_state.tab1_recommended_book
        rec_num       = st.session_state.tab1_recommended_num

        if isinstance(similar_books, str):
            st.write(similar_books)
        else:
            book_list   = similar_books.index.tolist()
            header_text = (
                f"Top <strong>{rec_num}</strong> recommendations "
                f"similar to '<strong>{rec_book}</strong>':"
            )
            render_book_cards(book_list, final_filtered_df, header_text)


# =========================================================================
# TAB 2 — User-Based Recommendations
# =========================================================================
with tab2:

    st.markdown("""
        <p style='font-size:15px; color:#c4c4c4; margin-bottom:18px;'>
            Enter a <strong style='color:#F7E7A1;'>User ID</strong> to get personalised book
            recommendations based on that user's reading history and ratings.
        </p>
    """, unsafe_allow_html=True)

    # --- Input widgets ---
    user_id_input = st.selectbox(
        "Select or enter a User ID:",
        options=all_user_ids,
        index=None,
        placeholder="Choose a User ID...",
        key='user_id_input'
    )

    num_user_recs = st.number_input(
        'Number of recommendations:',
        min_value=1, max_value=50, value=10,
        key='num_recs_tab2'
    )

    show_history = st.checkbox(
        "📚 Show user's reading history alongside recommendations",
        value=True,
        key='show_history'
    )

    # Session state for tab2
    if 'tab2_recommendations' not in st.session_state:
        st.session_state.tab2_recommendations = None
    if 'tab2_user_history' not in st.session_state:
        st.session_state.tab2_user_history = None
    if 'tab2_user_id' not in st.session_state:
        st.session_state.tab2_user_id = None
    if 'tab2_status' not in st.session_state:
        st.session_state.tab2_status = None
    if 'tab2_num_recs' not in st.session_state:
        st.session_state.tab2_num_recs = None

    if st.button('Get Recommendations', key='btn_tab2'):
        if user_id_input is not None:
            with st.spinner(f"🔍 Finding recommendations for User {user_id_input}..."):
                recommendations, user_history, status = get_user_recommendations(
                    user_id_input,
                    final_filtered_df,
                    cosine_sim_df,
                    k=num_user_recs
                )
            st.session_state.tab2_recommendations = recommendations
            st.session_state.tab2_user_history    = user_history
            st.session_state.tab2_user_id         = user_id_input
            st.session_state.tab2_status          = status
            st.session_state.tab2_num_recs        = num_user_recs
        else:
            st.warning("⚠️ Please select a User ID.")

    # --- Display results ---
    if st.session_state.tab2_status is not None:
        status       = st.session_state.tab2_status
        uid          = st.session_state.tab2_user_id
        user_history = st.session_state.tab2_user_history
        recs         = st.session_state.tab2_recommendations
        rec_num      = st.session_state.tab2_num_recs

        if status == "no_history":
            st.error(f"❌ User ID **{uid}** has no interaction history in the database.")

        elif status == "no_recommendations":
            st.warning(
                f"⚠️ Could not generate recommendations for User **{uid}**. "
                "The books they interacted with may have no similar neighbours."
            )

        else:
            # -----------------------------------------------------------------
            # USER SUMMARY BOX
            # -----------------------------------------------------------------
            total_interactions = len(
                final_filtered_df[final_filtered_df['userId'] == uid]
            )
            num_rated = len(user_history) if user_history is not None else 0
            avg_rating = (
                round(user_history['rating'].mean(), 2)
                if user_history is not None and num_rated > 0 else "N/A"
            )

            st.markdown(f"""
                <div class='user-info-box'>
                    <h4>👤 User Profile — ID: {uid}</h4>
                    <p>📖 Total books interacted with: <span>{total_interactions}</span></p>
                    <p>⭐ Books explicitly rated: <span>{num_rated}</span></p>
                    <p>📊 Average rating given: <span>{avg_rating} / 10</span></p>
                </div>
            """, unsafe_allow_html=True)

            # -----------------------------------------------------------------
            # READING HISTORY TABLE (optional)
            # -----------------------------------------------------------------
            if show_history and user_history is not None and num_rated > 0:
                st.markdown(
                    "<div class='recommendation-header'>"
                    f"📚 Reading History for User <strong>{uid}</strong> "
                    f"(Top rated books shown):"
                    "</div>",
                    unsafe_allow_html=True
                )
                st.write("")

                # Build HTML table
                rows_html = ""
                for rank, (_, row) in enumerate(user_history.iterrows(), 1):
                    stars = "⭐" * min(int(row['rating']) // 2, 5)
                    rows_html += f"""
                        <tr>
                            <td style='text-align:center;'>{rank}</td>
                            <td>{row['title']}</td>
                            <td class='rating-star'>{row['rating']} / 10 &nbsp; {stars}</td>
                        </tr>
                    """

                st.markdown(f"""
                    <table class='history-table'>
                        <thead>
                            <tr>
                                <th style='text-align:center; width:60px;'>#</th>
                                <th>Book Title</th>
                                <th>Rating</th>
                            </tr>
                        </thead>
                        <tbody>
                            {rows_html}
                        </tbody>
                    </table>
                """, unsafe_allow_html=True)

                st.markdown("<br>", unsafe_allow_html=True)

            elif show_history:
                st.info(
                    "ℹ️ This user has no explicit ratings — "
                    "recommendations are based on implicit interactions only."
                )

            # -----------------------------------------------------------------
            # RECOMMENDED BOOKS GRID
            # -----------------------------------------------------------------
            header_text = (
                f"✨ Top <strong>{rec_num}</strong> Personalised Recommendations "
                f"for User <strong>{uid}</strong>:"
            )
            render_book_cards(recs, final_filtered_df, header_text)
