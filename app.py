import streamlit as st
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from huggingface_hub import hf_hub_download
import warnings

warnings.filterwarnings('ignore')

# =============================================================================
# DATA LOADING & PREPARATION
# =============================================================================

@st.cache_data
def load_and_prepare_data():
    # Load your final filtered dataframe from Hugging Face
    final_filtered_df_path = hf_hub_download(
        repo_id="IamPradeep/BRS_DATA",
        filename="final_filtered_df.csv",
        repo_type="dataset"
    )
    final_filtered_df = pd.read_csv(final_filtered_df_path)

    # Load the dataframe containing book URLs from Hugging Face
    book_urls_df_path = hf_hub_download(
        repo_id="IamPradeep/BRS_DATA",
        filename="Books.csv",
        repo_type="dataset"
    )
    book_urls_df = pd.read_csv(book_urls_df_path)
    book_urls_df.rename(columns={'Book-Title': 'title'}, inplace=True)

    # Merge the dataframes on the title
    final_filtered_df = final_filtered_df.merge(book_urls_df, on='title', how='left')

    # URL replacements for known broken images
    url1 = 'http://images.amazon.com/images/P/0690040784.01.LZZZZZZZ.jpg'
    url2 = 'http://images.amazon.com/images/P/0451172817.01.LZZZZZZZ.jpg'
    url3 = 'http://images.amazon.com/images/P/0312084986.01.LZZZZZZZ.jpg'
    url4 = 'http://images.amazon.com/images/P/1590400356.01.LZZZZZZZ.jpg'

    final_filtered_df.loc[final_filtered_df['title'] == 'Jacob Have I Loved', 'Image-URL-L'] = url1
    final_filtered_df.loc[final_filtered_df['title'] == 'Needful Things', 'Image-URL-L'] = url2
    final_filtered_df.loc[final_filtered_df['title'] == 'All Creatures Great and Small', 'Image-URL-L'] = url3
    final_filtered_df.loc[final_filtered_df['title'] == "The Kitchen God's Wife", 'Image-URL-L'] = url4

    # -------------------------------------------------------------------------
    # BUILD SIMILARITY MATRIX USING ONLY EXPLICIT RATINGS (> 0)
    # -------------------------------------------------------------------------
    explicit_ratings_df = final_filtered_df[final_filtered_df['rating'] > 0]
    book_user_mat = explicit_ratings_df.pivot_table(
        index='title', columns='userId', values='rating'
    ).fillna(0)

    cosine_sim = cosine_similarity(book_user_mat)
    cosine_sim_df = pd.DataFrame(
        cosine_sim,
        index=book_user_mat.index,
        columns=book_user_mat.index
    )

    return final_filtered_df, cosine_sim_df


final_filtered_df, cosine_sim_df = load_and_prepare_data()

# =============================================================================
# RECOMMENDATION FUNCTIONS
# =============================================================================

def get_top_similar_books(book_title, n=10):
    """Item-Item similarity based recommendations for a given book title."""
    if book_title not in cosine_sim_df.index:
        return "⚠️ Book not found in the database."
    similar_scores = cosine_sim_df[book_title]
    similar_books = similar_scores.sort_values(ascending=False)[1:n + 1]
    return similar_books


def get_user_recommendations(user_id, df, sim_matrix, k=10):
    """
    Generates personalised book recommendations for a specific user
    using Item-Item collaborative filtering.
    """
    # Everything the user has touched (rated or implicit)
    user_history_all = df[df['userId'] == user_id]['title'].tolist()

    # Subset with explicit ratings for display purposes
    user_history_rated = (
        df[(df['userId'] == user_id) & (df['rating'] > 0)][['title', 'rating']]
        .drop_duplicates(subset='title')
        .sort_values(by='rating', ascending=False)
    )

    if len(user_history_all) == 0:
        return None, None  # Cold-start: no history at all

    # Build candidate scores by accumulating cosine similarity from each
    # book the user has interacted with
    scores = {}
    for item in user_history_all:
        if item in sim_matrix.index:
            similar_items = sim_matrix[item].sort_values(ascending=False)[1:50]
            for sim_item, score in similar_items.items():
                if sim_item not in user_history_all:
                    scores[sim_item] = scores.get(sim_item, 0) + score

    if not scores:
        return [], user_history_rated

    sorted_scores = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    top_recommendations = [x[0] for x in sorted_scores[:k]]

    return top_recommendations, user_history_rated


# =============================================================================
# SHARED CSS STYLES
# =============================================================================

st.markdown("""
    <style>
    html, body, [class*="css"], [class*="st-"],
    h1, h2, h3, h4, h5, h6, p, div, span,
    label, input, button, select, option, textarea {
        font-family: 'Tiempos', 'Tiempos Text', Georgia,
                     'Times New Roman', serif !important;
    }

    /* ── Subheader ── */
    .subheader {
        font-size: 22px;
        font-weight: bold;
        margin-bottom: 10px;
        color: #1a73e8;
        text-align: center;
    }

    /* ── Buttons ── */
    .stButton > button {
        font-family: 'Tiempos', 'Tiempos Text', Georgia,
                     'Times New Roman', serif !important;
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
    .stButton > button:active { transform: scale(0.98); }

    /* ── Book Card ── */
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

    /* ── Misc ── */
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
    .book-image-area { padding: 35px 20px 20px 20px; }
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

    /* ── User History Table ── */
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
    }
    .history-table td {
        padding: 9px 14px;
        border-bottom: 1px solid #2b2b2b;
        color: #e0e0e0;
        background: #1e1e1e;
    }
    .history-table tr:hover td { background: #2a2a2a; }

    /* ── Star Rating ── */
    .star-rating { color: #FFD700; font-size: 15px; }

    /* ── Info / Warning Boxes ── */
    .info-box {
        background: #1e3a5f;
        border-left: 5px solid #1a73e8;
        border-radius: 8px;
        padding: 14px 18px;
        margin: 12px 0;
        color: #cde;
        font-size: 14px;
    }
    .warning-box {
        background: #3a2a00;
        border-left: 5px solid #ff8a00;
        border-radius: 8px;
        padding: 14px 18px;
        margin: 12px 0;
        color: #ffe0a0;
        font-size: 14px;
    }

    /* ── Tab Styling ── */
    button[data-baseweb="tab"] {
        font-size: 17px !important;
        font-weight: bold !important;
        padding: 12px 28px !important;
    }
    </style>
""", unsafe_allow_html=True)

# =============================================================================
# PAGE HEADER
# =============================================================================

st.markdown("""
    <h1 style='font-size: 40px; text-align: center;
               margin-bottom: 5px; padding-bottom: 0px;'>
        Book Recommendation System
    </h1>
    <p class='subheader'>Let Us Help You Choose Your Next Book!</p>
""", unsafe_allow_html=True)

st.image(
    'https://img.freepik.com/premium-vector/bookcase-with-books_182089-197.jpg',
    use_container_width=True
)

# =============================================================================
# HELPER: render a row of book cards
# =============================================================================

def render_book_cards(book_list, df, start_index=0):
    """
    Renders book cards in rows of 3.

    Parameters
    ----------
    book_list  : list of book titles  (or a pandas Index / Series.index)
    df         : master dataframe with metadata & image URLs
    start_index: offset for the badge number (useful when combining sections)
    """
    book_list = list(book_list)          # normalise to plain list
    for i in range(0, len(book_list), 3):
        cols = st.columns(3)
        for j in range(3):
            if i + j < len(book_list):
                book = book_list[i + j]
                rows = df[df['title'] == book]
                if rows.empty:
                    continue
                book_info = rows.iloc[0]

                safe_title  = str(book).replace('"', '&quot;').replace("'", "&#39;")
                safe_author = str(book_info['Book-Author']).replace('"', '&quot;').replace("'", "&#39;")

                with cols[j]:
                    st.markdown(f"""
                    <div class='book-column'>
                        <div class='recommendation-badge'>{start_index + i + j + 1}</div>
                        <div class='book-image-area'>
                            <img src='{book_info['Image-URL-L']}'
                                 style='height:290px; width:auto; display:block;'>
                        </div>
                        <div class='book-info'>
                            <div class='premium-title'
                                 title="{safe_title}">{book}</div>
                            <div class='premium-divider'></div>
                            <div class='premium-author'
                                 title="{safe_author}">{book_info['Book-Author']}</div>
                            <div class='premium-year'>{book_info['Year-Of-Publication']}</div>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)

        if i < len(book_list) - 3:
            st.markdown("<br><hr><br>", unsafe_allow_html=True)


# =============================================================================
# TABS
# =============================================================================

tab1, tab2 = st.tabs(["📖  Book-Based Recommendations",
                       "👤  User-Based Recommendations"])

# ─────────────────────────────────────────────────────────────────────────────
# TAB 1 — Book-Based (Item-Item similarity)
# ─────────────────────────────────────────────────────────────────────────────
with tab1:

    st.markdown("### 📚 Find Books Similar to a Title You Love")
    st.markdown("""
        <div class='info-box'>
            🔍 Select any book from the dropdown below and we will find the
            most similar books based on <strong>collaborative filtering
            (cosine similarity)</strong> across all user ratings.
        </div>
    """, unsafe_allow_html=True)

    all_books = sorted(final_filtered_df['title'].unique().tolist())

    book_title = st.selectbox(
        'Enter a book title:',
        all_books,
        index=None,
        placeholder="Choose or enter a book title...",
        key='book_title'
    )

    num_recommendations = st.number_input(
        'Number of recommendations:',
        min_value=1, max_value=50, value=10,
        key='num_rec_book'
    )

    # Session state keys for Tab 1
    for key in ['recommendations', 'recommended_book', 'recommended_num']:
        if key not in st.session_state:
            st.session_state[key] = None

    if st.button('🔎 Recommend Books', key='btn_book'):
        if book_title:
            similar_books = get_top_similar_books(book_title, num_recommendations)
            st.session_state.recommendations  = similar_books
            st.session_state.recommended_book = book_title
            st.session_state.recommended_num  = num_recommendations
        else:
            st.session_state.recommendations = None
            st.warning("⚠️ Please select or enter a book title.")

    if st.session_state.recommendations is not None:
        similar_books = st.session_state.recommendations
        rec_book      = st.session_state.recommended_book
        rec_num       = st.session_state.recommended_num

        if isinstance(similar_books, str):
            st.error(similar_books)
        else:
            st.markdown(
                f"<div class='recommendation-header'>Top <strong>{rec_num}</strong> "
                f"recommendations for '<strong>{rec_book}</strong>':</div>",
                unsafe_allow_html=True
            )
            st.write("")
            render_book_cards(similar_books.index, final_filtered_df)
            st.markdown(
                "<div class='extra-space'></div><div class='extra-space'></div>",
                unsafe_allow_html=True
            )
            st.image(
                'https://github.com/MarpakaPradeepSai/Employee-Churn-Prediction/'
                'blob/main/Data/Images%20&%20GIFs/thank-you-33.gif?raw=true',
                use_container_width=True
            )

# ─────────────────────────────────────────────────────────────────────────────
# TAB 2 — User-Based (Personalised)
# ─────────────────────────────────────────────────────────────────────────────
with tab2:

    st.markdown("### 👤 Get Personalised Recommendations for a User")
    st.markdown("""
        <div class='info-box'>
            🎯 Enter a <strong>User ID</strong> to receive personalised book
            recommendations based on that user's reading history and ratings.
            The system uses <strong>Item-Item Collaborative Filtering</strong>
            — books rated highly by similar readers are surfaced first.
        </div>
    """, unsafe_allow_html=True)

    # ── Valid user IDs ──
    valid_user_ids = sorted(
        final_filtered_df[final_filtered_df['rating'] > 0]['userId'].unique().tolist()
    )

    # Show a sample of valid IDs so testers know what to enter
    with st.expander("💡 Show sample valid User IDs"):
        sample_ids = valid_user_ids[:20]
        st.write(", ".join(str(uid) for uid in sample_ids))

    user_id_input = st.number_input(
        'Enter User ID:',
        min_value=int(min(valid_user_ids)),
        max_value=int(max(valid_user_ids)),
        value=int(valid_user_ids[0]),
        step=1,
        key='user_id_input'
    )

    num_user_recs = st.number_input(
        'Number of recommendations:',
        min_value=1, max_value=50, value=10,
        key='num_rec_user'
    )

    show_history = st.checkbox(
        "📋 Show user's reading history alongside recommendations",
        value=True,
        key='show_history'
    )

    # Session state keys for Tab 2
    for key in ['user_recs', 'user_history_df', 'queried_user_id']:
        if key not in st.session_state:
            st.session_state[key] = None

    if st.button('🎯 Get My Recommendations', key='btn_user'):
        uid = int(user_id_input)

        if uid not in valid_user_ids:
            st.session_state.user_recs        = "invalid"
            st.session_state.user_history_df  = None
            st.session_state.queried_user_id  = uid
        else:
            with st.spinner(f"⏳ Generating recommendations for User {uid} …"):
                recs, history_df = get_user_recommendations(
                    uid, final_filtered_df, cosine_sim_df, k=num_user_recs
                )
            st.session_state.user_recs       = recs
            st.session_state.user_history_df = history_df
            st.session_state.queried_user_id = uid

    # ── Display results ──
    if st.session_state.user_recs is not None:
        uid      = st.session_state.queried_user_id
        recs     = st.session_state.user_recs
        hist_df  = st.session_state.user_history_df

        # Invalid user
        if recs == "invalid":
            st.markdown(f"""
                <div class='warning-box'>
                    ⚠️ User ID <strong>{uid}</strong> was not found in the
                    database or has no explicit ratings. Please try a different ID.
                </div>
            """, unsafe_allow_html=True)

        # Cold-start (history is None)
        elif recs is None:
            st.markdown(f"""
                <div class='warning-box'>
                    ⚠️ User ID <strong>{uid}</strong> has no interaction
                    history. We cannot generate personalised recommendations
                    for new users yet.
                </div>
            """, unsafe_allow_html=True)

        # No neighbours found (empty list)
        elif len(recs) == 0:
            st.markdown(f"""
                <div class='warning-box'>
                    ⚠️ We could not find enough similar books for User
                    <strong>{uid}</strong>. The user may have rated only
                    very obscure titles with no neighbours in the similarity
                    matrix.
                </div>
            """, unsafe_allow_html=True)

        else:
            # ── Reading History ──
            if show_history and hist_df is not None and len(hist_df) > 0:
                st.markdown(
                    f"<div class='recommendation-header'>"
                    f"📚 Reading History of User <strong>{uid}</strong> "
                    f"(top rated):</div>",
                    unsafe_allow_html=True
                )
                st.write("")

                def stars(rating):
                    full  = int(rating)
                    empty = 10 - full
                    return "⭐" * full + "☆" * empty

                rows_html = ""
                for _, row in hist_df.head(10).iterrows():
                    rows_html += f"""
                        <tr>
                            <td>{row['title']}</td>
                            <td>
                                <span class='star-rating'>{stars(row['rating'])}</span>
                                &nbsp;<strong style='color:#F7E7A1;'>{int(row['rating'])}/10</strong>
                            </td>
                        </tr>
                    """

                st.markdown(f"""
                    <table class='history-table'>
                        <thead>
                            <tr>
                                <th>📖 Book Title</th>
                                <th>⭐ Rating</th>
                            </tr>
                        </thead>
                        <tbody>
                            {rows_html}
                        </tbody>
                    </table>
                """, unsafe_allow_html=True)

                if len(hist_df) > 10:
                    st.markdown(
                        f"<p style='color:#888; font-size:13px; margin-top:-10px;'>"
                        f"… and {len(hist_df) - 10} more rated books.</p>",
                        unsafe_allow_html=True
                    )

                st.markdown("<br><hr><br>", unsafe_allow_html=True)

            # ── Recommendations ──
            st.markdown(
                f"<div class='recommendation-header'>"
                f"✨ Top <strong>{len(recs)}</strong> Personalised Recommendations "
                f"for User <strong>{uid}</strong>:</div>",
                unsafe_allow_html=True
            )
            st.write("")
            render_book_cards(recs, final_filtered_df)
            st.markdown(
                "<div class='extra-space'></div><div class='extra-space'></div>",
                unsafe_allow_html=True
            )
            st.image(
                'https://github.com/MarpakaPradeepSai/Employee-Churn-Prediction/'
                'blob/main/Data/Images%20&%20GIFs/thank-you-33.gif?raw=true',
                use_container_width=True
            )
