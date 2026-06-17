# Deployment Code
import streamlit as st
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from huggingface_hub import hf_hub_download
import warnings

warnings.filterwarnings('ignore')

# ------------------------------ Load Data ------------------------------
@st.cache_data
def load_and_prepare_data():
    # Load final filtered dataframe
    final_filtered_df_path = hf_hub_download(
        repo_id="IamPradeep/BRS_DATA", filename="final_filtered_df.csv", repo_type="dataset"
    )
    final_filtered_df = pd.read_csv(final_filtered_df_path)

    # Load book URLs dataframe
    book_urls_df_path = hf_hub_download(
        repo_id="IamPradeep/BRS_DATA", filename="Books.csv", repo_type="dataset"
    )
    book_urls_df = pd.read_csv(book_urls_df_path)
    book_urls_df.rename(columns={'Book-Title': 'title'}, inplace=True)

    # Drop duplicate titles to prevent row multiplication
    book_urls_df = book_urls_df.drop_duplicates(subset=['title'], keep='first')

    # Merge the dataframes
    final_filtered_df = final_filtered_df.merge(
        book_urls_df[['title', 'Book-Author', 'Year-Of-Publication', 'Image-URL-L']],
        on='title', how='left'
    )

    # Fix broken image URLs
    replacements = {
        'Jacob Have I Loved': 'http://images.amazon.com/images/P/0690040784.01.LZZZZZZZ.jpg',
        'Needful Things': 'http://images.amazon.com/images/P/0451172817.01.LZZZZZZZ.jpg',
        'All Creatures Great and Small': 'http://images.amazon.com/images/P/0312084986.01.LZZZZZZZ.jpg',
        "The Kitchen God's Wife": 'http://images.amazon.com/images/P/1590400356.01.LZZZZZZZ.jpg'
    }
    for title, url in replacements.items():
        final_filtered_df.loc[final_filtered_df['title'] == title, 'Image-URL-L'] = url

    # Build similarity matrix using explicit ratings (>0)
    explicit_ratings_df = final_filtered_df[final_filtered_df['rating'] > 0]
    book_user_mat = explicit_ratings_df.pivot_table(
        index='title', columns='userId', values='rating'
    ).fillna(0)

    cosine_sim = cosine_similarity(book_user_mat)
    cosine_sim_df = pd.DataFrame(
        cosine_sim, index=book_user_mat.index, columns=book_user_mat.index
    )

    return final_filtered_df, cosine_sim_df

final_filtered_df, cosine_sim_df = load_and_prepare_data()

# ------------------------------ Helper Functions ------------------------------
def get_top_similar_books(book_title, n=10):
    """Get similar books based on book title"""
    if book_title not in cosine_sim_df.index:
        return "⚠️ Book not found in the database."
    similar_scores = cosine_sim_df[book_title]
    similar_books = similar_scores.sort_values(ascending=False)[1:n+1]
    return similar_books

def get_user_recommendations(user_id, df, sim_matrix, k=10):
    """Generate personalized recommendations for a specific user."""
    user_history_all = df[df['userId'] == user_id]['title'].unique().tolist()
    user_history_rated = df[df['userId'] == user_id][['title', 'rating']].sort_values(
        by='rating', ascending=False
    )
    user_history_rated = user_history_rated.drop_duplicates(subset=['title'])

    if len(user_history_all) == 0:
        return None, None

    scores = {}
    for item in user_history_all:
        if item in sim_matrix.index:
            similar_items = sim_matrix[item].sort_values(ascending=False)[1:50]
            for sim_item, score in similar_items.items():
                if sim_item not in user_history_all:
                    scores[sim_item] = scores.get(sim_item, 0) + score

    sorted_scores = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    top_recommendations = [x[0] for x in sorted_scores[:k]]
    return top_recommendations, user_history_rated

def display_book_cards(books_list, start_index=0):
    """Display books in an elegant card layout with hover effects."""
    for i in range(0, len(books_list), 3):
        cols = st.columns(3)
        for j in range(3):
            if i + j < len(books_list):
                book = books_list[i + j]
                book_info = final_filtered_df[final_filtered_df['title'] == book].iloc[0]

                safe_title = str(book).replace('"', '&quot;').replace("'", "&#39;")
                safe_author = str(book_info['Book-Author']).replace('"', '&quot;').replace("'", "&#39;")
                rank = start_index + i + j + 1

                with cols[j]:
                    st.markdown(f"""
                    <div class='book-card'>
                        <div class='card-badge'>{rank}</div>
                        <div class='card-image'>
                            <img src='{book_info['Image-URL-L']}' alt='Cover' onerror="this.onerror=null;this.src='https://via.placeholder.com/150x220?text=No+Image';">
                        </div>
                        <div class='card-info'>
                            <div class='card-title' title="{safe_title}">{book}</div>
                            <div class='card-author' title="{safe_author}">{book_info['Book-Author']}</div>
                            <div class='card-year'>{book_info['Year-Of-Publication']}</div>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
        if i < len(books_list) - 3:
            st.markdown("<br><hr><br>", unsafe_allow_html=True)

# ------------------------------ UI Configuration ------------------------------
st.set_page_config(
    page_title="Book Recommendation System",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ------------------------------ Custom CSS Injections ------------------------------
st.markdown("""
<link href="https://fonts.googleapis.com/css2?family=Playfair+Display:wght@400;600;700&family=Poppins:wght@300;400;500;600&display=swap" rel="stylesheet">
<style>
    /* ============ GLOBAL RESETS ============ */
    body {
        font-family: 'Poppins', sans-serif;
        background: linear-gradient(135deg, #0B0F19 0%, #1A1F2E 100%);
        color: #E0E0E0;
    }
    .stApp {
        background: transparent;
    }
    /* Main content container */
    .main {
        background: rgba(20, 25, 35, 0.85);
        backdrop-filter: blur(20px);
        border-radius: 24px;
        padding: 2rem 3rem;
        margin: 1rem auto;
        max-width: 1200px;
        box-shadow: 0 25px 50px -12px rgba(0,0,0,0.5);
    }

    /* ============ TYPOGRAPHY ============ */
    h1, h2, h3, h4, h5, h6 {
        font-family: 'Playfair Display', serif;
        font-weight: 700;
        color: #F7E7A1;
    }
    .subheader {
        font-family: 'Poppins', sans-serif;
        font-size: 1.5rem;
        font-weight: 300;
        text-align: center;
        margin-bottom: 2rem;
        color: #B0B7C3;
        letter-spacing: 0.5px;
    }
    hr {
        border: none;
        height: 2px;
        background: linear-gradient(90deg, transparent, #B2BEB5, transparent);
        margin: 2rem 0;
    }

    /* ============ BUTTONS ============ */
    .stButton > button {
        font-family: 'Poppins', sans-serif;
        font-weight: 600;
        font-size: 1rem;
        background: linear-gradient(135deg, #FF8A00, #E52E71);
        color: white !important;
        border: none;
        border-radius: 50px;
        padding: 12px 28px;
        transition: all 0.3s ease;
        box-shadow: 0 4px 15px rgba(229, 46, 113, 0.4);
        letter-spacing: 0.5px;
    }
    .stButton > button:hover {
        transform: translateY(-3px);
        box-shadow: 0 8px 25px rgba(229, 46, 113, 0.6);
        background: linear-gradient(135deg, #FF9A2E, #EE3D7E);
    }
    .stButton > button:active {
        transform: translateY(1px);
    }

    /* ============ TABS ============ */
    .stTabs [data-baseweb="tab-list"] {
        gap: 12px;
        justify-content: center;
        border-bottom: 2px solid rgba(255,255,255,0.1);
        padding-bottom: 10px;
    }
    .stTabs [data-baseweb="tab"] {
        font-family: 'Playfair Display', serif;
        font-size: 1.2rem;
        font-weight: 600;
        background: rgba(255,255,255,0.05);
        border-radius: 30px 30px 0 0;
        padding: 12px 30px;
        color: #C0C0C0 !important;
        border: 1px solid rgba(255,255,255,0.1);
        border-bottom: none;
        transition: all 0.3s ease;
    }
    .stTabs [data-baseweb="tab"]:hover {
        background: rgba(255,255,255,0.1);
        color: #F7E7A1 !important;
    }
    .stTabs [aria-selected="true"] {
        background: linear-gradient(135deg, #1E3A5F, #2E5A8F) !important;
        color: white !important;
        border-color: #F7E7A1;
        box-shadow: 0 -5px 15px rgba(46, 90, 143, 0.4);
    }

    /* ============ BOOK CARDS ============ */
    .book-card {
        position: relative;
        background: rgba(255,255,255,0.04);
        backdrop-filter: blur(12px);
        border: 1px solid rgba(255,255,255,0.1);
        border-radius: 20px;
        overflow: hidden;
        transition: transform 0.3s ease, box-shadow 0.3s ease;
        box-shadow: 0 10px 30px rgba(0,0,0,0.3);
        margin: 10px;
    }
    .book-card:hover {
        transform: translateY(-10px) scale(1.02);
        box-shadow: 0 20px 40px rgba(0,0,0,0.5), 0 0 0 1px rgba(255,138,0,0.3);
    }
    .card-badge {
        position: absolute;
        top: -15px;
        /* Center horizontally and push slightly right*/
        left: 50%;
        transform: translateX(-50%);
        width: 44px;
        height: 44px;
        border-radius: 50%;
        background: linear-gradient(135deg, #28a745, #20c997);
        display: flex;
        align-items: center;
        justify-content: center;
        font-family: 'Poppins', sans-serif;
        font-weight: 700;
        font-size: 1.1rem;
        color: white;
        z-index: 10;
        box-shadow: 0 4px 12px rgba(40,167,69,0.5);
    }
    .card-image {
        padding: 2rem 1.5rem 1rem;
        display: flex;
        justify-content: center;
        background: rgba(0,0,0,0.2);
    }
    .card-image img {
        height: 240px;
        width: auto;
        object-fit: cover;
        border-radius: 8px;
        transition: transform 0.3s ease;
        box-shadow: 0 8px 20px rgba(0,0,0,0.5);
    }
    .book-card:hover .card-image img {
        transform: scale(1.05);
    }
    .card-info {
        background: rgba(10, 15, 25, 0.8);
        backdrop-filter: blur(10px);
        padding: 1.2rem;
        text-align: center;
    }
    .card-title {
        font-family: 'Playfair Display', serif;
        font-size: 1rem;
        font-weight: 600;
        color: #F7E7A1;
        margin-bottom: 0.5rem;
        white-space: nowrap;
        overflow: hidden;
        text-overflow: ellipsis;
    }
    .card-author {
        font-family: 'Poppins', sans-serif;
        font-size: 0.85rem;
        color: #B0B7C3;
        font-style: italic;
        margin-bottom: 0.3rem;
        white-space: nowrap;
        overflow: hidden;
        text-overflow: ellipsis;
    }
    .card-year {
        font-family: 'Poppins', sans-serif;
        font-size: 0.75rem;
        color: #8A8D91;
        text-transform: uppercase;
        letter-spacing: 1.5px;
        font-weight: 500;
    }

    /* ============ EXPANDER ============ */
    [data-testid="stExpander"] details {
        border: none;
        border-radius: 16px;
        background: rgba(255,255,255,0.03);
    }
    [data-testid="stExpander"] details[open] {
        border: 1px solid rgba(255,138,0,0.3);
    }
    [data-testid="stExpander"] summary {
        font-family: 'Playfair Display', serif;
        font-size: 1.2rem;
        padding: 1rem 1.5rem;
        background: rgba(255,255,255,0.05);
        color: #F7E7A1;
        border-radius: 12px;
    }

    /* ============ INPUTS & SELECTS ============ */
    .stSelectbox div[data-baseweb="select"] {
        font-family: 'Poppins', sans-serif;
        background: rgba(255,255,255,0.05);
        border-radius: 12px;
        border-color: rgba(255,138,0,0.2);
    }
    .stNumberInput input {
        background: rgba(255,255,255,0.05);
        border-color: rgba(255,138,0,0.2);
        color: white;
        font-family: 'Poppins', sans-serif;
    }

    /* ============ SCROLL TO TOP BUTTON ============ */
    #scroll-top-btn {
        display: none;
        position: fixed;
        bottom: 30px;
        right: 30px;
        z-index: 999;
        width: 50px;
        height: 50px;
        border-radius: 50%;
        background: linear-gradient(135deg, #FF8A00, #E52E71);
        color: white;
        font-size: 1.5rem;
        border: none;
        outline: none;
        cursor: pointer;
        box-shadow: 0 4px 15px rgba(229,46,113,0.5);
        transition: all 0.3s ease;
    }
    #scroll-top-btn:hover {
        transform: translateY(-5px);
        box-shadow: 0 8px 25px rgba(229,46,113,0.8);
    }

    /* ============ RECOMMENDATION HEADER ============ */
    .recommendation-header {
        font-family: 'Playfair Display', serif;
        background: linear-gradient(90deg, rgba(255,138,0,0.1) 0%, rgba(229,46,113,0.1) 100%);
        border-left: 5px solid #F7E7A1;
        padding: 0.8rem 1.5rem;
        margin: 1rem 0 2rem;
        border-radius: 0 12px 12px 0;
        display: inline-block;
        font-size: 1.1rem;
        color: #F7E7A1;
    }

    /* ============ EXTRA MARGINS ============ */
    .extra-space {
        margin-top: 3rem;
    }
</style>
""", unsafe_allow_html=True)

# ------------------------------ Scroll-to-Top Button (JS) ------------------------------
st.components.v1.html("""
<script>
// Smooth scroll function
window.addEventListener('scroll', function() {
    var btn = document.getElementById('scroll-top-btn');
    if (window.pageYOffset > 300) {
        btn.style.display = 'block';
    } else {
        btn.style.display = 'none';
    }
});

document.addEventListener('DOMContentLoaded', function() {
    var btn = document.createElement('button');
    btn.id = 'scroll-top-btn';
    btn.innerHTML = '↑';
    btn.title = 'Back to top';
    btn.onclick = function() {
        window.scrollTo({top: 0, behavior: 'smooth'});
    };
    document.body.appendChild(btn);
});
</script>
""", height=0)

# ------------------------------ Main Page Content ------------------------------
st.markdown("""
    <h1 style='font-size: 3.2rem; text-align: center; margin-bottom: 0.5rem;'>
        📚 Book Recommendation System
    </h1>
    <p class='subheader'>Let Us Help You Choose Your Next Book!</p>
""", unsafe_allow_html=True)

# Banner image
st.image(
    'https://img.freepik.com/premium-vector/bookcase-with-books_182089-197.jpg',
    use_container_width=True
)

# ------------------------------ TABS ------------------------------
tab1, tab2 = st.tabs([
    "📚 Book-to-Book Recommendations",
    "👤 User-Specific Recommendations"
])

# ============== TAB 1: Book-to-Book ==============
with tab1:
    st.markdown("<h3 style='text-align: center; margin-bottom: 1.5rem;'>Find Similar Books</h3>", unsafe_allow_html=True)
    st.write("Select a book and discover similar titles based on user preferences and ratings.")

    all_books = sorted(final_filtered_df['title'].unique().tolist())
    col1, col2 = st.columns([2, 1])
    with col1:
        book_title = st.selectbox(
            'Enter a book title:',
            all_books,
            index=None,
            placeholder="Choose or enter a book title...",
            key='book_title'
        )
    with col2:
        num_recs_book = st.number_input(
            'Number of recommendations:',
            min_value=1, max_value=50, value=10,
            key='num_recs_book'
        )

    # Initialize session state if not present
    if 'recommendations' not in st.session_state:
        st.session_state.recommendations = None
        st.session_state.recommended_book = None
        st.session_state.recommended_num = None

    if st.button('✨ Recommend Books', key='btn_book_recs'):
        if book_title:
            with st.spinner('🔍 Finding the best matches...'):
                similar_books = get_top_similar_books(book_title, num_recs_book)
                st.session_state.recommendations = similar_books
                st.session_state.recommended_book = book_title
                st.session_state.recommended_num = num_recs_book
        else:
            st.session_state.recommendations = None
            st.warning("⚠️ Please select or enter a book title.")

    if st.session_state.recommendations is not None:
        similar_books = st.session_state.recommendations
        rec_book = st.session_state.recommended_book
        rec_num = st.session_state.recommended_num

        if isinstance(similar_books, str):
            st.error(similar_books)
        else:
            st.markdown(
                f"<div class='recommendation-header'>Top {rec_num} recommendations for '<strong>{rec_book}</strong>'</div>",
                unsafe_allow_html=True
            )
            st.write("")
            books_list = similar_books.index.tolist()
            display_book_cards(books_list)

    # Thank you animation
    st.markdown("<div class='extra-space'></div>", unsafe_allow_html=True)
    st.image(
        'https://github.com/MarpakaPradeepSai/Employee-Churn-Prediction/blob/main/Data/Images%20&%20GIFs/thank-you-33.gif?raw=true',
        use_container_width=True
    )

# ============== TAB 2: User-Specific Recommendations ==============
with tab2:
    st.markdown("<h3 style='text-align: center; margin-bottom: 1.5rem;'>Personalized Recommendations</h3>", unsafe_allow_html=True)
    st.write("Enter a User ID to get personalized book recommendations based on their reading history.")

    all_user_ids = sorted(final_filtered_df['userId'].unique().tolist())

    col1, col2 = st.columns([2, 1])
    with col1:
        user_id_input = st.selectbox(
            'Select or enter a User ID:',
            all_user_ids,
            index=None,
            placeholder="Choose a User ID...",
            key='user_id_select'
        )
    with col2:
        num_user_recs = st.number_input(
            'Number of recommendations:',
            min_value=1, max_value=50, value=10,
            key='num_user_recs'
        )

    if 'user_recommendations' not in st.session_state:
        st.session_state.user_recommendations = None
        st.session_state.user_history_display = None
        st.session_state.current_user_id = None

    if st.button('🎯 Get Personalized Recommendations', key='btn_user_recs'):
        if user_id_input:
            with st.spinner('🧠 Analyzing reading patterns...'):
                recommendations, user_history = get_user_recommendations(
                    user_id_input, final_filtered_df, cosine_sim_df, k=num_user_recs
                )
                if recommendations is None:
                    st.warning(f"⚠️ User ID {user_id_input} has no interaction history in the database.")
                    st.session_state.user_recommendations = None
                    st.session_state.user_history_display = None
                else:
                    st.session_state.user_recommendations = recommendations
                    st.session_state.user_history_display = user_history
                    st.session_state.current_user_id = user_id_input
        else:
            st.warning("⚠️ Please select or enter a User ID.")

    if st.session_state.user_recommendations is not None:
        user_id_display = st.session_state.current_user_id
        recommendations = st.session_state.user_recommendations
        user_history = st.session_state.user_history_display

        # Display user history in a sleek expander
        if user_history is not None and len(user_history) > 0:
            with st.expander("📖 View User's Reading History", expanded=False):
                history_df = user_history.copy()
                history_df.reset_index(drop=True, inplace=True)
                history_df.index = history_df.index + 1
                history_df.columns = ['Book Title', 'Rating']
                # Style the rating column
                def highlight_rating(val):
                    color = '#28a745' if val > 0 else '#E52E71'
                    return f'color: {color}; font-weight: bold;'
                styled_df = history_df.style.applymap(highlight_rating, subset=['Rating'])
                st.dataframe(styled_df, use_container_width=True)
                st.caption("ℹ️ *A rating of \"0\" indicates an **interacted** but **unrated** book.*")

        st.markdown("<br>", unsafe_allow_html=True)

        if len(recommendations) > 0:
            st.markdown(
                f"<div class='recommendation-header'>Top {len(recommendations)} Personalized Recommendations for User ID: <strong>{user_id_display}</strong></div>",
                unsafe_allow_html=True
            )
            st.write("")
            display_book_cards(recommendations)
        else:
            st.info("No recommendations available for this user at the moment.")

    # Thank you animation
    st.markdown("<div class='extra-space'></div>", unsafe_allow_html=True)
    st.image(
        'https://github.com/MarpakaPradeepSai/Employee-Churn-Prediction/blob/main/Data/Images%20&%20GIFs/thank-you-33.gif?raw=true',
        use_container_width=True
    )
