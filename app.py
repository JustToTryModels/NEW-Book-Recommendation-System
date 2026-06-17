# Deployment Code
import streamlit as st
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from huggingface_hub import hf_hub_download
import warnings
import random

warnings.filterwarnings('ignore')

# Page Configuration must be the first Streamlit command
st.set_page_config(page_title="Lumiere Books", page_icon="📚", layout="wide", initial_sidebar_state="collapsed")

@st.cache_data
def load_and_prepare_data():
    # Load your final filtered dataframe from Hugging Face
    final_filtered_df_path = hf_hub_download(repo_id="IamPradeep/BRS_DATA", filename="final_filtered_df.csv", repo_type="dataset")
    final_filtered_df = pd.read_csv(final_filtered_df_path)

    # Load the dataframe containing book URLs from Hugging Face
    book_urls_df_path = hf_hub_download(repo_id="IamPradeep/BRS_DATA", filename="Books.csv", repo_type="dataset")
    book_urls_df = pd.read_csv(book_urls_df_path)
    book_urls_df.rename(columns={'Book-Title': 'title'}, inplace=True)

    # Drop duplicate titles before merging to prevent row multiplication
    book_urls_df = book_urls_df.drop_duplicates(subset=['title'], keep='first')

    # Merge the dataframes on the title
    final_filtered_df = final_filtered_df.merge(book_urls_df[['title', 'Book-Author', 'Year-Of-Publication', 'Image-URL-L']], on='title', how='left')

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

    # BUILD SIMILARITY MATRIX USING ONLY EXPLICIT RATINGS (>0)               
    explicit_ratings_df = final_filtered_df[final_filtered_df['rating'] > 0]
    book_user_mat = explicit_ratings_df.pivot_table(index='title', columns='userId', values='rating').fillna(0)

    # Calculate the cosine similarity matrix
    cosine_sim = cosine_similarity(book_user_mat)
    cosine_sim_df = pd.DataFrame(cosine_sim, index=book_user_mat.index, columns=book_user_mat.index)

    return final_filtered_df, cosine_sim_df

# -------------------------------------------------------------------------
# CUSTOM PREMIUM CSS & THEME
# -------------------------------------------------------------------------
st.markdown("""
    <link href="https://fonts.googleapis.com/css2?family=Playfair+Display:wght@400;700;900&family=Inter:wght@300;400;600;700&display=swap" rel="stylesheet">
    <style>
    /* Global Dark Theme & Glassmorphism */
    .stApp {
        background: radial-gradient(circle at top left, #1a1c2e 0%, #0a0b14 40%, #050609 100%);
        color: #e0e0e0;
        font-family: 'Inter', sans-serif;
    }
    
    /* Hide Streamlit default elements for a cleaner app look */
    #MainMenu, footer, header {visibility: hidden;}
    .stApp > div {padding-top: 1rem;}
    
    /* Custom Scrollbar */
    ::-webkit-scrollbar { width: 8px; height: 8px; }
    ::-webkit-scrollbar-track { background: #0a0b14; }
    ::-webkit-scrollbar-thumb { background: #00c6ff; border-radius: 10px; }
    ::-webkit-scrollbar-thumb:hover { background: #0072ff; }

    /* Typography */
    h1, h2, h3, .premium-title { font-family: 'Playfair Display', serif !important; }
    
    /* Hero Header */
    .hero-header {
        background: rgba(255, 255, 255, 0.03);
        backdrop-filter: blur(15px);
        border: 1px solid rgba(255, 255, 255, 0.1);
        border-radius: 24px;
        padding: 2.5rem;
        text-align: center;
        margin-bottom: 2rem;
        box-shadow: 0 8px 32px rgba(0, 0, 0, 0.4);
        position: relative;
        overflow: hidden;
    }
    .hero-header::before {
        content: '';
        position: absolute;
        top: -50%; left: -50%; width: 200%; height: 200%;
        background: conic-gradient(from 0deg, transparent 0%, rgba(0, 198, 255, 0.1) 20%, transparent 40%);
        animation: rotate 8s linear infinite;
        z-index: 0;
    }
    @keyframes rotate { from {transform: rotate(0deg);} to {transform: rotate(360deg);} }
    
    .hero-title {
        font-size: 3.5rem;
        font-weight: 900;
        background: linear-gradient(90deg, #ffffff, #00c6ff, #ffffff);
        background-size: 200% auto;
        -webkit-background-clip: text;
        color: transparent;
        animation: shine 5s linear infinite;
        margin-bottom: 0.5rem;
        position: relative;
        z-index: 1;
    }
    @keyframes shine { to { background-position: 200% center; } }
    
    .hero-sub {
        font-size: 1.2rem;
        color: #a0a0a0;
        font-weight: 300;
        letter-spacing: 1px;
        position: relative;
        z-index: 1;
    }

    /* Streamlit Component Styling */
    .stTabs [data-baseweb="tab-list"] {
        background: rgba(255, 255, 255, 0.05);
        border-radius: 16px;
        padding: 6px;
        gap: 4px;
        border: 1px solid rgba(255, 255, 255, 0.1);
        backdrop-filter: blur(10px);
    }
    .stTabs [data-baseweb="tab"] {
        background-color: transparent;
        border-radius: 12px;
        padding: 10px 20px;
        font-size: 1rem;
        font-weight: 600;
        color: #a0a0a0 !important;
        border: none !important;
        transition: all 0.3s ease;
    }
    .stTabs [aria-selected="true"] {
        background: linear-gradient(135deg, #0072ff 0%, #00c6ff 100%) !important;
        color: white !important;
        box-shadow: 0 4px 15px rgba(0, 114, 255, 0.4);
    }
    .stTabs [data-baseweb="tab-border"] { display: none !important; }
    .stTabs [data-baseweb="tab-highlight"] { display: none !important; }

    /* Inputs & Buttons */
    .stSelectbox label, .stNumberInput label {
        color: #c0c0c0 !important;
        font-weight: 500;
        letter-spacing: 0.5px;
    }
    
    div[data-baseweb="select"] > div, div[data-baseweb="spinbutton"] {
        background-color: rgba(255, 255, 255, 0.05) !important;
        border: 1px solid rgba(255, 255, 255, 0.1) !important;
        border-radius: 12px !important;
        transition: all 0.3s ease;
    }
    div[data-baseweb="select"] > div:hover, div[data-baseweb="spinbutton"]:hover {
        border: 1px solid #00c6ff !important;
        background-color: rgba(0, 198, 255, 0.05) !important;
    }

    .stButton > button {
        background: linear-gradient(135deg, #0072ff 0%, #00c6ff 100%);
        color: white !important;
        border: none;
        border-radius: 12px;
        padding: 12px 28px;
        font-weight: 700;
        font-size: 1.05rem;
        letter-spacing: 0.5px;
        transition: all 0.3s ease;
        box-shadow: 0 4px 15px rgba(0, 114, 255, 0.3);
        width: 100%;
        margin-top: 10px;
    }
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 25px rgba(0, 114, 255, 0.5);
        background: linear-gradient(135deg, #00c6ff 0%, #0072ff 100%);
    }
    .stButton > button:active { transform: translateY(0); }

    /* Book Cards */
    .book-grid {
        display: flex;
        flex-wrap: wrap;
        gap: 2rem;
        justify-content: flex-start;
    }
    .book-column {
        flex: 1 1 300px;
        max-width: 320px;
        position: relative;
        background: rgba(255, 255, 255, 0.03);
        border: 1px solid rgba(255, 255, 255, 0.1);
        border-radius: 18px;
        backdrop-filter: blur(10px);
        transition: all 0.4s cubic-bezier(0.175, 0.885, 0.32, 1.275);
        overflow: hidden;
        animation: fadeInUp 0.6s ease-out forwards;
        opacity: 0;
    }
    @keyframes fadeInUp {
        from { opacity: 0; transform: translateY(30px); }
        to { opacity: 1; transform: translateY(0); }
    }
    .book-column:hover {
        transform: translateY(-10px) scale(1.02);
        border-color: rgba(0, 198, 255, 0.5);
        box-shadow: 0 15px 40px rgba(0, 198, 255, 0.2);
        background: rgba(255, 255, 255, 0.06);
    }
    .recommendation-badge {
        position: absolute;
        top: 15px;
        right: 15px;
        width: 36px;
        height: 36px;
        border-radius: 50%;
        background: linear-gradient(135deg, #ff8a00, #e52e71);
        color: white;
        display: flex;
        align-items: center;
        justify-content: center;
        font-size: 1rem;
        font-weight: 700;
        z-index: 10;
        box-shadow: 0 4px 10px rgba(229, 46, 113, 0.4);
        font-family: 'Inter', sans-serif;
    }
    .book-image-area {
        padding: 25px;
        display: flex;
        justify-content: center;
        align-items: center;
        height: 280px;
        background: radial-gradient(circle at center, rgba(255,255,255,0.05) 0%, transparent 70%);
    }
    .book-image-area img {
        max-height: 250px;
        width: auto;
        border-radius: 6px;
        box-shadow: 0 10px 20px rgba(0,0,0,0.5);
        transition: transform 0.4s ease;
    }
    .book-column:hover .book-image-area img {
        transform: scale(1.05) rotate(-1deg);
    }
    .book-info {
        padding: 1.2rem 1.5rem 1.5rem 1.5rem;
        border-top: 1px solid rgba(255, 255, 255, 0.1);
    }
    .premium-title {
        font-size: 1.1rem;
        font-weight: 700;
        color: #ffffff;
        margin-bottom: 0.5rem;
        line-height: 1.4;
        display: -webkit-box;
        -webkit-line-clamp: 2;
        -webkit-box-orient: vertical;
        overflow: hidden;
        text-overflow: ellipsis;
        min-height: 2.8rem;
    }
    .premium-divider {
        width: 30px;
        height: 3px;
        background: linear-gradient(90deg, #0072ff, #00c6ff);
        margin-bottom: 1rem;
        border-radius: 2px;
    }
    .premium-author {
        font-size: 0.9rem;
        color: #00c6ff;
        font-weight: 500;
        white-space: nowrap;
        overflow: hidden;
        text-overflow: ellipsis;
        margin-bottom: 0.25rem;
    }
    .premium-year {
        font-size: 0.8rem;
        color: #888;
        font-weight: 300;
    }

    /* Reading History Custom Styling */
    .history-container {
        background: rgba(255, 255, 255, 0.03);
        border: 1px solid rgba(255, 255, 255, 0.1);
        border-radius: 12px;
        padding: 1rem;
        margin-top: 1rem;
        max-height: 300px;
        overflow-y: auto;
    }
    .history-item {
        display: flex;
        justify-content: space-between;
        padding: 0.75rem;
        margin-bottom: 0.5rem;
        background: rgba(255,255,255,0.02);
        border-radius: 8px;
        border-left: 3px solid #0072ff;
        transition: all 0.2s;
    }
    .history-item:hover { background: rgba(0, 198, 255, 0.05); }
    .history-title { color: #e0e0e0; font-weight: 500; max-width: 70%; }
    .history-rating { font-weight: 700; color: #ff8a00; }
    .history-rating.unrated { color: #555; font-style: italic; font-weight: 400; }

    /* Expander Styling */
    .streamlit-expander {
        background: transparent !important;
        border: none !important;
    }
    .streamlit-expander details {
        background: rgba(255, 255, 255, 0.03) !important;
        border: 1px solid rgba(255, 255, 255, 0.1) !important;
        border-radius: 12px !important;
        backdrop-filter: blur(10px);
    }
    .streamlit-expander summary {
        font-family: 'Playfair Display', serif !important;
        font-size: 1.1rem !important;
        font-weight: 700;
        color: #00c6ff;
        padding: 1rem;
    }
    .streamlit-expander summary span { color: #fff; }
    
    .recommendation-header {
        font-size: 1.5rem;
        font-family: 'Playfair Display', serif;
        color: #ffffff;
        border-left: 4px solid #00c6ff;
        padding-left: 1rem;
        margin: 2rem 0;
    }
    
    .divider-glow {
        height: 1px;
        background: linear-gradient(90deg, transparent, rgba(0, 198, 255, 0.5), transparent);
        margin: 3rem 0;
    }
    </style>
""", unsafe_allow_html=True)

# -------------------------------------------------------------------------
# DATA LOADING (With Spinner)
# -------------------------------------------------------------------------
with st.spinner("Loading the grand library..."):
    final_filtered_df, cosine_sim_df = load_and_prepare_data()

# -------------------------------------------------------------------------
# HERO SECTION
# -------------------------------------------------------------------------
st.markdown("""
    <div class="hero-header">
        <h1 class="hero-title">Lumiere Books</h1>
        <p class="hero-sub">Discover your next literary obsession powered by AI recommendations</p>
    </div>
""", unsafe_allow_html=True)

# -------------------------------------------------------------------------
# HELPER FUNCTIONS
# -------------------------------------------------------------------------

def get_top_similar_books(book_title, n=10):
    if book_title not in cosine_sim_df.index:
        return "⚠️ Book not found in the database."
    
    similar_scores = cosine_sim_df[book_title]
    similar_books = similar_scores.sort_values(ascending=False)[1:n+1]
    return similar_books

def get_user_recommendations(user_id, df, sim_matrix, k=10):
    user_history_all = df[df['userId'] == user_id]['title'].unique().tolist()
    user_history_rated = df[df['userId'] == user_id][['title', 'rating']].sort_values(by='rating', ascending=False)
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
    cols = st.columns(3)
    for idx, book in enumerate(books_list):
        book_info = final_filtered_df[final_filtered_df['title'] == book].iloc[0]
        
        safe_title = str(book).replace('"', '&quot;').replace("'", "&#39;")
        safe_author = str(book_info['Book-Author']).replace('"', '&quot;').replace("'", "&#39;")
        img_url = book_info['Image-URL-L'] if pd.notna(book_info['Image-URL-L']) else "https://via.placeholder.com/150"
        
        animation_delay = (idx % 3) * 0.15
        
        col_idx = idx % 3
        with cols[col_idx]:
            st.markdown(f"""
            <div class="book-column" style="animation-delay: {animation_delay}s;">
                <div class="recommendation-badge">{start_index + idx + 1}</div>
                <div class="book-image-area">
                    <img src="{img_url}" alt="{safe_title}" onerror="this.src='https://via.placeholder.com/150x200/1a1c2e/ffffff?text=No+Cover';">
                </div>
                <div class="book-info">
                    <div class="premium-title" title="{safe_title}">{book}</div>
                    <div class="premium-divider"></div>
                    <div class="premium-author" title="{safe_author}">{book_info['Book-Author']}</div>
                    <div class="premium-year">{int(book_info['Year-Of-Publication']) if pd.notna(book_info['Year-Of-Publication']) else 'N/A'}</div>
                </div>
            </div>
            """, unsafe_allow_html=True)
        
        # Add a divider after every 3 books if there are more to come, handled naturally by columns now.
        # But to ensure vertical spacing on larger screens:
        if idx % 3 == 2 and idx < len(books_list) - 1:
            # Invisible spacer
            st.markdown("<div style='height: 2rem;'></div>", unsafe_allow_html=True)

# -------------------------------------------------------------------------
# TABS LAYOUT
# -------------------------------------------------------------------------

tab1, tab2 = st.tabs(["📚 Book-to-Book", "👤 Personalized"])

# -------------------------------------------------------------------------
# TAB 1: BOOK-TO-BOOK
# -------------------------------------------------------------------------
with tab1:
    col_a, col_b = st.columns([3, 1])
    
    with col_a:
        all_books = sorted(final_filtered_df['title'].unique().tolist())
        book_title = st.selectbox('Choose a book you love:', all_books, index=None, 
                                  placeholder="Type or select a title...", key='book_title')
    
    with col_b:
        num_recommendations = st.number_input('How many suggestions?', min_value=1, max_value=30, value=6, key='num_recs_book')

    if st.button('✨ Discover Similar Books', key='btn_book_recs'):
        if book_title:
            with st.spinner("Finding literary matches..."):
                similar_books = get_top_similar_books(book_title, num_recommendations)
                st.session_state.recommendations = similar_books
                st.session_state.recommended_book = book_title
                st.session_state.recommended_num = num_recommendations
        else:
            st.warning("Please select a book first.")
    
    # Surprise Me Button
    if st.button("🎲 Surprise Me! (Random Book)", key='btn_surprise_book'):
        with st.spinner("Picking a random adventure..."):
            book_title = random.choice(all_books)
            st.session_state.book_title = book_title
            similar_books = get_top_similar_books(book_title, num_recommendations)
            st.session_state.recommendations = similar_books
            st.session_state.recommended_book = book_title
            st.session_state.recommended_num = num_recommendations
            st.rerun()

    if 'recommendations' in st.session_state and st.session_state.recommendations is not None:
        similar_books = st.session_state.recommendations
        rec_book = st.session_state.recommended_book
        rec_num = st.session_state.recommended_num
        
        if isinstance(similar_books, str):
            st.write(similar_books)
        else:
            st.markdown(f"<div class='recommendation-header'>If you loved <strong style='color:#00c6ff'>{rec_book}</strong>, you'll adore these:</div>", unsafe_allow_html=True)
            books_list = similar_books.index.tolist()
            display_book_cards(books_list)
            
            st.markdown("<div class='divider-glow'></div>", unsafe_allow_html=True)
            st.markdown("<p style='text-align:center; color:#666; font-style:italic;'>Happy Reading! 📖</p>", unsafe_allow_html=True)


# -------------------------------------------------------------------------
# TAB 2: USER-SPECIFIC
# -------------------------------------------------------------------------
with tab2:
    st.markdown("<p style='color:#a0a0a0; margin-bottom:1rem;'>Enter your reading profile to get hyper-personalized recommendations based on your unique taste.</p>", unsafe_allow_html=True)
    
    all_user_ids = sorted(final_filtered_df['userId'].unique().tolist())
    
    col1, col2 = st.columns([3, 1])
    
    with col1:
        user_id_input = st.selectbox('Select your User ID:', all_user_ids, index=None, 
                                     placeholder="Find your User ID...", key='user_id_select')
    
    with col2:
        num_user_recs = st.number_input('Number of books:', min_value=1, max_value=30, value=6, key='num_recs_user')
    
    btn_col1, btn_col2 = st.columns(2)
    
    with btn_col1:
        if st.button('🎯 Get My Recommendations', key='btn_user_recs'):
            if user_id_input:
                with st.spinner("Analyzing your reading DNA..."):
                    recommendations, user_history = get_user_recommendations(user_id_input, final_filtered_df, cosine_sim_df, k=num_user_recs)
                    if recommendations is None:
                        st.warning(f"User ID {user_id_input} has no interaction history.")
                        st.session_state.user_recommendations = None
                        st.session_state.user_history_display = None
                    else:
                        st.session_state.user_recommendations = recommendations
                        st.session_state.user_history_display = user_history
                        st.session_state.current_user_id = user_id_input
            else:
                st.warning("Please select a User ID.")

    with btn_col2:
        if st.button("🎲 Random Profile", key='btn_surprise_user'):
            with st.spinner("Selecting a random reader..."):
                random_user = random.choice(all_user_ids)
                st.session_state.user_id_select = random_user
                recommendations, user_history = get_user_recommendations(random_user, final_filtered_df, cosine_sim_df, k=num_user_recs)
                if recommendations:
                    st.session_state.user_recommendations = recommendations
                    st.session_state.user_history_display = user_history
                    st.session_state.current_user_id = random_user
                    st.rerun()

    if 'user_recommendations' in st.session_state and st.session_state.user_recommendations is not None:
        user_id_display = st.session_state.current_user_id
        recommendations = st.session_state.user_recommendations
        user_history = st.session_state.user_history_display
        
        # Custom Reading History Display
        if user_history is not None and len(user_history) > 0:
            with st.expander("📖 View Reading History", expanded=False):
                history_html = "<div class='history-container'>"
                for _, row in user_history.head(15).iterrows():
                    rating = row['rating']
                    rating_class = "" if rating > 0 else "unrated"
                    rating_text = f"⭐ {rating}/10" if rating > 0 else "Interacted"
                    history_html += f"""
                    <div class="history-item">
                        <span class="history-title">{row['title']}</span>
                        <span class="history-rating {rating_class}">{rating_text}</span>
                    </div>
                    """
                history_html += "</div>"
                history_html += "<p style='color:#666; font-size:0.8rem; margin-top:0.5rem;'>* A rating of 0 indicates an interacted but unrated book.</p>"
                st.markdown(history_html, unsafe_allow_html=True)
        
        st.markdown(f"<div class='recommendation-header'>Curated for User <strong style='color:#00c6ff'>#{user_id_display}</strong></div>", unsafe_allow_html=True)
        
        if len(recommendations) > 0:
            display_book_cards(recommendations)
            st.markdown("<div class='divider-glow'></div>", unsafe_allow_html=True)
            st.markdown("<p style='text-align:center; color:#666; font-style:italic;'>Happy Reading! 📖</p>", unsafe_allow_html=True)
        else:
            st.info("No recommendations available for this user at the moment.")
