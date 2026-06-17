import streamlit as st
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from huggingface_hub import hf_hub_download
import warnings

warnings.filterwarnings('ignore')

# Set page configuration for a premium, wide feel
st.set_page_config(page_title="Bookly - Recommendation System", page_icon="📚", layout="wide")

@st.cache_data
def load_and_prepare_data():
    # Load your final filtered dataframe from Hugging Face
    final_filtered_df_path = hf_hub_download(repo_id="IamPradeep/BRS_DATA", filename="final_filtered_df.csv", repo_type="dataset")
    final_filtered_df = pd.read_csv(final_filtered_df_path)

    # Load the dataframe containing book URLs from Hugging Face
    book_urls_df_path = hf_hub_download(repo_id="IamPradeep/BRS_DATA", filename="Books.csv", repo_type="dataset")
    book_urls_df = pd.read_csv(book_urls_df_path)
    book_urls_df.rename(columns={'Book-Title': 'title'}, inplace=True)

    # Fix duplicate titles before merging to prevent row multiplication
    book_urls_df = book_urls_df.drop_duplicates(subset=['title'], keep='first')

    # Merge dataframes
    final_filtered_df = final_filtered_df.merge(book_urls_df[['title', 'Book-Author', 'Year-Of-Publication', 'Image-URL-L']], on='title', how='left')

    # URL Fallback corrections
    urls = {
        'Jacob Have I Loved': 'http://images.amazon.com/images/P/0690040784.01.LZZZZZZZ.jpg',
        'Needful Things': 'http://images.amazon.com/images/P/0451172817.01.LZZZZZZZ.jpg',
        'All Creatures Great and Small': 'http://images.amazon.com/images/P/0312084986.01.LZZZZZZZ.jpg',
        "The Kitchen God's Wife": 'http://images.amazon.com/images/P/1590400356.01.LZZZZZZZ.jpg'
    }
    for title, url in urls.items():
        final_filtered_df.loc[final_filtered_df['title'] == title, 'Image-URL-L'] = url

    # Build similarity matrix using only explicit ratings (>0)
    explicit_ratings_df = final_filtered_df[final_filtered_df['rating'] > 0]
    book_user_mat = explicit_ratings_df.pivot_table(index='title', columns='userId', values='rating').fillna(0)

    cosine_sim = cosine_similarity(book_user_mat)
    cosine_sim_df = pd.DataFrame(cosine_sim, index=book_user_mat.index, columns=book_user_mat.index)

    return final_filtered_df, cosine_sim_df

final_filtered_df, cosine_sim_df = load_and_prepare_data()

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
    """Displays books in a high-fidelity glassmorphic grid layout"""
    # Fallback image if cover art fails to load or is NaN
    fallback_img = "https://images.unsplash.com/photo-1543002588-bfa74002ed7e?w=500&auto=format&fit=crop&q=60"
    
    for i in range(0, len(books_list), 4):
        cols = st.columns(4)
        for j in range(4):
            if i + j < len(books_list):
                book = books_list[i + j]
                book_data = final_filtered_df[final_filtered_df['title'] == book]
                
                if not book_data.empty:
                    book_info = book_data.iloc[0]
                    author = book_info['Book-Author']
                    year = book_info['Year-Of-Publication']
                    img_url = book_info['Image-URL-L'] if pd.notna(book_info['Image-URL-L']) else fallback_img
                else:
                    author, year, img_url = "Unknown Author", "N/A", fallback_img
                
                with cols[j]:
                    st.markdown(f"""
                    <div class="book-card">
                        <div class="rank-badge">#{start_index + i + j + 1}</div>
                        <div class="image-container">
                            <img src="{img_url}" onerror="this.src='{fallback_img}';">
                        </div>
                        <div class="card-content">
                            <div class="book-title" title="{book}">{book}</div>
                            <div class="book-author">{author}</div>
                            <div class="book-year">Released: {year}</div>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)

# -------------------------------------------------------------------------
# STYLED INJECTION (UI Customizations)
# -------------------------------------------------------------------------

st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@300;400;600;700&display=swap');
    
    /* Global Overrides */
    * {
        font-family: 'Plus Jakarta Sans', sans-serif !important;
    }
    
    /* Seamless Header Integration */
    .main-title {
        background: linear-gradient(90deg, #FF4B4B, #FF8364);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
        font-weight: 800;
        font-size: 3rem !important;
        margin-bottom: 0px;
    }
    .main-subtitle {
        text-align: center;
        color: #7A869A;
        font-size: 1.15rem;
        margin-bottom: 2rem;
    }
    
    /* Modern Glassmorphic Book Card Design */
    .book-card {
        background: rgba(255, 255, 255, 0.06);
        border: 1px solid rgba(255, 255, 255, 0.12);
        border-radius: 18px;
        padding: 16px;
        text-align: center;
        position: relative;
        transition: all 0.4s cubic-bezier(0.16, 1, 0.3, 1);
        margin-top: 15px;
        margin-bottom: 15px;
        box-shadow: 0 4px 30px rgba(0, 0, 0, 0.05);
        backdrop-filter: blur(5px);
    }
    
    /* Light/Dark mode resilience helper */
    @media (prefers-color-scheme: light) {
        .book-card {
            background: rgba(0, 0, 0, 0.02);
            border: 1px solid rgba(0, 0, 0, 0.08);
        }
        .card-content .book-title { color: #172B4D !important; }
    }
    
    .book-card:hover {
        transform: translateY(-8px) scale(1.02);
        box-shadow: 0 12px 40px rgba(255, 75, 75, 0.15);
        border-color: rgba(255, 75, 75, 0.4);
    }
    
    .image-container {
        height: 240px;
        display: flex;
        align-items: center;
        justify-content: center;
        border-radius: 12px;
        overflow: hidden;
        margin-bottom: 15px;
    }
    
    .image-container img {
        max-height: 100%;
        max-width: 100%;
        object-fit: contain;
        transition: transform 0.5s ease;
    }
    
    .book-card:hover .image-container img {
        transform: scale(1.08);
    }
    
    .rank-badge {
        position: absolute;
        top: -12px;
        left: 16px;
        background: linear-gradient(135deg, #FF4B4B, #FF8364);
        color: white;
        padding: 4px 14px;
        border-radius: 20px;
        font-size: 0.85rem;
        font-weight: 700;
        box-shadow: 0 4px 10px rgba(255, 75, 75, 0.3);
    }
    
    .card-content {
        text-align: left;
        padding: 0 4px;
    }
    
    .book-title {
        font-weight: 700;
        font-size: 1rem !important;
        color: #F4F5F7;
        margin-bottom: 6px;
        display: -webkit-box;
        -webkit-line-clamp: 2;
        -webkit-box-orient: vertical;
        overflow: hidden;
        height: 2.6rem;
        line-height: 1.3;
    }
    
    .book-author {
        color: #FF8364;
        font-size: 0.85rem;
        font-weight: 500;
        white-space: nowrap;
        overflow: hidden;
        text-overflow: ellipsis;
        margin-bottom: 4px;
    }
    
    .book-year {
        color: #7A869A;
        font-size: 0.75rem;
        font-weight: 600;
        text-transform: uppercase;
    }
    
    /* Header Section Callouts */
    .section-header {
        border-left: 4px solid #FF4B4B;
        padding-left: 12px;
        margin: 25px 0 15px 0;
        font-size: 1.25rem;
        font-weight: 700;
    }
    
    /* Clean Tab Design Adjustments */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
    }
    .stTabs [data-baseweb="tab"] {
        padding: 10px 20px;
        border-radius: 8px;
        font-weight: 600;
    }
    </style>
""", unsafe_allow_html=True)

# -------------------------------------------------------------------------
# APPLICATION LANDING HERO
# -------------------------------------------------------------------------
st.markdown('<h1 class="main-title">Bookly AI</h1>', unsafe_allow_html=True)
st.markdown('<p class="main-subtitle">Next-generation discovery engine powered by intuitive hyper-collaborative modeling</p>', unsafe_allow_html=True)

# Setup UI Core Navigation Tabs
tab1, tab2 = st.tabs(["📚 Item Exploration Matcher", "👤 High-Dimensional User Profiler"])

# -------------------------------------------------------------------------
# TAB 1: ITEM EXPLORATION MATCHER (Book-to-Book)
# -------------------------------------------------------------------------
with tab1:
    st.markdown('<div class="section-header">Discover Latent Book Correlations</div>', unsafe_allow_html=True)
    
    col_input1, col_input2 = st.columns([3, 1])
    with col_input1:
        all_books = sorted(final_filtered_df['title'].unique().tolist())
        book_title = st.selectbox('Target Anchor Book:', all_books, index=None, 
                                  placeholder="Type or select a novel title catalog...", key='book_title')
    with col_input2:
        num_recommendations = st.number_input('Target Deep Match Count:', min_value=1, max_value=40, value=8, key='num_recs_book')
    
    if st.button('Generate Item Vector Connections →', type='primary', key='btn_book_recs'):
        if book_title:
            with st.spinner("Processing structural matrix projections..."):
                similar_books = get_top_similar_books(book_title, num_recommendations)
                
                if isinstance(similar_books, str):
                    st.error(similar_books)
                else:
                    st.markdown(f'<div class="section-header">Top Vector Recommendations Found for: <i>{book_title}</i></div>', unsafe_allow_html=True)
                    display_book_cards(similar_books.index.tolist())
        else:
            st.warning("Please assign a valid anchor title target.")

# -------------------------------------------------------------------------
# TAB 2: HIGH-DIMENSIONAL USER PROFILER (User-Specific)
# -------------------------------------------------------------------------
with tab2:
    st.markdown('<div class="section-header">User Behavioral Recommender Stack</div>', unsafe_allow_html=True)
    
    all_user_ids = sorted(final_filtered_df['userId'].unique().tolist())
    col_u1, col_u2 = st.columns([3, 1])
    
    with col_u1:
        user_id_input = st.selectbox('Target Profile ID:', all_user_ids, index=None, placeholder="Select a historical user footprint...")
    with col_u2:
        num_user_recs = st.number_input('Volume Scale Allocation:', min_value=1, max_value=40, value=8)
        
    if st.button('Synthesize Matrix Inferences →', type='primary', key='btn_user_recs'):
        if user_id_input:
            with st.spinner("Compiling global interest vectors..."):
                recommendations, user_history = get_user_recommendations(user_id_input, final_filtered_df, cosine_sim_df, k=num_user_recs)
                
                if recommendations is None:
                    st.error(f"User ID: {user_id_input} contains insufficient baseline interaction steps.")
                else:
                    # Render historical engagement metrics cleanly
                    with st.expander("📊 Inspect Historical Identity Vectors", expanded=False):
                        history_df = user_history.copy().reset_index(drop=True)
                        history_df.index = history_df.index + 1
                        history_df.columns = ['Interacted Title Match', 'Assigned Explicit Weight']
                        st.dataframe(history_df, use_container_width=True)
                    
                    st.markdown(f'<div class="section-header">Personalized Recommendation Matrix for Profile: User {user_id_input}</div>', unsafe_allow_html=True)
                    if len(recommendations) > 0:
                        display_book_cards(recommendations)
                    else:
                        st.info("System matches are complete. Explicit scoring thresholds resulted in empty array passes.")
        else:
            st.warning("Please define a target runtime user ID framework.")
