import streamlit as st
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from huggingface_hub import hf_hub_download
import warnings
import time
import random

warnings.filterwarnings('ignore')

# Page configuration
st.set_page_config(
    page_title="📚 BookVerse - Discover Your Next Read",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# -------------------------------------------------------------------------
# ENHANCED CSS WITH ANIMATIONS & BEAUTIFUL STYLING
# -------------------------------------------------------------------------
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Playfair+Display:wght@400;500;600;700;800;900&family=Inter:wght@300;400;500;600;700&family=Cormorant+Garamond:wght@300;400;500;600;700&display=swap');
    
    /* Global Styles */
    .stApp {
        background: linear-gradient(135deg, #0f0c29 0%, #302b63 50%, #24243e 100%);
    }
    
    .main .block-container {
        padding-top: 2rem;
        max-width: 1400px;
    }
    
    /* Hide Streamlit branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    
    /* Custom Title */
    .hero-title {
        font-family: 'Playfair Display', serif !important;
        font-size: 4.5rem;
        font-weight: 900;
        text-align: center;
        background: linear-gradient(135deg, #f6d365 0%, #fda085 50%, #ff9a9e 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        margin-bottom: 0.5rem;
        letter-spacing: -1px;
        animation: fadeInDown 1s ease-out;
        text-shadow: 0 0 30px rgba(246, 211, 101, 0.3);
    }
    
    .hero-subtitle {
        font-family: 'Inter', sans-serif !important;
        font-size: 1.25rem;
        text-align: center;
        color: #e0e0e0;
        margin-bottom: 2.5rem;
        font-weight: 300;
        letter-spacing: 2px;
        animation: fadeInUp 1s ease-out 0.3s both;
    }
    
    .hero-tagline {
        font-family: 'Cormorant Garamond', serif !important;
        font-size: 1.1rem;
        text-align: center;
        color: #b8b8b8;
        font-style: italic;
        margin-bottom: 2rem;
    }
    
    /* Animations */
    @keyframes fadeInDown {
        from { opacity: 0; transform: translateY(-30px); }
        to { opacity: 1; transform: translateY(0); }
    }
    
    @keyframes fadeInUp {
        from { opacity: 0; transform: translateY(30px); }
        to { opacity: 1; transform: translateY(0); }
    }
    
    @keyframes slideIn {
        from { opacity: 0; transform: scale(0.9); }
        to { opacity: 1; transform: scale(1); }
    }
    
    @keyframes pulse {
        0%, 100% { transform: scale(1); }
        50% { transform: scale(1.05); }
    }
    
    @keyframes shimmer {
        0% { background-position: -200% center; }
        100% { background-position: 200% center; }
    }
    
    @keyframes float {
        0%, 100% { transform: translateY(0px); }
        50% { transform: translateY(-10px); }
    }
    
    @keyframes glow {
        0%, 100% { box-shadow: 0 0 20px rgba(246, 211, 101, 0.3); }
        50% { box-shadow: 0 0 40px rgba(246, 211, 101, 0.6); }
    }
    
    /* Tabs Styling */
    .stTabs {
        background: transparent !important;
    }
    
    .stTabs [data-baseweb="tab-list"] {
        gap: 12px;
        justify-content: center;
        background: rgba(255, 255, 255, 0.03);
        padding: 12px;
        border-radius: 50px;
        backdrop-filter: blur(20px);
        border: 1px solid rgba(255, 255, 255, 0.08);
        box-shadow: 0 8px 32px rgba(0, 0, 0, 0.3);
    }
    
    .stTabs [data-baseweb="tab"] {
        font-family: 'Inter', sans-serif !important;
        height: 50px;
        background: transparent !important;
        border-radius: 30px !important;
        padding: 0 28px !important;
        font-size: 15px !important;
        font-weight: 500 !important;
        color: #b8b8b8 !important;
        border: none !important;
        transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1) !important;
    }
    
    .stTabs [data-baseweb="tab"]:hover {
        background: rgba(255, 255, 255, 0.08) !important;
        color: #fff !important;
        transform: translateY(-2px);
    }
    
    .stTabs [aria-selected="true"] {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%) !important;
        color: white !important;
        box-shadow: 0 8px 20px rgba(102, 126, 234, 0.4) !important;
        font-weight: 600 !important;
    }
    
    .stTabs [data-baseweb="tab-border"] { display: none !important; }
    .stTabs [data-baseweb="tab-highlight"] { display: none !important; }
    
    /* Select Box Styling */
    .stSelectbox > div > div {
        background: rgba(255, 255, 255, 0.05) !important;
        border: 1px solid rgba(255, 255, 255, 0.1) !important;
        border-radius: 12px !important;
        color: white !important;
        backdrop-filter: blur(10px);
        transition: all 0.3s ease !important;
    }
    
    .stSelectbox > div > div:hover {
        border-color: rgba(246, 211, 101, 0.5) !important;
        box-shadow: 0 0 20px rgba(246, 211, 101, 0.1) !important;
    }
    
    .stSelectbox label, .stNumberInput label {
        font-family: 'Inter', sans-serif !important;
        color: #e0e0e0 !important;
        font-weight: 500 !important;
        font-size: 0.95rem !important;
    }
    
    /* Number Input */
    .stNumberInput > div > div > input {
        background: rgba(255, 255, 255, 0.05) !important;
        border: 1px solid rgba(255, 255, 255, 0.1) !important;
        border-radius: 12px !important;
        color: white !important;
        font-family: 'Inter', sans-serif !important;
    }
    
    /* Button Styling */
    .stButton > button {
        font-family: 'Inter', sans-serif !important;
        font-size: 16px !important;
        font-weight: 600 !important;
        background: linear-gradient(135deg, #f6d365 0%, #fda085 100%) !important;
        color: #1a1a2e !important;
        border: none !important;
        border-radius: 30px !important;
        padding: 14px 36px !important;
        cursor: pointer !important;
        transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1) !important;
        width: 100%;
        box-shadow: 0 8px 25px rgba(246, 211, 101, 0.3) !important;
        letter-spacing: 0.5px;
        text-transform: none !important;
    }
    
    .stButton > button:hover {
        transform: translateY(-3px) scale(1.02) !important;
        box-shadow: 0 12px 35px rgba(246, 211, 101, 0.5) !important;
        background: linear-gradient(135deg, #fda085 0%, #f6d365 100%) !important;
    }
    
    .stButton > button:active {
        transform: translateY(-1px) scale(0.99) !important;
    }
    
    /* Book Card Styling */
    .book-card {
        position: relative;
        background: linear-gradient(145deg, rgba(255,255,255,0.07) 0%, rgba(255,255,255,0.02) 100%);
        backdrop-filter: blur(20px);
        border: 1px solid rgba(255, 255, 255, 0.1);
        border-radius: 20px;
        padding: 0;
        margin-top: 35px;
        margin-bottom: 20px;
        transition: all 0.4s cubic-bezier(0.4, 0, 0.2, 1);
        overflow: hidden;
        animation: slideIn 0.6s ease-out backwards;
        box-shadow: 0 10px 40px rgba(0, 0, 0, 0.3);
    }
    
    .book-card:hover {
        transform: translateY(-10px) scale(1.02);
        box-shadow: 0 25px 60px rgba(0, 0, 0, 0.5), 0 0 30px rgba(246, 211, 101, 0.2);
        border-color: rgba(246, 211, 101, 0.4);
    }
    
    .book-card::before {
        content: '';
        position: absolute;
        top: 0;
        left: -100%;
        width: 100%;
        height: 100%;
        background: linear-gradient(90deg, transparent, rgba(255,255,255,0.05), transparent);
        transition: left 0.6s;
    }
    
    .book-card:hover::before {
        left: 100%;
    }
    
    .book-image-wrapper {
        background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
        padding: 30px 20px 25px 20px;
        display: flex;
        justify-content: center;
        align-items: center;
        min-height: 280px;
        position: relative;
        overflow: hidden;
    }
    
    .book-image-wrapper::after {
        content: '';
        position: absolute;
        bottom: 0;
        left: 0;
        right: 0;
        height: 60px;
        background: linear-gradient(to top, rgba(30, 30, 50, 0.95), transparent);
    }
    
    .book-image-wrapper img {
        max-height: 230px;
        max-width: 90%;
        object-fit: contain;
        filter: drop-shadow(0 10px 25px rgba(0,0,0,0.3));
        transition: transform 0.4s ease;
    }
    
    .book-card:hover .book-image-wrapper img {
        transform: scale(1.08) rotate(-2deg);
    }
    
    .book-details {
        background: linear-gradient(180deg, rgba(30, 30, 50, 0.95) 0%, rgba(20, 20, 35, 0.98) 100%);
        padding: 18px 16px;
        text-align: center;
    }
    
    .book-title {
        font-family: 'Playfair Display', serif !important;
        font-size: 1.05rem;
        font-weight: 700;
        color: #f6d365;
        margin-bottom: 6px;
        line-height: 1.3;
        display: -webkit-box;
        -webkit-line-clamp: 2;
        -webkit-box-orient: vertical;
        overflow: hidden;
        min-height: 2.6em;
    }
    
    .book-divider {
        width: 40px;
        height: 2px;
        background: linear-gradient(90deg, #f6d365, #fda085);
        margin: 8px auto;
        border-radius: 2px;
    }
    
    .book-author {
        font-family: 'Cormorant Garamond', serif !important;
        font-size: 0.95rem;
        color: #d0d0d0;
        font-style: italic;
        margin-bottom: 6px;
        overflow: hidden;
        text-overflow: ellipsis;
        white-space: nowrap;
    }
    
    .book-year {
        font-family: 'Inter', sans-serif !important;
        font-size: 0.75rem;
        color: #888;
        text-transform: uppercase;
        letter-spacing: 1.5px;
        font-weight: 500;
    }
    
    .book-rank-badge {
        position: absolute;
        top: -18px;
        left: 50%;
        transform: translateX(-50%);
        width: 50px;
        height: 50px;
        border-radius: 50%;
        background: linear-gradient(135deg, #f6d365 0%, #fda085 100%);
        color: #1a1a2e;
        display: flex;
        align-items: center;
        justify-content: center;
        font-size: 1.2rem;
        font-weight: 800;
        z-index: 10;
        box-shadow: 0 6px 20px rgba(246, 211, 101, 0.4);
        border: 3px solid rgba(255, 255, 255, 0.2);
        font-family: 'Inter', sans-serif !important;
    }
    
    .book-rank-badge.top-3 {
        background: linear-gradient(135deg, #ffd700 0%, #ff8c00 100%);
        animation: glow 2s ease-in-out infinite;
    }
    
    /* Section Headers */
    .section-header {
        font-family: 'Playfair Display', serif !important;
        font-size: 1.8rem;
        font-weight: 700;
        color: #f6d365;
        text-align: center;
        margin: 2.5rem 0 1.5rem 0;
        position: relative;
    }
    
    .section-header::after {
        content: '';
        display: block;
        width: 80px;
        height: 3px;
        background: linear-gradient(90deg, #f6d365, #fda085);
        margin: 12px auto 0;
        border-radius: 3px;
    }
    
    .recommendation-header {
        font-family: 'Inter', sans-serif !important;
        font-size: 1.15rem;
        color: #e0e0e0;
        padding: 16px 24px;
        background: linear-gradient(135deg, rgba(102, 126, 234, 0.15) 0%, rgba(118, 75, 162, 0.15) 100%);
        border-left: 4px solid #f6d365;
        border-radius: 12px;
        margin: 1.5rem 0;
        backdrop-filter: blur(10px);
        border: 1px solid rgba(255, 255, 255, 0.05);
    }
    
    .recommendation-header strong {
        color: #f6d365;
        font-weight: 700;
    }
    
    /* Stats Cards */
    .stat-card {
        background: linear-gradient(135deg, rgba(255,255,255,0.05) 0%, rgba(255,255,255,0.02) 100%);
        backdrop-filter: blur(20px);
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 16px;
        padding: 20px;
        text-align: center;
        transition: all 0.3s ease;
    }
    
    .stat-card:hover {
        transform: translateY(-5px);
        border-color: rgba(246, 211, 101, 0.3);
        box-shadow: 0 10px 30px rgba(0, 0, 0, 0.3);
    }
    
    .stat-number {
        font-family: 'Playfair Display', serif !important;
        font-size: 2.5rem;
        font-weight: 800;
        background: linear-gradient(135deg, #f6d365 0%, #fda085 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
    }
    
    .stat-label {
        font-family: 'Inter', sans-serif !important;
        font-size: 0.85rem;
        color: #b8b8b8;
        text-transform: uppercase;
        letter-spacing: 1.5px;
        margin-top: 5px;
    }
    
    /* Expander */
    [data-testid="stExpander"] details {
        background: rgba(255, 255, 255, 0.03) !important;
        border: 1px solid rgba(255, 255, 255, 0.08) !important;
        border-radius: 12px !important;
        backdrop-filter: blur(10px);
    }
    
    [data-testid="stExpander"] summary {
        background: linear-gradient(135deg, rgba(102, 126, 234, 0.1) 0%, rgba(118, 75, 162, 0.1) 100%) !important;
        border-radius: 12px !important;
        color: #f6d365 !important;
        font-family: 'Inter', sans-serif !important;
        font-weight: 600 !important;
    }
    
    [data-testid="stExpander"] summary:hover {
        background: linear-gradient(135deg, rgba(102, 126, 234, 0.2) 0%, rgba(118, 75, 162, 0.2) 100%) !important;
    }
    
    /* Dataframe */
    .stDataFrame {
        background: rgba(255, 255, 255, 0.03) !important;
        border-radius: 12px !important;
        overflow: hidden;
    }
    
    /* Hero Image Container */
    .hero-image {
        border-radius: 24px;
        overflow: hidden;
        box-shadow: 0 20px 60px rgba(0, 0, 0, 0.4);
        margin: 2rem 0;
        position: relative;
    }
    
    .hero-image::after {
        content: '';
        position: absolute;
        inset: 0;
        background: linear-gradient(180deg, transparent 60%, rgba(15, 12, 41, 0.4) 100%);
        pointer-events: none;
    }
    
    /* Warning & Info Messages */
    .stAlert {
        background: rgba(255, 255, 255, 0.05) !important;
        border-radius: 12px !important;
        backdrop-filter: blur(10px);
        border: 1px solid rgba(255, 255, 255, 0.1) !important;
        color: #e0e0e0 !important;
        font-family: 'Inter', sans-serif !important;
    }
    
    /* Footer */
    .custom-footer {
        text-align: center;
        padding: 3rem 0 1rem 0;
        color: #888;
        font-family: 'Inter', sans-serif;
        font-size: 0.9rem;
    }
    
    .custom-footer a {
        color: #f6d365;
        text-decoration: none;
    }
    
    /* Loading Spinner */
    .stSpinner > div {
        border-color: #f6d365 !important;
    }
    
    /* Caption */
    .stCaption {
        color: #b8b8b8 !important;
        font-family: 'Inter', sans-serif !important;
        font-style: italic;
    }
    
    /* Welcome message */
    .welcome-box {
        background: linear-gradient(135deg, rgba(102, 126, 234, 0.1) 0%, rgba(118, 75, 162, 0.1) 100%);
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 20px;
        padding: 30px;
        margin: 2rem 0;
        backdrop-filter: blur(20px);
        text-align: center;
    }
    
    .welcome-box p {
        color: #e0e0e0;
        font-family: 'Inter', sans-serif;
        font-size: 1.05rem;
        line-height: 1.7;
        margin: 0;
    }
    
    /* Subtle scrollbar */
    ::-webkit-scrollbar {
        width: 8px;
        height: 8px;
    }
    ::-webkit-scrollbar-track {
        background: rgba(0,0,0,0.2);
    }
    ::-webkit-scrollbar-thumb {
        background: linear-gradient(135deg, #f6d365, #fda085);
        border-radius: 4px;
    }
    ::-webkit-scrollbar-thumb:hover {
        background: linear-gradient(135deg, #fda085, #f6d365);
    }
</style>
""", unsafe_allow_html=True)

# -------------------------------------------------------------------------
# DATA LOADING
# -------------------------------------------------------------------------
@st.cache_data
def load_and_prepare_data():
    final_filtered_df_path = hf_hub_download(repo_id="IamPradeep/BRS_DATA", filename="final_filtered_df.csv", repo_type="dataset")
    final_filtered_df = pd.read_csv(final_filtered_df_path)
    
    book_urls_df_path = hf_hub_download(repo_id="IamPradeep/BRS_DATA", filename="Books.csv", repo_type="dataset")
    book_urls_df = pd.read_csv(book_urls_df_path)
    book_urls_df.rename(columns={'Book-Title': 'title'}, inplace=True)
    book_urls_df = book_urls_df.drop_duplicates(subset=['title'], keep='first')
    
    final_filtered_df = final_filtered_df.merge(book_urls_df[['title', 'Book-Author', 'Year-Of-Publication', 'Image-URL-L']], on='title', how='left')
    
    # Fix broken URLs
    url_replacements = {
        'Jacob Have I Loved': 'http://images.amazon.com/images/P/0690040784.01.LZZZZZZZ.jpg',
        'Needful Things': 'http://images.amazon.com/images/P/0451172817.01.LZZZZZZZ.jpg',
        'All Creatures Great and Small': 'http://images.amazon.com/images/P/0312084986.01.LZZZZZZZ.jpg',
        "The Kitchen God's Wife": 'http://images.amazon.com/images/P/1590400356.01.LZZZZZZZ.jpg'
    }
    
    for book_title, url in url_replacements.items():
        final_filtered_df.loc[final_filtered_df['title'] == book_title, 'Image-URL-L'] = url
    
    # Build similarity matrix
    explicit_ratings_df = final_filtered_df[final_filtered_df['rating'] > 0]
    book_user_mat = explicit_ratings_df.pivot_table(index='title', columns='userId', values='rating').fillna(0)
    cosine_sim = cosine_similarity(book_user_mat)
    cosine_sim_df = pd.DataFrame(cosine_sim, index=book_user_mat.index, columns=book_user_mat.index)
    
    return final_filtered_df, cosine_sim_df

# Loading state
with st.spinner('✨ Preparing your literary journey...'):
    final_filtered_df, cosine_sim_df = load_and_prepare_data()

# -------------------------------------------------------------------------
# HELPER FUNCTIONS
# -------------------------------------------------------------------------
def get_top_similar_books(book_title, n=10):
    if book_title not in cosine_sim_df.index:
        return "⚠️ Book not found in the database."
    similar_scores = cosine_sim_df[book_title]
    return similar_scores.sort_values(ascending=False)[1:n+1]

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
    for i in range(0, len(books_list), 3):
        cols = st.columns(3, gap="medium")
        for j in range(3):
            if i + j < len(books_list):
                book = books_list[i + j]
                book_matches = final_filtered_df[final_filtered_df['title'] == book]
                if len(book_matches) == 0:
                    continue
                book_info = book_matches.iloc[0]
                
                rank = start_index + i + j + 1
                rank_class = "top-3" if rank <= 3 else ""
                
                # Animation delay
                delay = (i + j) * 0.1
                
                with cols[j]:
                    st.markdown(f"""
                    <div class='book-card' style='animation-delay: {delay}s;'>
                        <div class='book-rank-badge {rank_class}'>{rank}</div>
                        <div class='book-image-wrapper'>
                            <img src='{book_info['Image-URL-L']}' alt='{book}'>
                        </div>
                        <div class='book-details'>
                            <div class='book-title'>{book}</div>
                            <div class='book-divider'></div>
                            <div class='book-author'>{book_info['Book-Author']}</div>
                            <div class='book-year'>{book_info['Year-Of-Publication']}</div>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)

# -------------------------------------------------------------------------
# HERO SECTION
# -------------------------------------------------------------------------
st.markdown("""
    <h1 class='hero-title'>📚 BookVerse</h1>
    <p class='hero-subtitle'>YOUR PERSONAL LIBRARY COMPANION</p>
    <p class='hero-tagline'>"A reader lives a thousand lives before he dies. The man who never reads lives only one."</p>
""", unsafe_allow_html=True)

# Stats Section
st.markdown("<br>", unsafe_allow_html=True)
col1, col2, col3, col4 = st.columns(4)

total_books = final_filtered_df['title'].nunique()
total_users = final_filtered_df['userId'].nunique()
total_ratings = len(final_filtered_df[final_filtered_df['rating'] > 0])
avg_rating = final_filtered_df[final_filtered_df['rating'] > 0]['rating'].mean()

with col1:
    st.markdown(f"""
    <div class='stat-card'>
        <div class='stat-number'>{total_books:,}</div>
        <div class='stat-label'>Books</div>
    </div>
    """, unsafe_allow_html=True)

with col2:
    st.markdown(f"""
    <div class='stat-card'>
        <div class='stat-number'>{total_users:,}</div>
        <div class='stat-label'>Readers</div>
    </div>
    """, unsafe_allow_html=True)

with col3:
    st.markdown(f"""
    <div class='stat-card'>
        <div class='stat-number'>{total_ratings:,}</div>
        <div class='stat-label'>Ratings</div>
    </div>
    """, unsafe_allow_html=True)

with col4:
    st.markdown(f"""
    <div class='stat-card'>
        <div class='stat-number'>{avg_rating:.1f}★</div>
        <div class='stat-label'>Avg Rating</div>
    </div>
    """, unsafe_allow_html=True)

st.markdown("<br><br>", unsafe_allow_html=True)

# Welcome Box
st.markdown("""
    <div class='welcome-box'>
        <p>🌟 Discover books tailored to your taste using our intelligent recommendation engine powered by <strong>collaborative filtering</strong> and <strong>cosine similarity</strong>. Choose your adventure below!</p>
    </div>
""", unsafe_allow_html=True)

# -------------------------------------------------------------------------
# TABS
# -------------------------------------------------------------------------
tab1, tab2 = st.tabs(["📖  Book-to-Book  ", "👤  Personalized for You  "])

# TAB 1: BOOK-TO-BOOK RECOMMENDATIONS
with tab1:
    st.markdown("<h2 class='section-header'>Find Books Similar to Your Favorite</h2>", unsafe_allow_html=True)
    st.markdown("<p style='text-align:center; color:#b8b8b8; font-family:Inter;'>Select a book you've enjoyed, and we'll suggest similar titles you might love.</p>", unsafe_allow_html=True)
    st.markdown("<br>", unsafe_allow_html=True)
    
    all_books = sorted(final_filtered_df['title'].unique().tolist())
    
    col1, col2 = st.columns([3, 1])
    with col1:
        book_title = st.selectbox('🔍 Choose a book:', all_books, index=None, 
                                  placeholder="Type or select a book title...", key='book_title')
    with col2:
        num_recommendations = st.number_input('📊 How many?', 
                                             min_value=1, max_value=50, value=10, key='num_recs_book')
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    if 'recommendations' not in st.session_state:
        st.session_state.recommendations = None
    if 'recommended_book' not in st.session_state:
        st.session_state.recommended_book = None
    if 'recommended_num' not in st.session_state:
        st.session_state.recommended_num = None
    
    if st.button('✨ Discover Similar Books', key='btn_book_recs', use_container_width=False):
        if book_title:
            with st.spinner('🔮 Curating your recommendations...'):
                time.sleep(0.8)
                similar_books = get_top_similar_books(book_title, num_recommendations)
                st.session_state.recommendations = similar_books
                st.session_state.recommended_book = book_title
                st.session_state.recommended_num = num_recommendations
        else:
            st.warning("⚠️ Please select a book title to get started.")
    
    if st.session_state.recommendations is not None:
        similar_books = st.session_state.recommendations
        rec_book = st.session_state.recommended_book
        rec_num = st.session_state.recommended_num
        
        if isinstance(similar_books, str):
            st.error(similar_books)
        else:
            st.markdown(f"""
            <div class='recommendation-header'>
                🎯 Top <strong>{rec_num}</strong> recommendations for readers who loved <strong>"{rec_book}"</strong>
            </div>
            """, unsafe_allow_html=True)
            
            books_list = similar_books.index.tolist()
            display_book_cards(books_list)
            
            st.markdown("<br><br>", unsafe_allow_html=True)
            st.balloons()

# TAB 2: USER-SPECIFIC RECOMMENDATIONS
with tab2:
    st.markdown("<h2 class='section-header'>Personalized Recommendations Just for You</h2>", unsafe_allow_html=True)
    st.markdown("<p style='text-align:center; color:#b8b8b8; font-family:Inter;'>Enter your User ID to get book suggestions based on your reading history.</p>", unsafe_allow_html=True)
    st.markdown("<br>", unsafe_allow_html=True)
    
    all_user_ids = sorted(final_filtered_df['userId'].unique().tolist())
    
    col1, col2 = st.columns([3, 1])
    with col1:
        user_id_input = st.selectbox('👤 Select your User ID:', all_user_ids, 
                                     index=None, placeholder="Choose a User ID...", key='user_id_select')
    with col2:
        num_user_recs = st.number_input('📊 How many?', 
                                       min_value=1, max_value=50, value=10, key='num_recs_user')
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    if 'user_recommendations' not in st.session_state:
        st.session_state.user_recommendations = None
    if 'user_history_display' not in st.session_state:
        st.session_state.user_history_display = None
    if 'current_user_id' not in st.session_state:
        st.session_state.current_user_id = None
    
    if st.button('🎁 Get My Personalized Picks', key='btn_user_recs'):
        if user_id_input:
            with st.spinner('🎨 Crafting your personalized experience...'):
                time.sleep(0.8)
                recommendations, user_history = get_user_recommendations(
                    user_id_input, final_filtered_df, cosine_sim_df, k=num_user_recs)
                
                if recommendations is None:
                    st.warning(f"⚠️ User ID {user_id_input} has no reading history in our database.")
                    st.session_state.user_recommendations = None
                    st.session_state.user_history_display = None
                else:
                    st.session_state.user_recommendations = recommendations
                    st.session_state.user_history_display = user_history
                    st.session_state.current_user_id = user_id_input
        else:
            st.warning("⚠️ Please select a User ID to continue.")
    
    if st.session_state.user_recommendations is not None:
        user_id_display = st.session_state.current_user_id
        recommendations = st.session_state.user_recommendations
        user_history = st.session_state.user_history_display
        
        if user_history is not None and len(user_history) > 0:
            with st.expander("📚  View Your Reading History"):
                history_df = user_history.copy()
                history_df.reset_index(drop=True, inplace=True)
                history_df.index = history_df.index + 1
                history_df.columns = ['📖 Book Title', '⭐ Rating']
                st.dataframe(history_df, use_container_width=True, height=300)
                st.caption("ℹ️ *A rating of 0 indicates an interacted but unrated book.*")
        
        st.markdown("<br>", unsafe_allow_html=True)
        
        if len(recommendations) > 0:
            st.markdown(f"""
            <div class='recommendation-header'>
                🎯 Top <strong>{len(recommendations)}</strong> Personalized Picks for User <strong>#{user_id_display}</strong>
            </div>
            """, unsafe_allow_html=True)
            
            display_book_cards(recommendations)
            
            st.markdown("<br><br>", unsafe_allow_html=True)
            st.balloons()
        else:
            st.info("📚 No recommendations available at the moment. Try exploring more books!")

# Footer
st.markdown("""
    <div class='custom-footer'>
        <p>✨ Crafted with ❤️ for book lovers everywhere ✨</p>
        <p style='font-size: 0.8rem; margin-top: 10px;'>Powered by Collaborative Filtering & Cosine Similarity</p>
    </div>
""", unsafe_allow_html=True)
