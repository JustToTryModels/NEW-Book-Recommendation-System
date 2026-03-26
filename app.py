import streamlit as st
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from huggingface_hub import hf_hub_download
import warnings

# --- Page Configuration ---
st.set_page_config(page_title="Lumina Books | Premium Recommendations", layout="wide")
warnings.filterwarnings('ignore')

@st.cache_data
def load_and_prepare_data():
    # Data Loading
    final_filtered_df_path = hf_hub_download(repo_id="IamPradeep/BRS_DATA", filename="final_filtered_df.csv", repo_type="dataset")
    final_filtered_df = pd.read_csv(final_filtered_df_path)

    book_urls_df_path = hf_hub_download(repo_id="IamPradeep/BRS_DATA", filename="Books.csv", repo_type="dataset")
    book_urls_df = pd.read_csv(book_urls_df_path)
    book_urls_df.rename(columns={'Book-Title': 'title'}, inplace=True)

    final_filtered_df = final_filtered_df.merge(book_urls_df, on='title', how='left')

    # Hardcoded URL Fixes
    replacements = {
        'Jacob Have I Loved': 'http://images.amazon.com/images/P/0690040784.01.LZZZZZZZ.jpg',
        'Needful Things': 'http://images.amazon.com/images/P/0451172817.01.LZZZZZZZ.jpg',
        'All Creatures Great and Small': 'http://images.amazon.com/images/P/0312084986.01.LZZZZZZZ.jpg',
        "The Kitchen God's Wife": 'http://images.amazon.com/images/P/1590400356.01.LZZZZZZZ.jpg'
    }
    for title, url in replacements.items():
        final_filtered_df.loc[final_filtered_df['title'] == title, 'Image-URL-L'] = url

    # Similarity Engine
    book_user_mat = final_filtered_df.pivot_table(index='title', columns='userId', values='rating').fillna(0)
    cosine_sim = cosine_similarity(book_user_mat)
    cosine_sim_df = pd.DataFrame(cosine_sim, index=book_user_mat.index, columns=book_user_mat.index)

    return final_filtered_df, cosine_sim_df

final_filtered_df, cosine_sim_df = load_and_prepare_data()

def get_top_similar_books(book_title, n=10):
    if book_title not in cosine_sim_df.index:
        return None
    similar_scores = cosine_sim_df[book_title]
    similar_books = similar_scores.sort_values(ascending=False)[1:n+1]
    return similar_books

# --- Premium Styling ---
st.markdown("""
    <style>
    /* Global Dark Theme Overrides */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;700&display=swap');
    
    .main {
        background-color: #0f0f0f;
        color: #ffffff;
        font-family: 'Inter', sans-serif;
    }
    
    /* Hero Banner */
    .hero-container {
        padding: 60px 0px;
        background: linear-gradient(rgba(0,0,0,0.5), rgba(15,15,15,1)), 
                    url('https://img.freepik.com/premium-vector/bookcase-with-books_182089-197.jpg');
        background-size: cover;
        background-position: center;
        text-align: center;
        border-radius: 20px;
        margin-bottom: 30px;
    }
    
    .hero-title {
        font-size: 55px !important;
        font-weight: 800 !important;
        letter-spacing: -1px;
        margin-bottom: 0px;
        background: -webkit-linear-gradient(#fff, #888);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }

    /* Card Styling */
    .book-card {
        background: #1a1a1a;
        border-radius: 12px;
        padding: 0px;
        margin-bottom: 25px;
        transition: all 0.4s cubic-bezier(0.175, 0.885, 0.32, 1.275);
        border: 1px solid #333;
        overflow: hidden;
        position: relative;
    }
    
    .book-card:hover {
        transform: scale(1.05);
        z-index: 10;
        border-color: #e52e71;
        box-shadow: 0px 10px 30px rgba(0,0,0,0.5);
    }
    
    .cover-img {
        width: 100%;
        height: 380px;
        object-fit: cover;
        transition: 0.3s;
    }

    .card-content {
        padding: 15px;
        background: linear-gradient(0deg, #1a1a1a 0%, rgba(26,26,26,0.8) 100%);
    }

    .book-title-text {
        font-size: 16px;
        font-weight: 700;
        white-space: nowrap;
        overflow: hidden;
        text-overflow: ellipsis;
        color: #fff;
    }

    .book-meta {
        font-size: 12px;
        color: #aaa;
        margin-top: 5px;
    }

    /* Premium Button */
    div.stButton > button {
        width: 100%;
        background: linear-gradient(90deg, #e50914, #ff0101);
        color: white;
        border: none;
        padding: 12px;
        border-radius: 5px;
        font-weight: bold;
        text-transform: uppercase;
        letter-spacing: 1px;
    }
    
    /* Horizontal Rule Replacement */
    .section-divider {
        height: 1px;
        background: linear-gradient(90deg, transparent, #333, transparent);
        margin: 40px 0;
    }

    /* Hide Streamlit components for a cleaner look */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    </style>
""", unsafe_allow_html=True)

# --- Sidebar Controls ---
with st.sidebar:
    st.markdown("## 🎬 Control Center")
    all_books = sorted(final_filtered_df['title'].unique().tolist())
    book_title = st.selectbox('Pick a Story:', all_books, index=None, placeholder="Search your library...")
    num_recommendations = st.slider('Number of Picks', 3, 30, 12)
    
    st.write("---")
    recommend_btn = st.button('Get Recommendations')

# --- Main UI ---
st.markdown("""
    <div class="hero-container">
        <h1 class="hero-title">LUMINA</h1>
        <p style='color: #ccc; font-size: 1.2rem;'>Your next great adventure starts here.</p>
    </div>
""", unsafe_allow_html=True)

if recommend_btn:
    if book_title:
        results = get_top_similar_books(book_title, num_recommendations)
        if results is not None:
            st.markdown(f"### ⚡ Trending because you liked: <span style='color:#e50914'>{book_title}</span>", unsafe_allow_html=True)
            
            # Use 4 columns for a Netflix-style grid
            cols_per_row = 4
            for i in range(0, len(results), cols_per_row):
                cols = st.columns(cols_per_row)
                for j in range(cols_per_row):
                    if i + j < len(results):
                        target_book = results.index[i + j]
                        info = final_filtered_df[final_filtered_df['title'] == target_book].iloc[0]
                        
                        with cols[j]:
                            st.markdown(f"""
                            <div class="book-card">
                                <img class="cover-img" src="{info['Image-URL-L']}">
                                <div class="card-content">
                                    <div class="book-title-text">{target_book}</div>
                                    <div class="book-meta">👤 {info['Book-Author']} • 📅 {info['Year-Of-Publication']}</div>
                                </div>
                            </div>
                            """, unsafe_allow_html=True)
            
            st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)
            st.image('https://github.com/MarpakaPradeepSai/Employee-Churn-Prediction/blob/main/Data/Images%20&%20GIFs/thank-you-33.gif?raw=true', width=200)
        else:
            st.error("Book not found in database.")
    else:
        st.warning("Please select a title from the sidebar first.")
else:
    # Default view when no recommendation is made
    st.markdown("#### 🍿 Recently Added to Library")
    # Show a few random books to fill the space
    sample_books = final_filtered_df.sample(4)
    cols = st.columns(4)
    for idx, (i, row) in enumerate(sample_books.iterrows()):
        with cols[idx]:
             st.markdown(f"""
                <div class="book-card">
                    <img class="cover-img" src="{row['Image-URL-L']}">
                    <div class="card-content">
                        <div class="book-title-text">{row['title']}</div>
                        <div class="book-meta">👤 {row['Book-Author']}</div>
                    </div>
                </div>
                """, unsafe_allow_html=True)
