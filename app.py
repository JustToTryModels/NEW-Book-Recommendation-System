import streamlit as st
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from huggingface_hub import hf_hub_download
import warnings

warnings.filterwarnings('ignore')

st.set_page_config(page_title="Book Recommendation System", page_icon="📚", layout="wide")

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

    # Create the book-user matrix
    book_user_mat = final_filtered_df.pivot_table(index='title', columns='userId', values='rating').fillna(0)

    # Calculate the cosine similarity matrix
    cosine_sim = cosine_similarity(book_user_mat)
    cosine_sim_df = pd.DataFrame(cosine_sim, index=book_user_mat.index, columns=book_user_mat.index)

    return final_filtered_df, cosine_sim_df

final_filtered_df, cosine_sim_df = load_and_prepare_data()

def get_top_similar_books(book_title, n=10):
    if book_title not in cosine_sim_df.index:
        return "⚠️ Book not found in the database."
    
    similar_scores = cosine_sim_df[book_title]
    similar_books = similar_scores.sort_values(ascending=False)[1:n+1]
    return similar_books

# Premium CSS Styling
st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Cormorant+Garamond:wght@400;500;600;700&family=Montserrat:wght@300;400;500;600&display=swap');
    
    /* Main Background */
    .stApp {
        background: linear-gradient(135deg, #1a1a2e 0%, #16213e 50%, #0f3460 100%);
    }
    
    /* Global Font Styling */
    html, body, [class*="css"], [class*="st-"], h1, h2, h3, h4, h5, h6, p, div, span, label {
        font-family: 'Montserrat', sans-serif !important;
        color: #e8e8e8;
    }
    
    /* Premium Header */
    .premium-header {
        text-align: center;
        padding: 40px 20px;
        background: linear-gradient(135deg, rgba(212, 175, 55, 0.1) 0%, rgba(255, 215, 0, 0.05) 100%);
        border-radius: 20px;
        margin-bottom: 30px;
        border: 1px solid rgba(212, 175, 55, 0.3);
        box-shadow: 0 10px 40px rgba(0, 0, 0, 0.3);
    }
    
    .premium-title {
        font-family: 'Cormorant Garamond', serif !important;
        font-size: 52px;
        font-weight: 700;
        background: linear-gradient(135deg, #d4af37 0%, #f4e5b0 50%, #d4af37 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        letter-spacing: 3px;
        margin-bottom: 10px;
        text-shadow: 0 0 40px rgba(212, 175, 55, 0.3);
    }
    
    .premium-subtitle {
        font-family: 'Montserrat', sans-serif !important;
        font-size: 18px;
        color: #b8b8b8;
        letter-spacing: 4px;
        text-transform: uppercase;
        font-weight: 300;
    }
    
    .tagline {
        font-family: 'Cormorant Garamond', serif !important;
        font-size: 24px;
        color: #d4af37;
        margin-top: 25px;
        font-style: italic;
        font-weight: 500;
    }
    
    /* Decorative Divider */
    .divider {
        display: flex;
        align-items: center;
        justify-content: center;
        margin: 30px 0;
    }
    
    .divider-line {
        height: 1px;
        width: 100px;
        background: linear-gradient(90deg, transparent, #d4af37, transparent);
    }
    
    .divider-icon {
        margin: 0 20px;
        font-size: 24px;
    }
    
    /* Input Styling */
    .stSelectbox > div > div {
        background: rgba(255, 255, 255, 0.05) !important;
        border: 1px solid rgba(212, 175, 55, 0.3) !important;
        border-radius: 12px !important;
        color: #e8e8e8 !important;
    }
    
    .stSelectbox > div > div:hover {
        border: 1px solid rgba(212, 175, 55, 0.6) !important;
    }
    
    .stNumberInput > div > div > input {
        background: rgba(255, 255, 255, 0.05) !important;
        border: 1px solid rgba(212, 175, 55, 0.3) !important;
        border-radius: 12px !important;
        color: #e8e8e8 !important;
    }
    
    /* Premium Button */
    .stButton > button {
        font-family: 'Montserrat', sans-serif !important;
        font-size: 16px;
        font-weight: 600;
        background: linear-gradient(135deg, #d4af37 0%, #aa8c2c 100%);
        color: #1a1a2e !important;
        border: none;
        border-radius: 50px;
        padding: 15px 40px;
        letter-spacing: 2px;
        text-transform: uppercase;
        cursor: pointer;
        transition: all 0.4s cubic-bezier(0.175, 0.885, 0.32, 1.275);
        box-shadow: 0 10px 30px rgba(212, 175, 55, 0.3);
    }
    
    .stButton > button:hover {
        transform: translateY(-3px) scale(1.02);
        box-shadow: 0 15px 40px rgba(212, 175, 55, 0.4);
        background: linear-gradient(135deg, #f4e5b0 0%, #d4af37 100%);
        color: #1a1a2e !important;
    }
    
    .stButton > button:active {
        transform: translateY(-1px) scale(1.01);
    }
    
    /* Book Card - Premium Glassmorphism */
    .book-card {
        position: relative;
        padding: 25px;
        background: linear-gradient(135deg, rgba(255, 255, 255, 0.1) 0%, rgba(255, 255, 255, 0.05) 100%);
        backdrop-filter: blur(10px);
        -webkit-backdrop-filter: blur(10px);
        border: 1px solid rgba(212, 175, 55, 0.2);
        border-radius: 20px;
        box-shadow: 0 15px 35px rgba(0, 0, 0, 0.3), inset 0 1px 0 rgba(255, 255, 255, 0.1);
        margin-bottom: 20px;
        transition: all 0.4s cubic-bezier(0.175, 0.885, 0.32, 1.275);
        overflow: hidden;
    }
    
    .book-card::before {
        content: '';
        position: absolute;
        top: 0;
        left: 0;
        right: 0;
        height: 3px;
        background: linear-gradient(90deg, #d4af37, #f4e5b0, #d4af37);
        opacity: 0;
        transition: opacity 0.4s ease;
    }
    
    .book-card:hover {
        transform: translateY(-10px);
        box-shadow: 0 25px 50px rgba(0, 0, 0, 0.4), 0 0 30px rgba(212, 175, 55, 0.2);
        border-color: rgba(212, 175, 55, 0.5);
    }
    
    .book-card:hover::before {
        opacity: 1;
    }
    
    /* Book Rank Badge */
    .rank-badge {
        position: absolute;
        top: 15px;
        left: 15px;
        width: 35px;
        height: 35px;
        background: linear-gradient(135deg, #d4af37 0%, #aa8c2c 100%);
        border-radius: 50%;
        display: flex;
        align-items: center;
        justify-content: center;
        font-size: 14px;
        font-weight: 700;
        color: #1a1a2e;
        box-shadow: 0 4px 15px rgba(212, 175, 55, 0.4);
        z-index: 10;
    }
    
    /* Book Image */
    .book-image-container {
        position: relative;
        display: flex;
        justify-content: center;
        margin-bottom: 20px;
        padding-top: 10px;
    }
    
    .book-image {
        height: 280px;
        width: auto;
        max-width: 100%;
        object-fit: contain;
        border-radius: 10px;
        box-shadow: 0 10px 30px rgba(0, 0, 0, 0.5);
        transition: transform 0.4s ease, box-shadow 0.4s ease;
    }
    
    .book-card:hover .book-image {
        transform: scale(1.03);
        box-shadow: 0 15px 40px rgba(0, 0, 0, 0.6);
    }
    
    /* Book Info */
    .book-info {
        text-align: center;
        padding-top: 15px;
        border-top: 1px solid rgba(212, 175, 55, 0.2);
    }
    
    .book-title {
        font-family: 'Cormorant Garamond', serif !important;
        font-size: 18px;
        font-weight: 600;
        color: #f4e5b0;
        margin-bottom: 12px;
        line-height: 1.4;
        display: block;
        max-height: 50px;
        overflow-x: auto;
        overflow-y: hidden;
        white-space: nowrap;
        padding-bottom: 5px;
    }
    
    .book-title::-webkit-scrollbar {
        height: 4px;
    }
    
    .book-title::-webkit-scrollbar-thumb {
        background: rgba(212, 175, 55, 0.5);
        border-radius: 10px;
    }
    
    .book-meta {
        display: flex;
        flex-direction: column;
        gap: 8px;
        align-items: center;
    }
    
    .meta-item {
        display: flex;
        align-items: center;
        gap: 8px;
        font-size: 13px;
        color: #a0a0a0;
    }
    
    .meta-icon {
        font-size: 14px;
    }
    
    .author-name {
        color: #c0c0c0;
        font-weight: 500;
    }
    
    .year {
        color: #888;
        font-size: 12px;
    }
    
    /* Section Header */
    .section-header {
        text-align: center;
        margin: 40px 0 30px 0;
        padding: 25px;
        background: linear-gradient(135deg, rgba(212, 175, 55, 0.1) 0%, rgba(255, 215, 0, 0.02) 100%);
        border-radius: 15px;
        border: 1px solid rgba(212, 175, 55, 0.2);
    }
    
    .results-title {
        font-family: 'Cormorant Garamond', serif !important;
        font-size: 28px;
        color: #d4af37;
        margin-bottom: 5px;
    }
    
    .results-subtitle {
        font-size: 14px;
        color: #888;
        letter-spacing: 1px;
    }
    
    /* Row Divider */
    .row-divider {
        height: 1px;
        background: linear-gradient(90deg, transparent, rgba(212, 175, 55, 0.3), transparent);
        margin: 35px 0;
    }
    
    /* Footer Space */
    .extra-space {
        margin-top: 60px;
    }
    
    /* Hide Streamlit Elements */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    
    /* Label Styling */
    .stSelectbox label, .stNumberInput label {
        font-family: 'Montserrat', sans-serif !important;
        font-size: 14px;
        color: #d4af37 !important;
        letter-spacing: 1px;
        text-transform: uppercase;
        font-weight: 500;
    }
    
    /* Banner Image Container */
    .banner-container {
        border-radius: 20px;
        overflow: hidden;
        box-shadow: 0 15px 40px rgba(0, 0, 0, 0.3);
        margin-bottom: 30px;
        border: 1px solid rgba(212, 175, 55, 0.2);
    }
    </style>
""", unsafe_allow_html=True)

# Premium Header
st.markdown("""
    <div class="premium-header">
        <div class="premium-title">📚 Book Recommendation System</div>
        <div class="premium-subtitle">Discover Your Next Literary Adventure</div>
        <div class="divider">
            <div class="divider-line"></div>
            <div class="divider-icon">✦</div>
            <div class="divider-line"></div>
        </div>
        <div class="tagline">Let Us Help You Choose Your Next Book</div>
    </div>
""", unsafe_allow_html=True)

# Banner Image
st.markdown('<div class="banner-container">', unsafe_allow_html=True)
st.image('https://img.freepik.com/premium-vector/bookcase-with-books_182089-197.jpg', use_container_width=True)
st.markdown('</div>', unsafe_allow_html=True)

# Input Section
st.markdown('<div class="divider"><div class="divider-line"></div><div class="divider-icon">📖</div><div class="divider-line"></div></div>', unsafe_allow_html=True)

all_books = final_filtered_df['title'].unique().tolist()

col1, col2 = st.columns([3, 1])
with col1:
    book_title = st.selectbox('Select a Book Title:', all_books, index=None, placeholder="Choose or search for a book...", key='book_title')
with col2:
    num_recommendations = st.number_input('Recommendations:', min_value=1, max_value=50, value=10)

st.markdown("<br>", unsafe_allow_html=True)

# Center the button
col1, col2, col3 = st.columns([1, 1, 1])
with col2:
    recommend_clicked = st.button('✨ Discover Books')

if 'recommendations' not in st.session_state:
    st.session_state.recommendations = None
if 'recommended_book' not in st.session_state:
    st.session_state.recommended_book = None
if 'recommended_num' not in st.session_state:
    st.session_state.recommended_num = None

if recommend_clicked:
    if book_title:
        similar_books = get_top_similar_books(book_title, num_recommendations)
        st.session_state.recommendations = similar_books
        st.session_state.recommended_book = book_title
        st.session_state.recommended_num = num_recommendations
    else:
        st.session_state.recommendations = None
        st.warning("⚠️ Please select a book title first.")

if st.session_state.recommendations is not None:
    similar_books = st.session_state.recommendations
    rec_book = st.session_state.recommended_book
    rec_num = st.session_state.recommended_num

    if isinstance(similar_books, str):
        st.error(similar_books)
    else:
        # Section Header
        st.markdown(f"""
            <div class="section-header">
                <div class="results-title">✨ Curated Recommendations ✨</div>
                <div class="results-subtitle">Top {rec_num} books similar to "<strong>{rec_book}</strong>"</div>
            </div>
        """, unsafe_allow_html=True)
        
        for i in range(0, len(similar_books), 3):
            cols = st.columns(3)
            for j in range(3):
                if i + j < len(similar_books):
                    book = similar_books.index[i + j]
                    book_info = final_filtered_df[final_filtered_df['title'] == book].iloc[0]
                    with cols[j]:
                        st.markdown(f"""
                        <div class="book-card">
                            <div class="rank-badge">{i + j + 1}</div>
                            <div class="book-image-container">
                                <img src="{book_info['Image-URL-L']}" class="book-image" alt="{book}">
                            </div>
                            <div class="book-info">
                                <div class="book-title">{book}</div>
                                <div class="book-meta">
                                    <div class="meta-item">
                                        <span class="meta-icon">✍️</span>
                                        <span class="author-name">{book_info['Book-Author']}</span>
                                    </div>
                                    <div class="meta-item">
                                        <span class="meta-icon">📅</span>
                                        <span class="year">{book_info['Year-Of-Publication']}</span>
                                    </div>
                                </div>
                            </div>
                        </div>
                        """, unsafe_allow_html=True)
            
            if i < len(similar_books) - 3:
                st.markdown('<div class="row-divider"></div>', unsafe_allow_html=True)

        st.markdown('<div class="extra-space"></div>', unsafe_allow_html=True)
        
        # Thank You Section
        st.markdown("""
            <div class="divider">
                <div class="divider-line"></div>
                <div class="divider-icon">📚</div>
                <div class="divider-line"></div>
            </div>
        """, unsafe_allow_html=True)
        
        st.image('https://github.com/MarpakaPradeepSai/Employee-Churn-Prediction/blob/main/Data/Images%20&%20GIFs/thank-you-33.gif?raw=true', use_container_width=True)
