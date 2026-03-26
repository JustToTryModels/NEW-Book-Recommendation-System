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

# Streamlit app
st.markdown("<h1 style='font-size: 48px; background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); -webkit-background-clip: text; -webkit-text-fill-color: transparent; font-weight: 800; letter-spacing: -1px; margin-bottom: 10px;'>Book Recommendation System</h1>", unsafe_allow_html=True)

# Premium CSS
st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');
    
    html, body, [class*="css"], [class*="st-"], h1, h2, h3, h4, h5, h6, p, div, span, label, input, button, select, option, textarea {
        font-family: 'Inter', 'Segoe UI', 'Helvetica Neue', Arial, sans-serif !important;
    }
    
    /* Main container background */
    .main {
        background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
    }
    
    .subheader {
        font-size: 24px;
        font-weight: 600;
        margin-bottom: 30px;
        color: #2d3748;
        letter-spacing: -0.5px;
        text-align: center;
    }
    
    /* Premium Button Styling */
    .stButton > button {
        font-family: 'Inter', sans-serif !important;
        font-size: 16px;
        font-weight: 600;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white !important;
        border: none;
        border-radius: 12px;
        padding: 14px 32px;
        cursor: pointer;
        transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
        box-shadow: 0 10px 25px rgba(102, 126, 234, 0.4);
        letter-spacing: 0.5px;
        text-transform: uppercase;
        font-size: 14px;
        width: 100%;
        margin-top: 20px;
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 15px 35px rgba(102, 126, 234, 0.5);
        background: linear-gradient(135deg, #764ba2 0%, #667eea 100%);
    }
    
    .stButton > button:active {
        transform: translateY(0px);
        box-shadow: 0 5px 15px rgba(102, 126, 234, 0.3);
    }
    
    /* Number Input & Select Box Styling */
    .stNumberInput > div > div > input,
    .stSelectbox > div > div > div {
        border-radius: 10px;
        border: 2px solid #e2e8f0;
        padding: 10px;
        font-size: 15px;
        transition: all 0.3s ease;
    }
    
    .stNumberInput > div > div > input:focus,
    .stSelectbox > div > div > div:focus {
        border-color: #667eea;
        box-shadow: 0 0 0 3px rgba(102, 126, 234, 0.1);
    }
    
    /* Book Card - Premium Design */
    .book-column {
        position: relative;
        padding: 24px;
        border-radius: 20px;
        background: linear-gradient(135deg, #ffffff 0%, #f8f9fa 100%);
        box-shadow: 0 10px 40px rgba(0, 0, 0, 0.08);
        margin-bottom: 20px;
        transition: all 0.4s cubic-bezier(0.4, 0, 0.2, 1);
        border: 1px solid rgba(255, 255, 255, 0.8);
        overflow: hidden;
    }
    
    .book-column::before {
        content: '';
        position: absolute;
        top: 0;
        left: 0;
        right: 0;
        height: 4px;
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        opacity: 0;
        transition: opacity 0.4s ease;
    }
    
    .book-column:hover::before {
        opacity: 1;
    }
    
    .book-column:hover {
        transform: translateY(-10px);
        box-shadow: 0 20px 60px rgba(0, 0, 0, 0.15);
        border-color: rgba(102, 126, 234, 0.2);
    }
    
    /* Image Container with Frame Effect */
    .image-container {
        position: relative;
        padding: 12px;
        background: linear-gradient(135deg, #f8f9fa 0%, #e9ecef 100%);
        border-radius: 16px;
        margin-bottom: 20px;
        box-shadow: inset 0 2px 8px rgba(0, 0, 0, 0.06);
    }
    
    .image-container img {
        object-fit: contain;
        max-height: 320px;
        width: auto;
        display: block;
        margin: 0 auto;
        border-radius: 8px;
        transition: transform 0.4s ease;
    }
    
    .book-column:hover .image-container img {
        transform: scale(1.03);
    }
    
    /* Book Info Section */
    .book-info {
        line-height: 1.6;
    }
    
    /* Title with gradient on hover */
    .scroll-title {
        display: block;
        font-size: 18px;
        font-weight: 700;
        white-space: nowrap;
        overflow-x: auto;
        padding-bottom: 8px;
        margin-bottom: 12px;
        color: #1a202c;
        letter-spacing: -0.3px;
        transition: all 0.3s ease;
    }
    
    .book-column:hover .scroll-title {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    
    .scroll-title::-webkit-scrollbar {
        height: 4px;
    }
    
    .scroll-title::-webkit-scrollbar-track {
        background: #f1f1f1;
        border-radius: 10px;
    }
    
    .scroll-title::-webkit-scrollbar-thumb {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        border-radius: 10px;
    }
    
    /* Info Container */
    .info-container {
        background: linear-gradient(135deg, #f8f9fa 0%, #ffffff 100%);
        border-left: 4px solid #667eea;
        padding: 12px 16px;
        margin-top: 12px;
        border-radius: 8px;
        box-shadow: 0 2px 8px rgba(0, 0, 0, 0.04);
    }
    
    .author-info {
        font-size: 14px;
        color: #4a5568;
        font-weight: 500;
        margin-bottom: 6px;
        display: flex;
        align-items: center;
        gap: 8px;
    }
    
    .year-info {
        font-size: 13px;
        color: #718096;
        font-weight: 400;
        display: flex;
        align-items: center;
        gap: 8px;
    }
    
    /* Divider */
    hr {
        border: none !important;
        height: 2px !important;
        background: linear-gradient(90deg, transparent, #667eea, transparent) !important;
        margin: 40px 0 !important;
        opacity: 0.3 !important;
    }
    
    /* Recommendation Header */
    .rec-header {
        font-size: 20px;
        font-weight: 700;
        color: #2d3748;
        text-align: center;
        padding: 20px;
        background: linear-gradient(135deg, #ffffff 0%, #f8f9fa 100%);
        border-radius: 16px;
        margin-bottom: 30px;
        box-shadow: 0 4px 16px rgba(0, 0, 0, 0.06);
        border-left: 5px solid #667eea;
    }
    
    /* Extra spacing */
    .extra-space {
        margin-top: 60px;
    }
    
    /* Rank badge */
    .rank-badge {
        position: absolute;
        top: 16px;
        left: 16px;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        width: 36px;
        height: 36px;
        border-radius: 50%;
        display: flex;
        align-items: center;
        justify-content: center;
        font-weight: 700;
        font-size: 14px;
        box-shadow: 0 4px 12px rgba(102, 126, 234, 0.4);
        z-index: 10;
    }
    
    /* Hero Image styling */
    .hero-image {
        border-radius: 24px;
        box-shadow: 0 20px 60px rgba(0, 0, 0, 0.1);
        margin-bottom: 40px;
    }
    
    </style>
""", unsafe_allow_html=True)

st.markdown("<p class='subheader'>✨ Discover Your Next Great Read ✨</p>", unsafe_allow_html=True)
st.markdown("<div class='hero-image'>", unsafe_allow_html=True)
st.image('https://img.freepik.com/premium-vector/bookcase-with-books_182089-197.jpg', use_container_width=True)
st.markdown("</div>", unsafe_allow_html=True)

all_books = final_filtered_df['title'].unique().tolist()
book_title = st.selectbox('📚 Select a book title:', all_books, index=None, placeholder="Choose or type to search...", key='book_title')

num_recommendations = st.number_input('🔢 Number of recommendations:', min_value=1, max_value=50, value=10)

if 'recommendations' not in st.session_state:
    st.session_state.recommendations = None
if 'recommended_book' not in st.session_state:
    st.session_state.recommended_book = None
if 'recommended_num' not in st.session_state:
    st.session_state.recommended_num = None

if st.button('✨ Get Recommendations'):
    if book_title:
        similar_books = get_top_similar_books(book_title, num_recommendations)
        st.session_state.recommendations = similar_books
        st.session_state.recommended_book = book_title
        st.session_state.recommended_num = num_recommendations
    else:
        st.session_state.recommendations = None
        st.warning("⚠️ Please select a book title to get recommendations.")

if st.session_state.recommendations is not None:
    similar_books = st.session_state.recommendations
    rec_book = st.session_state.recommended_book
    rec_num = st.session_state.recommended_num

    if isinstance(similar_books, str):
        st.write(similar_books)
    else:
        st.markdown(f"<div class='rec-header'>🎯 Top {rec_num} Recommendations for <strong>'{rec_book}'</strong></div>", unsafe_allow_html=True)
        
        for i in range(0, len(similar_books), 3):
            cols = st.columns(3)
            for j in range(3):
                if i + j < len(similar_books):
                    book = similar_books.index[i + j]
                    book_info = final_filtered_df[final_filtered_df['title'] == book].iloc[0]
                    with cols[j]:
                        st.markdown(f"""
                        <div class='book-column'>
                            <div class='rank-badge'>{i + j + 1}</div>
                            <div class='image-container'>
                                <img src='{book_info['Image-URL-L']}'>
                            </div>
                            <div class='book-info'>
                                <div class='scroll-title'>{book}</div>
                                <div class='info-container'>
                                    <div class='author-info'>
                                        <span style='font-size: 16px;'>✍️</span>
                                        <span>{book_info['Book-Author']}</span>
                                    </div>
                                    <div class='year-info'>
                                        <span style='font-size: 16px;'>📅</span>
                                        <span>{book_info['Year-Of-Publication']}</span>
                                    </div>
                                </div>
                            </div>
                        </div>
                        """, unsafe_allow_html=True)
            if i < len(similar_books) - 3:
                st.markdown("<hr>", unsafe_allow_html=True)

        st.markdown("<div class='extra-space'></div>", unsafe_allow_html=True)
        st.image('https://github.com/MarpakaPradeepSai/Employee-Churn-Prediction/blob/main/Data/Images%20&%20GIFs/thank-you-33.gif?raw=true', use_container_width=True)
