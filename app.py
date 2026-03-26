import streamlit as st
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from huggingface_hub import hf_hub_download
import warnings

warnings.filterwarnings('ignore')

@st.cache_data
def load_and_prepare_data():
    final_filtered_df_path = hf_hub_download(repo_id="IamPradeep/BRS_DATA", filename="final_filtered_df.csv", repo_type="dataset")
    final_filtered_df = pd.read_csv(final_filtered_df_path)

    book_urls_df_path = hf_hub_download(repo_id="IamPradeep/BRS_DATA", filename="Books.csv", repo_type="dataset")
    book_urls_df = pd.read_csv(book_urls_df_path)
    book_urls_df.rename(columns={'Book-Title': 'title'}, inplace=True)

    final_filtered_df = final_filtered_df.merge(book_urls_df, on='title', how='left')

    url1 = 'http://images.amazon.com/images/P/0690040784.01.LZZZZZZZ.jpg'
    url2 = 'http://images.amazon.com/images/P/0451172817.01.LZZZZZZZ.jpg'
    url3 = 'http://images.amazon.com/images/P/0312084986.01.LZZZZZZZ.jpg'
    url4 = 'http://images.amazon.com/images/P/1590400356.01.LZZZZZZZ.jpg'

    final_filtered_df.loc[final_filtered_df['title'] == 'Jacob Have I Loved', 'Image-URL-L'] = url1
    final_filtered_df.loc[final_filtered_df['title'] == 'Needful Things', 'Image-URL-L'] = url2
    final_filtered_df.loc[final_filtered_df['title'] == 'All Creatures Great and Small', 'Image-URL-L'] = url3
    final_filtered_df.loc[final_filtered_df['title'] == "The Kitchen God's Wife", 'Image-URL-L'] = url4

    book_user_mat = final_filtered_df.pivot_table(index='title', columns='userId', values='rating').fillna(0)
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

# ====================== PREMIUM STREAMLIT APP ======================
st.set_page_config(page_title="BRS • Luxe", layout="wide", initial_sidebar_state="collapsed")

st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Playfair+Display:wght@400;700&family=Inter:wght@300;400;500;600&display=swap');

    html, body, [class*="css"], [class*="st-"] {
        font-family: 'Inter', system-ui, sans-serif !important;
    }
    
    h1 {
        font-family: 'Playfair Display', serif !important;
        font-size: 58px !important;
        font-weight: 700;
        background: linear-gradient(90deg, #d4af77, #f5e8c7, #d4af77);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        letter-spacing: -2.5px;
        margin-bottom: 8px !important;
    }
    
    .subheader {
        font-size: 24px;
        color: #c9b89f;
        font-weight: 400;
        letter-spacing: 4px;
        margin-bottom: 30px !important;
    }

    /* Premium Button */
    .stButton > button {
        font-family: 'Inter', sans-serif !important;
        font-size: 17px;
        font-weight: 600;
        background: linear-gradient(135deg, #2c1810, #5c4033);
        color: #f5e8c7 !important;
        border: 2px solid #d4af77;
        border-radius: 50px;
        padding: 14px 42px;
        letter-spacing: 2px;
        box-shadow: 0 10px 35px rgba(212, 175, 119, 0.25);
        transition: all 0.4s cubic-bezier(0.4, 0, 0.2, 1);
    }
    
    .stButton > button:hover {
        transform: translateY(-4px) scale(1.03);
        box-shadow: 0 20px 45px rgba(212, 175, 119, 0.35);
        border-color: #e8d5a8;
    }

    /* Premium Book Card */
    .book-column {
        background: #161618;
        border-radius: 20px;
        overflow: hidden;
        box-shadow: 0 15px 40px rgba(0, 0, 0, 0.7),
                    inset 0 0 0 1px rgba(212, 175, 119, 0.15);
        transition: all 0.5s cubic-bezier(0.4, 0, 0.2, 1);
        position: relative;
        margin-top: 38px;
        height: 100%;
    }
    
    .book-column:hover {
        transform: translateY(-12px);
        box-shadow: 0 25px 60px rgba(0, 0, 0, 0.8),
                    inset 0 0 0 1px rgba(212, 175, 119, 0.4);
    }

    .recommendation-badge {
        position: absolute;
        top: -24px;
        left: 50%;
        transform: translateX(-50%);
        width: 52px;
        height: 52px;
        background: linear-gradient(135deg, #d4af77, #f5e8c7);
        color: #1a1208;
        border: 3px solid #161618;
        border-radius: 50%;
        display: flex;
        align-items: center;
        justify-content: center;
        font-size: 22px;
        font-weight: 700;
        box-shadow: 0 8px 25px rgba(212, 175, 119, 0.5);
        z-index: 20;
        font-family: 'Playfair Display', serif;
    }

    .book-image-area {
        padding: 35px 25px 20px 25px;
        background: #0f0f11;
        position: relative;
    }
    
    .book-image-area img {
        border-radius: 12px;
        box-shadow: 0 10px 30px rgba(0, 0, 0, 0.6);
        transition: transform 0.6s ease;
        height: 310px;
        object-fit: contain;
        width: 100%;
    }
    
    .book-column:hover .book-image-area img {
        transform: scale(1.04);
    }

    .book-info {
        background: #161618;
        padding: 24px 20px 28px 20px;
        text-align: center;
        border-top: 1px solid rgba(212, 175, 119, 0.15);
    }

    .premium-title {
        font-family: 'Playfair Display', serif;
        font-size: 17.5px;
        font-weight: 700;
        color: #f5e8c7;
        margin-bottom: 10px;
        line-height: 1.35;
        min-height: 48px;
        display: -webkit-box;
        -webkit-line-clamp: 2;
        -webkit-box-orient: vertical;
        overflow: hidden;
    }

    .premium-divider {
        width: 48px;
        height: 2.5px;
        background: linear-gradient(90deg, transparent, #d4af77, transparent);
        margin: 0 auto 14px auto;
    }

    .premium-author {
        font-size: 14.2px;
        color: #b8a88a;
        font-style: italic;
        margin-bottom: 6px;
    }

    .premium-year {
        font-size: 12.5px;
        color: #76654a;
        letter-spacing: 1.5px;
        font-weight: 500;
    }

    .recommendation-header {
        font-size: 21px;
        color: #d4af77;
        font-family: 'Playfair Display', serif;
        margin-bottom: 15px;
        padding-left: 8px;
        border-left: 4px solid #d4af77;
    }

    hr {
        border: none;
        height: 1px;
        background: linear-gradient(90deg, transparent, rgba(212, 175, 119, 0.2), transparent);
        margin: 45px 0;
    }

    .extra-space {
        margin-top: 60px;
    }
    </style>
""", unsafe_allow_html=True)

st.markdown("<h1>Book Recommendation System</h1>", unsafe_allow_html=True)
st.markdown("<p class='subheader'>CURATED FOR DISCERNING READERS</p>", unsafe_allow_html=True)

st.image('https://img.freepik.com/premium-vector/bookcase-with-books_182089-197.jpg', use_container_width=True)

all_books = final_filtered_df['title'].unique().tolist()
book_title = st.selectbox('Select a book you loved:', all_books, index=None, placeholder="Choose a book title...", key='book_title')

num_recommendations = st.number_input('Number of recommendations:', min_value=1, max_value=50, value=10)

if 'recommendations' not in st.session_state:
    st.session_state.recommendations = None
if 'recommended_book' not in st.session_state:
    st.session_state.recommended_book = None
if 'recommended_num' not in st.session_state:
    st.session_state.recommended_num = None

if st.button('Get Recommendations'):
    if book_title:
        similar_books = get_top_similar_books(book_title, num_recommendations)
        st.session_state.recommendations = similar_books
        st.session_state.recommended_book = book_title
        st.session_state.recommended_num = num_recommendations
    else:
        st.session_state.recommendations = None
        st.error("Please select a book title.")

if st.session_state.recommendations is not None:
    similar_books = st.session_state.recommendations
    rec_book = st.session_state.recommended_book
    rec_num = st.session_state.recommended_num

    if isinstance(similar_books, str):
        st.write(similar_books)
    else:
        st.markdown(f"<div class='recommendation-header'>Top {rec_num} recommendations for <strong>'{rec_book}'</strong></div>", unsafe_allow_html=True)
        
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
                                <img src='{book_info['Image-URL-L']}' alt='{safe_title}'>
                            </div>
                            <div class='book-info'>
                                <div class='premium-title' title="{safe_title}">{book}</div>
                                <div class='premium-divider'></div>
                                <div class='premium-author'>by {book_info['Book-Author']}</div>
                                <div class='premium-year'>{book_info['Year-Of-Publication']}</div>
                            </div>
                        </div>
                        """, unsafe_allow_html=True)
            if i < len(similar_books) - 3:
                st.markdown("<hr>", unsafe_allow_html=True)

        st.markdown("<div class='extra-space'></div>", unsafe_allow_html=True)
        st.image('https://github.com/MarpakaPradeepSai/Employee-Churn-Prediction/blob/main/Data/Images%20&%20GIFs/thank-you-33.gif?raw=true', use_container_width=True)
