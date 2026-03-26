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

    # URL fixes
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
st.markdown("<h1 style='font-size: 42px; text-align: center; margin-bottom: 8px;'>Book Recommendation System</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center; font-size: 20px; color: #b8b8b8; margin-bottom: 30px;'>Discover your next literary obsession</p>", unsafe_allow_html=True)

st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Playfair+Display:wght@400;700&family=Inter:wght@300;400;500;600&display=swap');

    html, body, [class*="css"], [class*="st-"] {
        font-family: 'Inter', system-ui, sans-serif;
    }
    
    h1, h2, h3 {
        font-family: 'Playfair Display', serif !important;
    }

    .stButton > button {
        font-family: 'Inter', sans-serif;
        font-weight: 600;
        background: linear-gradient(90deg, #c9a227, #e8c670);
        color: #1a1a1a !important;
        border: none;
        border-radius: 30px;
        padding: 12px 28px;
        font-size: 16px;
        transition: all 0.3s ease;
    }
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 10px 25px rgba(201, 162, 39, 0.3);
    }

    .book-column {
        background: linear-gradient(145deg, #1c1c1c, #282828);
        border: 1px solid #3a3a3a;
        border-radius: 20px;
        overflow: hidden;
        box-shadow: 0 15px 35px rgba(0, 0, 0, 0.7);
        transition: all 0.4s cubic-bezier(0.4, 0, 0.2, 1);
        position: relative;
        height: 100%;
    }
    
    .book-column:hover {
        transform: translateY(-15px);
        box-shadow: 0 30px 60px rgba(201, 162, 39, 0.2);
        border-color: #d4af37;
    }

    .recommendation-badge {
        position: absolute;
        top: -18px;
        left: 50%;
        transform: translateX(-50%);
        width: 52px;
        height: 52px;
        background: linear-gradient(135deg, #d4af37, #f0d48a);
        color: #1a1a1a;
        border: 3px solid #1c1c1c;
        border-radius: 50%;
        display: flex;
        align-items: center;
        justify-content: center;
        font-size: 20px;
        font-weight: 800;
        z-index: 20;
        box-shadow: 0 8px 20px rgba(212, 175, 55, 0.4);
        font-family: 'Playfair Display', serif;
    }

    .book-image-area {
        padding: 35px 25px 20px 25px;
        background: #161616;
        position: relative;
    }

    .book-image-area img {
        border-radius: 12px;
        box-shadow: 0 10px 30px rgba(0, 0, 0, 0.6);
        transition: transform 0.5s ease;
        width: 100% !important;
        height: 310px !important;
        object-fit: cover;
    }

    .book-column:hover .book-image-area img {
        transform: scale(1.04);
    }

    .book-info {
        padding: 24px 20px 28px 20px;
        text-align: center;
        background: linear-gradient(to bottom, #1f1f1f, #1a1a1a);
        border-top: 1px solid #3a3a3a;
    }

    .premium-title {
        font-family: 'Playfair Display', serif;
        font-size: 17.5px;
        font-weight: 700;
        color: #f0f0f0;
        line-height: 1.35;
        margin-bottom: 10px;
        min-height: 48px;
        display: -webkit-box;
        -webkit-line-clamp: 2;
        -webkit-box-orient: vertical;
        overflow: hidden;
    }

    .premium-divider {
        width: 42px;
        height: 2.5px;
        background: linear-gradient(90deg, transparent, #d4af37, transparent);
        margin: 0 auto 12px auto;
    }

    .premium-author {
        font-size: 14px;
        color: #b8b8b8;
        font-style: italic;
        margin-bottom: 6px;
        font-weight: 400;
    }

    .premium-year {
        font-size: 12.5px;
        color: #777777;
        letter-spacing: 1.5px;
        font-weight: 500;
        text-transform: uppercase;
    }

    .recommendation-header {
        font-size: 18px;
        font-weight: 600;
        color: #d4af37;
        margin: 30px 0 20px 0;
        padding-bottom: 12px;
        border-bottom: 2px solid #3a3a3a;
        font-family: 'Playfair Display', serif;
    }

    .extra-space {
        margin-top: 60px;
    }
    </style>
""", unsafe_allow_html=True)

st.image('https://img.freepik.com/premium-vector/bookcase-with-books_182089-197.jpg', use_container_width=True)

all_books = final_filtered_df['title'].unique().tolist()
book_title = st.selectbox('Select a book title:', all_books, index=None, placeholder="Choose or search a book...", key='book_title')

num_recommendations = st.number_input('Number of recommendations:', min_value=1, max_value=50, value=10)

if 'recommendations' not in st.session_state:
    st.session_state.recommendations = None
if 'recommended_book' not in st.session_state:
    st.session_state.recommended_book = None
if 'recommended_num' not in st.session_state:
    st.session_state.recommended_num = None

if st.button('Get Recommendations', use_container_width=True):
    if book_title:
        similar_books = get_top_similar_books(book_title, num_recommendations)
        st.session_state.recommendations = similar_books
        st.session_state.recommended_book = book_title
        st.session_state.recommended_num = num_recommendations
    else:
        st.warning("⚠️ Please select a book title.")

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
                                <div class='premium-author'>{book_info['Book-Author']}</div>
                                <div class='premium-year'>{book_info['Year-Of-Publication']}</div>
                            </div>
                        </div>
                        """, unsafe_allow_html=True)
            
            if i < len(similar_books) - 3:
                st.markdown("<br>", unsafe_allow_html=True)

        st.markdown("<div class='extra-space'></div>", unsafe_allow_html=True)
        st.image('https://github.com/MarpakaPradeepSai/Employee-Churn-Prediction/blob/main/Data/Images%20&%20GIFs/thank-you-33.gif?raw=true', use_container_width=True)
