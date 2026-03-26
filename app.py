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

st.markdown("<h1 style='font-size: 40px;'>Book Recommendation System</h1>", unsafe_allow_html=True)

st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Playfair+Display:wght@400;500;600;700;800;900&family=Lora:ital,wght@0,400;0,500;0,600;0,700;1,400;1,500&family=Cormorant+Garamond:ital,wght@0,300;0,400;0,500;0,600;0,700;1,300;1,400&display=swap');

    html, body, [class*="css"], [class*="st-"], h1, h2, h3, h4, h5, h6, p, div, span, label, input, button, select, option, textarea {
        font-family: 'Lora', 'Playfair Display', Georgia, 'Times New Roman', serif !important;
    }

    .subheader {
        font-size: 22px;
        font-weight: bold;
        margin-bottom: 20px;
        color: #1a73e8;
    }

    .stButton > button {
        font-family: 'Lora', 'Playfair Display', Georgia, serif !important;
        font-size: 16px;
        background: linear-gradient(135deg, #ff8a00, #e52e71);
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
    .stButton > button:active {
        transform: scale(0.98);
    }

    /* ===== PREMIUM BOOK CARD ===== */
    .book-column {
        position: relative;
        padding: 0;
        border-radius: 18px;
        margin-top: 35px;
        margin-bottom: 20px;
        overflow: visible;
        background: linear-gradient(145deg, #1a1a2e, #16213e, #0f3460);
        border: 1px solid rgba(255, 255, 255, 0.06);
        box-shadow:
            0 10px 40px rgba(0, 0, 0, 0.45),
            0 2px 10px rgba(0, 0, 0, 0.25),
            inset 0 1px 0 rgba(255, 255, 255, 0.05);
        transition: transform 0.4s cubic-bezier(0.25, 0.46, 0.45, 0.94), box-shadow 0.4s ease;
    }
    .book-column:hover {
        transform: translateY(-10px) scale(1.02);
        box-shadow:
            0 20px 60px rgba(229, 46, 113, 0.2),
            0 10px 30px rgba(0, 0, 0, 0.4),
            inset 0 1px 0 rgba(255, 255, 255, 0.08);
    }

    /* Animated gradient border glow on hover */
    .book-column::before {
        content: '';
        position: absolute;
        top: -2px; left: -2px; right: -2px; bottom: -2px;
        border-radius: 20px;
        background: linear-gradient(135deg, #ff8a00, #e52e71, #9b59b6, #3498db, #e52e71, #ff8a00);
        background-size: 400% 400%;
        z-index: -1;
        opacity: 0;
        transition: opacity 0.4s ease;
        animation: borderGlow 4s ease infinite;
    }
    .book-column:hover::before {
        opacity: 1;
    }
    @keyframes borderGlow {
        0% { background-position: 0% 50%; }
        50% { background-position: 100% 50%; }
        100% { background-position: 0% 50%; }
    }

    /* Shimmer overlay */
    .book-column::after {
        content: '';
        position: absolute;
        top: 0; left: -100%; right: 0; bottom: 0;
        width: 60%;
        background: linear-gradient(
            90deg,
            transparent,
            rgba(255, 255, 255, 0.03),
            transparent
        );
        transform: skewX(-15deg);
        transition: left 0.8s ease;
        border-radius: 18px;
        pointer-events: none;
        z-index: 2;
    }
    .book-column:hover::after {
        left: 130%;
    }

    /* ===== BADGE ===== */
    .recommendation-badge {
        position: absolute;
        top: -20px;
        left: 50%;
        transform: translateX(-50%);
        width: 44px;
        height: 44px;
        border-radius: 50%;
        background: linear-gradient(135deg, #f7971e, #ffd200);
        color: #1a1a2e;
        border: 3px solid #1a1a2e;
        display: flex;
        align-items: center;
        justify-content: center;
        font-size: 17px;
        font-weight: 800;
        z-index: 15;
        box-shadow:
            0 4px 15px rgba(247, 151, 30, 0.5),
            0 0 20px rgba(255, 210, 0, 0.2);
        font-family: 'Playfair Display', serif !important;
        letter-spacing: -0.5px;
    }

    /* ===== IMAGE AREA ===== */
    .book-image-area {
        padding: 40px 25px 20px 25px;
        display: flex;
        align-items: center;
        justify-content: center;
        position: relative;
        overflow: hidden;
        border-radius: 18px 18px 0 0;
    }
    /* Subtle radial glow behind image */
    .book-image-area::before {
        content: '';
        position: absolute;
        width: 70%;
        height: 70%;
        top: 50%;
        left: 50%;
        transform: translate(-50%, -50%);
        background: radial-gradient(ellipse, rgba(229, 46, 113, 0.08), transparent 70%);
        pointer-events: none;
        z-index: 0;
    }

    .book-image-area img {
        object-fit: contain;
        max-height: 290px;
        width: auto;
        display: block;
        margin: 0 auto;
        border-radius: 6px;
        position: relative;
        z-index: 1;
        box-shadow: 0 8px 25px rgba(0, 0, 0, 0.4);
        transition: transform 0.4s ease, box-shadow 0.4s ease;
    }
    .book-column:hover .book-image-area img {
        transform: scale(1.04);
        box-shadow: 0 12px 35px rgba(0, 0, 0, 0.5);
    }

    /* ===== BOOK INFO BLOCK ===== */
    .book-info {
        background: linear-gradient(180deg, rgba(15, 52, 96, 0.5), rgba(10, 10, 30, 0.95));
        padding: 22px 18px 24px 18px;
        border-radius: 0 0 18px 18px;
        border-top: 1px solid rgba(255, 255, 255, 0.05);
        text-align: center;
        display: flex;
        flex-direction: column;
        justify-content: flex-start;
        align-items: center;
        min-height: 145px;
        position: relative;
    }
    /* Top accent line */
    .book-info::before {
        content: '';
        position: absolute;
        top: 0;
        left: 15%;
        right: 15%;
        height: 2px;
        background: linear-gradient(90deg, transparent, rgba(229, 46, 113, 0.5), rgba(255, 138, 0, 0.5), transparent);
        border-radius: 2px;
    }

    /* ===== TITLE ===== */
    .premium-title {
        font-family: 'Playfair Display', serif !important;
        font-size: 16.5px;
        font-weight: 700;
        color: #f0e6d3;
        margin-bottom: 10px;
        line-height: 1.35;
        width: 100%;
        white-space: nowrap;
        overflow-x: auto;
        overflow-y: hidden;
        display: block;
        padding-bottom: 6px;
        letter-spacing: 0.3px;
        text-shadow: 0 1px 3px rgba(0, 0, 0, 0.3);
    }
    .premium-title::-webkit-scrollbar {
        height: 4px;
    }
    .premium-title::-webkit-scrollbar-track {
        background: rgba(255, 255, 255, 0.03);
        border-radius: 10px;
    }
    .premium-title::-webkit-scrollbar-thumb {
        background: linear-gradient(90deg, #e52e71, #ff8a00);
        border-radius: 10px;
    }

    /* ===== DIVIDER ===== */
    .premium-divider {
        width: 45px;
        height: 3px;
        background: linear-gradient(90deg, #f7971e, #e52e71, #9b59b6);
        margin: 4px auto 14px auto;
        border-radius: 10px;
        position: relative;
        overflow: hidden;
    }
    .premium-divider::after {
        content: '';
        position: absolute;
        top: 0; left: -100%;
        width: 50%;
        height: 100%;
        background: linear-gradient(90deg, transparent, rgba(255,255,255,0.6), transparent);
        animation: dividerShine 3s ease-in-out infinite;
    }
    @keyframes dividerShine {
        0% { left: -100%; }
        50% { left: 150%; }
        100% { left: 150%; }
    }

    /* ===== AUTHOR ===== */
    .premium-author {
        font-family: 'Cormorant Garamond', serif !important;
        font-size: 14.5px;
        color: #b8c5d6;
        font-style: italic;
        font-weight: 500;
        margin-bottom: 8px;
        width: 100%;
        white-space: nowrap;
        overflow: hidden;
        text-overflow: ellipsis;
        letter-spacing: 0.4px;
    }

    /* ===== YEAR ===== */
    .premium-year {
        display: inline-block;
        font-size: 11px;
        color: #a0a8b4;
        text-transform: uppercase;
        letter-spacing: 2.5px;
        font-weight: 600;
        padding: 4px 14px;
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 20px;
        background: rgba(255, 255, 255, 0.03);
        margin-top: 4px;
    }

    /* ===== SIMILARITY SCORE BADGE ===== */
    .similarity-score {
        position: absolute;
        top: 40px;
        right: 12px;
        background: linear-gradient(135deg, rgba(46, 204, 113, 0.15), rgba(39, 174, 96, 0.1));
        border: 1px solid rgba(46, 204, 113, 0.3);
        color: #2ecc71;
        font-size: 11px;
        font-weight: 700;
        padding: 4px 10px;
        border-radius: 20px;
        letter-spacing: 0.5px;
        z-index: 10;
        backdrop-filter: blur(8px);
        font-family: 'Lora', serif !important;
    }

    /* ===== GENERAL OVERRIDES ===== */
    img {
        object-fit: contain;
        max-height: 300px;
        width: auto;
        display: block;
        margin: 0 auto;
    }
    hr {
        border: none !important;
        height: 2px !important;
        background: linear-gradient(90deg, transparent, rgba(229, 46, 113, 0.3), rgba(255, 138, 0, 0.3), rgba(155, 89, 182, 0.3), transparent) !important;
        margin-top: 30px !important;
        margin-bottom: 30px !important;
        opacity: 1 !important;
        border-radius: 999px !important;
    }
    .extra-space {
        margin-top: 50px;
    }
    .recommendation-header {
        font-size: 15px;
        border-left: 5px solid #B2BEB5;
        padding-left: 12px;
        margin-left: 5px;
    }
    </style>
""", unsafe_allow_html=True)

st.markdown("<p class='subheader'>Let Us Help You Choose Your Next Book!</p>", unsafe_allow_html=True)
st.image('https://img.freepik.com/premium-vector/bookcase-with-books_182089-197.jpg', use_container_width=True)

all_books = final_filtered_df['title'].unique().tolist()
book_title = st.selectbox('Enter a book title:', all_books, index=None, placeholder="Choose or enter a book title...", key='book_title')

num_recommendations = st.number_input('Enter the number of recommendations:', min_value=1, max_value=50, value=10)

if 'recommendations' not in st.session_state:
    st.session_state.recommendations = None
if 'recommended_book' not in st.session_state:
    st.session_state.recommended_book = None
if 'recommended_num' not in st.session_state:
    st.session_state.recommended_num = None

if st.button('Recommend books'):
    if book_title:
        similar_books = get_top_similar_books(book_title, num_recommendations)
        st.session_state.recommendations = similar_books
        st.session_state.recommended_book = book_title
        st.session_state.recommended_num = num_recommendations
    else:
        st.session_state.recommendations = None
        st.write("⚠️ Please select or enter a book title.")

if st.session_state.recommendations is not None:
    similar_books = st.session_state.recommendations
    rec_book = st.session_state.recommended_book
    rec_num = st.session_state.recommended_num

    if isinstance(similar_books, str):
        st.write(similar_books)
    else:
        st.markdown(f"<div class='recommendation-header'>Top {rec_num} recommendations for '<strong>{rec_book}</strong>':</div>", unsafe_allow_html=True)
        st.write("")

        for i in range(0, len(similar_books), 3):
            cols = st.columns(3)
            for j in range(3):
                if i + j < len(similar_books):
                    book = similar_books.index[i + j]
                    score = similar_books.values[i + j]
                    book_info = final_filtered_df[final_filtered_df['title'] == book].iloc[0]

                    safe_title = str(book).replace('"', '&quot;').replace("'", "&#39;")
                    safe_author = str(book_info['Book-Author']).replace('"', '&quot;').replace("'", "&#39;")

                    with cols[j]:
                        st.markdown(f"""
                        <div class='book-column'>
                            <div class='recommendation-badge'>{i + j + 1}</div>
                            <div class='similarity-score'>★ {score:.0%} Match</div>
                            <div class='book-image-area'>
                                <img src='{book_info['Image-URL-L']}' alt='{safe_title}'>
                            </div>
                            <div class='book-info'>
                                <div class='premium-title' title="{safe_title}">{book}</div>
                                <div class='premium-divider'></div>
                                <div class='premium-author' title="{safe_author}">By {book_info['Book-Author']}</div>
                                <div class='premium-year'>📅 {book_info['Year-Of-Publication']}</div>
                            </div>
                        </div>
                        """, unsafe_allow_html=True)
            if i < len(similar_books) - 3:
                st.markdown("<br><hr><br>", unsafe_allow_html=True)

        st.markdown("<div class='extra-space'></div><div class='extra-space'></div>", unsafe_allow_html=True)
        st.image('https://github.com/MarpakaPradeepSai/Employee-Churn-Prediction/blob/main/Data/Images%20&%20GIFs/thank-you-33.gif?raw=true', use_container_width=True)
