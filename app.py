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
st.markdown("<h1 style='font-size: 40px;'>Book Recommendation System</h1>", unsafe_allow_html=True)

st.markdown("""
    <style>
    html, body, [class*="css"], [class*="st-"], h1, h2, h3, h4, h5, h6, p, div, span, label, input, button, select, option, textarea {
        font-family: 'Tiempos', 'Tiempos Text', Georgia, 'Times New Roman', serif !important;
    }
    .subheader {
        font-size: 22px;
        font-weight: bold;
        margin-bottom: 20px;
        color: #1a73e8;
    }
    .stButton > button {
        font-family: 'Tiempos', 'Tiempos Text', Georgia, 'Times New Roman', serif !important;
        font-size: 16px;
        background: linear-gradient(90deg, #ff8a00, #e52e71);
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

    /* ── Book Info Section (Premium Redesign) ── */
    .book-info {
        margin: 0;
        padding: 0;
        background: linear-gradient(180deg, #1e1e2f 0%, #2b2b3d 100%);
        border-radius: 0 0 10px 10px;
        border-top: 3px solid #e52e71;
        overflow: hidden;
    }

    .book-title-area {
        padding: 14px 14px 10px 14px;
        text-align: center;
    }

    .scroll-title {
        display: block;
        font-size: 15px;
        font-weight: 700;
        white-space: nowrap;
        overflow-x: auto;
        color: #f5e6a3;
        letter-spacing: 0.3px;
        line-height: 1.3;
        padding-bottom: 4px;
    }
    .scroll-title::-webkit-scrollbar {
        height: 5px;
    }
    .scroll-title::-webkit-scrollbar-track {
        background: transparent;
    }
    .scroll-title::-webkit-scrollbar-thumb {
        background: rgba(245, 230, 163, 0.3);
        border-radius: 10px;
    }

    .info-separator {
        height: 1px;
        margin: 0 14px;
        background: linear-gradient(90deg, transparent, rgba(245, 230, 163, 0.35), transparent);
    }

    .book-meta {
        padding: 10px 14px 14px 14px;
        display: flex;
        flex-direction: column;
        gap: 7px;
    }

    .meta-item {
        display: flex;
        align-items: flex-start;
        gap: 9px;
    }

    .meta-icon {
        flex-shrink: 0;
        width: 26px;
        height: 26px;
        border-radius: 6px;
        background: rgba(255, 255, 255, 0.07);
        display: flex;
        align-items: center;
        justify-content: center;
        font-size: 13px;
        border: 1px solid rgba(255, 255, 255, 0.08);
    }

    .meta-text-block {
        display: flex;
        flex-direction: column;
        min-width: 0;
    }

    .meta-label {
        font-size: 9px;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 1px;
        color: rgba(176, 196, 222, 0.55);
        line-height: 1.2;
    }

    .meta-value {
        font-size: 13px;
        font-weight: 500;
        color: #d0d8e8;
        line-height: 1.35;
        white-space: nowrap;
        overflow-x: auto;
    }
    .meta-value::-webkit-scrollbar {
        height: 4px;
    }
    .meta-value::-webkit-scrollbar-track {
        background: transparent;
    }
    .meta-value::-webkit-scrollbar-thumb {
        background: rgba(208, 216, 232, 0.2);
        border-radius: 10px;
    }

    /* ── Year Highlight Badge ── */
    .year-highlight {
        margin-top: 4px;
        padding: 6px 14px;
        display: inline-block;
        background: rgba(255, 138, 0, 0.12);
        border: 1px solid rgba(255, 138, 0, 0.25);
        border-radius: 20px;
        font-size: 11.5px;
        font-weight: 600;
        color: #f0c060;
        letter-spacing: 1.2px;
        text-transform: uppercase;
        text-align: center;
    }

    .year-section {
        text-align: center;
        padding: 2px 14px 14px 14px;
    }

    img {
        object-fit: contain;
        max-height: 300px;
        width: auto;
        display: block;
        margin: 0 auto;
    }
    hr {
        border: none !important;
        border-top: 10px solid #B2BEB5 !important;
        margin-top: 25px !important;
        margin-bottom: 25px !important;
        opacity: 1 !important;
        border-radius: 999px !important;
    }
    .book-column {
        position: relative;
        padding: 0;
        border: 2px solid #2b2b2b;
        border-radius: 12px;
        background-color: rgba(128, 128, 128, 0.05);
        box-shadow: 0 4px 8px rgba(0, 0, 0, 0.1);
        margin-top: 28px;
        margin-bottom: 15px;
        transition: transform 0.2s ease, box-shadow 0.2s ease;
        overflow: visible;
    }
    .book-column:hover {
        transform: translateY(-5px);
        box-shadow: 0 8px 16px rgba(0, 0, 0, 0.2);
    }
    .book-image-area {
        padding: 35px 20px 20px 20px;
    }
    .recommendation-badge {
        position: absolute;
        top: -22px;
        left: 50%;
        transform: translateX(-50%);
        width: 48px;
        height: 48px;
        border-radius: 50%;
        background: #28a745;
        color: white;
        border: 2px solid #2b2b2b;
        display: flex;
        align-items: center;
        justify-content: center;
        font-size: 18px;
        font-weight: bold;
        z-index: 10;
        box-shadow: 0 4px 10px rgba(0, 0, 0, 0.25);
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
                    book_info = final_filtered_df[final_filtered_df['title'] == book].iloc[0]
                    with cols[j]:
                        st.markdown(f"""
                        <div class='book-column'>
                            <div class='recommendation-badge'>{i + j + 1}</div>
                            <div class='book-image-area'>
                                <img src='{book_info['Image-URL-L']}' style='height:290px; width:auto; display:block;'>
                            </div>
                            <div class='book-info'>
                                <div class='book-title-area'>
                                    <div class='scroll-title'>{book}</div>
                                </div>
                                <div class='info-separator'></div>
                                <div class='book-meta'>
                                    <div class='meta-item'>
                                        <div class='meta-icon'>👤</div>
                                        <div class='meta-text-block'>
                                            <span class='meta-label'>Author</span>
                                            <span class='meta-value'>{book_info['Book-Author']}</span>
                                        </div>
                                    </div>
                                </div>
                                <div class='year-section'>
                                    <span class='year-highlight'>📅 Published · {book_info['Year-Of-Publication']}</span>
                                </div>
                            </div>
                        </div>
                        """, unsafe_allow_html=True)
            if i < len(similar_books) - 3:
                st.markdown("<br><hr><br>", unsafe_allow_html=True)

        st.markdown("<div class='extra-space'></div><div class='extra-space'></div>", unsafe_allow_html=True)
        st.image('https://github.com/MarpakaPradeepSai/Employee-Churn-Prediction/blob/main/Data/Images%20&%20GIFs/thank-you-33.gif?raw=true', use_container_width=True)
