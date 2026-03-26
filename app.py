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

# Streamlit app configuration
st.markdown("""
    <style>
    /* IMPORT PREMIUM FONTS */
    @import url('https://fonts.googleapis.com/css2?family=Playfair+Display:wght@700&family=Lato:wght@300;400;700&display=swap');

    /* GLOBAL STYLES */
    html, body, [class*="css"], [class*="st-"] {
        font-family: 'Lato', sans-serif;
    }
    
    h1, h2, h3 {
        font-family: 'Playfair Display', serif;
    }

    /* MAIN TITLE STYLING */
    .main-title {
        font-family: 'Playfair Display', serif;
        font-size: 3.5rem;
        background: linear-gradient(90deg, #FFD700, #E5E4E2);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.5);
        margin-bottom: 10px;
    }

    .subheader {
        font-size: 1.5rem;
        color: #B0B0B0;
        font-weight: 300;
        margin-bottom: 30px;
        letter-spacing: 1px;
    }

    /* BUTTON STYLING */
    .stButton > button {
        font-family: 'Lato', sans-serif;
        font-size: 16px;
        background: linear-gradient(135deg, #d4af37 0%, #a67c00 100%);
        color: white !important;
        border: none;
        border-radius: 5px;
        padding: 12px 30px;
        font-weight: bold;
        text-transform: uppercase;
        letter-spacing: 2px;
        cursor: pointer;
        transition: all 0.3s ease;
        box-shadow: 0 4px 15px rgba(212, 175, 55, 0.4);
    }
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 20px rgba(212, 175, 55, 0.6);
        background: linear-gradient(135deg, #ebd174 0%, #b88a00 100%);
    }

    /* CARD CONTAINER */
    .premium-card {
        background: #1a1a1a;
        border: 1px solid #333;
        border-radius: 12px;
        overflow: hidden;
        position: relative;
        transition: all 0.4s cubic-bezier(0.175, 0.885, 0.32, 1.275);
        box-shadow: 0 10px 20px rgba(0,0,0,0.3);
        height: 100%;
        display: flex;
        flex-direction: column;
    }
    
    .premium-card:hover {
        transform: translateY(-10px);
        box-shadow: 0 20px 40px rgba(0,0,0,0.6);
        border-color: #d4af37; /* Gold border on hover */
    }

    /* IMAGE AREA */
    .card-img-wrapper {
        width: 100%;
        height: 280px;
        background-color: #121212;
        display: flex;
        align-items: center;
        justify-content: center;
        padding: 20px;
        position: relative;
        overflow: hidden;
    }

    .card-img-wrapper img {
        height: 100%;
        width: auto;
        object-fit: contain;
        box-shadow: 0 10px 25px rgba(0,0,0,0.5);
        transition: transform 0.5s ease;
        z-index: 1;
    }

    .premium-card:hover .card-img-wrapper img {
        transform: scale(1.08);
    }

    /* RANK BADGE */
    .rank-badge {
        position: absolute;
        top: 15px;
        left: 15px;
        width: 40px;
        height: 40px;
        background: linear-gradient(135deg, #d4af37, #f3e5ab, #a67c00);
        color: #222;
        border-radius: 50%;
        display: flex;
        align-items: center;
        justify-content: center;
        font-family: 'Playfair Display', serif;
        font-weight: 900;
        font-size: 18px;
        z-index: 10;
        box-shadow: 0 4px 10px rgba(0,0,0,0.5);
        border: 2px solid #fff;
    }

    /* CONTENT AREA */
    .card-content {
        padding: 18px 15px;
        text-align: center;
        background: linear-gradient(to bottom, #1e1e1e, #141414);
        flex-grow: 1;
        display: flex;
        flex-direction: column;
        justify-content: flex-start;
        border-top: 1px solid #333;
    }

    /* SCROLLABLE TITLE */
    .book-title {
        font-family: 'Playfair Display', serif;
        font-size: 18px;
        color: #ffffff;
        margin-bottom: 8px;
        white-space: nowrap;
        overflow-x: auto;
        overflow-y: hidden;
        padding-bottom: 5px;
    }
    
    /* Custom Scrollbar for Title */
    .book-title::-webkit-scrollbar {
        height: 4px;
    }
    .book-title::-webkit-scrollbar-track {
        background: #2a2a2a; 
        border-radius: 2px;
    }
    .book-title::-webkit-scrollbar-thumb {
        background: #555; 
        border-radius: 2px;
    }
    .book-title::-webkit-scrollbar-thumb:hover {
        background: #d4af37; 
    }

    /* DECORATIVE DIVIDER */
    .gold-divider {
        height: 2px;
        width: 40px;
        background: linear-gradient(90deg, transparent, #d4af37, transparent);
        margin: 5px auto 10px auto;
    }

    /* AUTHOR & YEAR */
    .book-author {
        font-family: 'Lato', sans-serif;
        font-size: 14px;
        color: #aaa;
        font-style: italic;
        margin-bottom: 5px;
        white-space: nowrap;
        overflow: hidden;
        text-overflow: ellipsis;
    }

    .book-year {
        font-family: 'Lato', sans-serif;
        font-size: 11px;
        color: #666;
        font-weight: 700;
        letter-spacing: 1.5px;
        text-transform: uppercase;
        margin-top: auto; /* Pushes to bottom if flex spacing needed */
    }

    /* SEPARATOR BETWEEN ROWS */
    hr.fancy-hr {
        border: 0;
        height: 1px;
        background-image: linear-gradient(to right, rgba(255, 255, 255, 0), rgba(212, 175, 55, 0.75), rgba(255, 255, 255, 0));
        margin: 40px 0;
    }
    
    .results-header {
        font-family: 'Playfair Display', serif;
        font-size: 24px;
        color: #fff;
        border-left: 4px solid #d4af37;
        padding-left: 15px;
        margin: 30px 0 20px 0;
    }
    </style>
""", unsafe_allow_html=True)

# App Header
st.markdown("<h1 class='main-title'>Literary Compass</h1>", unsafe_allow_html=True)
st.markdown("<p class='subheader'>Curating your next masterpiece reading experience.</p>", unsafe_allow_html=True)
st.image('https://img.freepik.com/premium-vector/bookcase-with-books_182089-197.jpg', use_container_width=True)

# Input Section
col1, col2 = st.columns([3, 1])
with col1:
    all_books = final_filtered_df['title'].unique().tolist()
    book_title = st.selectbox('Select a Masterpiece:', all_books, index=None, placeholder="Search your library...", key='book_title')
with col2:
    num_recommendations = st.number_input('Count:', min_value=1, max_value=50, value=10)

if 'recommendations' not in st.session_state:
    st.session_state.recommendations = None
if 'recommended_book' not in st.session_state:
    st.session_state.recommended_book = None
if 'recommended_num' not in st.session_state:
    st.session_state.recommended_num = None

if st.button('Discover Recommendations'):
    if book_title:
        similar_books = get_top_similar_books(book_title, num_recommendations)
        st.session_state.recommendations = similar_books
        st.session_state.recommended_book = book_title
        st.session_state.recommended_num = num_recommendations
    else:
        st.session_state.recommendations = None
        st.warning("Please select a book title to proceed.")

# Results Display
if st.session_state.recommendations is not None:
    similar_books = st.session_state.recommendations
    rec_book = st.session_state.recommended_book
    rec_num = st.session_state.recommended_num

    if isinstance(similar_books, str):
        st.error(similar_books)
    else:
        # Styled Header for Results
        st.markdown(f"<div class='results-header'>Curated Selections based on '<em>{rec_book}</em>'</div>", unsafe_allow_html=True)
        
        # Grid System
        for i in range(0, len(similar_books), 3):
            cols = st.columns(3)
            for j in range(3):
                if i + j < len(similar_books):
                    book_idx = i + j
                    book_name = similar_books.index[book_idx]
                    book_info = final_filtered_df[final_filtered_df['title'] == book_name].iloc[0]
                    
                    # Safe strings for HTML attributes
                    safe_title = str(book_name).replace('"', '&quot;').replace("'", "&#39;")
                    safe_author = str(book_info['Book-Author']).replace('"', '&quot;').replace("'", "&#39;")
                    
                    with cols[j]:
                        st.markdown(f"""
                        <div class='premium-card'>
                            <div class='rank-badge'>{book_idx + 1}</div>
                            <div class='card-img-wrapper'>
                                <img src='{book_info['Image-URL-L']}' alt="Book Cover">
                            </div>
                            <div class='card-content'>
                                <div class='book-title' title="{safe_title}">{book_name}</div>
                                <div class='gold-divider'></div>
                                <div class='book-author' title="{safe_author}">By {book_info['Book-Author']}</div>
                                <div class='book-year'>{book_info['Year-Of-Publication']}</div>
                            </div>
                        </div>
                        """, unsafe_allow_html=True)
            
            # Add fancy divider between rows (except last row)
            if i < len(similar_books) - 3:
                st.markdown("<hr class='fancy-hr'>", unsafe_allow_html=True)

        st.markdown("<br><br>", unsafe_allow_html=True)
        st.image('https://github.com/MarpakaPradeepSai/Employee-Churn-Prediction/blob/main/Data/Images%20&%20GIFs/thank-you-33.gif?raw=true', use_container_width=True)
