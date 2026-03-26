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

# Streamlit app styling
st.markdown("""
    <style>
    /* Base Typography & Background */
    html, body, [class*="css"], [class*="st-"], h1, h2, h3, h4, h5, h6, p, div, span, label, input, button, select, option, textarea {
        font-family: 'Tiempos', 'Tiempos Text', Georgia, 'Times New Roman', serif !important;
    }
    
    /* Main Header Styling */
    .premium-header {
        font-size: 48px;
        font-weight: 800;
        text-align: center;
        background: linear-gradient(45deg, #ff8a00, #e52e71);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 10px;
        letter-spacing: 1px;
        padding-top: 20px;
    }
    
    .subheader {
        font-size: 20px;
        font-weight: 500;
        text-align: center;
        margin-bottom: 30px;
        color: #a0aab5;
        letter-spacing: 0.5px;
    }

    /* Premium Button Styling */
    .stButton > button {
        font-family: 'Tiempos', 'Tiempos Text', Georgia, 'Times New Roman', serif !important;
        font-size: 16px;
        background: linear-gradient(135deg, #ff8a00 0%, #e52e71 100%);
        color: white !important;
        border: 1px solid rgba(255, 255, 255, 0.2);
        border-radius: 30px;
        padding: 12px 28px;
        font-weight: bold;
        cursor: pointer;
        transition: all 0.3s cubic-bezier(0.175, 0.885, 0.32, 1.275);
        display: inline-flex;
        align-items: center;
        justify-content: center;
        width: auto;
        min-width: 120px;
        box-shadow: 0 8px 20px rgba(229, 46, 113, 0.3);
    }
    .stButton > button:hover {
        transform: translateY(-3px) scale(1.02);
        box-shadow: 0 12px 25px rgba(229, 46, 113, 0.5);
        border: 1px solid rgba(255, 255, 255, 0.4);
    }
    .stButton > button:active {
        transform: translateY(1px) scale(0.98);
    }
    
    /* Sleek Book Card Container */
    .book-column {
        position: relative;
        padding: 0;
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 18px;
        background: linear-gradient(145deg, #1c1c1c, #121212);
        box-shadow: 0 10px 30px rgba(0, 0, 0, 0.4);
        margin-top: 35px;
        margin-bottom: 20px;
        transition: all 0.4s cubic-bezier(0.25, 0.8, 0.25, 1);
        display: flex;
        flex-direction: column;
        height: 100%;
        overflow: visible;
    }
    .book-column:hover {
        transform: translateY(-8px);
        box-shadow: 0 15px 40px rgba(229, 46, 113, 0.25);
        border-color: rgba(229, 46, 113, 0.4);
    }

    /* Rank Badge - Gold/Peach Metallic Gradient */
    .recommendation-badge {
        position: absolute;
        top: -22px;
        left: 50%;
        transform: translateX(-50%);
        width: 46px;
        height: 46px;
        border-radius: 50%;
        background: linear-gradient(135deg, #f6d365 0%, #fda085 100%);
        color: #111;
        border: 3px solid #1a1a1a;
        display: flex;
        align-items: center;
        justify-content: center;
        font-size: 18px;
        font-weight: 800;
        z-index: 10;
        box-shadow: 0 5px 15px rgba(253, 160, 133, 0.4);
    }

    /* Book Cover Image Area */
    .book-image-area {
        padding: 40px 20px 25px 20px;
        display: flex;
        justify-content: center;
        align-items: center;
        background: radial-gradient(circle at top, rgba(255,255,255,0.05) 0%, rgba(0,0,0,0) 70%);
        border-radius: 18px 18px 0 0;
    }
    .book-image-area img {
        height: 270px;
        width: auto;
        object-fit: contain;
        border-radius: 6px;
        box-shadow: 5px 15px 25px rgba(0,0,0,0.6);
        transition: transform 0.4s ease;
    }
    .book-column:hover .book-image-area img {
        transform: scale(1.04) rotate(1deg);
    }

    /* Bottom Info Card Area */
    .book-info {
        background: rgba(20, 20, 20, 0.8);
        padding: 25px 20px;
        border-radius: 0 0 18px 18px;
        border-top: 1px solid rgba(255, 255, 255, 0.05);
        text-align: center;
        display: flex;
        flex-direction: column;
        justify-content: flex-start;
        align-items: center;
        flex-grow: 1;
        backdrop-filter: blur(10px);
    }
    
    .premium-title {
        font-size: 18px;
        font-weight: 700;
        color: #f8f9fa;
        margin-bottom: 8px;
        line-height: 1.3;
        width: 100%;
        /* 2-line clamp for an elegant look without scrollbars */
        display: -webkit-box;
        -webkit-line-clamp: 2;
        -webkit-box-orient: vertical;
        overflow: hidden;
        text-overflow: ellipsis;
        white-space: normal;
    }

    .premium-divider {
        width: 40px;
        height: 2px;
        background: linear-gradient(90deg, #ff8a00, #e52e71);
        margin: 12px 0;
        border-radius: 5px;
        opacity: 0.8;
    }

    .premium-author {
        font-size: 14px;
        color: #b0b0b0;
        font-style: italic;
        margin-bottom: 12px;
        width: 100%;
        white-space: nowrap;
        overflow: hidden;
        text-overflow: ellipsis;
        font-weight: 300;
    }

    .premium-year {
        font-size: 12px;
        color: #ff8a00;
        text-transform: uppercase;
        letter-spacing: 2px;
        font-weight: 700;
        margin-top: auto;
    }
    
    /* Fading Gradient Divider */
    hr {
        border: none !important;
        height: 1px !important;
        background: linear-gradient(90deg, transparent, rgba(229, 46, 113, 0.5), transparent) !important;
        margin: 40px 0 !important;
        opacity: 1 !important;
    }
    
    .extra-space {
        margin-top: 50px;
    }
    
    /* Result Header Styling */
    .recommendation-header {
        font-size: 20px;
        font-weight: 600;
        border-left: 4px solid #e52e71;
        padding-left: 15px;
        margin-left: 5px;
        margin-bottom: 25px;
        color: #f8f9fa;
        background: linear-gradient(90deg, rgba(229,46,113,0.1) 0%, transparent 100%);
        padding-top: 5px;
        padding-bottom: 5px;
        border-radius: 0 5px 5px 0;
    }
    </style>
""", unsafe_allow_html=True)

st.markdown("<div class='premium-header'>Book Recommendation System</div>", unsafe_allow_html=True)
st.markdown("<p class='subheader'>Let Us Help You Choose Your Next Masterpiece</p>", unsafe_allow_html=True)
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
        st.markdown(f"<div class='recommendation-header'>Top {rec_num} recommendations for '<strong>{rec_book}</strong>'</div>", unsafe_allow_html=True)
        st.write("")
        
        for i in range(0, len(similar_books), 3):
            cols = st.columns(3)
            for j in range(3):
                if i + j < len(similar_books):
                    book = similar_books.index[i + j]
                    book_info = final_filtered_df[final_filtered_df['title'] == book].iloc[0]
                    
                    # Prevent quotes in variables from breaking HTML attributes occasionally
                    safe_title = str(book).replace('"', '&quot;').replace("'", "&#39;")
                    safe_author = str(book_info['Book-Author']).replace('"', '&quot;').replace("'", "&#39;")
                    
                    with cols[j]:
                        st.markdown(f"""
                        <div class='book-column'>
                            <div class='recommendation-badge'>{i + j + 1}</div>
                            <div class='book-image-area'>
                                <img src='{book_info['Image-URL-L']}'>
                            </div>
                            <div class='book-info'>
                                <div class='premium-title' title="{safe_title}">{book}</div>
                                <div class='premium-divider'></div>
                                <div class='premium-author' title="{safe_author}">By {book_info['Book-Author']}</div>
                                <div class='premium-year'>{book_info['Year-Of-Publication']}</div>
                            </div>
                        </div>
                        """, unsafe_allow_html=True)
            if i < len(similar_books) - 3:
                st.markdown("<hr>", unsafe_allow_html=True)

        st.markdown("<div class='extra-space'></div><div class='extra-space'></div>", unsafe_allow_html=True)
        st.image('https://github.com/MarpakaPradeepSai/Employee-Churn-Prediction/blob/main/Data/Images%20&%20GIFs/thank-you-33.gif?raw=true', use_container_width=True)
