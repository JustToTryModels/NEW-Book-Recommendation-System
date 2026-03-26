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
    @import url('https://fonts.googleapis.com/css2?family=Playfair+Display:wght@400;500;600;700&family=Cormorant+Garamond:wght@400;500;600;700&family=Lora:wght@400;500;600;700&display=swap');
    
    html, body, [class*="css"], [class*="st-"], h1, h2, h3, h4, h5, h6, p, div, span, label, input, button, select, option, textarea {
        font-family: 'Playfair Display', 'Cormorant Garamond', 'Lora', Georgia, serif !important;
    }
    .subheader {
        font-size: 22px;
        font-weight: bold;
        margin-bottom: 20px;
        color: #1a73e8;
    }
    .stButton > button {
        font-family: 'Playfair Display', Georgia, serif !important;
        font-size: 16px;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 50%, #f093fb 100%);
        color: white !important;
        border: none;
        border-radius: 30px;
        padding: 12px 28px;
        font-weight: 600;
        cursor: pointer;
        transition: all 0.4s cubic-bezier(0.175, 0.885, 0.32, 1.275);
        display: inline-flex;
        align-items: center;
        justify-content: center;
        margin: 4px 2px;
        width: auto;
        min-width: 100px;
        box-shadow: 0 4px 15px rgba(102, 126, 234, 0.4);
        letter-spacing: 0.5px;
        text-transform: uppercase;
    }
    .stButton > button:hover {
        transform: translateY(-3px) scale(1.02);
        box-shadow: 0 8px 25px rgba(102, 126, 234, 0.6);
        color: white !important;
        background: linear-gradient(135deg, #764ba2 0%, #667eea 50%, #f093fb 100%);
    }
    .stButton > button:active {
        transform: translateY(0) scale(0.98);
    }
    
    /* Ultra Premium Book Card */
    .book-column {
        position: relative;
        padding: 0;
        border-radius: 24px;
        background: linear-gradient(145deg, #1a1a2e 0%, #16213e 50%, #0f0f23 100%);
        margin-top: 35px;
        margin-bottom: 20px;
        transition: all 0.5s cubic-bezier(0.23, 1, 0.32, 1);
        overflow: visible;
        border: 1px solid rgba(255, 255, 255, 0.08);
        box-shadow: 
            0 10px 40px rgba(0, 0, 0, 0.4),
            0 0 0 1px rgba(255, 255, 255, 0.05),
            inset 0 1px 0 rgba(255, 255, 255, 0.1);
    }
    
    .book-column::before {
        content: '';
        position: absolute;
        top: 0;
        left: 0;
        right: 0;
        bottom: 0;
        border-radius: 24px;
        background: linear-gradient(135deg, 
            rgba(102, 126, 234, 0.1) 0%, 
            rgba(118, 75, 162, 0.05) 50%, 
            rgba(240, 147, 251, 0.1) 100%);
        opacity: 0;
        transition: opacity 0.5s ease;
        pointer-events: none;
    }
    
    .book-column:hover {
        transform: translateY(-12px) scale(1.02);
        box-shadow: 
            0 25px 60px rgba(102, 126, 234, 0.3),
            0 15px 40px rgba(0, 0, 0, 0.4),
            0 0 60px rgba(240, 147, 251, 0.15),
            inset 0 1px 0 rgba(255, 255, 255, 0.2);
        border-color: rgba(102, 126, 234, 0.3);
    }
    
    .book-column:hover::before {
        opacity: 1;
    }
    
    /* Premium Book Image Area */
    .book-image-area {
        padding: 45px 25px 25px 25px;
        position: relative;
        display: flex;
        justify-content: center;
        align-items: center;
        min-height: 320px;
    }
    
    .book-image-area::after {
        content: '';
        position: absolute;
        bottom: 0;
        left: 10%;
        right: 10%;
        height: 20px;
        background: radial-gradient(ellipse at center, rgba(0,0,0,0.3) 0%, transparent 70%);
        filter: blur(8px);
    }
    
    .book-image-area img {
        height: 280px !important;
        width: auto !important;
        max-width: 100%;
        object-fit: contain;
        display: block;
        border-radius: 8px;
        box-shadow: 
            0 15px 35px rgba(0, 0, 0, 0.5),
            0 5px 15px rgba(0, 0, 0, 0.3),
            -5px 0 20px rgba(0, 0, 0, 0.2),
            5px 0 20px rgba(0, 0, 0, 0.2);
        transition: all 0.5s cubic-bezier(0.23, 1, 0.32, 1);
        position: relative;
        z-index: 2;
    }
    
    .book-column:hover .book-image-area img {
        transform: scale(1.05) rotateY(-5deg);
        box-shadow: 
            0 25px 50px rgba(0, 0, 0, 0.6),
            0 10px 25px rgba(102, 126, 234, 0.2),
            -8px 0 25px rgba(0, 0, 0, 0.3),
            8px 0 25px rgba(0, 0, 0, 0.3);
    }
    
    /* Luxurious Book Info Section */
    .book-info {
        background: linear-gradient(180deg, 
            rgba(26, 26, 46, 0.95) 0%, 
            rgba(15, 15, 35, 0.98) 100%);
        padding: 28px 20px 32px 20px;
        border-radius: 0 0 24px 24px;
        text-align: center;
        display: flex;
        flex-direction: column;
        justify-content: flex-start;
        align-items: center;
        min-height: 160px;
        position: relative;
        border-top: 1px solid rgba(255, 255, 255, 0.05);
    }
    
    .book-info::before {
        content: '';
        position: absolute;
        top: 0;
        left: 20%;
        right: 20%;
        height: 1px;
        background: linear-gradient(90deg, 
            transparent, 
            rgba(102, 126, 234, 0.5), 
            rgba(240, 147, 251, 0.5), 
            transparent);
    }
    
    /* Premium Title Styling */
    .premium-title {
        font-family: 'Playfair Display', Georgia, serif !important;
        font-size: 17px;
        font-weight: 600;
        color: #ffffff;
        margin-bottom: 12px;
        line-height: 1.5;
        width: 100%;
        white-space: nowrap;
        overflow-x: auto;
        overflow-y: hidden;
        display: block;
        padding-bottom: 8px;
        text-shadow: 0 2px 10px rgba(0, 0, 0, 0.3);
        letter-spacing: 0.3px;
    }

    .premium-title::-webkit-scrollbar {
        height: 4px;
    }

    .premium-title::-webkit-scrollbar-track {
        background: rgba(255, 255, 255, 0.05);
        border-radius: 10px;
    }

    .premium-title::-webkit-scrollbar-thumb {
        background: linear-gradient(90deg, #667eea, #764ba2);
        border-radius: 10px;
    }

    /* Elegant Divider */
    .premium-divider {
        width: 50px;
        height: 3px;
        background: linear-gradient(90deg, #667eea, #764ba2, #f093fb);
        margin: 8px 0 16px 0;
        border-radius: 10px;
        position: relative;
        box-shadow: 0 0 15px rgba(102, 126, 234, 0.5);
    }
    
    .premium-divider::before,
    .premium-divider::after {
        content: '◆';
        position: absolute;
        top: 50%;
        transform: translateY(-50%);
        font-size: 6px;
        color: #f093fb;
    }
    
    .premium-divider::before {
        left: -12px;
    }
    
    .premium-divider::after {
        right: -12px;
    }

    /* Author Styling */
    .premium-author {
        font-family: 'Cormorant Garamond', Georgia, serif !important;
        font-size: 15px;
        color: #b8c5d6;
        font-style: italic;
        margin-bottom: 10px;
        width: 100%;
        white-space: nowrap;
        overflow: hidden;
        text-overflow: ellipsis;
        letter-spacing: 0.5px;
        text-shadow: 0 1px 5px rgba(0, 0, 0, 0.2);
    }

    /* Year Badge Styling */
    .premium-year {
        font-family: 'Lora', Georgia, serif !important;
        font-size: 11px;
        color: #8892a6;
        text-transform: uppercase;
        letter-spacing: 2.5px;
        font-weight: 600;
        padding: 6px 16px;
        background: linear-gradient(135deg, rgba(102, 126, 234, 0.15), rgba(118, 75, 162, 0.15));
        border-radius: 20px;
        border: 1px solid rgba(102, 126, 234, 0.2);
        margin-top: 5px;
    }
    
    /* Spectacular Recommendation Badge */
    .recommendation-badge {
        position: absolute;
        top: -18px;
        left: 50%;
        transform: translateX(-50%);
        width: 56px;
        height: 56px;
        border-radius: 50%;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 50%, #f093fb 100%);
        color: white;
        border: 4px solid #0f0f23;
        display: flex;
        align-items: center;
        justify-content: center;
        font-family: 'Playfair Display', Georgia, serif !important;
        font-size: 20px;
        font-weight: 700;
        z-index: 15;
        box-shadow: 
            0 8px 25px rgba(102, 126, 234, 0.5),
            0 4px 15px rgba(0, 0, 0, 0.3),
            inset 0 2px 4px rgba(255, 255, 255, 0.3);
        transition: all 0.4s cubic-bezier(0.23, 1, 0.32, 1);
    }
    
    .recommendation-badge::before {
        content: '';
        position: absolute;
        width: 100%;
        height: 100%;
        border-radius: 50%;
        background: linear-gradient(135deg, rgba(255,255,255,0.3), transparent);
        top: 0;
        left: 0;
    }
    
    .recommendation-badge::after {
        content: '';
        position: absolute;
        width: 68px;
        height: 68px;
        border-radius: 50%;
        border: 2px solid rgba(102, 126, 234, 0.3);
        animation: pulse-ring 2s ease-out infinite;
    }
    
    @keyframes pulse-ring {
        0% {
            transform: scale(0.9);
            opacity: 1;
        }
        100% {
            transform: scale(1.3);
            opacity: 0;
        }
    }
    
    .book-column:hover .recommendation-badge {
        transform: translateX(-50%) scale(1.1) rotate(5deg);
        box-shadow: 
            0 12px 35px rgba(102, 126, 234, 0.6),
            0 6px 20px rgba(240, 147, 251, 0.4);
    }
    
    /* Premium Horizontal Divider */
    hr {
        border: none !important;
        height: 4px !important;
        background: linear-gradient(90deg, 
            transparent, 
            rgba(102, 126, 234, 0.3), 
            rgba(118, 75, 162, 0.5), 
            rgba(240, 147, 251, 0.3), 
            transparent) !important;
        margin-top: 35px !important;
        margin-bottom: 35px !important;
        opacity: 1 !important;
        border-radius: 999px !important;
        position: relative;
    }
    
    .extra-space {
        margin-top: 50px;
    }
    
    .recommendation-header {
        font-family: 'Playfair Display', Georgia, serif !important;
        font-size: 16px;
        border-left: 4px solid;
        border-image: linear-gradient(180deg, #667eea, #764ba2, #f093fb) 1;
        padding-left: 15px;
        margin-left: 5px;
        color: #e0e0e0;
        text-shadow: 0 1px 3px rgba(0, 0, 0, 0.2);
    }
    
    img {
        object-fit: contain;
        max-height: 300px;
        width: auto;
        display: block;
        margin: 0 auto;
    }
    
    /* Shine Animation on Hover */
    .book-column::after {
        content: '';
        position: absolute;
        top: 0;
        left: -100%;
        width: 100%;
        height: 100%;
        background: linear-gradient(
            90deg,
            transparent,
            rgba(255, 255, 255, 0.05),
            transparent
        );
        transition: left 0.7s ease;
        border-radius: 24px;
        pointer-events: none;
    }
    
    .book-column:hover::after {
        left: 100%;
    }
    
    /* Star decoration for top cards */
    .book-column[data-rank="1"]::before,
    .book-column[data-rank="2"]::before,
    .book-column[data-rank="3"]::before {
        content: '★';
        position: absolute;
        top: 8px;
        right: 12px;
        font-size: 14px;
        color: gold;
        text-shadow: 0 0 10px rgba(255, 215, 0, 0.5);
        z-index: 5;
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
                    
                    # Prevent quotes in variables from breaking HTML attributes occasionally
                    safe_title = str(book).replace('"', '&quot;').replace("'", "&#39;")
                    safe_author = str(book_info['Book-Author']).replace('"', '&quot;').replace("'", "&#39;")
                    rank = i + j + 1
                    
                    with cols[j]:
                        st.markdown(f"""
                        <div class='book-column' data-rank="{rank}">
                            <div class='recommendation-badge'>{rank}</div>
                            <div class='book-image-area'>
                                <img src='{book_info['Image-URL-L']}'>
                            </div>
                            <div class='book-info'>
                                <div class='premium-title' title="{safe_title}">{book}</div>
                                <div class='premium-divider'></div>
                                <div class='premium-author' title="{safe_author}">By {book_info['Book-Author']}</div>
                                <div class='premium-year'>Published {book_info['Year-Of-Publication']}</div>
                            </div>
                        </div>
                        """, unsafe_allow_html=True)
            if i < len(similar_books) - 3:
                st.markdown("<br><hr><br>", unsafe_allow_html=True)

        st.markdown("<div class='extra-space'></div><div class='extra-space'></div>", unsafe_allow_html=True)
        st.image('https://github.com/MarpakaPradeepSai/Employee-Churn-Prediction/blob/main/Data/Images%20&%20GIFs/thank-you-33.gif?raw=true', use_container_width=True)
