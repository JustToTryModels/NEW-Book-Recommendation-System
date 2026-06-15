import streamlit as st
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from huggingface_hub import hf_hub_download
import warnings

warnings.filterwarnings('ignore')

# =========================================================================
# 1. DATA LOADING & SIMILARITY MATRIX (Cached for performance)
# =========================================================================
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

    # URL to replace (Fixing broken image links)
    url1 = 'http://images.amazon.com/images/P/0690040784.01.LZZZZZZZ.jpg'
    url2 = 'http://images.amazon.com/images/P/0451172817.01.LZZZZZZZ.jpg'
    url3 = 'http://images.amazon.com/images/P/0312084986.01.LZZZZZZZ.jpg'
    url4 = 'http://images.amazon.com/images/P/1590400356.01.LZZZZZZZ.jpg'

    final_filtered_df.loc[final_filtered_df['title'] == 'Jacob Have I Loved', 'Image-URL-L'] = url1
    final_filtered_df.loc[final_filtered_df['title'] == 'Needful Things', 'Image-URL-L'] = url2
    final_filtered_df.loc[final_filtered_df['title'] == 'All Creatures Great and Small', 'Image-URL-L'] = url3
    final_filtered_df.loc[final_filtered_df['title'] == "The Kitchen God's Wife", 'Image-URL-L'] = url4

    # -------------------------------------------------------------------------
    #  BUILD SIMILARITY MATRIX USING ONLY EXPLICIT RATINGS (>0)               
    # -------------------------------------------------------------------------
    explicit_ratings_df = final_filtered_df[final_filtered_df['rating'] > 0]
    book_user_mat = explicit_ratings_df.pivot_table(index='title', columns='userId', values='rating').fillna(0)

    # Calculate the cosine similarity matrix
    cosine_sim = cosine_similarity(book_user_mat)
    cosine_sim_df = pd.DataFrame(cosine_sim, index=book_user_mat.index, columns=book_user_mat.index)

    return final_filtered_df, cosine_sim_df

final_filtered_df, cosine_sim_df = load_and_prepare_data()

# =========================================================================
# 2. RECOMMENDATION LOGIC FUNCTIONS
# =========================================================================
def get_top_similar_books(book_title, n=10):
    if book_title not in cosine_sim_df.index:
        return "⚠️ Book not found in the database."
    
    similar_scores = cosine_sim_df[book_title]
    similar_books = similar_scores.sort_values(ascending=False)[1:n+1]
    return similar_books

def get_user_recommendations(user_id, df, sim_matrix, k=10):
    """
    Generates personalized recommendations for a specific user.
    """
    # Get all books the user interacted with (to exclude them from recommendations)
    user_history_all = df[df['userId'] == user_id]['title'].tolist()
    # Get books the user explicitly rated > 0 to show as context/history
    user_history_rated = df[(df['userId'] == user_id) & (df['rating'] > 0)][['title', 'rating']].sort_values(by='rating', ascending=False)

    if len(user_history_all) == 0:
        return [], user_history_rated, False # False indicates user not found

    # Aggregate similarity scores for all books similar to the user's history
    scores = {}
    for item in user_history_all:
        if item in sim_matrix.index:
            similar_items = sim_matrix[item].sort_values(ascending=False)[1:50] 
            
            for sim_item, score in similar_items.items():
                if sim_item not in user_history_all:
                    scores[sim_item] = scores.get(sim_item, 0) + score

    # Sort and return top K
    sorted_scores = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    top_recommendations = [x[0] for x in sorted_scores[:k]]
    
    return top_recommendations, user_history_rated, True

# =========================================================================
# 3. STREAMLIT UI & CSS
# =========================================================================
st.markdown("""
    <h1 style='font-size: 40px; text-align: center; margin-bottom: 5px; padding-bottom: 0px;'>
        Book Recommendation System
    </h1>
    <p class='subheader'>Let Us Help You Choose Your Next Book!</p>
""", unsafe_allow_html=True)

st.image('https://img.freepik.com/premium-vector/bookcase-with-books_182089-197.jpg', use_container_width=True)

st.markdown("""
    <style>
    html, body, [class*="css"], [class*="st-"], h1, h2, h3, h4, h5, h6, p, div, span, label, input, button, select, option, textarea {
        font-family: 'Tiempos', 'Tiempos Text', Georgia, 'Times New Roman', serif !important;
    }
    .subheader {
        font-size: 22px;
        font-weight: bold;
        margin-bottom: 10px;
        color: #1a73e8;
        text-align: center;
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
    
    /* Book Info Block */
    .book-info {
        background: #1e1e1e;
        padding: 20px 15px;
        border-radius: 0 0 10px 10px;
        border-top: 3px solid #e52e71;
        text-align: center;
        display: flex;
        flex-direction: column;
        justify-content: flex-start;
        align-items: center;
        min-height: 150px;
        height: 150px;
        box-sizing: border-box;
    }
    
    .premium-title {
        font-size: 16px;
        font-weight: bold;
        color: #F7E7A1;
        margin-bottom: 8px;
        line-height: 1.4;
        width: 100%;
        white-space: nowrap;
        overflow-x: auto;
        overflow-y: hidden;
        display: block;
        padding-bottom: 5px;
        height: 38px;
        box-sizing: border-box;
    }

    .premium-title::-webkit-scrollbar { height: 6px; }
    .premium-title::-webkit-scrollbar-thumb { background: #ccc; border-radius: 10px; }

    .premium-divider {
        width: 35px;
        height: 3px;
        background: linear-gradient(90deg, #ff8a00, #e52e71);
        margin: 6px 0 12px 0;
        border-radius: 5px;
    }

    .premium-author {
        font-size: 13.5px;
        color: #c4c4c4;
        font-style: italic;
        margin-bottom: 6px;
        width: 100%;
        white-space: nowrap;
        overflow: hidden;
        text-overflow: ellipsis;
    }

    .premium-year {
        font-size: 11.5px;
        color: #888888;
        text-transform: uppercase;
        letter-spacing: 1.2px;
        font-weight: 600;
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
    .extra-space { margin-top: 50px; }
    .recommendation-header {
        font-size: 15px;
        border-left: 5px solid #B2BEB5;
        padding-left: 12px;
        margin-left: 5px;
    }
    </style>
""", unsafe_allow_html=True)

# Session state initialization
if 'recommendations' not in st.session_state: st.session_state.recommendations = None
if 'recommended_book' not in st.session_state: st.session_state.recommended_book = None
if 'recommended_num' not in st.session_state: st.session_state.recommended_num = None

if 'user_recs' not in st.session_state: st.session_state.user_recs = None
if 'user_history' not in st.session_state: st.session_state.user_history = None
if 'user_id' not in st.session_state: st.session_state.user_id = None
if 'user_recs_num' not in st.session_state: st.session_state.user_recs_num = None
if 'user_found' not in st.session_state: st.session_state.user_found = None

# Create Tabs to separate the two recommendation systems
tab1, tab2 = st.tabs(["📚 Book-to-Book Recommendations", "👤 User-Specific Recommendations"])

# =========================================================================
# TAB 1: BOOK-TO-BOOK RECOMMENDATIONS
# =========================================================================
with tab1:
    all_books = sorted(final_filtered_df['title'].unique().tolist())
    book_title = st.selectbox('Enter a book title:', all_books, index=None, placeholder="Choose or enter a book title...", key='book_title')
    num_recommendations = st.number_input('Enter the number of recommendations:', min_value=1, max_value=50, value=10, key='book_rec_num')

    if st.button('Recommend books', key='btn_book'):
        if book_title:
            similar_books = get_top_similar_books(book_title, num_recommendations)
            st.session_state.recommendations = similar_books
            st.session_state.recommended_book = book_title
            st.session_state.recommended_num = num_recommendations
        else:
            st.session_state.recommendations = None
            st.warning("⚠️ Please select or enter a book title.")

    if st.session_state.recommendations is not None:
        similar_books = st.session_state.recommendations
        rec_book = st.session_state.recommended_book
        rec_num = st.session_state.recommended_num

        if isinstance(similar_books, str):
            st.warning(similar_books)
        else:
            st.markdown(f"<div class='recommendation-header'>Top {rec_num} recommendations for '<strong>{rec_book}</strong>':</div>", unsafe_allow_html=True)
            st.write("")
            
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
                                    <img src='{book_info['Image-URL-L']}' style='height:290px; width:auto; display:block;'>
                                </div>
                                <div class='book-info'>
                                    <div class='premium-title' title="{safe_title}">{book}</div>
                                    <div class='premium-divider'></div>
                                    <div class='premium-author' title="{safe_author}">{book_info['Book-Author']}</div>
                                    <div class='premium-year'>{book_info['Year-Of-Publication']}</div>
                                </div>
                            </div>
                            """, unsafe_allow_html=True)
                if i < len(similar_books) - 3:
                    st.markdown("<br><hr><br>", unsafe_allow_html=True)

            st.markdown("<div class='extra-space'></div><div class='extra-space'></div>", unsafe_allow_html=True)
            st.image('https://github.com/MarpakaPradeepSai/Employee-Churn-Prediction/blob/main/Data/Images%20&%20GIFs/thank-you-33.gif?raw=true', use_container_width=True)


# =========================================================================
# TAB 2: USER-SPECIFIC RECOMMENDATIONS
# =========================================================================
with tab2:
    st.markdown("### Find Personalized Recommendations for a Specific User")
    user_id_input = st.text_input("Enter User ID:", placeholder="e.g., 277427", key='user_id_input')
    num_user_recs = st.number_input("Enter the number of recommendations:", min_value=1, max_value=50, value=10, key='user_recs_num')
    
    if st.button("Get User Recommendations", key='btn_user'):
        if user_id_input:
            try:
                user_id = int(user_id_input.strip())
                recs, history, user_found = get_user_recommendations(user_id, final_filtered_df, cosine_sim_df, k=num_user_recs)
                st.session_state.user_recs = recs
                st.session_state.user_history = history
                st.session_state.user_id = user_id
                st.session_state.user_recs_num = num_user_recs
                st.session_state.user_found = user_found
            except ValueError:
                st.error("⚠️ Please enter a valid numeric User ID.")
        else:
            st.warning("⚠️ Please enter a User ID.")
            
    if st.session_state.user_recs is not None:
        recs = st.session_state.user_recs
        history = st.session_state.user_history
        uid = st.session_state.user_id
        unum = st.session_state.user_recs_num
        user_found = st.session_state.user_found
        
        if not user_found:
            st.error(f"❌ User ID {uid} not found in the database.")
        else:
            # Display User's Reading History Context
            st.markdown(f"#### 📖 Top Rated Books by User {uid}")
            if len(history) > 0:
                hist_cols = st.columns(min(len(history), 5))
                for i in range(min(len(history), 5)):
                    row = history.iloc[i]
                    book = row['title']
                    book_info = final_filtered_df[final_filtered_df['title'] == book].iloc[0]
                    
                    with hist_cols[i]:
                        st.markdown(f"""
                        <div style="text-align: center; padding: 15px; border: 1px solid #444; border-radius: 10px; background-color: #2a2a2a; color: white; height: 220px; display: flex; flex-direction: column; justify-content: center; align-items: center;">
                            <img src="{book_info['Image-URL-L']}" style="height: 110px; object-fit: contain; margin-bottom: 10px;">
                            <strong style="font-size: 14px; display: block; margin-bottom: 5px; color: #F7E7A1;">{book}</strong>
                            <span style="color: #ccc; font-size: 13px;">Your Rating: ⭐ {row['rating']}</span>
                        </div>
                        """, unsafe_allow_html=True)
                if len(history) > 5:
                    st.caption(f"... and {len(history) - 5} more explicitly rated books.")
            else:
                st.info("This user has no explicit ratings (>0), only implicit interactions. Recommendations are dynamically generated based on all their interactions.")
                
            st.markdown("---")
            
            # Display Recommended Books
            st.markdown(f"#### ✨ Top {unum} Recommendations for User {uid}")
            if len(recs) > 0:
                for i in range(0, len(recs), 3):
                    cols = st.columns(3)
                    for j in range(3):
                        if i + j < len(recs):
                            book = recs[i + j]
                            book_info = final_filtered_df[final_filtered_df['title'] == book].iloc[0]
                            
                            safe_title = str(book).replace('"', '&quot;').replace("'", "&#39;")
                            safe_author = str(book_info['Book-Author']).replace('"', '&quot;').replace("'", "&#39;")
                            
                            with cols[j]:
                                st.markdown(f"""
                                <div class='book-column'>
                                    <div class='recommendation-badge'>{i + j + 1}</div>
                                    <div class='book-image-area'>
                                        <img src='{book_info['Image-URL-L']}' style='height:290px; width:auto; display:block;'>
                                    </div>
                                    <div class='book-info'>
                                        <div class='premium-title' title="{safe_title}">{book}</div>
                                        <div class='premium-divider'></div>
                                        <div class='premium-author' title="{safe_author}">{book_info['Book-Author']}</div>
                                        <div class='premium-year'>{book_info['Year-Of-Publication']}</div>
                                    </div>
                                </div>
                                """, unsafe_allow_html=True)
                    if i < len(recs) - 3:
                        st.markdown("<br><hr><br>", unsafe_allow_html=True)
            else:
                st.warning("No recommendations could be generated. The user might be completely new or their books currently have no similar neighbors in the matrix.")
