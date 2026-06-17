import streamlit as st
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from huggingface_hub import hf_hub_download
import warnings

warnings.filterwarnings('ignore')

# -------------------------------------------------------------------------
# PAGE CONFIGURATION
# -------------------------------------------------------------------------
st.set_page_config(page_title="Next Chapter | Book Recommender", page_icon="✨", layout="wide", initial_sidebar_state="collapsed")

# -------------------------------------------------------------------------
# DATA LOADING & PREPARATION
# -------------------------------------------------------------------------
@st.cache_data(show_spinner=False)
def load_and_prepare_data():
    # Load your final filtered dataframe from Hugging Face
    final_filtered_df_path = hf_hub_download(repo_id="IamPradeep/BRS_DATA", filename="final_filtered_df.csv", repo_type="dataset")
    final_filtered_df = pd.read_csv(final_filtered_df_path)

    # Load the dataframe containing book URLs from Hugging Face
    book_urls_df_path = hf_hub_download(repo_id="IamPradeep/BRS_DATA", filename="Books.csv", repo_type="dataset")
    book_urls_df = pd.read_csv(book_urls_df_path)
    book_urls_df.rename(columns={'Book-Title': 'title'}, inplace=True)

    # ✅ FIX 1: Drop duplicate titles before merging to prevent row multiplication!
    book_urls_df = book_urls_df.drop_duplicates(subset=['title'], keep='first')

    # Merge the dataframes on the title
    final_filtered_df = final_filtered_df.merge(book_urls_df[['title', 'Book-Author', 'Year-Of-Publication', 'Image-URL-L']], on='title', how='left')

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

    # -------------------------------------------------------------------------
    #  BUILD SIMILARITY MATRIX USING ONLY EXPLICIT RATINGS (>0)               
    # -------------------------------------------------------------------------
    explicit_ratings_df = final_filtered_df[final_filtered_df['rating'] > 0]
    book_user_mat = explicit_ratings_df.pivot_table(index='title', columns='userId', values='rating').fillna(0)

    # Calculate the cosine similarity matrix
    cosine_sim = cosine_similarity(book_user_mat)
    cosine_sim_df = pd.DataFrame(cosine_sim, index=book_user_mat.index, columns=book_user_mat.index)

    return final_filtered_df, cosine_sim_df

with st.spinner("📚 Dusting off the bookshelves... Fetching your data!"):
    final_filtered_df, cosine_sim_df = load_and_prepare_data()

# -------------------------------------------------------------------------
# HELPER FUNCTIONS
# -------------------------------------------------------------------------

def get_top_similar_books(book_title, n=10):
    """Get similar books based on book title"""
    if book_title not in cosine_sim_df.index:
        return None
    similar_scores = cosine_sim_df[book_title]
    similar_books = similar_scores.sort_values(ascending=False)[1:n+1]
    return similar_books

def get_user_recommendations(user_id, df, sim_matrix, k=10):
    """Generates personalized recommendations for a specific user."""
    user_history_all = df[df['userId'] == user_id]['title'].unique().tolist()
    user_history_rated = df[df['userId'] == user_id][['title', 'rating']].sort_values(by='rating', ascending=False)
    user_history_rated = user_history_rated.drop_duplicates(subset=['title'])

    if len(user_history_all) == 0:
        return None, None

    scores = {}
    for item in user_history_all:
        if item in sim_matrix.index:
            similar_items = sim_matrix[item].sort_values(ascending=False)[1:50] 
            for sim_item, score in similar_items.items():
                if sim_item not in user_history_all:
                    scores[sim_item] = scores.get(sim_item, 0) + score

    sorted_scores = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    top_recommendations = [x[0] for x in sorted_scores[:k]]
    return top_recommendations, user_history_rated

def display_book_cards(books_list, start_index=0):
    """Display books in a 4-column highly interactive card layout"""
    # 4 columns for a wider, more immersive layout
    for i in range(0, len(books_list), 4):
        cols = st.columns(4)
        for j in range(4):
            if i + j < len(books_list):
                book = books_list[i + j]
                book_info = final_filtered_df[final_filtered_df['title'] == book].iloc[0]
                
                safe_title = str(book).replace('"', '&quot;').replace("'", "&#39;")
                safe_author = str(book_info['Book-Author']).replace('"', '&quot;').replace("'", "&#39;")
                
                with cols[j]:
                    st.markdown(f"""
                    <div class='book-column'>
                        <div class='recommendation-badge'>{start_index + i + j + 1}</div>
                        <div class='book-image-area'>
                            <img src='{book_info['Image-URL-L']}' class='book-cover'>
                        </div>
                        <div class='book-info'>
                            <div class='premium-title' title="{safe_title}">{book}</div>
                            <div class='premium-divider'></div>
                            <div class='premium-author' title="{safe_author}">✍️ {book_info['Book-Author']}</div>
                            <div class='premium-year'>📅 {book_info['Year-Of-Publication']}</div>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)

# -------------------------------------------------------------------------
# SUPERIOR CSS STYLING
# -------------------------------------------------------------------------
st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;600;800&display=swap');

    /* Global Typography */
    html, body, [class*="css"] {
        font-family: 'Outfit', sans-serif !important;
    }
    
    /* Main Header Styling */
    .main-header {
        font-size: 55px;
        font-weight: 800;
        text-align: center;
        background: linear-gradient(135deg, #FF6B6B 0%, #4ECDC4 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: -15px;
        padding-top: 20px;
    }
    .subheader {
        font-size: 20px;
        font-weight: 400;
        margin-bottom: 40px;
        color: #8892B0;
        text-align: center;
        letter-spacing: 1px;
    }

    /* Input & Select Box styling */
    .stSelectbox label, .stNumberInput label {
        font-size: 16px !important;
        font-weight: 600 !important;
        color: #FF6B6B !important;
    }

    /* Buttons */
    .stButton > button {
        background: linear-gradient(90deg, #FF6B6B, #FF8E53);
        color: white !important;
        border: none;
        border-radius: 30px;
        padding: 12px 28px;
        font-size: 16px;
        font-weight: 600;
        letter-spacing: 1px;
        cursor: pointer;
        transition: all 0.3s ease;
        box-shadow: 0 4px 15px rgba(255, 107, 107, 0.4);
        width: 100%;
        margin-top: 28px;
    }
    .stButton > button:hover {
        transform: translateY(-3px);
        box-shadow: 0 8px 25px rgba(255, 107, 107, 0.6);
    }
    .stButton > button:active {
        transform: translateY(1px);
    }

    /* Book Card Glassmorphism Design */
    .book-column {
        position: relative;
        background: rgba(26, 26, 46, 0.8);
        backdrop-filter: blur(12px);
        -webkit-backdrop-filter: blur(12px);
        border: 1px solid rgba(255, 255, 255, 0.1);
        border-radius: 20px;
        box-shadow: 0 10px 30px rgba(0, 0, 0, 0.15);
        margin-top: 30px;
        margin-bottom: 25px;
        transition: all 0.4s cubic-bezier(0.175, 0.885, 0.32, 1.275);
        overflow: visible;
        display: flex;
        flex-direction: column;
        height: 100%;
    }
    .book-column:hover {
        transform: translateY(-10px) scale(1.02);
        box-shadow: 0 20px 40px rgba(78, 205, 196, 0.2), 0 0 20px rgba(255, 107, 107, 0.2);
        border: 1px solid rgba(255, 107, 107, 0.3);
    }

    /* Book Image & 3D Spine effect */
    .book-image-area {
        padding: 40px 20px 20px 20px;
        display: flex;
        justify-content: center;
        align-items: center;
    }
    .book-cover {
        height: 250px;
        width: auto;
        border-radius: 4px 12px 12px 4px;
        box-shadow: -5px 0 15px rgba(0,0,0,0.4), inset 4px 0 10px rgba(255,255,255,0.2);
        transition: transform 0.3s ease;
    }
    .book-column:hover .book-cover {
        transform: rotateY(-10deg) scale(1.05);
    }

    /* Badge */
    .recommendation-badge {
        position: absolute;
        top: -20px;
        left: 50%;
        transform: translateX(-50%);
        width: 45px;
        height: 45px;
        border-radius: 50%;
        background: linear-gradient(135deg, #FF6B6B, #FF8E53);
        color: white;
        display: flex;
        align-items: center;
        justify-content: center;
        font-size: 20px;
        font-weight: 800;
        z-index: 10;
        box-shadow: 0 5px 15px rgba(255, 107, 107, 0.5);
        border: 3px solid #1a1a2e;
    }

    /* Book Info Section */
    .book-info {
        background: rgba(15, 15, 26, 0.6);
        padding: 20px;
        border-radius: 0 0 20px 20px;
        text-align: center;
        flex-grow: 1;
        display: flex;
        flex-direction: column;
        justify-content: space-between;
        align-items: center;
    }
    
    .premium-title {
        font-size: 16px;
        font-weight: 800;
        color: #FFFFFF;
        margin-bottom: 8px;
        line-height: 1.3;
        display: -webkit-box;
        -webkit-line-clamp: 2;
        -webkit-box-orient: vertical;
        overflow: hidden;
        text-overflow: ellipsis;
        min-height: 42px;
    }

    .premium-divider {
        width: 40px;
        height: 4px;
        background: linear-gradient(90deg, #4ECDC4, #556270);
        margin: 10px 0;
        border-radius: 5px;
    }

    .premium-author {
        font-size: 14px;
        color: #A0AABF;
        font-weight: 400;
        margin-bottom: 5px;
        width: 100%;
        white-space: nowrap;
        overflow: hidden;
        text-overflow: ellipsis;
    }

    .premium-year {
        font-size: 12px;
        color: #4ECDC4;
        font-weight: 600;
        letter-spacing: 1px;
    }
    
    /* Tabs Redesign */
    .stTabs [data-baseweb="tab-list"] {
        gap: 15px;
        justify-content: center;
        background-color: transparent;
        padding-bottom: 10px;
    }
    .stTabs [data-baseweb="tab"] {
        background-color: transparent !important;
        border: 2px solid #333 !important;
        border-radius: 30px !important;
        padding: 10px 25px !important;
        color: #8892B0 !important;
        transition: all 0.3s ease;
    }
    .stTabs [aria-selected="true"] {
        background: rgba(255, 107, 107, 0.1) !important;
        border: 2px solid #FF6B6B !important;
        color: #FF6B6B !important;
        box-shadow: 0 0 15px rgba(255, 107, 107, 0.2);
    }
    
    .recommendation-header {
        font-size: 22px;
        font-weight: 600;
        color: #FFFFFF;
        text-align: center;
        margin-top: 30px;
        margin-bottom: 10px;
    }
    .recommendation-header span {
        color: #FF6B6B;
    }
    
    /* Expander styling */
    [data-testid="stExpander"] details {
        border: 1px solid rgba(255, 255, 255, 0.1) !important;
        border-radius: 12px !important;
        background: rgba(26, 26, 46, 0.5) !important;
    }
    [data-testid="stExpander"] summary {
        color: #4ECDC4 !important;
        font-weight: 600 !important;
        font-size: 16px !important;
    }
    </style>
""", unsafe_allow_html=True)

# -------------------------------------------------------------------------
# STREAMLIT APP UI & LAYOUT
# -------------------------------------------------------------------------

st.markdown("<div class='main-header'>Next Chapter</div>", unsafe_allow_html=True)
st.markdown("<div class='subheader'>Let AI curate your perfect reading list.</div>", unsafe_allow_html=True)

# Top Image/Banner (Using Streamlit Columns to center it)
_, img_col, _ = st.columns([1,2,1])
with img_col:
    # A cleaner, more modern illustration vector URL
    st.image('https://raw.githubusercontent.com/MarpakaPradeepSai/Employee-Churn-Prediction/main/Data/Images%20&%20GIFs/book-banner.png', use_container_width=True, clamp=True)
    # Note: Using an implicit fallback if URL fails, but standard image behaves normally.

st.markdown("<br>", unsafe_allow_html=True)

# Tabs
tab1, tab2 = st.tabs(["📚 Discover Similar Books", "👤 For You (Personalized)"])

# -------------------------------------------------------------------------
# TAB 1: BOOK-TO-BOOK RECOMMENDATIONS
# -------------------------------------------------------------------------
with tab1:
    col1, col2, col3 = st.columns([3, 1, 1])
    
    all_books = sorted(final_filtered_df['title'].unique().tolist())
    with col1:
        book_title = st.selectbox('Which book did you recently love?', all_books, index=None, placeholder="Search for a book...")
    with col2:
        num_recommendations = st.number_input('How many?', min_value=1, max_value=40, value=8)
    with col3:
        find_btn = st.button('✨ Spark Ideas')
    
    if find_btn:
        if book_title:
            similar_books = get_top_similar_books(book_title, num_recommendations)
            if similar_books is None:
                st.error("⚠️ Book not found in the database. Try another one!")
            else:
                st.toast('Curating your customized list! 🪄', icon='✨')
                st.markdown(f"<div class='recommendation-header'>Top {num_recommendations} Matches for <span>'{book_title}'</span></div><hr style='border-top: 1px solid #333; margin: 20px 0;'>", unsafe_allow_html=True)
                
                books_list = similar_books.index.tolist()
                display_book_cards(books_list)
        else:
            st.warning("⚠️ Please select a book first.")

# -------------------------------------------------------------------------
# TAB 2: USER-SPECIFIC RECOMMENDATIONS
# -------------------------------------------------------------------------
with tab2:
    all_user_ids = sorted(final_filtered_df['userId'].unique().tolist())
    
    ucol1, ucol2, ucol3 = st.columns([2, 1, 1])
    with ucol1:
        user_id_input = st.selectbox('Select User Profile ID:', all_user_ids, index=None, placeholder="Choose a User ID...")
    with ucol2:
        num_user_recs = st.number_input('Books to fetch:', min_value=1, max_value=40, value=8, key='user_num')
    with ucol3:
        user_btn = st.button('🚀 Get Matches')
    
    if user_btn:
        if user_id_input:
            recommendations, user_history = get_user_recommendations(user_id_input, final_filtered_df, cosine_sim_df, k=num_user_recs)
            
            if recommendations is None:
                st.warning(f"⚠️ User {user_id_input} hasn't read enough books yet. Tell them to read more!")
            else:
                # Trigger a fun celebration for the user!
                st.balloons()
                st.toast(f'Welcome back, User {user_id_input}! Found {len(recommendations)} books for you.', icon='📚')
                
                # Expandable Reading History beautifully formatted
                if user_history is not None and not user_history.empty:
                    with st.expander("📖 Glance at User's Past Reads"):
                        history_df = user_history.copy().reset_index(drop=True)
                        history_df.index += 1
                        st.dataframe(
                            history_df,
                            use_container_width=True,
                            column_config={
                                "title": st.column_config.TextColumn("Book Title"),
                                "rating": st.column_config.ProgressColumn("Rating Score", format="%d", min_value=0, max_value=10)
                            }
                        )
                        st.caption("ℹ️ *A rating of 0 indicates an interaction (read/viewed) without a formal rating score.*")
                
                if len(recommendations) > 0:
                    st.markdown(f"<div class='recommendation-header'>Your Hand-Picked Reading List, <span>User {user_id_input}</span></div><hr style='border-top: 1px solid #333; margin: 20px 0;'>", unsafe_allow_html=True)
                    display_book_cards(recommendations)
                else:
                    st.info("No new recommendations available at the moment based on this history.")
        else:
            st.warning("⚠️ Please select a User ID to begin.")
