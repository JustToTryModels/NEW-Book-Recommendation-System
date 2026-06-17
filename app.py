# Deployment Code
import streamlit as st
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from huggingface_hub import hf_hub_download
import warnings
import time

warnings.filterwarnings('ignore')

# -------------------------------------------------------------------------
# PAGE CONFIGURATION
# -------------------------------------------------------------------------
st.set_page_config(
    page_title="BookWise - Your Personal Book Companion",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="collapsed"
)

@st.cache_data
def load_and_prepare_data():
    # Load your final filtered dataframe from Hugging Face
    final_filtered_df_path = hf_hub_download(repo_id="IamPradeep/BRS_DATA", filename="final_filtered_df.csv", repo_type="dataset")
    final_filtered_df = pd.read_csv(final_filtered_df_path)

    # Load the dataframe containing book URLs from Hugging Face
    book_urls_df_path = hf_hub_download(repo_id="IamPradeep/BRS_DATA", filename="Books.csv", repo_type="dataset")
    book_urls_df = pd.read_csv(book_urls_df_path)
    book_urls_df.rename(columns={'Book-Title': 'title'}, inplace=True)

    # Drop duplicate titles before merging
    book_urls_df = book_urls_df.drop_duplicates(subset=['title'], keep='first')

    # Merge the dataframes on the title
    final_filtered_df = final_filtered_df.merge(book_urls_df[['title', 'Book-Author', 'Year-Of-Publication', 'Image-URL-L']], on='title', how='left')

    # URL replacements
    url1 = 'http://images.amazon.com/images/P/0690040784.01.LZZZZZZZ.jpg'
    url2 = 'http://images.amazon.com/images/P/0451172817.01.LZZZZZZZ.jpg'
    url3 = 'http://images.amazon.com/images/P/0312084986.01.LZZZZZZZ.jpg'
    url4 = 'http://images.amazon.com/images/P/1590400356.01.LZZZZZZZ.jpg'

    final_filtered_df.loc[final_filtered_df['title'] == 'Jacob Have I Loved', 'Image-URL-L'] = url1
    final_filtered_df.loc[final_filtered_df['title'] == 'Needful Things', 'Image-URL-L'] = url2
    final_filtered_df.loc[final_filtered_df['title'] == 'All Creatures Great and Small', 'Image-URL-L'] = url3
    final_filtered_df.loc[final_filtered_df['title'] == "The Kitchen God's Wife", 'Image-URL-L'] = url4

    # Build similarity matrix
    explicit_ratings_df = final_filtered_df[final_filtered_df['rating'] > 0]
    book_user_mat = explicit_ratings_df.pivot_table(index='title', columns='userId', values='rating').fillna(0)

    cosine_sim = cosine_similarity(book_user_mat)
    cosine_sim_df = pd.DataFrame(cosine_sim, index=book_user_mat.index, columns=book_user_mat.index)

    return final_filtered_df, cosine_sim_df

# Show loading animation
with st.spinner('🔮 Loading the magic of books...'):
    final_filtered_df, cosine_sim_df = load_and_prepare_data()

# -------------------------------------------------------------------------
# HELPER FUNCTIONS
# -------------------------------------------------------------------------

def get_top_similar_books(book_title, n=10):
    """Get similar books based on book title"""
    if book_title not in cosine_sim_df.index:
        return "⚠️ Book not found in the database."
    
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
    """Display books in a premium card layout with animations"""
    for i in range(0, len(books_list), 3):
        cols = st.columns(3)
        for j in range(3):
            if i + j < len(books_list):
                book = books_list[i + j]
                book_info = final_filtered_df[final_filtered_df['title'] == book].iloc[0]
                
                safe_title = str(book).replace('"', '&quot;').replace("'", "&#39;")
                safe_author = str(book_info['Book-Author']).replace('"', '&quot;').replace("'", "&#39;")
                
                rank = start_index + i + j + 1
                
                # Determine badge color based on rank
                if rank == 1:
                    badge_color = "linear-gradient(135deg, #FFD700, #FFA500)"
                    badge_icon = "👑"
                elif rank == 2:
                    badge_color = "linear-gradient(135deg, #C0C0C0, #808080)"
                    badge_icon = "🥈"
                elif rank == 3:
                    badge_color = "linear-gradient(135deg, #CD7F32, #8B4513)"
                    badge_icon = "🥉"
                else:
                    badge_color = "linear-gradient(135deg, #667eea, #764ba2)"
                    badge_icon = ""
                
                with cols[j]:
                    st.markdown(f"""
                    <div class='book-column animate-fade-in' style='animation-delay: {j * 0.1}s;'>
                        <div class='recommendation-badge' style='background: {badge_color};'>
                            <span class='badge-icon'>{badge_icon}</span>
                            <span class='badge-number'>{rank}</span>
                        </div>
                        <div class='book-image-area'>
                            <div class='image-wrapper'>
                                <img src='{book_info['Image-URL-L']}' class='book-image' alt="{safe_title}">
                                <div class='image-overlay'>
                                    <div class='overlay-text'>View Details</div>
                                </div>
                            </div>
                        </div>
                        <div class='book-info'>
                            <div class='premium-title' title="{safe_title}">{book}</div>
                            <div class='premium-divider'></div>
                            <div class='premium-author' title="{safe_author}">
                                <span class='author-icon'>✍️</span> {book_info['Book-Author']}
                            </div>
                            <div class='premium-year'>
                                <span class='year-icon'>📅</span> {book_info['Year-Of-Publication']}
                            </div>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
        
        if i < len(books_list) - 3:
            st.markdown("<div class='section-divider'></div>", unsafe_allow_html=True)

# -------------------------------------------------------------------------
# ENHANCED CSS STYLING
# -------------------------------------------------------------------------

st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Playfair+Display:wght@400;700;900&family=Poppins:wght@300;400;500;600;700&family=Cormorant+Garamond:wght@300;400;600;700&display=swap');
    
    /* Global Styles */
    * {
        transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
    }
    
    .main {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        background-attachment: fixed;
    }
    
    .stApp {
        background: transparent;
    }
    
    .block-container {
        padding-top: 2rem;
        padding-bottom: 3rem;
        max-width: 1400px;
    }
    
    /* Animated Background Particles */
    .main::before {
        content: "";
        position: fixed;
        top: 0;
        left: 0;
        width: 100%;
        height: 100%;
        background-image: 
            radial-gradient(circle at 20% 50%, rgba(255, 255, 255, 0.05) 0%, transparent 50%),
            radial-gradient(circle at 80% 80%, rgba(255, 255, 255, 0.05) 0%, transparent 50%),
            radial-gradient(circle at 40% 20%, rgba(255, 255, 255, 0.03) 0%, transparent 50%);
        pointer-events: none;
        z-index: 0;
    }
    
    /* Typography */
    h1, h2, h3, h4, h5, h6 {
        font-family: 'Playfair Display', serif !important;
        font-weight: 700 !important;
        letter-spacing: 0.5px;
    }
    
    p, label, div, span, li, .stMarkdown {
        font-family: 'Poppins', sans-serif !important;
    }
    
    /* Animations */
    @keyframes fadeIn {
        from {
            opacity: 0;
            transform: translateY(30px);
        }
        to {
            opacity: 1;
            transform: translateY(0);
        }
    }
    
    @keyframes slideInFromLeft {
        from {
            opacity: 0;
            transform: translateX(-50px);
        }
        to {
            opacity: 1;
            transform: translateX(0);
        }
    }
    
    @keyframes float {
        0%, 100% {
            transform: translateY(0px);
        }
        50% {
            transform: translateY(-10px);
        }
    }
    
    @keyframes shimmer {
        0% {
            background-position: -1000px 0;
        }
        100% {
            background-position: 1000px 0;
        }
    }
    
    @keyframes pulse {
        0%, 100% {
            transform: scale(1);
        }
        50% {
            transform: scale(1.05);
        }
    }
    
    .animate-fade-in {
        animation: fadeIn 0.6s ease-out forwards;
        opacity: 0;
    }
    
    /* Header Section */
    .hero-container {
        background: linear-gradient(135deg, rgba(255,255,255,0.95) 0%, rgba(255,255,255,0.9) 100%);
        border-radius: 30px;
        padding: 60px 40px;
        margin-bottom: 40px;
        box-shadow: 0 20px 60px rgba(0,0,0,0.3);
        text-align: center;
        position: relative;
        overflow: hidden;
        backdrop-filter: blur(10px);
    }
    
    .hero-container::before {
        content: "";
        position: absolute;
        top: -50%;
        left: -50%;
        width: 200%;
        height: 200%;
        background: linear-gradient(45deg, transparent, rgba(255,255,255,0.1), transparent);
        animation: shimmer 3s infinite;
    }
    
    .hero-title {
        font-size: 72px !important;
        font-weight: 900 !important;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        margin-bottom: 15px;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.1);
        animation: fadeIn 1s ease-out;
        letter-spacing: -2px;
        line-height: 1.2;
    }
    
    .hero-subtitle {
        font-size: 24px;
        color: #555;
        font-weight: 300;
        margin-top: 10px;
        animation: fadeIn 1s ease-out 0.2s both;
        font-family: 'Cormorant Garamond', serif !important;
        font-style: italic;
    }
    
    .hero-description {
        font-size: 16px;
        color: #777;
        margin-top: 20px;
        max-width: 800px;
        margin-left: auto;
        margin-right: auto;
        line-height: 1.6;
        animation: fadeIn 1s ease-out 0.4s both;
    }
    
    /* Stats Bar */
    .stats-container {
        display: flex;
        justify-content: center;
        gap: 40px;
        margin-top: 30px;
        flex-wrap: wrap;
    }
    
    .stat-item {
        text-align: center;
        padding: 15px 30px;
        background: rgba(102, 126, 234, 0.1);
        border-radius: 15px;
        min-width: 150px;
    }
    
    .stat-number {
        font-size: 32px;
        font-weight: 700;
        color: #667eea;
        display: block;
    }
    
    .stat-label {
        font-size: 14px;
        color: #666;
        margin-top: 5px;
        text-transform: uppercase;
        letter-spacing: 1px;
    }
    
    /* Buttons */
    .stButton > button {
        font-family: 'Poppins', sans-serif !important;
        font-size: 16px;
        font-weight: 600;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white !important;
        border: none;
        border-radius: 50px;
        padding: 15px 40px;
        cursor: pointer;
        box-shadow: 0 10px 30px rgba(102, 126, 234, 0.4);
        transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
        position: relative;
        overflow: hidden;
        text-transform: uppercase;
        letter-spacing: 1px;
    }
    
    .stButton > button::before {
        content: "";
        position: absolute;
        top: 50%;
        left: 50%;
        width: 0;
        height: 0;
        border-radius: 50%;
        background: rgba(255, 255, 255, 0.2);
        transform: translate(-50%, -50%);
        transition: width 0.6s, height 0.6s;
    }
    
    .stButton > button:hover::before {
        width: 300px;
        height: 300px;
    }
    
    .stButton > button:hover {
        transform: translateY(-3px);
        box-shadow: 0 15px 40px rgba(102, 126, 234, 0.6);
    }
    
    .stButton > button:active {
        transform: translateY(-1px);
        box-shadow: 0 5px 20px rgba(102, 126, 234, 0.4);
    }
    
    /* Select Boxes */
    .stSelectbox > div > div {
        background: white;
        border: 2px solid #e0e0e0;
        border-radius: 15px;
        transition: all 0.3s ease;
        font-family: 'Poppins', sans-serif;
    }
    
    .stSelectbox > div > div:hover {
        border-color: #667eea;
        box-shadow: 0 5px 15px rgba(102, 126, 234, 0.2);
    }
    
    .stSelectbox > div > div:focus-within {
        border-color: #667eea;
        box-shadow: 0 5px 20px rgba(102, 126, 234, 0.3);
    }
    
    /* Number Input */
    .stNumberInput > div > div {
        background: white;
        border: 2px solid #e0e0e0;
        border-radius: 15px;
        transition: all 0.3s ease;
    }
    
    .stNumberInput > div > div:hover {
        border-color: #667eea;
        box-shadow: 0 5px 15px rgba(102, 126, 234, 0.2);
    }
    
    /* Tabs */
    .stTabs {
        background: transparent;
        margin-top: 30px;
    }
    
    .stTabs [data-baseweb="tab-list"] {
        gap: 15px;
        justify-content: center;
        background: rgba(255, 255, 255, 0.9);
        padding: 15px;
        border-radius: 25px;
        box-shadow: 0 10px 30px rgba(0,0,0,0.1);
    }
    
    .stTabs [data-baseweb="tab"] {
        font-family: 'Poppins', sans-serif !important;
        height: 60px;
        background: transparent;
        border-radius: 15px;
        padding: 0px 30px;
        font-size: 16px;
        font-weight: 600;
        color: #555 !important;
        border: 2px solid transparent;
        transition: all 0.3s ease;
    }
    
    .stTabs [data-baseweb="tab"] p {
        color: #555 !important;
        font-weight: 600;
    }
    
    .stTabs [data-baseweb="tab"]:hover {
        background: rgba(102, 126, 234, 0.1);
        transform: translateY(-2px);
    }
    
    .stTabs [aria-selected="true"] {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%) !important;
        color: white !important;
        border: 2px solid transparent !important;
        box-shadow: 0 8px 25px rgba(102, 126, 234, 0.4);
    }
    
    .stTabs [aria-selected="true"] p {
        color: white !important;
    }
    
    /* Tab Content */
    .stTabs [data-baseweb="tab-panel"] {
        background: rgba(255, 255, 255, 0.95);
        padding: 40px;
        border-radius: 25px;
        margin-top: 20px;
        box-shadow: 0 10px 40px rgba(0,0,0,0.1);
        backdrop-filter: blur(10px);
    }
    
    /* Book Cards */
    .book-column {
        position: relative;
        padding: 0;
        border: none;
        border-radius: 25px;
        background: linear-gradient(145deg, #ffffff, #f5f5f5);
        box-shadow: 0 15px 35px rgba(0,0,0,0.15);
        margin-top: 40px;
        margin-bottom: 25px;
        transition: all 0.4s cubic-bezier(0.4, 0, 0.2, 1);
        overflow: hidden;
    }
    
    .book-column::before {
        content: "";
        position: absolute;
        top: 0;
        left: 0;
        right: 0;
        height: 4px;
        background: linear-gradient(90deg, #667eea, #764ba2, #f093fb, #667eea);
        background-size: 200% 100%;
        animation: shimmer 3s linear infinite;
        opacity: 0;
        transition: opacity 0.3s ease;
    }
    
    .book-column:hover::before {
        opacity: 1;
    }
    
    .book-column:hover {
        transform: translateY(-12px) scale(1.02);
        box-shadow: 0 25px 50px rgba(102, 126, 234, 0.3);
    }
    
    .book-image-area {
        padding: 45px 25px 25px 25px;
        background: linear-gradient(180deg, rgba(102, 126, 234, 0.05) 0%, transparent 100%);
    }
    
    .image-wrapper {
        position: relative;
        overflow: hidden;
        border-radius: 15px;
        background: white;
        box-shadow: 0 10px 30px rgba(0,0,0,0.1);
    }
    
    .book-image {
        height: 320px !important;
        width: 100% !important;
        object-fit: contain;
        display: block;
        transition: transform 0.5s cubic-bezier(0.4, 0, 0.2, 1);
    }
    
    .book-column:hover .book-image {
        transform: scale(1.05);
    }
    
    .image-overlay {
        position: absolute;
        top: 0;
        left: 0;
        right: 0;
        bottom: 0;
        background: linear-gradient(135deg, rgba(102, 126, 234, 0.9), rgba(118, 75, 162, 0.9));
        display: flex;
        align-items: center;
        justify-content: center;
        opacity: 0;
        transition: opacity 0.4s ease;
    }
    
    .book-column:hover .image-overlay {
        opacity: 1;
    }
    
    .overlay-text {
        color: white;
        font-size: 18px;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 2px;
        transform: translateY(20px);
        transition: transform 0.4s ease;
    }
    
    .book-column:hover .overlay-text {
        transform: translateY(0);
    }
    
    .book-info {
        background: linear-gradient(180deg, #1a1a1a 0%, #2d2d2d 100%);
        padding: 25px 20px;
        border-radius: 0 0 25px 25px;
        text-align: center;
        min-height: 170px;
        height: 170px;
        display: flex;
        flex-direction: column;
        justify-content: center;
        align-items: center;
    }
    
    .premium-title {
        font-size: 17px;
        font-weight: 700;
        color: #FFD700;
        margin-bottom: 12px;
        line-height: 1.4;
        width: 100%;
        white-space: nowrap;
        overflow-x: auto;
        overflow-y: hidden;
        padding-bottom: 8px;
        max-height: 48px;
        font-family: 'Playfair Display', serif !important;
        letter-spacing: 0.5px;
    }
    
    .premium-title::-webkit-scrollbar {
        height: 6px;
    }
    
    .premium-title::-webkit-scrollbar-track {
        background: rgba(255, 255, 255, 0.1);
        border-radius: 10px;
    }
    
    .premium-title::-webkit-scrollbar-thumb {
        background: linear-gradient(90deg, #667eea, #764ba2);
        border-radius: 10px;
    }
    
    .premium-divider {
        width: 50px;
        height: 3px;
        background: linear-gradient(90deg, #667eea, #764ba2);
        margin: 10px auto;
        border-radius: 5px;
    }
    
    .premium-author {
        font-size: 14px;
        color: #d0d0d0;
        font-style: italic;
        margin-bottom: 10px;
        width: 100%;
        white-space: nowrap;
        overflow: hidden;
        text-overflow: ellipsis;
        display: flex;
        align-items: center;
        justify-content: center;
        gap: 6px;
    }
    
    .author-icon {
        font-size: 14px;
    }
    
    .premium-year {
        font-size: 13px;
        color: #999;
        text-transform: uppercase;
        letter-spacing: 1.5px;
        font-weight: 600;
        display: flex;
        align-items: center;
        justify-content: center;
        gap: 6px;
    }
    
    .year-icon {
        font-size: 13px;
    }
    
    /* Recommendation Badge */
    .recommendation-badge {
        position: absolute;
        top: -28px;
        left: 50%;
        transform: translateX(-50%);
        width: 56px;
        height: 56px;
        border-radius: 50%;
        background: linear-gradient(135deg, #667eea, #764ba2);
        color: white;
        border: 4px solid white;
        display: flex;
        align-items: center;
        justify-content: center;
        font-size: 18px;
        font-weight: 700;
        z-index: 10;
        box-shadow: 0 8px 20px rgba(0, 0, 0, 0.3);
        animation: float 3s ease-in-out infinite;
    }
    
    .badge-icon {
        position: absolute;
        font-size: 20px;
        top: -8px;
        right: -8px;
    }
    
    .badge-number {
        font-family: 'Poppins', sans-serif;
        font-weight: 700;
    }
    
    /* Section Divider */
    .section-divider {
        height: 2px;
        background: linear-gradient(90deg, transparent, rgba(102, 126, 234, 0.5), transparent);
        margin: 40px 0;
        border-radius: 2px;
    }
    
    /* Recommendation Header */
    .recommendation-header {
        font-size: 22px;
        font-weight: 600;
        color: #333;
        border-left: 6px solid #667eea;
        padding-left: 20px;
        margin: 30px 0 20px 0;
        background: linear-gradient(90deg, rgba(102, 126, 234, 0.1), transparent);
        padding-top: 15px;
        padding-bottom: 15px;
        border-radius: 0 15px 15px 0;
        animation: slideInFromLeft 0.6s ease-out;
    }
    
    /* Expander */
    [data-testid="stExpander"] {
        background: white;
        border-radius: 20px;
        border: 2px solid #e0e0e0;
        overflow: hidden;
        margin: 20px 0;
        box-shadow: 0 5px 15px rgba(0,0,0,0.05);
        transition: all 0.3s ease;
    }
    
    [data-testid="stExpander"]:hover {
        border-color: #667eea;
        box-shadow: 0 8px 25px rgba(102, 126, 234, 0.15);
    }
    
    [data-testid="stExpander"] summary {
        background: linear-gradient(135deg, #f8f9ff 0%, #e8ebff 100%);
        font-weight: 600;
        font-size: 16px;
        padding: 20px;
        color: #667eea;
        cursor: pointer;
    }
    
    [data-testid="stExpander"] summary:hover {
        background: linear-gradient(135deg, #e8ebff 0%, #d8ddff 100%);
    }
    
    [data-testid="stExpander"][open] {
        border-color: #667eea;
    }
    
    /* DataFrames */
    .stDataFrame {
        border-radius: 15px;
        overflow: hidden;
        box-shadow: 0 5px 15px rgba(0,0,0,0.1);
    }
    
    /* Alerts */
    .stAlert {
        border-radius: 15px;
        border: none;
        box-shadow: 0 5px 15px rgba(0,0,0,0.1);
    }
    
    /* Info, Warning, Success boxes */
    .stInfo, .stWarning, .stSuccess {
        background: white;
        border-radius: 15px;
        padding: 20px;
        border-left: 6px solid;
        box-shadow: 0 5px 15px rgba(0,0,0,0.1);
    }
    
    /* Spinner */
    .stSpinner > div {
        border-color: #667eea !important;
        border-right-color: transparent !important;
    }
    
    /* Input Section Container */
    .input-container {
        background: rgba(255, 255, 255, 0.5);
        padding: 30px;
        border-radius: 20px;
        margin-bottom: 30px;
        box-shadow: 0 5px 20px rgba(0,0,0,0.05);
    }
    
    /* Icon Emojis */
    .section-icon {
        font-size: 28px;
        margin-right: 10px;
        vertical-align: middle;
    }
    
    /* Responsive Design */
    @media (max-width: 768px) {
        .hero-title {
            font-size: 42px !important;
        }
        
        .hero-subtitle {
            font-size: 18px;
        }
        
        .stats-container {
            gap: 20px;
        }
        
        .stat-item {
            min-width: 120px;
            padding: 12px 20px;
        }
    }
    </style>
    """, unsafe_allow_html=True)

# -------------------------------------------------------------------------
# HERO SECTION
# -------------------------------------------------------------------------

st.markdown(f"""
    <div class='hero-container'>
        <h1 class='hero-title'>📚 BookWise</h1>
        <p class='hero-subtitle'>Your Personal AI-Powered Reading Companion</p>
        <p class='hero-description'>
            Discover your next favorite book through our intelligent recommendation engine. 
            Powered by advanced machine learning algorithms and collaborative filtering.
        </p>
        <div class='stats-container'>
            <div class='stat-item'>
                <span class='stat-number'>{len(final_filtered_df['title'].unique()):,}</span>
                <span class='stat-label'>Books</span>
            </div>
            <div class='stat-item'>
                <span class='stat-number'>{len(final_filtered_df['userId'].unique()):,}</span>
                <span class='stat-label'>Readers</span>
            </div>
            <div class='stat-item'>
                <span class='stat-number'>{len(final_filtered_df):,}</span>
                <span class='stat-label'>Ratings</span>
            </div>
        </div>
    </div>
""", unsafe_allow_html=True)

# -------------------------------------------------------------------------
# TABS FOR DIFFERENT RECOMMENDATION TYPES
# -------------------------------------------------------------------------

tab1, tab2 = st.tabs(["📚 Book Recommendations", "👤 Personalized Discovery"])

# -------------------------------------------------------------------------
# TAB 1: BOOK-TO-BOOK RECOMMENDATIONS
# -------------------------------------------------------------------------
with tab1:
    st.markdown("<h3 style='text-align: center; color: #333; margin-bottom: 20px;'><span class='section-icon'>🔍</span>Find Similar Books</h3>", unsafe_allow_html=True)
    st.markdown("<p style='text-align: center; color: #666; font-size: 15px; margin-bottom: 30px;'>Select a book you love and we'll recommend similar titles based on reader preferences and ratings.</p>", unsafe_allow_html=True)
    
    # Input Container
    st.markdown("<div class='input-container'>", unsafe_allow_html=True)
    
    all_books = sorted(final_filtered_df['title'].unique().tolist())
    
    col1, col2 = st.columns([3, 1])
    
    with col1:
        book_title = st.selectbox(
            '📖 Select a book title:', 
            all_books, 
            index=None, 
            placeholder="Choose or type a book title...", 
            key='book_title'
        )
    
    with col2:
        num_recommendations = st.number_input(
            '🎯 Number of recommendations:', 
            min_value=1, 
            max_value=50, 
            value=10, 
            key='num_recs_book'
        )
    
    st.markdown("</div>", unsafe_allow_html=True)
    
    # Center the button
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        recommend_button = st.button('✨ Get Recommendations', key='btn_book_recs', use_container_width=True)
    
    if 'recommendations' not in st.session_state:
        st.session_state.recommendations = None
    if 'recommended_book' not in st.session_state:
        st.session_state.recommended_book = None
    if 'recommended_num' not in st.session_state:
        st.session_state.recommended_num = None
    
    if recommend_button:
        if book_title:
            with st.spinner('🔮 Finding perfect matches for you...'):
                time.sleep(0.5)  # Brief pause for effect
                similar_books = get_top_similar_books(book_title, num_recommendations)
                st.session_state.recommendations = similar_books
                st.session_state.recommended_book = book_title
                st.session_state.recommended_num = num_recommendations
        else:
            st.warning("⚠️ Please select a book title first!")
    
    if st.session_state.recommendations is not None:
        similar_books = st.session_state.recommendations
        rec_book = st.session_state.recommended_book
        rec_num = st.session_state.recommended_num
        
        if isinstance(similar_books, str):
            st.error(similar_books)
        else:
            st.markdown(f"""
                <div class='recommendation-header'>
                    🎯 Top {rec_num} recommendations for '<strong>{rec_book}</strong>'
                </div>
            """, unsafe_allow_html=True)
            
            books_list = similar_books.index.tolist()
            display_book_cards(books_list)
            
            # Thank you section
            st.markdown("<br><br>", unsafe_allow_html=True)
            st.markdown("""
                <div style='text-align: center; padding: 40px; background: linear-gradient(135deg, rgba(102, 126, 234, 0.1), rgba(118, 75, 162, 0.1)); border-radius: 20px; margin-top: 40px;'>
                    <h3 style='color: #667eea; margin-bottom: 15px;'>🌟 Happy Reading! 🌟</h3>
                    <p style='color: #666; font-size: 16px;'>We hope you find your next favorite book!</p>
                </div>
            """, unsafe_allow_html=True)

# -------------------------------------------------------------------------
# TAB 2: USER-SPECIFIC RECOMMENDATIONS
# -------------------------------------------------------------------------
with tab2:
    st.markdown("<h3 style='text-align: center; color: #333; margin-bottom: 20px;'><span class='section-icon'>🎭</span>Personalized Reading Journey</h3>", unsafe_allow_html=True)
    st.markdown("<p style='text-align: center; color: #666; font-size: 15px; margin-bottom: 30px;'>Enter a User ID to receive curated book recommendations based on reading history and preferences.</p>", unsafe_allow_html=True)
    
    # Input Container
    st.markdown("<div class='input-container'>", unsafe_allow_html=True)
    
    all_user_ids = sorted(final_filtered_df['userId'].unique().tolist())
    
    col1, col2 = st.columns([3, 1])
    
    with col1:
        user_id_input = st.selectbox(
            '👤 Select or enter a User ID:', 
            all_user_ids, 
            index=None, 
            placeholder="Choose a User ID...", 
            key='user_id_select'
        )
    
    with col2:
        num_user_recs = st.number_input(
            '🎯 Number of recommendations:', 
            min_value=1, 
            max_value=50, 
            value=10, 
            key='num_recs_user'
        )
    
    st.markdown("</div>", unsafe_allow_html=True)
    
    # Center the button
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        user_recommend_button = st.button('✨ Discover My Books', key='btn_user_recs', use_container_width=True)
    
    if 'user_recommendations' not in st.session_state:
        st.session_state.user_recommendations = None
    if 'user_history_display' not in st.session_state:
        st.session_state.user_history_display = None
    if 'current_user_id' not in st.session_state:
        st.session_state.current_user_id = None
    
    if user_recommend_button:
        if user_id_input:
            with st.spinner('🎨 Crafting personalized recommendations...'):
                time.sleep(0.5)
                recommendations, user_history = get_user_recommendations(
                    user_id_input, 
                    final_filtered_df, 
                    cosine_sim_df, 
                    k=num_user_recs
                )
                
                if recommendations is None:
                    st.warning(f"⚠️ User ID {user_id_input} has no interaction history in our database.")
                    st.session_state.user_recommendations = None
                    st.session_state.user_history_display = None
                else:
                    st.session_state.user_recommendations = recommendations
                    st.session_state.user_history_display = user_history
                    st.session_state.current_user_id = user_id_input
        else:
            st.warning("⚠️ Please select or enter a User ID!")
    
    if st.session_state.user_recommendations is not None:
        user_id_display = st.session_state.current_user_id
        recommendations = st.session_state.user_recommendations
        user_history = st.session_state.user_history_display
        
        # Display User's Reading History
        if user_history is not None and len(user_history) > 0:
            with st.expander(f"📖 View Reading History for User {user_id_display} ({len(user_history)} books)"):
                history_df = user_history.copy()
                history_df.reset_index(drop=True, inplace=True)
                history_df.index = history_df.index + 1
                history_df.columns = ['📚 Book Title', '⭐ Rating']
                
                st.dataframe(
                    history_df, 
                    use_container_width=True,
                    height=min(400, (len(history_df) + 1) * 35 + 3)
                )
                st.caption("💡 *Note: A rating of \"0\" indicates an interacted but unrated book.*")
        
        st.markdown("<br>", unsafe_allow_html=True)
        
        # Display Recommendations
        if len(recommendations) > 0:
            st.markdown(f"""
                <div class='recommendation-header'>
                    🎁 Top {len(recommendations)} Personalized Picks for User <strong>{user_id_display}</strong>
                </div>
            """, unsafe_allow_html=True)
            
            display_book_cards(recommendations)
            
            # Thank you section
            st.markdown("<br><br>", unsafe_allow_html=True)
            st.markdown("""
                <div style='text-align: center; padding: 40px; background: linear-gradient(135deg, rgba(102, 126, 234, 0.1), rgba(118, 75, 162, 0.1)); border-radius: 20px; margin-top: 40px;'>
                    <h3 style='color: #667eea; margin-bottom: 15px;'>🌟 Happy Reading! 🌟</h3>
                    <p style='color: #666; font-size: 16px;'>We hope these recommendations light up your reading journey!</p>
                </div>
            """, unsafe_allow_html=True)
        else:
            st.info("📭 No recommendations available for this user at the moment.")

# -------------------------------------------------------------------------
# FOOTER
# -------------------------------------------------------------------------

st.markdown("<br><br>", unsafe_allow_html=True)
st.markdown("""
    <div style='text-align: center; padding: 30px; background: rgba(255, 255, 255, 0.8); border-radius: 20px; margin-top: 50px;'>
        <p style='color: #666; font-size: 14px; margin-bottom: 10px;'>
            Made with ❤️ using Streamlit & Machine Learning
        </p>
        <p style='color: #999; font-size: 12px;'>
            Powered by Collaborative Filtering & Cosine Similarity
        </p>
    </div>
""", unsafe_allow_html=True)
