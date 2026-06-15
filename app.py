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
    """
    Generates personalized recommendations for a specific user.
    Returns: (recommendations_list, user_history_dataframe)
    """
    # Get User's History
    user_history_all = df[df['userId'] == user_id]['title'].tolist()
    user_history_rated = df[df['userId'] == user_id][['title', 'rating']].sort_values(by='rating', ascending=False)
    
    # Remove duplicates from user history
    user_history_rated = user_history_rated.drop_duplicates(subset=['title'])

    if len(user_history_all) == 0:
        return None, None

    # Generate Candidates based on similarity
    scores = {}
    for item in user_history_all:
        if item in sim_matrix.index:
            similar_items = sim_matrix[item].sort_values(ascending=False)[1:50] 
            
            for sim_item, score in similar_items.items():
                if sim_item not in user_history_all:
                    scores[sim_item] = scores.get(sim_item, 0) + score

    # Sort and Return Top K
    sorted_scores = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    top_recommendations = [x[0] for x in sorted_scores[:k]]
    
    return top_recommendations, user_history_rated

def display_book_cards(books_list, start_index=0):
    """Display books in a card layout"""
    for i in range(0, len(books_list), 3):
        cols = st.columns(3)
        for j in range(3):
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
        if i < len(books_list) - 3:
            st.markdown("<br><hr><br>", unsafe_allow_html=True)

# -------------------------------------------------------------------------
# STREAMLIT APP UI
# -------------------------------------------------------------------------

# Combined Title and Subtitle
st.markdown("""
    <h1 style='font-size: 40px; text-align: center; margin-bottom: 5px; padding-bottom: 0px;'>
        Book Recommendation System
    </h1>
    <p class='subheader'>Let Us Help You Choose Your Next Book!</p>
""", unsafe_allow_html=True)

st.image('https://img.freepik.com/premium-vector/bookcase-with-books_182089-197.jpg', use_container_width=True)

# CSS Styling
st.markdown("""
    <style>
    /* Targeted elements for font application to prevent breaking core UI icon elements */
    h1, h2, h3, h4, h5, h6, p, label, .subheader, .premium-title, .premium-author, .premium-year, .book-info, .recommendation-header, [data-testid="stTable"], [data-testid="stDataFrame"] * {
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

    .premium-title::-webkit-scrollbar {
        height: 6px;
    }

    .premium-title::-webkit-scrollbar-thumb {
        background: #ccc;
        border-radius: 10px;
    }

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
    .extra-space {
        margin-top: 50px;
    }
    .recommendation-header {
        font-size: 15px;
        border-left: 5px solid #B2BEB5;
        padding-left: 12px;
        margin-left: 5px;
    }
    
    /* Center the tabs */
    .stTabs [data-baseweb="tab-list"] {
        gap: 24px;
        justify-content: center;
    }
    
    .stTabs [data-baseweb="tab"] {
        font-family: 'Tiempos', 'Tiempos Text', Georgia, 'Times New Roman', serif !important;
        height: 50px;
        background-color: #f0f2f6;
        border-radius: 10px 10px 0px 0px;
        padding: 0px 20px;
        font-size: 16px;
        font-weight: bold;
    }
    
    .stTabs [aria-selected="true"] {
        background: linear-gradient(90deg, #ff8a00, #e52e71);
        color: white;
    }

    /* Style the drop-down (expander) header to light pink */
    .stExpander {
        background-color: #FFD1DC !important;
        border-radius: 8px;
    }
    .stExpander > details > summary {
        background-color: #FFD1DC !important;
        border-radius: 8px;
    }

    /* Style dataframe/table column headers to light pink */
    [data-testid="stDataFrame"] thead tr th, [data-testid="stTable"] thead tr th {
        background-color: #FFD1DC !important;
        color: black !important;
    }
    </style>
""", unsafe_allow_html=True)

# -------------------------------------------------------------------------
# TABS FOR DIFFERENT RECOMMENDATION TYPES
# -------------------------------------------------------------------------

tab1, tab2 = st.tabs(["📚 Book-to-Book Recommendations", "👤 User-Specific Recommendations"])

# -------------------------------------------------------------------------
# TAB 1: BOOK-TO-BOOK RECOMMENDATIONS
# -------------------------------------------------------------------------
with tab1:
    st.markdown("<h3 style='text-align: center;'>Find Similar Books</h3>", unsafe_allow_html=True)
    st.write("Select a book and discover similar titles based on user preferences and ratings.")
    
    all_books = sorted(final_filtered_df['title'].unique().tolist())
    book_title = st.selectbox('Enter a book title:', all_books, index=None, 
                              placeholder="Choose or enter a book title...", key='book_title')
    
    num_recommendations = st.number_input('Enter the number of recommendations:', 
                                         min_value=1, max_value=50, value=10, key='num_recs_book')
    
    if 'recommendations' not in st.session_state:
        st.session_state.recommendations = None
    if 'recommended_book' not in st.session_state:
        st.session_state.recommended_book = None
    if 'recommended_num' not in st.session_state:
        st.session_state.recommended_num = None
    
    if st.button('Recommend books', key='btn_book_recs'):
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
            st.write(similar_books)
        else:
            st.markdown(f"<div class='recommendation-header'>Top {rec_num} recommendations for '<strong>{rec_book}</strong>':</div>", 
                        unsafe_allow_html=True)
            st.write("")
            
            books_list = similar_books.index.tolist()
            display_book_cards(books_list)
            
            st.markdown("<div class='extra-space'></div><div class='extra-space'></div>", unsafe_allow_html=True)
            st.image('https://github.com/MarpakaPradeepSai/Employee-Churn-Prediction/blob/main/Data/Images%20&%20GIFs/thank-you-33.gif?raw=true', 
                    use_container_width=True)

# -------------------------------------------------------------------------
# TAB 2: USER-SPECIFIC RECOMMENDATIONS
# -------------------------------------------------------------------------
with tab2:
    st.markdown("<h3 style='text-align: center;'>Personalized Recommendations for Users</h3>", unsafe_allow_html=True)
    st.write("Enter a User ID to get personalized book recommendations based on their reading history.")
    
    # Get all unique user IDs
    all_user_ids = sorted(final_filtered_df['userId'].unique().tolist())
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        user_id_input = st.selectbox('Select or enter a User ID:', all_user_ids, 
                                     index=None, placeholder="Choose a User ID...", key='user_id_select')
    
    with col2:
        num_user_recs = st.number_input('Number of recommendations:', 
                                       min_value=1, max_value=50, value=10, key='num_recs_user')
    
    if 'user_recommendations' not in st.session_state:
        st.session_state.user_recommendations = None
    if 'user_history_display' not in st.session_state:
        st.session_state.user_history_display = None
    if 'current_user_id' not in st.session_state:
        st.session_state.current_user_id = None
    
    if st.button('Get Personalized Recommendations', key='btn_user_recs'):
        if user_id_input:
            recommendations, user_history = get_user_recommendations(user_id_input, final_filtered_df, 
                                                                     cosine_sim_df, k=num_user_recs)
            
            if recommendations is None:
                st.warning(f"⚠️ User ID {user_id_input} has no interaction history in the database.")
                st.session_state.user_recommendations = None
                st.session_state.user_history_display = None
            else:
                st.session_state.user_recommendations = recommendations
                st.session_state.user_history_display = user_history
                st.session_state.current_user_id = user_id_input
        else:
            st.warning("⚠️ Please select or enter a User ID.")
    
    if st.session_state.user_recommendations is not None:
        user_id_display = st.session_state.current_user_id
        recommendations = st.session_state.user_recommendations
        user_history = st.session_state.user_history_display
        
        # Display User's Reading History
        if user_history is not None and len(user_history) > 0:
            with st.expander("📖 View User's Reading History"):
                # Create a simple table
                history_df = user_history.copy()
                history_df.reset_index(drop=True, inplace=True)
                history_df.index = history_df.index + 1
                history_df.columns = ['Book Title', 'Rating']
                
                st.dataframe(history_df, use_container_width=True)
                st.markdown("<p style='color: gray; font-size: 14px; font-style: italic; margin-top: 8px;'>ℹ️ Note: A rating of 0 indicates an interacted but unrated book.</p>", unsafe_allow_html=True)
        
        st.markdown("<br>", unsafe_allow_html=True)
        
        # Display Recommendations with combined heading
        if len(recommendations) > 0:
            st.markdown(f"<div class='recommendation-header'>Top {len(recommendations)} Personalized Recommendations for User ID: <strong>{user_id_display}</strong></div>", 
                        unsafe_allow_html=True)
            st.write("")
            
            display_book_cards(recommendations)
            
            st.markdown("<div class='extra-space'></div><div class='extra-space'></div>", unsafe_allow_html=True)
            st.image('https://github.com/MarpakaPradeepSai/Employee-Churn-Prediction/blob/main/Data/Images%20&%20GIFs/thank-you-33.gif?raw=true', 
                    use_container_width=True)
        else:
            st.info("No recommendations available for this user at the moment.")
