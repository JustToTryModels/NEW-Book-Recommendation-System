import streamlit as st
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from huggingface_hub import hf_hub_download

import warnings
warnings.filterwarnings('ignore')

# Load your final filtered dataframe from Hugging Face
final_filtered_df_path = hf_hub_download(repo_id="Zlib2/Data", filename="final_filtered_df.csv", repo_type="dataset")
final_filtered_df = pd.read_csv(final_filtered_df_path)

# Load the dataframe containing book URLs from Hugging Face
book_urls_df_path = hf_hub_download(repo_id="Zlib2/Data", filename="Books.csv", repo_type="dataset")
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

def get_top_similar_books(book_title, n=10):
    # Check if the book and user exist in our data
    if book_title not in cosine_sim_df.index:
        return "Book not found in the database."
    
    # Get the similarity scores for the given book
    similar_scores = cosine_sim_df[book_title]
    
    # Sort the books by similarity score and return the top n (excluding the book itself)
    similar_books = similar_scores.sort_values(ascending=False)[1:n+1]
    return similar_books

# Function to get book suggestions based on user input
def get_book_suggestions(input_text):
    return final_filtered_df[final_filtered_df['title'].str.contains(input_text, case=False, na=False)]['title'].unique().tolist()

# Streamlit app
st.title('Book Recommendation System')

# Define CSS for button styles and other formatting tweaks
st.markdown("""
    <style>
    .subheader {
        font-size: 22px;
        font-weight: bold;
        margin-bottom: 20px;
        color: #1a73e8;
    }
    .stButton button {
        font-family: 'Courier New', Courier, monospace;
        font-size: 16px;
        background-color: #4CAF50;
        color: white !important;
        border: none;
        padding: 10px 20px;
        text-align: center;
        text-decoration: none;
        display: inline-block;
        margin: 4px 2px;
        cursor: pointer;
    }
    .stButton button:hover {
        background-color: #45a049;
    }
    .book-info {
        line-height: 1.2;
        margin-bottom: 15px;
    }
    .book-title {
        margin-bottom: 30px;
    }
    .author-info {
        margin-top: 5px;
        font-size: 12px;
    }
    .year-info {
        font-size: 11px;
        margin-top: 3px;
        margin-left: 10px;
        color: #777;
    }
    img {
        object-fit: contain;
        max-height: 300px;
        width: auto;
        display: block;
        margin: 0 auto;
    }
    .spacer {
        margin-bottom: 15px;
    }
    hr {
        border: 0;
        height: 2px;
        background: #333;
        margin-top: 15px;
        margin-bottom: 15px;
    }
    .column-divider {
        border-right: 2px solid #333;
        height: 100%;
        position: absolute;
        right: 0;
        top: 0;
    }
    .book-column {
        position: relative;
        padding-right: 10px;
    }
    .extra-space {
        margin-top: 50px;
    }
    </style>
""", unsafe_allow_html=True)

st.markdown("<p class='subheader'>Let Us Help You Choose Your Next Book!</p>", unsafe_allow_html=True)
st.image('https://img.freepik.com/premium-vector/bookcase-with-books_182089-197.jpg', use_container_width=True)

# Create a selectbox for book title with autocomplete
all_books = final_filtered_df['title'].unique().tolist()
book_title = st.selectbox('Enter a book title:', [''] + all_books, key='book_title')

num_recommendations = st.number_input('Enter the number of recommendations:', min_value=1, max_value=50, value=10)

if st.button('Recommend books'):
    if book_title:
        if book_title != '':
            similar_books = get_top_similar_books(book_title, num_recommendations)
            if isinstance(similar_books, str):
                st.write(similar_books)
            else:
                st.markdown(f"<div style='font-size:15px;'>Top {num_recommendations} recommendations for '<strong>{book_title}</strong>':</div>", unsafe_allow_html=True)
                st.write("")
                
                # Display books in rows with images, horizontal and vertical lines
                for i in range(0, len(similar_books), 3):
                    cols = st.columns(3)
                    for j in range(3):
                        if i + j < len(similar_books):
                            book = similar_books.index[i + j]
                            book_info = final_filtered_df[final_filtered_df['title'] == book].iloc[0]
                            with cols[j]:
                                st.markdown(f"""
                                <div class='book-column'>
                                    <div class='book-info'>
                                        <strong>{i + j + 1}. {book}</strong><br>
                                        <div class='author-info' style='margin-left: 10px;'>by {book_info['Book-Author']}</div>
                                        <div class='year-info'>{book_info['Year-Of-Publication']}</div>
                                    </div>
                                    <img src='{book_info['Image-URL-L']}' style='height:290px; width:auto; display:block;'>
                                    {'' if j == 2 else '<div class="column-divider"></div>'}
                                </div>
                                """, unsafe_allow_html=True)
                    if i < len(similar_books) - 3:
                        st.markdown("<br>", unsafe_allow_html=True)  # Line space above horizontal line
                        st.markdown("<hr>", unsafe_allow_html=True)  # Horizontal line between rows
                        st.markdown("<br>", unsafe_allow_html=True)  # Line space below horizontal line

                # Add extra space between books and final image
                st.markdown("<div class='extra-space'></div>", unsafe_allow_html=True)
                st.markdown("<div class='extra-space'></div>", unsafe_allow_html=True)
                
                # Display the final images
                st.image('https://theweekjunior.co.uk/sites/default/files/inline-images/DM-596_TWJ_Landing_page_text_02.png', use_container_width=True)
                st.image('https://media2.giphy.com/media/v1.Y2lkPTc5MGI3NjExNHV3M2w0bW9mMnNzeG05NTBzcjFzc29uY21yeWw1aXhnZm40bzlleSZlcD12MV9pbnRlcm5hbF9naWZfYnlfaWQmY3Q9Zw/wIVA0zh5pt0G5YtcAL/giphy.webp', use_container_width=True)
        else:
            st.write("Please enter a book title.")
