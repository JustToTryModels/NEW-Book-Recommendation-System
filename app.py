import streamlit as st
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from huggingface_hub import hf_hub_download
import warnings
from html import escape

st.set_page_config(
    page_title="LuxeReads | Premium Book Recommendations",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="collapsed"
)

warnings.filterwarnings("ignore")

FALLBACK_COVER = "https://placehold.co/600x900/0b1020/E5E7EB?text=No+Cover"


def pick_col(columns, candidates):
    for c in candidates:
        if c in columns:
            return c
    return None


def clean_text(value, default="N/A"):
    if pd.isna(value):
        return default
    value = str(value).strip()
    return value if value else default


@st.cache_data(show_spinner=False)
def load_and_prepare_data():
    # Load your final filtered dataframe from Hugging Face
    final_filtered_df_path = hf_hub_download(
        repo_id="IamPradeep/BRS_DATA",
        filename="final_filtered_df.csv",
        repo_type="dataset"
    )
    final_filtered_df = pd.read_csv(final_filtered_df_path)

    # Load the dataframe containing book URLs from Hugging Face
    book_urls_df_path = hf_hub_download(
        repo_id="IamPradeep/BRS_DATA",
        filename="Books.csv",
        repo_type="dataset"
    )
    book_urls_df = pd.read_csv(book_urls_df_path)
    book_urls_df.rename(columns={'Book-Title': 'title'}, inplace=True)

    # Merge the dataframes on the title
    final_filtered_df = final_filtered_df.merge(book_urls_df, on='title', how='left')

    # Fix specific image URLs
    final_filtered_df.loc[final_filtered_df['title'] == 'Jacob Have I Loved', 'Image-URL-L'] = \
        'http://images.amazon.com/images/P/0690040784.01.LZZZZZZZ.jpg'
    final_filtered_df.loc[final_filtered_df['title'] == 'Needful Things', 'Image-URL-L'] = \
        'http://images.amazon.com/images/P/0451172817.01.LZZZZZZZ.jpg'
    final_filtered_df.loc[final_filtered_df['title'] == 'All Creatures Great and Small', 'Image-URL-L'] = \
        'http://images.amazon.com/images/P/0312084986.01.LZZZZZZZ.jpg'
    final_filtered_df.loc[final_filtered_df['title'] == "The Kitchen God's Wife", 'Image-URL-L'] = \
        'http://images.amazon.com/images/P/1590400356.01.LZZZZZZZ.jpg'

    # Create the book-user matrix
    book_user_mat = final_filtered_df.pivot_table(
        index='title',
        columns='userId',
        values='rating'
    ).fillna(0)

    # Calculate cosine similarity
    cosine_sim = cosine_similarity(book_user_mat)
    cosine_sim_df = pd.DataFrame(
        cosine_sim,
        index=book_user_mat.index,
        columns=book_user_mat.index
    )

    return final_filtered_df, cosine_sim_df


final_filtered_df, cosine_sim_df = load_and_prepare_data()

author_col = pick_col(
    final_filtered_df.columns,
    ['Book-Author', 'Book-Author_x', 'Book-Author_y', 'author']
)
year_col = pick_col(
    final_filtered_df.columns,
    ['Year-Of-Publication', 'Year-Of-Publication_x', 'Year-Of-Publication_y', 'year']
)
image_col = pick_col(
    final_filtered_df.columns,
    ['Image-URL-L', 'Image-URL-L_x', 'Image-URL-L_y']
)

book_catalog = final_filtered_df.drop_duplicates(subset='title').copy()
all_books = sorted(book_catalog['title'].dropna().unique().tolist())

total_books = len(all_books)
total_users = int(final_filtered_df['userId'].nunique()) if 'userId' in final_filtered_df.columns else 0
avg_rating = round(float(final_filtered_df['rating'].mean()), 2) if 'rating' in final_filtered_df.columns else 0.0


def get_book_meta(title):
    row_df = book_catalog[book_catalog['title'] == title]
    if row_df.empty:
        return {
            "title": title,
            "author": "Unknown author",
            "year": "N/A",
            "image": FALLBACK_COVER
        }

    row = row_df.iloc[0]

    author = clean_text(row[author_col], "Unknown author") if author_col else "Unknown author"
    year = clean_text(row[year_col], "N/A") if year_col else "N/A"
    image = clean_text(row[image_col], FALLBACK_COVER) if image_col else FALLBACK_COVER

    try:
        year = str(int(float(year)))
    except:
        pass

    if not isinstance(image, str) or not image.startswith("http"):
        image = FALLBACK_COVER

    return {
        "title": clean_text(title, "Unknown title"),
        "author": author,
        "year": year,
        "image": image
    }


def get_top_similar_books(book_title, n=10):
    if book_title not in cosine_sim_df.index:
        return "⚠️ Book not found in the database."

    similar_scores = cosine_sim_df[book_title]
    similar_books = similar_scores.sort_values(ascending=False)[1:n+1]
    return similar_books


# ---------- Premium CSS ----------
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

html, body, [class*="css"], [class*="st-"] {
    font-family: 'Inter', sans-serif !important;
}

.stApp {
    background:
        radial-gradient(circle at 10% 10%, rgba(124, 58, 237, 0.22), transparent 22%),
        radial-gradient(circle at 90% 10%, rgba(6, 182, 212, 0.18), transparent 22%),
        radial-gradient(circle at 50% 85%, rgba(236, 72, 153, 0.12), transparent 20%),
        linear-gradient(180deg, #030712 0%, #07111f 45%, #030712 100%);
    color: #f8fafc;
}

.block-container {
    max-width: 1320px;
    padding-top: 1.2rem;
    padding-bottom: 3rem;
}

#MainMenu, footer, header {
    visibility: hidden;
}

div[data-testid="stToolbar"],
div[data-testid="stDecoration"] {
    display: none !important;
}

.topbar {
    display: flex;
    justify-content: space-between;
    align-items: center;
    padding: 0.3rem 0 1.1rem 0;
    gap: 1rem;
    flex-wrap: wrap;
}

.brand {
    font-size: 1.35rem;
    font-weight: 800;
    color: white;
    letter-spacing: 0.02em;
}

.brand span {
    background: linear-gradient(135deg, #8b5cf6, #22d3ee, #f472b6);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
}

.nav {
    display: flex;
    gap: 0.7rem;
    flex-wrap: wrap;
}

.nav span {
    color: rgba(255,255,255,0.74);
    font-size: 0.92rem;
    padding: 0.48rem 0.85rem;
    border: 1px solid rgba(255,255,255,0.08);
    border-radius: 999px;
    background: rgba(255,255,255,0.04);
    backdrop-filter: blur(8px);
}

.hero-card {
    position: relative;
    overflow: hidden;
    border-radius: 30px;
    padding: 2.4rem;
    background: linear-gradient(135deg, rgba(15,23,42,0.92), rgba(2,6,23,0.96));
    border: 1px solid rgba(255,255,255,0.10);
    box-shadow: 0 28px 80px rgba(0,0,0,0.42);
    margin-bottom: 1.2rem;
}

.hero-card::before {
    content: "";
    position: absolute;
    top: -100px;
    right: -80px;
    width: 380px;
    height: 380px;
    background: radial-gradient(circle, rgba(124,58,237,0.28), transparent 62%);
    pointer-events: none;
}

.hero-card::after {
    content: "";
    position: absolute;
    bottom: -120px;
    left: -90px;
    width: 360px;
    height: 360px;
    background: radial-gradient(circle, rgba(6,182,212,0.20), transparent 62%);
    pointer-events: none;
}

.hero-content {
    position: relative;
    z-index: 2;
    max-width: 760px;
}

.kicker {
    display: inline-block;
    margin-bottom: 0.9rem;
    padding: 0.38rem 0.75rem;
    border-radius: 999px;
    border: 1px solid rgba(255,255,255,0.12);
    background: rgba(255,255,255,0.06);
    color: #cbd5e1;
    font-size: 0.78rem;
    font-weight: 700;
    letter-spacing: 0.18em;
}

.hero-title {
    margin: 0;
    color: white;
    font-size: clamp(2.2rem, 4vw, 4.25rem);
    line-height: 1.02;
    letter-spacing: -0.04em;
    font-weight: 800;
}

.hero-title span {
    background: linear-gradient(135deg, #a78bfa, #22d3ee, #f472b6);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
}

.hero-subtitle {
    margin-top: 1rem;
    color: rgba(255,255,255,0.78);
    font-size: 1.02rem;
    line-height: 1.7;
    max-width: 680px;
}

.hero-pills {
    display: flex;
    flex-wrap: wrap;
    gap: 0.7rem;
    margin-top: 1.15rem;
}

.hero-pill {
    padding: 0.55rem 0.9rem;
    border-radius: 999px;
    background: rgba(255,255,255,0.06);
    border: 1px solid rgba(255,255,255,0.12);
    color: #e2e8f0;
    font-size: 0.9rem;
}

.stats-row {
    display: grid;
    grid-template-columns: repeat(3, 1fr);
    gap: 1rem;
    margin: 1rem 0 1.55rem;
}

.stat-card {
    padding: 1rem 1.2rem;
    border-radius: 22px;
    background: rgba(255,255,255,0.05);
    border: 1px solid rgba(255,255,255,0.08);
    backdrop-filter: blur(14px);
    box-shadow: 0 14px 32px rgba(0,0,0,0.22);
}

.stat-number {
    display: block;
    color: white;
    font-weight: 800;
    font-size: 1.65rem;
}

.stat-label {
    display: block;
    color: rgba(255,255,255,0.65);
    margin-top: 0.2rem;
    font-size: 0.92rem;
}

div[data-testid="stForm"] {
    background: linear-gradient(180deg, rgba(255,255,255,0.06), rgba(255,255,255,0.035));
    border: 1px solid rgba(255,255,255,0.10);
    border-radius: 26px;
    padding: 1.1rem 1.15rem 0.65rem;
    box-shadow: 0 18px 44px rgba(0,0,0,0.28);
    margin-bottom: 1rem;
}

.form-title {
    color: white;
    font-size: 1.06rem;
    font-weight: 700;
    margin-bottom: 0.25rem;
}

.form-subtitle {
    color: rgba(255,255,255,0.66);
    font-size: 0.92rem;
    margin-bottom: 0.95rem;
}

.stSelectbox label, .stNumberInput label {
    color: #e2e8f0 !important;
    font-weight: 600 !important;
}

div[data-baseweb="select"] > div,
div[data-baseweb="input"] > div {
    background: rgba(15,23,42,0.76) !important;
    border: 1px solid rgba(255,255,255,0.12) !important;
    border-radius: 16px !important;
    min-height: 54px !important;
    color: white !important;
}

div[data-baseweb="select"] span,
div[data-baseweb="input"] input {
    color: white !important;
}

div[data-baseweb="select"]:hover > div,
div[data-baseweb="input"]:hover > div {
    border-color: rgba(99,102,241,0.55) !important;
    box-shadow: 0 0 0 1px rgba(99,102,241,0.18);
}

ul[role="listbox"] {
    background: #0f172a !important;
    color: white !important;
    border: 1px solid rgba(255,255,255,0.08) !important;
}

.stFormSubmitButton > button,
.stButton > button {
    width: 100%;
    min-height: 54px;
    border-radius: 16px;
    border: none;
    font-weight: 800;
    color: white;
    background: linear-gradient(135deg, #7c3aed 0%, #ec4899 52%, #06b6d4 100%);
    box-shadow: 0 16px 38px rgba(124,58,237,0.36);
    transition: all 0.25s ease;
}

.stFormSubmitButton > button:hover,
.stButton > button:hover {
    transform: translateY(-2px) scale(1.01);
    box-shadow: 0 22px 46px rgba(124,58,237,0.46);
}

.section-heading {
    color: white;
    font-size: 1.15rem;
    font-weight: 800;
    margin: 1.3rem 0 0.9rem;
}

.spotlight {
    display: grid;
    grid-template-columns: 220px 1fr;
    gap: 1.4rem;
    align-items: center;
    padding: 1.2rem;
    border-radius: 28px;
    background: linear-gradient(135deg, rgba(255,255,255,0.06), rgba(255,255,255,0.03));
    border: 1px solid rgba(255,255,255,0.08);
    box-shadow: 0 24px 56px rgba(0,0,0,0.26);
    margin-bottom: 1.15rem;
}

.spotlight-cover img {
    width: 100%;
    height: 320px;
    object-fit: cover;
    border-radius: 20px;
    box-shadow: 0 20px 42px rgba(0,0,0,0.36);
}

.spotlight-title {
    color: white;
    font-size: clamp(1.6rem, 2.1vw, 2.35rem);
    font-weight: 800;
    line-height: 1.08;
    margin: 0.2rem 0 0.6rem;
}

.spotlight-desc {
    color: rgba(255,255,255,0.76);
    font-size: 1rem;
    line-height: 1.7;
    margin-bottom: 1rem;
}

.meta-row {
    display: flex;
    flex-wrap: wrap;
    gap: 0.72rem;
}

.meta-chip {
    padding: 0.55rem 0.86rem;
    border-radius: 999px;
    background: rgba(255,255,255,0.06);
    border: 1px solid rgba(255,255,255,0.1);
    color: #e2e8f0;
    font-size: 0.9rem;
}

.rec-card {
    position: relative;
    overflow: hidden;
    border-radius: 24px;
    padding: 0.9rem;
    background: linear-gradient(180deg, rgba(255,255,255,0.06), rgba(255,255,255,0.03));
    border: 1px solid rgba(255,255,255,0.08);
    box-shadow: 0 18px 42px rgba(0,0,0,0.30);
    transition: transform 0.22s ease, box-shadow 0.22s ease, border-color 0.22s ease;
    height: 100%;
}

.rec-card:hover {
    transform: translateY(-6px);
    box-shadow: 0 24px 56px rgba(0,0,0,0.35);
    border-color: rgba(99,102,241,0.35);
}

.rec-card img {
    width: 100%;
    height: 290px;
    object-fit: cover;
    border-radius: 18px;
    margin-bottom: 0.85rem;
}

.rank-chip {
    display: inline-flex;
    align-items: center;
    padding: 0.35rem 0.68rem;
    border-radius: 999px;
    background: linear-gradient(135deg, rgba(124,58,237,0.92), rgba(236,72,153,0.92));
    color: white;
    font-size: 0.78rem;
    font-weight: 700;
    margin-bottom: 0.7rem;
}

.rec-title {
    color: white;
    font-size: 1rem;
    font-weight: 700;
    line-height: 1.3;
    min-height: 2.6em;
    display: -webkit-box;
    -webkit-line-clamp: 2;
    -webkit-box-orient: vertical;
    overflow: hidden;
    margin-bottom: 0.45rem;
}

.rec-author, .rec-year {
    color: rgba(255,255,255,0.68);
    font-size: 0.92rem;
}

.match-text {
    margin-top: 0.75rem;
    color: #cbd5e1;
    font-size: 0.84rem;
    font-weight: 700;
}

.match-bar {
    width: 100%;
    height: 8px;
    border-radius: 999px;
    background: rgba(255,255,255,0.08);
    overflow: hidden;
    margin-top: 0.35rem;
}

.match-fill {
    height: 100%;
    border-radius: 999px;
    background: linear-gradient(90deg, #7c3aed, #06b6d4, #22c55e);
}

.footer-note {
    text-align: center;
    color: rgba(255,255,255,0.55);
    margin-top: 2rem;
    font-size: 0.92rem;
}

@media (max-width: 900px) {
    .stats-row {
        grid-template-columns: 1fr;
    }
    .spotlight {
        grid-template-columns: 1fr;
    }
    .spotlight-cover img {
        height: 340px;
    }
    .nav {
        display: none;
    }
}
</style>
""", unsafe_allow_html=True)


# ---------- Header / Hero ----------
st.markdown(f"""
<div class="topbar">
    <div class="brand"><span>Luxe</span>Reads</div>
    <div class="nav">
        <span>Discover</span>
        <span>Top Picks</span>
        <span>Curated Taste</span>
        <span>Premium Library</span>
    </div>
</div>

<div class="hero-card">
    <div class="hero-content">
        <div class="kicker">PREMIUM BOOK DISCOVERY</div>
        <h1 class="hero-title">Find your next <span>unputdownable</span> book.</h1>
        <p class="hero-subtitle">
            A cinematic recommendation experience inspired by premium streaming platforms —
            elegant visuals, curated results, and taste-based book discovery powered by reader similarity.
        </p>
        <div class="hero-pills">
            <div class="hero-pill">✨ Curated recommendations</div>
            <div class="hero-pill">📚 Rich book metadata</div>
            <div class="hero-pill">⚡ Instant similarity engine</div>
        </div>
    </div>
</div>

<div class="stats-row">
    <div class="stat-card">
        <span class="stat-number">{total_books:,}</span>
        <span class="stat-label">Books in collection</span>
    </div>
    <div class="stat-card">
        <span class="stat-number">{total_users:,}</span>
        <span class="stat-label">Readers analyzed</span>
    </div>
    <div class="stat-card">
        <span class="stat-number">{avg_rating}</span>
        <span class="stat-label">Average rating</span>
    </div>
</div>
""", unsafe_allow_html=True)


# ---------- Session State ----------
if 'recommendations' not in st.session_state:
    st.session_state.recommendations = None
if 'recommended_book' not in st.session_state:
    st.session_state.recommended_book = None
if 'recommended_num' not in st.session_state:
    st.session_state.recommended_num = None


# ---------- Search Form ----------
with st.form("premium_recommendation_form", clear_on_submit=False):
    st.markdown("<div class='form-title'>Build your personalized shelf</div>", unsafe_allow_html=True)
    st.markdown("<div class='form-subtitle'>Choose a book you love and let the engine curate similar titles for you.</div>", unsafe_allow_html=True)

    c1, c2, c3 = st.columns([3.8, 1.2, 1.3])

    with c1:
        book_title = st.selectbox(
            'Choose a book title',
            all_books,
            index=None,
            placeholder="Start typing to search for a book...",
            key='book_title'
        )

    with c2:
        num_recommendations = st.number_input(
            'How many picks?',
            min_value=1,
            max_value=50,
            value=10,
            step=1,
            key='num_recommendations'
        )

    with c3:
        st.markdown("<div style='height: 31px;'></div>", unsafe_allow_html=True)
        submitted = st.form_submit_button("✨ Curate Now", use_container_width=True)

if submitted:
    if book_title:
        with st.spinner("Curating premium recommendations for you..."):
            similar_books = get_top_similar_books(book_title, int(num_recommendations))
            st.session_state.recommendations = similar_books
            st.session_state.recommended_book = book_title
            st.session_state.recommended_num = int(num_recommendations)
    else:
        st.session_state.recommendations = None
        st.warning("⚠️ Please choose a book title first.")


# ---------- Results ----------
if st.session_state.recommendations is not None:
    similar_books = st.session_state.recommendations
    rec_book = st.session_state.recommended_book
    rec_num = st.session_state.recommended_num

    if isinstance(similar_books, str):
        st.warning(similar_books)
    else:
        selected_meta = get_book_meta(rec_book)
        avg_match = int(round(similar_books.mean() * 100)) if len(similar_books) > 0 else 0

        st.markdown("<div class='section-heading'>Because you selected</div>", unsafe_allow_html=True)

        st.markdown(f"""
        <div class="spotlight">
            <div class="spotlight-cover">
                <img src="{escape(selected_meta['image'])}" alt="{escape(selected_meta['title'])}">
            </div>
            <div>
                <div class="kicker">SPOTLIGHT TITLE</div>
                <div class="spotlight-title">{escape(selected_meta['title'])}</div>
                <div class="spotlight-desc">
                    Readers with similar preferences also showed strong affinity for the books below.
                    These recommendations are ranked using cosine similarity on reader-rating behavior.
                </div>
                <div class="meta-row">
                    <div class="meta-chip">👤 {escape(selected_meta['author'])}</div>
                    <div class="meta-chip">📅 {escape(selected_meta['year'])}</div>
                    <div class="meta-chip">🎯 Avg. match {avg_match}%</div>
                    <div class="meta-chip">📚 Top {rec_num} recommendations</div>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("<div class='section-heading'>Top recommendations for you</div>", unsafe_allow_html=True)

        for i in range(0, len(similar_books), 4):
            cols = st.columns(4, gap="large")
            for j in range(4):
                if i + j < len(similar_books):
                    book = similar_books.index[i + j]
                    score = float(similar_books.iloc[i + j])
                    score_pct = max(1, min(100, int(round(score * 100))))
                    meta = get_book_meta(book)

                    with cols[j]:
                        st.markdown(f"""
                        <div class="rec-card">
                            <img src="{escape(meta['image'])}" alt="{escape(meta['title'])}">
                            <div class="rank-chip">#{i + j + 1}</div>
                            <div class="rec-title">{escape(meta['title'])}</div>
                            <div class="rec-author">👤 {escape(meta['author'])}</div>
                            <div class="rec-year">📅 {escape(meta['year'])}</div>
                            <div class="match-text">{score_pct}% taste match</div>
                            <div class="match-bar">
                                <div class="match-fill" style="width:{score_pct}%"></div>
                            </div>
                        </div>
                        """, unsafe_allow_html=True)

        st.markdown("<div class='footer-note'>Crafted for immersive book discovery ✨</div>", unsafe_allow_html=True)

else:
    st.markdown("""
    <div class="footer-note">
        Choose a title above to unlock a premium recommendation experience.
    </div>
    """, unsafe_allow_html=True)
