import streamlit as st
import pandas as pd
import numpy as np
import html
from sklearn.metrics.pairwise import cosine_similarity
from huggingface_hub import hf_hub_download
import warnings

warnings.filterwarnings("ignore")

# -----------------------------------------------------------------------------
# PAGE CONFIG
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="Book Recommendation System",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded",
)

# -----------------------------------------------------------------------------
# PREMIUM CSS (Glassmorphism + Smooth + Clean)
# -----------------------------------------------------------------------------
st.markdown("""
<style>
/* --- Global --- */
:root{
  --bg0:#0b1220;
  --bg1:#0e1a33;
  --card:#0f1b2d99;
  --card2:#0f1b2dcc;
  --text:#eaf0ff;
  --muted:#b8c2e0;
  --muted2:#91a0c9;
  --line:#223256;
  --brand1:#ff8a00;
  --brand2:#e52e71;
  --brand3:#1a73e8;
  --good:#28a745;
  --shadow: 0 18px 55px rgba(0,0,0,.35);
  --shadow2: 0 10px 30px rgba(0,0,0,.25);
  --radius: 18px;
}

/* Streamlit base */
.stApp {
  background:
    radial-gradient(1200px 600px at 15% 10%, rgba(26,115,232,.30), transparent 60%),
    radial-gradient(900px 500px at 85% 15%, rgba(229,46,113,.25), transparent 60%),
    radial-gradient(800px 600px at 55% 85%, rgba(255,138,0,.18), transparent 60%),
    linear-gradient(180deg, var(--bg0), var(--bg1));
  color: var(--text);
}

header, footer {visibility:hidden;}
.block-container { padding-top: 1.2rem; padding-bottom: 3rem; }

/* --- Hero --- */
.hero {
  border: 1px solid rgba(255,255,255,.08);
  border-radius: 22px;
  background: linear-gradient(135deg, rgba(255,255,255,.06), rgba(255,255,255,.02));
  box-shadow: var(--shadow);
  overflow: hidden;
  position: relative;
  padding: 22px 22px 18px 22px;
}
.hero::after{
  content:"";
  position:absolute; inset:-2px;
  background: linear-gradient(135deg, rgba(255,138,0,.35), rgba(229,46,113,.25), rgba(26,115,232,.22));
  filter: blur(28px);
  z-index:0;
  opacity:.65;
}
.hero-inner{ position:relative; z-index:1; display:flex; gap:18px; align-items:center;}
.hero-title{
  font-family: ui-serif, Georgia, 'Times New Roman', serif;
  font-size: 42px;
  line-height: 1.05;
  margin: 0;
  letter-spacing: .2px;
}
.hero-sub{
  color: var(--muted);
  font-size: 16px;
  margin-top: 6px;
}
.hero-pillbar{
  margin-top: 10px;
  display:flex; flex-wrap:wrap; gap:10px;
}
.pill{
  font-size: 12px;
  color: var(--text);
  border:1px solid rgba(255,255,255,.10);
  background: rgba(15,27,45,.45);
  border-radius: 999px;
  padding: 6px 10px;
}

/* --- Tabs (centered + premium) --- */
.stTabs [data-baseweb="tab-list"] {
  gap: 14px;
  justify-content: center;
}
.stTabs [data-baseweb="tab"]{
  background: rgba(255,255,255,.06);
  border: 1px solid rgba(255,255,255,.10);
  border-bottom: none;
  border-radius: 14px 14px 0 0;
  height: 48px;
  padding: 0 16px;
}
.stTabs [aria-selected="true"]{
  background: linear-gradient(135deg, rgba(255,138,0,.25), rgba(229,46,113,.22), rgba(26,115,232,.18));
  border: 1px solid rgba(255,255,255,.18);
}

/* --- Inputs / Buttons --- */
label, p, h1, h2, h3, h4, h5, h6 { color: var(--text) !important; }
.stSelectbox, .stNumberInput, .stTextInput { color: var(--text); }

.stButton > button{
  border: none;
  border-radius: 999px;
  padding: 10px 16px;
  color: white !important;
  font-weight: 700;
  background: linear-gradient(90deg, var(--brand1), var(--brand2));
  box-shadow: 0 10px 30px rgba(229,46,113,.15);
  transition: transform .14s ease, box-shadow .14s ease, filter .14s ease;
}
.stButton > button:hover{
  transform: translateY(-1px) scale(1.02);
  filter: brightness(1.03);
  box-shadow: 0 16px 44px rgba(0,0,0,.35);
}
.stButton > button:active{ transform: translateY(0px) scale(.995); }

/* --- Section header --- */
.section {
  margin-top: 18px;
  border: 1px solid rgba(255,255,255,.08);
  border-radius: var(--radius);
  background: rgba(15,27,45,.32);
  padding: 16px 16px 6px 16px;
}
.section h3{
  margin: 0;
  font-size: 18px;
  letter-spacing: .2px;
}
.section .hint{
  margin-top: 6px;
  color: var(--muted);
  font-size: 13.5px;
}

/* --- Cards --- */
.grid { margin-top: 8px; }
.card {
  position: relative;
  border-radius: 18px;
  border: 1px solid rgba(255,255,255,.10);
  background: linear-gradient(180deg, rgba(15,27,45,.55), rgba(15,27,45,.28));
  box-shadow: var(--shadow2);
  overflow: hidden;
  transition: transform .16s ease, box-shadow .16s ease, border-color .16s ease;
  height: 470px;
}
.card:hover{
  transform: translateY(-6px);
  box-shadow: 0 24px 70px rgba(0,0,0,.45);
  border-color: rgba(255,255,255,.18);
}
.rank{
  position:absolute;
  top: 12px;
  left: 12px;
  z-index: 2;
  width: 36px;
  height: 36px;
  border-radius: 999px;
  display:flex;
  align-items:center;
  justify-content:center;
  font-weight: 800;
  color: #07111f;
  background: linear-gradient(135deg, #63ffb5, #28a745);
  border: 1px solid rgba(0,0,0,.2);
}
.score{
  position:absolute;
  top: 12px;
  right: 12px;
  z-index: 2;
  font-size: 12px;
  font-weight: 800;
  color: #07111f;
  background: linear-gradient(135deg, #ffd28a, #ff8a00);
  padding: 6px 10px;
  border-radius: 999px;
  border: 1px solid rgba(0,0,0,.18);
}
.cover-wrap{
  display:flex;
  align-items:center;
  justify-content:center;
  padding: 46px 16px 12px 16px;
  height: 320px;
  background:
     radial-gradient(600px 280px at 50% 10%, rgba(26,115,232,.20), transparent 62%),
     radial-gradient(600px 280px at 35% 10%, rgba(229,46,113,.18), transparent 60%);
}
.cover {
  max-height: 290px;
  width: auto;
  object-fit: contain;
  filter: drop-shadow(0 18px 25px rgba(0,0,0,.35));
}
.meta{
  padding: 14px 14px 14px 14px;
  border-top: 1px solid rgba(255,255,255,.08);
  background: rgba(10,16,30,.25);
  height: 150px;
  display:flex;
  flex-direction:column;
  justify-content:space-between;
}
.title{
  font-family: ui-serif, Georgia, 'Times New Roman', serif;
  font-weight: 800;
  font-size: 16px;
  color: #F7E7A1;
  line-height: 1.25;
  max-height: 42px;
  overflow: hidden;
}
.author{
  color: var(--muted);
  font-style: italic;
  font-size: 13px;
  margin-top: 6px;
  white-space: nowrap;
  overflow:hidden;
  text-overflow: ellipsis;
}
.chips{
  display:flex;
  gap:8px;
  flex-wrap:wrap;
  margin-top: 10px;
}
.chip{
  font-size: 12px;
  padding: 6px 10px;
  border-radius: 999px;
  border:1px solid rgba(255,255,255,.10);
  background: rgba(255,255,255,.04);
  color: var(--text);
}
.divider-soft{
  height: 1px;
  background: linear-gradient(90deg, transparent, rgba(255,255,255,.16), transparent);
  margin: 14px 0;
}

/* --- Expander --- */
[data-testid="stExpander"] summary {
  background: rgba(255,255,255,.06) !important;
  border-radius: 14px !important;
  border: 1px solid rgba(255,255,255,.10) !important;
}
[data-testid="stExpander"] details[open] summary {
  border-bottom-left-radius: 0 !important;
  border-bottom-right-radius: 0 !important;
}
[data-testid="stExpander"] details[open] {
  border-radius: 14px !important;
  border: 1px solid rgba(255,255,255,.10) !important;
}

/* --- Small helper --- */
.muted { color: var(--muted) !important; font-size: 13px; }
</style>
""", unsafe_allow_html=True)

# -----------------------------------------------------------------------------
# DATA LOADING + PREP
# -----------------------------------------------------------------------------
@st.cache_data(show_spinner=False)
def load_and_prepare_data():
    final_filtered_df_path = hf_hub_download(
        repo_id="IamPradeep/BRS_DATA",
        filename="final_filtered_df.csv",
        repo_type="dataset",
    )
    final_filtered_df = pd.read_csv(final_filtered_df_path)

    book_urls_df_path = hf_hub_download(
        repo_id="IamPradeep/BRS_DATA",
        filename="Books.csv",
        repo_type="dataset",
    )
    book_urls_df = pd.read_csv(book_urls_df_path)
    book_urls_df.rename(columns={"Book-Title": "title"}, inplace=True)
    book_urls_df = book_urls_df.drop_duplicates(subset=["title"], keep="first")

    final_filtered_df = final_filtered_df.merge(
        book_urls_df[["title", "Book-Author", "Year-Of-Publication", "Image-URL-L"]],
        on="title",
        how="left",
    )

    # Fix a few broken covers (your logic)
    url1 = "http://images.amazon.com/images/P/0690040784.01.LZZZZZZZ.jpg"
    url2 = "http://images.amazon.com/images/P/0451172817.01.LZZZZZZZ.jpg"
    url3 = "http://images.amazon.com/images/P/0312084986.01.LZZZZZZZ.jpg"
    url4 = "http://images.amazon.com/images/P/1590400356.01.LZZZZZZZ.jpg"
    final_filtered_df.loc[final_filtered_df["title"] == "Jacob Have I Loved", "Image-URL-L"] = url1
    final_filtered_df.loc[final_filtered_df["title"] == "Needful Things", "Image-URL-L"] = url2
    final_filtered_df.loc[final_filtered_df["title"] == "All Creatures Great and Small", "Image-URL-L"] = url3
    final_filtered_df.loc[final_filtered_df["title"] == "The Kitchen God's Wife", "Image-URL-L"] = url4

    # Build similarity matrix with explicit ratings only
    explicit = final_filtered_df[final_filtered_df["rating"] > 0]
    book_user_mat = explicit.pivot_table(index="title", columns="userId", values="rating").fillna(0)

    cosine_sim = cosine_similarity(book_user_mat)
    cosine_sim_df = pd.DataFrame(cosine_sim, index=book_user_mat.index, columns=book_user_mat.index)

    # Book metadata table (fast lookup, avoids repeated df filtering)
    meta = (final_filtered_df
            .drop_duplicates(subset=["title"])
            [["title", "Book-Author", "Year-Of-Publication", "Image-URL-L"]]
            .set_index("title"))

    return final_filtered_df, cosine_sim_df, meta

final_filtered_df, cosine_sim_df, book_meta = load_and_prepare_data()

# -----------------------------------------------------------------------------
# HELPER FUNCTIONS
# -----------------------------------------------------------------------------
def get_top_similar_books(book_title, n=10):
    if book_title not in cosine_sim_df.index:
        return None
    similar_scores = cosine_sim_df[book_title].sort_values(ascending=False)
    similar_scores = similar_scores.iloc[1:n+1]  # exclude itself
    return pd.DataFrame({"title": similar_scores.index, "score": similar_scores.values})

def get_user_recommendations(user_id, df, sim_matrix, k=10, candidate_pool=60):
    user_df = df[df["userId"] == user_id][["title", "rating"]].drop_duplicates("title")
    if user_df.empty:
        return None, None, None

    history_titles = user_df["title"].unique().tolist()
    # weight by rating (if 0, treat as weak signal)
    weights = {row["title"]: (row["rating"] if row["rating"] > 0 else 1.0) for _, row in user_df.iterrows()}

    scores = {}
    for t in history_titles:
        if t not in sim_matrix.index:
            continue
        sims = sim_matrix[t].sort_values(ascending=False).iloc[1:candidate_pool+1]
        for cand, s in sims.items():
            if cand in history_titles:
                continue
            scores[cand] = scores.get(cand, 0.0) + (s * weights.get(t, 1.0))

    if not scores:
        return [], user_df.sort_values("rating", ascending=False), pd.DataFrame(columns=["title", "score"])

    ranked = (pd.DataFrame(list(scores.items()), columns=["title", "score"])
              .sort_values("score", ascending=False)
              .head(k)
              .reset_index(drop=True))
    recs = ranked["title"].tolist()
    return recs, user_df.sort_values("rating", ascending=False), ranked

def _safe(x):
    return html.escape("" if pd.isna(x) else str(x), quote=True)

def render_cards(items_df, per_row=3, page=1, page_size=9, show_score=True):
    """
    items_df columns: title, score(optional)
    """
    if items_df is None or len(items_df) == 0:
        st.info("No items to display.")
        return

    total = len(items_df)
    pages = max(1, int(np.ceil(total / page_size)))
    page = max(1, min(page, pages))

    start = (page - 1) * page_size
    end = min(start + page_size, total)

    nav1, nav2, nav3 = st.columns([1, 2, 1])
    with nav1:
        prev = st.button("← Prev", key=f"prev_{hash(str(items_df.shape))}", disabled=(page == 1))
    with nav2:
        st.markdown(f"<div class='muted' style='text-align:center;'>Page <b>{page}</b> / {pages} • Showing <b>{start+1}-{end}</b> of {total}</div>", unsafe_allow_html=True)
    with nav3:
        nxt = st.button("Next →", key=f"next_{hash(str(items_df.shape))}", disabled=(page == pages))

    if prev:
        page -= 1
    if nxt:
        page += 1

    # draw grid
    for idx in range(start, end, per_row):
        cols = st.columns(per_row, gap="large")
        for j in range(per_row):
            k = idx + j
            if k >= end:
                break

            row = items_df.iloc[k]
            title = row["title"]
            score = row["score"] if ("score" in items_df.columns) else None

            if title in book_meta.index:
                author = book_meta.loc[title, "Book-Author"]
                year = book_meta.loc[title, "Year-Of-Publication"]
                img = book_meta.loc[title, "Image-URL-L"]
            else:
                author, year, img = "Unknown", "", ""

            card_html = f"""
            <div class="card">
              <div class="rank">{k+1}</div>
              {"<div class='score'>Score " + f"{float(score):.3f}" + "</div>" if (show_score and score is not None) else ""}
              <div class="cover-wrap">
                <img class="cover" src="{_safe(img)}" loading="lazy"
                     onerror="this.style.display='none';" />
              </div>
              <div class="meta">
                <div>
                  <div class="title" title="{_safe(title)}">{_safe(title)}</div>
                  <div class="author" title="{_safe(author)}">{_safe(author)}</div>
                  <div class="chips">
                    <span class="chip">📅 {_safe(year)}</span>
                    <span class="chip">📖 Similar taste</span>
                  </div>
                </div>
              </div>
            </div>
            """
            with cols[j]:
                st.markdown(card_html, unsafe_allow_html=True)

    return page

# -----------------------------------------------------------------------------
# HERO HEADER
# -----------------------------------------------------------------------------
c1, c2 = st.columns([2.2, 1], gap="large")
with c1:
    st.markdown("""
    <div class="hero">
      <div class="hero-inner">
        <div>
          <h1 class="hero-title">Book Recommendation System</h1>
          <div class="hero-sub">Smooth, personalized, and book‑to‑book recommendations powered by collaborative filtering.</div>
          <div class="hero-pillbar">
            <span class="pill">Cosine Similarity</span>
            <span class="pill">Explicit Ratings Matrix</span>
            <span class="pill">Fast UI + Pagination</span>
            <span class="pill">Premium Cards</span>
          </div>
        </div>
      </div>
    </div>
    """, unsafe_allow_html=True)

with c2:
    st.image(
        "https://img.freepik.com/premium-vector/bookcase-with-books_182089-197.jpg",
        use_container_width=True,
    )

st.markdown("<div class='divider-soft'></div>", unsafe_allow_html=True)

# -----------------------------------------------------------------------------
# SIDEBAR CONTROLS (INTERACTIVE)
# -----------------------------------------------------------------------------
with st.sidebar:
    st.markdown("## ⚙️ Experience Controls")
    st.caption("Make the UI feel exactly how you want.")
    per_row = st.slider("Cards per row", min_value=2, max_value=5, value=3, step=1)
    page_size = st.slider("Cards per page", min_value=6, max_value=30, value=9, step=3)
    st.markdown("<div class='divider-soft'></div>", unsafe_allow_html=True)

    st.markdown("## 📦 Data Overview")
    st.write(f"**Users:** {final_filtered_df['userId'].nunique():,}")
    st.write(f"**Books:** {final_filtered_df['title'].nunique():,}")
    st.write(f"**Interactions:** {len(final_filtered_df):,}")

# -----------------------------------------------------------------------------
# TABS
# -----------------------------------------------------------------------------
tab1, tab2 = st.tabs(["📚 Book‑to‑Book", "👤 User‑Specific"])

# -----------------------------------------------------------------------------
# TAB 1: BOOK-TO-BOOK
# -----------------------------------------------------------------------------
with tab1:
    st.markdown("""
    <div class="section">
      <h3>Find Similar Books</h3>
      <div class="hint">Search a title, then get smooth ranked recommendations with similarity scores.</div>
    </div>
    """, unsafe_allow_html=True)

    all_books = sorted(final_filtered_df["title"].dropna().unique().tolist())

    s1, s2, s3 = st.columns([2.2, 1, 1], gap="large")
    with s1:
        query = st.text_input("Type to filter titles", value="", placeholder="e.g., Harry Potter, Stephen King ...")
        filtered = [b for b in all_books if query.lower() in b.lower()] if query else all_books
        book_title = st.selectbox(
            "Choose a book title",
            filtered,
            index=None,
            placeholder="Select a title...",
            key="book_title_select",
        )

    with s2:
        num_recs = st.number_input("Recommendations", min_value=1, max_value=50, value=10)

    with s3:
        surprise = st.button("🎲 Surprise me")

    if surprise:
        book_title = np.random.choice(all_books)
        st.toast(f"Surprise pick: {book_title}")

    if "page_book" not in st.session_state:
        st.session_state.page_book = 1

    go = st.button("✨ Recommend Similar Books", key="btn_book_recs")

    if go and book_title:
        with st.spinner("Finding similar books..."):
            rec_df = get_top_similar_books(book_title, int(num_recs))
        if rec_df is None:
            st.warning("Book not found in similarity matrix.")
        else:
            st.markdown("<div class='divider-soft'></div>", unsafe_allow_html=True)
            st.markdown(
                f"<div class='muted'>Showing top <b>{len(rec_df)}</b> recommendations for: <b>{html.escape(book_title)}</b></div>",
                unsafe_allow_html=True,
            )
            st.session_state.page_book = 1
            st.session_state.book_rec_df = rec_df

    if "book_rec_df" in st.session_state and st.session_state.book_rec_df is not None:
        st.markdown("<div class='grid'></div>", unsafe_allow_html=True)
        st.session_state.page_book = render_cards(
            st.session_state.book_rec_df,
            per_row=per_row,
            page=st.session_state.page_book,
            page_size=page_size,
            show_score=True,
        )

# -----------------------------------------------------------------------------
# TAB 2: USER-SPECIFIC
# -----------------------------------------------------------------------------
with tab2:
    st.markdown("""
    <div class="section">
      <h3>Personalized Recommendations</h3>
      <div class="hint">Choose a user ID and generate recommendations from their history (weighted by ratings).</div>
    </div>
    """, unsafe_allow_html=True)

    all_user_ids = sorted(final_filtered_df["userId"].dropna().unique().tolist())

    u1, u2, u3 = st.columns([2.2, 1, 1], gap="large")
    with u1:
        user_id = st.selectbox(
            "Select a User ID",
            all_user_ids,
            index=None,
            placeholder="Choose a user...",
            key="user_id_select",
        )
    with u2:
        k = st.number_input("Recommendations", min_value=1, max_value=50, value=10)
    with u3:
        pool = st.number_input("Candidate pool", min_value=20, max_value=200, value=60, step=10)

    if "page_user" not in st.session_state:
        st.session_state.page_user = 1

    if st.button("🚀 Get Personalized Recommendations", key="btn_user_recs"):
        if not user_id:
            st.warning("Please select a User ID.")
        else:
            with st.spinner("Building personalized list..."):
                recs, hist, ranked = get_user_recommendations(
                    user_id, final_filtered_df, cosine_sim_df, k=int(k), candidate_pool=int(pool)
                )
            st.session_state.user_recs = recs
            st.session_state.user_hist = hist
            st.session_state.user_ranked = ranked
            st.session_state.current_user = user_id
            st.session_state.page_user = 1
            st.toast("Recommendations ready!")

    if "user_ranked" in st.session_state and st.session_state.user_ranked is not None:
        user_id_display = st.session_state.current_user
        hist = st.session_state.user_hist
        ranked = st.session_state.user_ranked

        # quick stats
        if hist is not None and not hist.empty:
            rated = hist[hist["rating"] > 0]
            c1, c2, c3 = st.columns(3)
            c1.metric("History size", f"{len(hist):,}")
            c2.metric("Rated books", f"{len(rated):,}")
            c3.metric("Avg rating", f"{rated['rating'].mean():.2f}" if len(rated) else "—")

            with st.expander("📖 View User Reading History"):
                show_only_rated = st.checkbox("Show only rated (>0)", value=False)
                view = hist.copy()
                if show_only_rated:
                    view = view[view["rating"] > 0]
                view = view.sort_values("rating", ascending=False).reset_index(drop=True)
                view.index = view.index + 1
                view.columns = ["Book Title", "Rating"]
                st.dataframe(view, use_container_width=True, height=320)

        st.markdown("<div class='divider-soft'></div>", unsafe_allow_html=True)
        st.markdown(
            f"<div class='muted'>Top <b>{len(ranked)}</b> recommendations for User: <b>{user_id_display}</b></div>",
            unsafe_allow_html=True,
        )

        if ranked is not None and len(ranked) > 0:
            st.session_state.page_user = render_cards(
                ranked,
                per_row=per_row,
                page=st.session_state.page_user,
                page_size=page_size,
                show_score=True,
            )
        else:
            st.info("No recommendations available for this user (not enough signals).")
