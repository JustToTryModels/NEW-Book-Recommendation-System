import streamlit as st
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from huggingface_hub import hf_hub_download
import warnings

warnings.filterwarnings('ignore')

# -------------------------------------------------------------------------
# PAGE CONFIG - Must be first Streamlit command
# -------------------------------------------------------------------------
st.set_page_config(
    page_title="BookVerse | AI Book Recommendations",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# -------------------------------------------------------------------------
# MASTER CSS - Full Premium Styling
# -------------------------------------------------------------------------
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Playfair+Display:ital,wght@0,400;0,600;0,700;0,800;1,400;1,600&family=Inter:wght@300;400;500;600;700&family=Crimson+Pro:ital,wght@0,400;0,600;1,400&display=swap');

/* ── ROOT VARIABLES ─────────────────────────────────── */
:root {
  --gold:       #C9A84C;
  --gold-light: #E8C97A;
  --gold-glow:  rgba(201,168,76,0.35);
  --navy:       #0D1B2A;
  --navy-mid:   #112240;
  --navy-card:  #162032;
  --navy-hover: #1D2D44;
  --accent:     #E63946;
  --accent2:    #FF6B6B;
  --teal:       #2EC4B6;
  --purple:     #7B2FBE;
  --text-main:  #F0EAD6;
  --text-muted: #8A9BB0;
  --text-dim:   #5A6A7A;
  --border:     rgba(201,168,76,0.2);
  --glass:      rgba(13,27,42,0.85);
  --radius-lg:  16px;
  --radius-md:  10px;
  --radius-sm:  6px;
  --shadow-gold: 0 0 30px rgba(201,168,76,0.15);
  --shadow-card: 0 8px 32px rgba(0,0,0,0.4);
  --transition: all 0.35s cubic-bezier(0.4,0,0.2,1);
}

/* ── GLOBAL RESET & BASE ────────────────────────────── */
*, *::before, *::after { box-sizing: border-box; }

html, body, [data-testid="stAppViewContainer"],
[data-testid="stApp"] {
  background: var(--navy) !important;
  color: var(--text-main) !important;
  font-family: 'Inter', sans-serif !important;
}

/* Animated starfield background */
[data-testid="stAppViewContainer"]::before {
  content: '';
  position: fixed;
  inset: 0;
  background:
    radial-gradient(ellipse at 20% 50%, rgba(123,47,190,0.08) 0%, transparent 60%),
    radial-gradient(ellipse at 80% 20%, rgba(46,196,182,0.06) 0%, transparent 50%),
    radial-gradient(ellipse at 50% 80%, rgba(201,168,76,0.05) 0%, transparent 55%);
  pointer-events: none;
  z-index: 0;
}

[data-testid="stMain"] > div { position: relative; z-index: 1; }

/* Hide Streamlit chrome */
#MainMenu, footer, header,
[data-testid="stToolbar"],
[data-testid="stDecoration"],
[data-testid="stStatusWidget"] { display: none !important; }

/* Scrollbar */
::-webkit-scrollbar { width: 6px; height: 6px; }
::-webkit-scrollbar-track { background: var(--navy-mid); }
::-webkit-scrollbar-thumb {
  background: linear-gradient(180deg, var(--gold), var(--purple));
  border-radius: 99px;
}

/* ── TYPOGRAPHY ─────────────────────────────────────── */
h1, h2, h3, h4 {
  font-family: 'Playfair Display', Georgia, serif !important;
  color: var(--text-main) !important;
}
p, span, label, div {
  font-family: 'Inter', sans-serif !important;
}

/* ── HERO SECTION ───────────────────────────────────── */
.hero-wrapper {
  position: relative;
  text-align: center;
  padding: 60px 20px 40px;
  margin-bottom: 10px;
  overflow: hidden;
}
.hero-wrapper::before {
  content: '';
  position: absolute;
  inset: 0;
  background: radial-gradient(ellipse at center top, rgba(201,168,76,0.12) 0%, transparent 65%);
  pointer-events: none;
}
.hero-badge {
  display: inline-flex;
  align-items: center;
  gap: 8px;
  background: linear-gradient(135deg, rgba(201,168,76,0.15), rgba(201,168,76,0.05));
  border: 1px solid var(--gold);
  border-radius: 99px;
  padding: 6px 18px;
  font-size: 11px;
  font-weight: 600;
  letter-spacing: 2.5px;
  text-transform: uppercase;
  color: var(--gold-light);
  margin-bottom: 24px;
  animation: fadeSlideDown 0.8s ease both;
}
.hero-badge span { font-size: 14px; }
.hero-title {
  font-family: 'Playfair Display', Georgia, serif !important;
  font-size: clamp(42px, 6vw, 80px) !important;
  font-weight: 800 !important;
  line-height: 1.1 !important;
  margin: 0 0 16px !important;
  background: linear-gradient(135deg, #F0EAD6 0%, var(--gold-light) 40%, var(--gold) 70%, #C17F24 100%);
  -webkit-background-clip: text !important;
  -webkit-text-fill-color: transparent !important;
  background-clip: text !important;
  animation: fadeSlideDown 0.9s ease both;
  text-shadow: none !important;
  filter: drop-shadow(0 0 40px rgba(201,168,76,0.3));
}
.hero-subtitle {
  font-size: 17px !important;
  color: var(--text-muted) !important;
  font-weight: 400 !important;
  letter-spacing: 0.3px !important;
  max-width: 560px !important;
  margin: 0 auto 32px !important;
  line-height: 1.7 !important;
  animation: fadeSlideDown 1.0s ease both;
}
.hero-divider {
  width: 80px;
  height: 2px;
  background: linear-gradient(90deg, transparent, var(--gold), transparent);
  margin: 0 auto 32px;
  animation: fadeSlideDown 1.1s ease both;
}
.hero-stats {
  display: flex;
  justify-content: center;
  gap: 40px;
  animation: fadeSlideDown 1.2s ease both;
}
.stat-item { text-align: center; }
.stat-number {
  font-family: 'Playfair Display', serif;
  font-size: 28px;
  font-weight: 700;
  color: var(--gold-light);
  display: block;
  line-height: 1;
}
.stat-label {
  font-size: 11px;
  color: var(--text-dim);
  text-transform: uppercase;
  letter-spacing: 1.5px;
  margin-top: 4px;
}
.stat-sep {
  width: 1px;
  background: var(--border);
  align-self: stretch;
}

/* ── HERO BANNER IMAGE ──────────────────────────────── */
.hero-image-container {
  position: relative;
  border-radius: 20px;
  overflow: hidden;
  margin: 0 0 40px;
  box-shadow: 0 20px 60px rgba(0,0,0,0.5), 0 0 0 1px var(--border);
}
.hero-image-container::after {
  content: '';
  position: absolute;
  inset: 0;
  background: linear-gradient(
    to bottom,
    transparent 40%,
    rgba(13,27,42,0.6) 80%,
    var(--navy) 100%
  );
  pointer-events: none;
}
.hero-image-container img {
  width: 100%;
  max-height: 340px;
  object-fit: cover;
  display: block;
}

/* ── TABS ───────────────────────────────────────────── */
[data-testid="stTabs"] {
  background: transparent !important;
}
.stTabs [data-baseweb="tab-list"] {
  gap: 0 !important;
  background: var(--navy-mid) !important;
  border-radius: var(--radius-lg) !important;
  padding: 6px !important;
  border: 1px solid var(--border) !important;
  width: fit-content !important;
  margin: 0 auto 32px !important;
  box-shadow: var(--shadow-gold);
}
.stTabs [data-baseweb="tab"] {
  background: transparent !important;
  border: none !important;
  border-radius: var(--radius-md) !important;
  padding: 12px 28px !important;
  font-family: 'Inter', sans-serif !important;
  font-size: 14px !important;
  font-weight: 600 !important;
  color: var(--text-muted) !important;
  letter-spacing: 0.3px !important;
  transition: var(--transition) !important;
  white-space: nowrap !important;
}
.stTabs [data-baseweb="tab"] p {
  color: inherit !important;
  font-size: inherit !important;
  font-weight: inherit !important;
  font-family: inherit !important;
}
.stTabs [data-baseweb="tab"]:hover {
  color: var(--gold-light) !important;
  background: rgba(201,168,76,0.08) !important;
}
.stTabs [aria-selected="true"] {
  background: linear-gradient(135deg, rgba(201,168,76,0.25), rgba(201,168,76,0.1)) !important;
  color: var(--gold-light) !important;
  box-shadow: 0 2px 12px rgba(201,168,76,0.2) !important;
  border: 1px solid rgba(201,168,76,0.3) !important;
}
.stTabs [aria-selected="true"] p { color: var(--gold-light) !important; }
.stTabs [data-baseweb="tab-border"],
.stTabs [data-baseweb="tab-highlight"] { display: none !important; }

/* ── SELECT BOX ─────────────────────────────────────── */
[data-testid="stSelectbox"] > div > div {
  background: var(--navy-card) !important;
  border: 1.5px solid var(--border) !important;
  border-radius: var(--radius-md) !important;
  color: var(--text-main) !important;
  font-family: 'Inter', sans-serif !important;
  transition: var(--transition) !important;
  box-shadow: 0 2px 12px rgba(0,0,0,0.2) !important;
}
[data-testid="stSelectbox"] > div > div:focus-within,
[data-testid="stSelectbox"] > div > div:hover {
  border-color: var(--gold) !important;
  box-shadow: 0 0 0 3px var(--gold-glow), 0 2px 12px rgba(0,0,0,0.2) !important;
}
[data-testid="stSelectbox"] svg { color: var(--gold) !important; fill: var(--gold) !important; }
[data-testid="stSelectbox"] input { color: var(--text-main) !important; }

/* Dropdown menu */
[data-baseweb="popover"] > div,
[data-baseweb="select"] [role="listbox"] {
  background: var(--navy-card) !important;
  border: 1px solid var(--border) !important;
  border-radius: var(--radius-md) !important;
  box-shadow: 0 16px 48px rgba(0,0,0,0.5) !important;
}
[data-baseweb="option"] {
  background: var(--navy-card) !important;
  color: var(--text-main) !important;
  font-family: 'Inter', sans-serif !important;
  transition: var(--transition) !important;
}
[data-baseweb="option"]:hover,
[data-baseweb="option"][aria-selected="true"] {
  background: rgba(201,168,76,0.12) !important;
  color: var(--gold-light) !important;
}

/* ── NUMBER INPUT ───────────────────────────────────── */
[data-testid="stNumberInput"] > div > div {
  background: var(--navy-card) !important;
  border: 1.5px solid var(--border) !important;
  border-radius: var(--radius-md) !important;
  color: var(--text-main) !important;
  transition: var(--transition) !important;
}
[data-testid="stNumberInput"] > div > div:focus-within {
  border-color: var(--gold) !important;
  box-shadow: 0 0 0 3px var(--gold-glow) !important;
}
[data-testid="stNumberInput"] input { color: var(--text-main) !important; font-family: 'Inter', sans-serif !important; }
[data-testid="stNumberInput"] button { color: var(--gold) !important; }

/* ── LABELS ─────────────────────────────────────────── */
[data-testid="stSelectbox"] label,
[data-testid="stNumberInput"] label {
  color: var(--text-muted) !important;
  font-size: 12px !important;
  font-weight: 600 !important;
  letter-spacing: 0.8px !important;
  text-transform: uppercase !important;
  margin-bottom: 6px !important;
}

/* ── BUTTONS ────────────────────────────────────────── */
.stButton > button {
  font-family: 'Inter', sans-serif !important;
  font-size: 14px !important;
  font-weight: 700 !important;
  letter-spacing: 0.8px !important;
  text-transform: uppercase !important;
  background: linear-gradient(135deg, var(--gold) 0%, #E8A820 50%, var(--gold) 100%) !important;
  background-size: 200% 200% !important;
  color: var(--navy) !important;
  border: none !important;
  border-radius: 99px !important;
  padding: 14px 36px !important;
  cursor: pointer !important;
  transition: var(--transition) !important;
  box-shadow: 0 4px 20px rgba(201,168,76,0.4), 0 0 0 0 rgba(201,168,76,0.2) !important;
  width: auto !important;
  display: inline-flex !important;
  align-items: center !important;
  justify-content: center !important;
  gap: 8px !important;
  animation: shimmer 3s infinite !important;
}
.stButton > button:hover {
  transform: translateY(-2px) scale(1.02) !important;
  box-shadow: 0 8px 30px rgba(201,168,76,0.55), 0 0 0 4px rgba(201,168,76,0.15) !important;
  color: var(--navy) !important;
}
.stButton > button:active {
  transform: translateY(0) scale(0.98) !important;
}

/* Center button */
.btn-center { display: flex; justify-content: center; margin: 24px 0; }

/* ── SECTION HEADERS ────────────────────────────────── */
.section-header {
  text-align: center;
  margin-bottom: 32px;
  padding: 0 20px;
}
.section-title {
  font-family: 'Playfair Display', Georgia, serif;
  font-size: 28px;
  font-weight: 700;
  color: var(--text-main);
  margin-bottom: 8px;
}
.section-desc {
  font-size: 14px;
  color: var(--text-muted);
  max-width: 520px;
  margin: 0 auto;
  line-height: 1.7;
}

/* ── RECOMMENDATION LABEL ───────────────────────────── */
.rec-label {
  display: flex;
  align-items: center;
  gap: 12px;
  background: linear-gradient(135deg, rgba(201,168,76,0.1), rgba(201,168,76,0.03));
  border: 1px solid var(--border);
  border-left: 4px solid var(--gold);
  border-radius: var(--radius-md);
  padding: 14px 20px;
  margin: 24px 0 28px;
  font-size: 14px;
  color: var(--text-muted);
  line-height: 1.5;
}
.rec-label strong { color: var(--gold-light); font-weight: 600; }
.rec-label-icon {
  font-size: 22px;
  flex-shrink: 0;
  filter: drop-shadow(0 0 6px rgba(201,168,76,0.5));
}

/* ── ROW SEPARATOR ──────────────────────────────────── */
.row-sep {
  display: flex;
  align-items: center;
  gap: 16px;
  margin: 36px 0;
}
.row-sep-line {
  flex: 1;
  height: 1px;
  background: linear-gradient(90deg, transparent, var(--border), transparent);
}
.row-sep-dot {
  width: 6px; height: 6px;
  background: var(--gold);
  border-radius: 50%;
  box-shadow: 0 0 8px var(--gold);
}

/* ── BOOK CARD ──────────────────────────────────────── */
.book-card-wrap {
  position: relative;
  margin-top: 30px;
  margin-bottom: 10px;
}

.book-card {
  background: var(--navy-card);
  border: 1px solid rgba(201,168,76,0.15);
  border-radius: var(--radius-lg);
  overflow: hidden;
  transition: var(--transition);
  cursor: pointer;
  position: relative;
  height: 100%;
  box-shadow: var(--shadow-card);
}
.book-card:hover {
  transform: translateY(-8px) scale(1.01);
  border-color: rgba(201,168,76,0.5);
  box-shadow: 0 20px 50px rgba(0,0,0,0.5), 0 0 0 1px rgba(201,168,76,0.3), var(--shadow-gold);
}
.book-card:hover .book-img-overlay { opacity: 1; }
.book-card:hover .book-card-shine { opacity: 1; }

.book-card-shine {
  position: absolute;
  top: 0; left: -100%;
  width: 60%; height: 100%;
  background: linear-gradient(
    105deg,
    transparent 40%,
    rgba(255,255,255,0.04) 50%,
    transparent 60%
  );
  opacity: 0;
  transition: opacity 0.5s ease;
  pointer-events: none;
  z-index: 2;
  animation: none;
}
.book-card:hover .book-card-shine {
  opacity: 1;
  animation: cardShine 0.8s ease forwards;
}

.badge-rank {
  position: absolute;
  top: -14px;
  left: 50%;
  transform: translateX(-50%);
  z-index: 10;
  width: 44px; height: 44px;
  border-radius: 50%;
  display: flex; align-items: center; justify-content: center;
  font-family: 'Playfair Display', serif;
  font-size: 16px;
  font-weight: 800;
  border: 2px solid var(--navy-card);
  box-shadow: 0 4px 16px rgba(0,0,0,0.4);
  transition: var(--transition);
}
.rank-1 { background: linear-gradient(135deg, #FFD700, #FFA500); color: #1a0a00; }
.rank-2 { background: linear-gradient(135deg, #C0C0C0, #A8A8A8); color: #1a1a1a; }
.rank-3 { background: linear-gradient(135deg, #CD7F32, #A0522D); color: #fff8f0; }
.rank-other { background: linear-gradient(135deg, var(--navy-mid), var(--navy-hover)); color: var(--gold-light); border: 1.5px solid var(--gold) !important; }

.book-img-container {
  position: relative;
  background: linear-gradient(145deg, #1a2a3a, #0f1e2e);
  padding: 28px 16px 16px;
  display: flex;
  align-items: center;
  justify-content: center;
  min-height: 240px;
  overflow: hidden;
}
.book-img-container::before {
  content: '';
  position: absolute;
  inset: 0;
  background:
    radial-gradient(ellipse at top center, rgba(201,168,76,0.06) 0%, transparent 60%);
  pointer-events: none;
}
.book-img-container img {
  height: 200px !important;
  width: auto !important;
  max-width: 140px !important;
  object-fit: contain !important;
  display: block !important;
  margin: 0 auto !important;
  border-radius: 4px !important;
  box-shadow: 0 8px 32px rgba(0,0,0,0.6), 4px 4px 0 rgba(0,0,0,0.3) !important;
  transition: var(--transition) !important;
  position: relative; z-index: 1;
}
.book-card:hover .book-img-container img {
  transform: scale(1.05) rotate(-1deg) !important;
  box-shadow: 0 16px 48px rgba(0,0,0,0.7), 4px 4px 0 rgba(0,0,0,0.3) !important;
}
.book-img-overlay {
  position: absolute;
  inset: 0;
  background: linear-gradient(to bottom, transparent 50%, rgba(13,27,42,0.8) 100%);
  opacity: 0;
  transition: var(--transition);
  z-index: 2;
}

.book-card-body {
  padding: 16px 16px 20px;
  background: linear-gradient(180deg, var(--navy-card) 0%, rgba(13,27,42,0.95) 100%);
  border-top: 1px solid rgba(201,168,76,0.12);
  position: relative;
}
.book-card-body::before {
  content: '';
  position: absolute;
  top: 0; left: 16px; right: 16px;
  height: 2px;
  background: linear-gradient(90deg, transparent, rgba(201,168,76,0.4), transparent);
}
.bk-title {
  font-family: 'Playfair Display', Georgia, serif;
  font-size: 14.5px;
  font-weight: 700;
  color: #F0EAD6;
  line-height: 1.4;
  margin-bottom: 8px;
  display: -webkit-box;
  -webkit-line-clamp: 2;
  -webkit-box-orient: vertical;
  overflow: hidden;
  text-overflow: ellipsis;
  min-height: 40px;
}
.bk-author {
  font-size: 12px;
  color: var(--text-muted);
  font-style: italic;
  margin-bottom: 10px;
  white-space: nowrap;
  overflow: hidden;
  text-overflow: ellipsis;
}
.bk-footer {
  display: flex;
  align-items: center;
  justify-content: space-between;
  margin-top: 10px;
}
.bk-year {
  display: inline-flex;
  align-items: center;
  gap: 4px;
  background: rgba(201,168,76,0.1);
  border: 1px solid rgba(201,168,76,0.2);
  border-radius: 99px;
  padding: 3px 10px;
  font-size: 10.5px;
  font-weight: 600;
  color: var(--gold);
  letter-spacing: 0.8px;
}
.bk-stars {
  display: flex;
  gap: 2px;
  font-size: 10px;
  color: var(--gold);
}

/* ── EXPANDER ────────────────────────────────────────── */
[data-testid="stExpander"] {
  background: var(--navy-card) !important;
  border: 1px solid var(--border) !important;
  border-radius: var(--radius-lg) !important;
  overflow: hidden !important;
  box-shadow: var(--shadow-card) !important;
  margin-bottom: 20px !important;
}
[data-testid="stExpander"] details { border: none !important; }
[data-testid="stExpander"] summary {
  background: linear-gradient(135deg, var(--navy-card), var(--navy-hover)) !important;
  color: var(--gold-light) !important;
  padding: 16px 20px !important;
  font-family: 'Inter', sans-serif !important;
  font-weight: 600 !important;
  font-size: 14px !important;
  border: none !important;
  cursor: pointer !important;
  transition: var(--transition) !important;
}
[data-testid="stExpander"] summary:hover {
  background: linear-gradient(135deg, var(--navy-hover), #243550) !important;
}
[data-testid="stExpander"] summary svg { color: var(--gold) !important; fill: var(--gold) !important; }
[data-testid="stExpander"] > div > div > div { padding: 16px !important; }

/* ── DATAFRAME ───────────────────────────────────────── */
[data-testid="stDataFrame"] {
  border-radius: var(--radius-md) !important;
  overflow: hidden !important;
  border: 1px solid var(--border) !important;
}
[data-testid="stDataFrame"] table { background: transparent !important; }
[data-testid="stDataFrame"] th {
  background: rgba(201,168,76,0.1) !important;
  color: var(--gold-light) !important;
  font-weight: 600 !important;
  font-size: 12px !important;
  letter-spacing: 0.5px !important;
  border-bottom: 1px solid var(--border) !important;
}
[data-testid="stDataFrame"] td {
  color: var(--text-main) !important;
  font-size: 13px !important;
  border-bottom: 1px solid rgba(255,255,255,0.04) !important;
}
[data-testid="stDataFrame"] tr:hover td {
  background: rgba(201,168,76,0.05) !important;
}

/* ── CAPTIONS & WARNINGS ────────────────────────────── */
[data-testid="stCaptionContainer"] p {
  color: var(--text-dim) !important;
  font-size: 12px !important;
  font-style: italic !important;
}
[data-testid="stAlert"] {
  background: rgba(230,57,70,0.1) !important;
  border: 1px solid rgba(230,57,70,0.3) !important;
  border-radius: var(--radius-md) !important;
  color: #ff8a94 !important;
}
[data-testid="stAlert"] p { color: #ff8a94 !important; }

/* Info box */
[data-testid="stAlert"][data-baseweb="notification"] {
  background: rgba(46,196,182,0.08) !important;
  border-color: rgba(46,196,182,0.3) !important;
}

/* ── DIVIDER ─────────────────────────────────────────── */
hr {
  border: none !important;
  height: 1px !important;
  background: linear-gradient(90deg, transparent, var(--border), transparent) !important;
  margin: 32px 0 !important;
  opacity: 1 !important;
}

/* ── THANK YOU SECTION ──────────────────────────────── */
.thankyou-wrap {
  position: relative;
  text-align: center;
  padding: 40px 20px;
  margin-top: 40px;
}
.thankyou-glow {
  position: absolute;
  inset: 0;
  background: radial-gradient(ellipse at center, rgba(201,168,76,0.08) 0%, transparent 65%);
  pointer-events: none;
  border-radius: 20px;
}

/* ── COLUMNS SPACING ─────────────────────────────────── */
[data-testid="stColumns"] { gap: 16px !important; }
[data-testid="column"] { padding: 0 !important; }

/* ── ANIMATIONS ──────────────────────────────────────── */
@keyframes fadeSlideDown {
  from { opacity: 0; transform: translateY(-20px); }
  to   { opacity: 1; transform: translateY(0); }
}
@keyframes fadeSlideUp {
  from { opacity: 0; transform: translateY(24px); }
  to   { opacity: 1; transform: translateY(0); }
}
@keyframes shimmer {
  0%   { background-position: 0% 50%; }
  50%  { background-position: 100% 50%; }
  100% { background-position: 0% 50%; }
}
@keyframes cardShine {
  from { left: -100%; }
  to   { left: 150%; }
}
@keyframes pulse {
  0%, 100% { box-shadow: 0 0 0 0 rgba(201,168,76,0.4); }
  50%       { box-shadow: 0 0 0 8px rgba(201,168,76,0); }
}
@keyframes float {
  0%, 100% { transform: translateY(0px); }
  50%       { transform: translateY(-6px); }
}
@keyframes glow-in {
  from { opacity: 0; filter: blur(10px); transform: scale(0.95); }
  to   { opacity: 1; filter: blur(0); transform: scale(1); }
}

/* Card entrance animation */
.card-anim {
  animation: fadeSlideUp 0.5s ease both;
}

/* Floating book icon */
.float-icon { animation: float 3s ease-in-out infinite; display: inline-block; }

/* ── RESPONSIVE ──────────────────────────────────────── */
@media (max-width: 768px) {
  .hero-title { font-size: 36px !important; }
  .hero-stats { gap: 20px; }
  .stTabs [data-baseweb="tab"] { padding: 10px 16px !important; font-size: 13px !important; }
}
</style>
""", unsafe_allow_html=True)

# -------------------------------------------------------------------------
# DATA LOADING
# -------------------------------------------------------------------------
@st.cache_data
def load_and_prepare_data():
    final_filtered_df_path = hf_hub_download(
        repo_id="IamPradeep/BRS_DATA", filename="final_filtered_df.csv", repo_type="dataset"
    )
    final_filtered_df = pd.read_csv(final_filtered_df_path)

    book_urls_df_path = hf_hub_download(
        repo_id="IamPradeep/BRS_DATA", filename="Books.csv", repo_type="dataset"
    )
    book_urls_df = pd.read_csv(book_urls_df_path)
    book_urls_df.rename(columns={'Book-Title': 'title'}, inplace=True)
    book_urls_df = book_urls_df.drop_duplicates(subset=['title'], keep='first')

    final_filtered_df = final_filtered_df.merge(
        book_urls_df[['title', 'Book-Author', 'Year-Of-Publication', 'Image-URL-L']],
        on='title', how='left'
    )

    # Fix specific image URLs
    fixes = {
        'Jacob Have I Loved':          'http://images.amazon.com/images/P/0690040784.01.LZZZZZZZ.jpg',
        'Needful Things':              'http://images.amazon.com/images/P/0451172817.01.LZZZZZZZ.jpg',
        'All Creatures Great and Small':'http://images.amazon.com/images/P/0312084986.01.LZZZZZZZ.jpg',
        "The Kitchen God's Wife":      'http://images.amazon.com/images/P/1590400356.01.LZZZZZZZ.jpg',
    }
    for title, url in fixes.items():
        final_filtered_df.loc[final_filtered_df['title'] == title, 'Image-URL-L'] = url

    # Build similarity matrix (explicit ratings only)
    explicit_df = final_filtered_df[final_filtered_df['rating'] > 0]
    book_user_mat = explicit_df.pivot_table(index='title', columns='userId', values='rating').fillna(0)
    cosine_sim = cosine_similarity(book_user_mat)
    cosine_sim_df = pd.DataFrame(cosine_sim, index=book_user_mat.index, columns=book_user_mat.index)

    return final_filtered_df, cosine_sim_df

with st.spinner("✨ Loading BookVerse..."):
    final_filtered_df, cosine_sim_df = load_and_prepare_data()

# Stats for hero
total_books  = final_filtered_df['title'].nunique()
total_users  = final_filtered_df['userId'].nunique()
total_ratings = len(final_filtered_df[final_filtered_df['rating'] > 0])

# -------------------------------------------------------------------------
# HELPER FUNCTIONS
# -------------------------------------------------------------------------
def get_top_similar_books(book_title, n=10):
    if book_title not in cosine_sim_df.index:
        return None
    similar_scores = cosine_sim_df[book_title]
    return similar_scores.sort_values(ascending=False)[1:n+1]

def get_user_recommendations(user_id, df, sim_matrix, k=10):
    user_history_all   = df[df['userId'] == user_id]['title'].unique().tolist()
    user_history_rated = df[df['userId'] == user_id][['title', 'rating']]\
                           .sort_values(by='rating', ascending=False)\
                           .drop_duplicates(subset=['title'])
    if not user_history_all:
        return None, None

    scores = {}
    for item in user_history_all:
        if item in sim_matrix.index:
            for sim_item, score in sim_matrix[item].sort_values(ascending=False)[1:50].items():
                if sim_item not in user_history_all:
                    scores[sim_item] = scores.get(sim_item, 0) + score

    sorted_scores      = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    top_recommendations = [x[0] for x in sorted_scores[:k]]
    return top_recommendations, user_history_rated

def get_rank_class(rank):
    return {1: "rank-1", 2: "rank-2", 3: "rank-3"}.get(rank, "rank-other")

def render_stars():
    import random
    count = random.randint(3, 5)
    return "★" * count + "☆" * (5 - count)

def display_book_cards(books_list):
    """Render premium book cards in a responsive 3-column grid."""
    for row_start in range(0, len(books_list), 3):
        row_books = books_list[row_start: row_start + 3]
        cols = st.columns(3, gap="medium")

        for col_idx, col in enumerate(cols):
            global_idx = row_start + col_idx
            if global_idx >= len(books_list):
                break

            book = books_list[global_idx]
            rank = global_idx + 1

            try:
                book_info = final_filtered_df[final_filtered_df['title'] == book].iloc[0]
            except IndexError:
                continue

            img_url   = str(book_info.get('Image-URL-L', ''))
            author    = str(book_info.get('Book-Author', 'Unknown'))
            year      = str(book_info.get('Year-Of-Publication', ''))
            safe_title  = book.replace('"', '&quot;').replace("'", "&#39;")
            safe_author = author.replace('"', '&quot;').replace("'", "&#39;")
            rank_cls  = get_rank_class(rank)
            delay     = (col_idx * 0.1)

            with col:
                st.markdown(f"""
                <div class="book-card-wrap" style="animation: fadeSlideUp 0.5s {delay}s ease both; opacity:0; animation-fill-mode:forwards;">
                  <div class="badge-rank {rank_cls}">{rank}</div>
                  <div class="book-card">
                    <div class="book-card-shine"></div>
                    <div class="book-img-container">
                      <img src="{img_url}"
                           alt="{safe_title}"
                           onerror="this.src='https://via.placeholder.com/140x200/1a2a3a/C9A84C?text=No+Cover'"/>
                      <div class="book-img-overlay"></div>
                    </div>
                    <div class="book-card-body">
                      <div class="bk-title" title="{safe_title}">{book}</div>
                      <div class="bk-author" title="{safe_author}">✍ {author}</div>
                      <div class="bk-footer">
                        <span class="bk-year">📅 {year if year and year != 'nan' else '—'}</span>
                        <span class="bk-stars">{'★★★★☆'}</span>
                      </div>
                    </div>
                  </div>
                </div>
                """, unsafe_allow_html=True)

        # Row separator (not after last row)
        if row_start + 3 < len(books_list):
            st.markdown("""
            <div class="row-sep">
              <div class="row-sep-line"></div>
              <div class="row-sep-dot"></div>
              <div class="row-sep-dot" style="opacity:0.5;width:4px;height:4px;"></div>
              <div class="row-sep-dot" style="opacity:0.25;width:3px;height:3px;"></div>
              <div class="row-sep-line"></div>
            </div>
            """, unsafe_allow_html=True)

# -------------------------------------------------------------------------
# HERO SECTION
# -------------------------------------------------------------------------
st.markdown(f"""
<div class="hero-wrapper">
  <div class="hero-badge"><span>✦</span> AI-Powered Discovery <span>✦</span></div>
  <h1 class="hero-title">BookVerse</h1>
  <p class="hero-subtitle">
    Discover your next great read through intelligent recommendations<br>
    powered by collaborative filtering & cosine similarity.
  </p>
  <div class="hero-divider"></div>
  <div class="hero-stats">
    <div class="stat-item">
      <span class="stat-number">{total_books:,}</span>
      <div class="stat-label">Books</div>
    </div>
    <div class="stat-sep"></div>
    <div class="stat-item">
      <span class="stat-number">{total_users:,}</span>
      <div class="stat-label">Readers</div>
    </div>
    <div class="stat-sep"></div>
    <div class="stat-item">
      <span class="stat-number">{total_ratings:,}</span>
      <div class="stat-label">Ratings</div>
    </div>
  </div>
</div>
""", unsafe_allow_html=True)

# Hero banner image
st.markdown('<div class="hero-image-container">', unsafe_allow_html=True)
st.image(
    'https://img.freepik.com/premium-vector/bookcase-with-books_182089-197.jpg',
    use_container_width=True
)
st.markdown('</div>', unsafe_allow_html=True)

# -------------------------------------------------------------------------
# SESSION STATE INIT
# -------------------------------------------------------------------------
for key in ['recommendations', 'recommended_book', 'recommended_num',
            'user_recommendations', 'user_history_display', 'current_user_id']:
    if key not in st.session_state:
        st.session_state[key] = None

# -------------------------------------------------------------------------
# TABS
# -------------------------------------------------------------------------
tab1, tab2 = st.tabs(["📚  Book-to-Book Recommendations", "👤  Personalized Recommendations"])

# ═════════════════════════════════════════════════════════════════
# TAB 1 — BOOK-TO-BOOK
# ═════════════════════════════════════════════════════════════════
with tab1:
    st.markdown("""
    <div class="section-header">
      <div class="section-title">Find Your Next Favourite</div>
      <div class="section-desc">
        Select any book and our AI engine will surface the most similar titles
        based on the collective taste of thousands of readers.
      </div>
    </div>
    """, unsafe_allow_html=True)

    # Input row
    c1, c2 = st.columns([3, 1], gap="medium")
    with c1:
        all_books  = sorted(final_filtered_df['title'].unique().tolist())
        book_title = st.selectbox(
            "Select a book",
            all_books, index=None,
            placeholder="🔍  Search or choose a book title…",
            key='book_title'
        )
    with c2:
        num_recommendations = st.number_input(
            "# Recommendations",
            min_value=1, max_value=50, value=10, key='num_recs_book'
        )

    st.markdown('<div class="btn-center">', unsafe_allow_html=True)
    clicked_book = st.button("✨  Discover Similar Books", key='btn_book_recs')
    st.markdown('</div>', unsafe_allow_html=True)

    if clicked_book:
        if book_title:
            result = get_top_similar_books(book_title, num_recommendations)
            if result is None:
                st.warning("⚠️ Book not found in the similarity index.")
            else:
                st.session_state.recommendations    = result
                st.session_state.recommended_book   = book_title
                st.session_state.recommended_num    = num_recommendations
        else:
            st.warning("⚠️ Please select a book title first.")

    # Results
    if st.session_state.recommendations is not None:
        rec_book = st.session_state.recommended_book
        rec_num  = st.session_state.recommended_num
        similar  = st.session_state.recommendations

        st.markdown(f"""
        <div class="rec-label">
          <span class="rec-label-icon">🎯</span>
          <span>Showing <strong>{rec_num} books</strong> similar to
          <strong>"{rec_book}"</strong> — ranked by collaborative similarity score.</span>
        </div>
        """, unsafe_allow_html=True)

        display_book_cards(similar.index.tolist())

        # Thank-you
        st.markdown('<div class="thankyou-wrap"><div class="thankyou-glow"></div>', unsafe_allow_html=True)
        st.image(
            'https://github.com/MarpakaPradeepSai/Employee-Churn-Prediction/blob/main/Data/Images%20&%20GIFs/thank-you-33.gif?raw=true',
            use_container_width=True
        )
        st.markdown('</div>', unsafe_allow_html=True)

# ═════════════════════════════════════════════════════════════════
# TAB 2 — USER-SPECIFIC
# ═════════════════════════════════════════════════════════════════
with tab2:
    st.markdown("""
    <div class="section-header">
      <div class="section-title">Personalised Just for You</div>
      <div class="section-desc">
        Enter a Reader ID to receive hand-crafted recommendations based on
        their unique reading history and rating patterns.
      </div>
    </div>
    """, unsafe_allow_html=True)

    all_user_ids = sorted(final_filtered_df['userId'].unique().tolist())

    cu1, cu2 = st.columns([3, 1], gap="medium")
    with cu1:
        user_id_input = st.selectbox(
            "Select a Reader ID",
            all_user_ids, index=None,
            placeholder="🔍  Choose a Reader ID…",
            key='user_id_select'
        )
    with cu2:
        num_user_recs = st.number_input(
            "# Recommendations",
            min_value=1, max_value=50, value=10, key='num_recs_user'
        )

    st.markdown('<div class="btn-center">', unsafe_allow_html=True)
    clicked_user = st.button("🚀  Get My Recommendations", key='btn_user_recs')
    st.markdown('</div>', unsafe_allow_html=True)

    if clicked_user:
        if user_id_input:
            recs, history = get_user_recommendations(
                user_id_input, final_filtered_df, cosine_sim_df, k=num_user_recs
            )
            if recs is None:
                st.warning(f"⚠️ Reader ID **{user_id_input}** has no interaction history.")
                st.session_state.user_recommendations  = None
                st.session_state.user_history_display  = None
            else:
                st.session_state.user_recommendations  = recs
                st.session_state.user_history_display  = history
                st.session_state.current_user_id       = user_id_input
        else:
            st.warning("⚠️ Please select a Reader ID first.")

    # Results
    if st.session_state.user_recommendations is not None:
        uid      = st.session_state.current_user_id
        recs     = st.session_state.user_recommendations
        history  = st.session_state.user_history_display

        # Reading history expander
        if history is not None and len(history) > 0:
            with st.expander(f"📖  View Reading History — {len(history)} book(s) on record"):
                h_df = history.copy().reset_index(drop=True)
                h_df.index  = h_df.index + 1
                h_df.columns = ['Book Title', 'Rating']
                st.dataframe(h_df, use_container_width=True, height=min(400, (len(h_df)+1)*35+40))
                st.caption("ℹ️ *A rating of 0 means the book was interacted with but not explicitly rated.*")

        st.markdown(f"""
        <div class="rec-label">
          <span class="rec-label-icon">🌟</span>
          <span>Showing <strong>{len(recs)} personalised picks</strong> for
          Reader <strong>#{uid}</strong> — curated from their reading DNA.</span>
        </div>
        """, unsafe_allow_html=True)

        if recs:
            display_book_cards(recs)

            st.markdown('<div class="thankyou-wrap"><div class="thankyou-glow"></div>', unsafe_allow_html=True)
            st.image(
                'https://github.com/MarpakaPradeepSai/Employee-Churn-Prediction/blob/main/Data/Images%20&%20GIFs/thank-you-33.gif?raw=true',
                use_container_width=True
            )
            st.markdown('</div>', unsafe_allow_html=True)
        else:
            st.info("No recommendations available for this reader at the moment.")
