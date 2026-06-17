<div align="center"> 
  
# 📚 Book Recommendation System 
  
[![Python](https://img.shields.io/badge/Python-3.10%2B-blue?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-App-FF4B4B?style=for-the-badge&logo=Streamlit&logoColor=white)](https://streamlit.io/)
[![Scikit-Learn](https://img.shields.io/badge/scikit--learn-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white)](https://scikit-learn.org/)
[![Hugging Face](https://img.shields.io/badge/Hugging%20Face-Datasets-F9AB00?style=for-the-badge&logo=huggingface&logoColor=white)](https://huggingface.co/)
[![License](https://img.shields.io/badge/License-MIT-green.svg?style=for-the-badge)](LICENSE)

</div>
  
A machine learning-powered recommendation engine designed to suggest books based on collaborative filtering techniques. By analyzing underlying patterns in user reading histories and explicit ratings, the system identifies similarities between items to provide highly accurate, personalized reading recommendations tailored to specific users.

<br>

---

## 📋 Table of Contents

- [What is Collaborative Filtering?](#-what-is-collaborative-filtering)
- [Project Overview](#-project-overview)
- [Live Demo](#-live-demo)
- [Dataset Overview](#-dataset-overview)
- [Methodology & Data Processing](#-methodology--data-processing)
- [User-Specific Personalized Recommendations](#-user-specific-personalized-recommendations)
- [Model Architecture & Evaluation](#-model-architecture--evaluation)
- [Key EDA Findings](#-key-eda-findings)
- [Installation & Usage](#-installation--usage)

<br>

---

## ❓ What is Collaborative Filtering?

<div align="center">
  <img src="https://github.com/MarpakaPradeepSai/Book-Recommendation-System/blob/main/Data/Images%20&%20GIF/CF2.webp?raw=true" alt="Collaborative Filtering" width="600"/>
</div>

**Collaborative filtering** is a technique used by recommender systems to predict a user's interests by collecting preferences from a broader user base. 

This project specifically utilizes **Item-Based Collaborative Filtering**, which recommends items to a target user based on the similarities between the items themselves. If User A likes *Harry Potter 1*, and the system identifies that *Harry Potter 1* is mathematically similar to *Harry Potter 2* (because users who read the first almost always read the second), it will recommend the latter.

**Advantages:**
- 🎯 **User-Centric:** Adapts to complex, nuanced user behaviors.
- 🚫 **No Item Metadata Needed:** Doesn't rely on tags or genres.
- ✨ **Serendipity:** Helps users discover unexpected but highly relevant books.

<br>

---

## 🎯 Project Overview

### Objective

Build a robust recommendation engine capable of processing millions of user interactions to suggest highly relevant books. The system evaluates different clustering and similarity algorithms to deliver context-aware, user-specific suggestions via a deployed web application.

<div align="center">

### 🛣️ Approach

| Component | Description |
|-----------|-------------|
| **Core Algorithm** | Item-Based Collaborative Filtering |
| **Similarity Metric** | Cosine Similarity |
| **Data Optimization** | Isolating `explicit_ratings_df` to prevent implicit data skew |
| **Sparsity Handling** | Threshold filtering (Active Users > 100 ratings, Popular Books > 50 ratings) |
| **Alternative Models Tested** | k-Nearest Neighbors (kNN), K-Means Clustering |
| **Deployment** | Streamlit Web Application |

</div>

<br>

---

## 🚀 Live Demo

Try the live recommendation engine here:

[![Open Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://book-recommendation-systemm.streamlit.app/) 

<div align="center">
  <table>
    <tr>
      <th>⚡ Find Similar Books</th>
      <th>✨ Personalized Recommendations for Users</th>
    </tr>
    <tr>
      <td><img src="https://github.com/MarpakaPradeepSai/Book-Recommendation-System/blob/main/Data/Images%20&%20GIF/Find_Similar_Books.gif?raw=true" alt="App Interface" height="400"/></td>
      <td><img src="https://github.com/MarpakaPradeepSai/Book-Recommendation-System/blob/main/Data/Images%20&%20GIF/Personalized_Recommendations.gif?raw=true" alt="Live Recommendations" height="400"/></td>
    </tr>
  </table>
</div>

> Simply input a `User ID` or select a book you've enjoyed, and the system will instantly generate a curated list of top recommendations complete with cover art, author details, and publication years!

<br>

---

## 📊 Dataset Overview

The project utilizes a comprehensive Book-Crossing dataset comprising three main components:

<div align="center">

| Component | Description | Volume |
|--------|-------|-------|
| **Users** | Anonymized user IDs and demographic data (Location, Age). | 278,858 records |
| **Books** | ISBN, Title, Author, Year, Publisher, and Cover Image URLs. | 271,360 records |
| **Ratings** | Explicit ratings (1-10) and implicit ratings (0). | 1,149,780 records |

</div>


Find the Datasets here:

[![Hugging Face Dataset](https://img.shields.io/badge/Dataset-HuggingFace-yellow?logo=huggingface)](https://huggingface.co/datasets/IamPradeep/BRS_DATA/tree/main)

---

## 🔬 Methodology & Data Processing

### 🧹 1. Data Cleaning & Engineering
- **Missing Values & Invalid Formats:** Cleaned invalid string entries in the `Year-Of-Publication` column and safely coerced them to numeric. Dropped rows with null `Book-Title` or `Book-Author` entries.
- **Handling Demographics:** Detected and assessed extreme outliers in the `Age` demographic using the IQR method to ensure accurate visualization of the user base.

### ⚖️ 2. Sparsity Reduction (The "Cold Start" Fix)
Recommendation matrices are notoriously sparse. To ensure high-quality recommendations and computational efficiency, the dataset was rigorously filtered:
1. **Active Users Only:** Filtered out users who rated fewer than **100 books**.
2. **Popular Books Only:** Filtered out books that received fewer than **50 ratings**.

### 🏗️ 3. Matrix Construction & `explicit_ratings_df`
A crucial step in the data pipeline was handling "implicit" ratings (interactions denoted by a `0`). 
- Treating a `0` interaction as a poor rating severely corrupts the mathematical calculations of a similarity matrix.
- To solve this, an **`explicit_ratings_df`** was isolated, containing *only* active ratings between `1` and `10`.
- The final User-Item pivot table was generated exclusively from this explicit dataframe, ensuring that Cosine calculations accurately reflect true user enjoyment rather than mere interactions.

<br>

---

## 👤 User-Specific Personalized Recommendations

Beyond finding "similar books", the system is engineered to generate highly personalized recommendations for individual users via a `get_recommendations()` pipeline:

1. **Context Fetching:** The system takes a specific `User-ID` and retrieves their entire interaction history (both explicit ratings and implicit interactions).
2. **Candidate Generation:** It queries the Item-Based similarity matrix to find the nearest neighbors (most similar books) for every book the user has highly rated.
3. **Critical Filtering:** The system actively tracks the user's history and cross-references the candidate list to **filter out books the user has already read/interacted with**. 
4. **Scoring & Ranking:** The remaining unseen candidates are ranked based on aggregated similarity scores, returning the definitive `Top K` customized recommendations.

<br>

---

## ⚔️ Model Architecture & Evaluation

Three distinct approaches were evaluated using a unified framework, predicting ratings and ranking quality across the dataset.

### 🚀 1. Cosine Similarity (Deployed Model)
- **Mechanism:** Calculates the cosine of the angle between two projected vectors (books) in the explicit multi-dimensional user space.
- **Evaluation Metrics (K=10):**
  - **MAE:** 1.0577
  - **RMSE:** 1.6181
  - **Precision@10:** 0.0326
  - **Recall@10:** 0.0868
  - **NDCG@10:** 0.1323
  - **HitRate@10:** 0.2293
  - **Coverage:** 0.7397
- **Why it was deployed:** Offered the highest item coverage and exact determinism matching kNN, but scaled perfectly for real-time Streamlit deployment. 

### 🤖 2. k-Nearest Neighbors (kNN)
- **Mechanism:** Utilized `sklearn.neighbors.NearestNeighbors` with `metric='cosine'` and `algorithm='brute'` on the explicit pivot table.
- **Evaluation Metrics (K=10):**
  - **MAE:** 1.0577
  - **RMSE:** 1.6181
  - **Precision@10:** 0.0326
  - **Recall@10:** 0.0868
  - **NDCG@10:** 0.1323
  - **HitRate@10:** 0.2293
  - **Coverage:** 0.7388
- **Verdict:** Yielded practically identical predictive performance to direct Cosine calculation, but with marginally less catalog coverage.

### 📊 3. K-Means Clustering
- **Mechanism:** Unsupervised grouping of similar books. Used Silhouette Score to find the optimal number of clusters ($k=2$, score = `0.3588`).
- **Evaluation Metrics (K=10):**
  - **MAE:** 1.1627
  - **RMSE:** 1.5868
  - **Precision@10:** 0.0258
  - **Recall@10:** 0.0712
  - **NDCG@10:** 0.1123
  - **HitRate@10:** 0.2003
  - **Coverage:** 0.6017
- **Verdict:** Hard-clustering forces books into strict groups, lowering hit rates and coverage compared to continuous neighborhood scoring.

<br>

---

## 💡 Key EDA Findings

Extensive Exploratory Data Analysis (EDA) revealed fascinating insights about reader behavior:

### 🏆 1. The Most Loved Books

<div align="center">

| Rank | Book Title | Perfect "10" Ratings | Insight |
|:---:|:---|:---:|:---|
| 1 | **The Da Vinci Code** | **160** | The undisputed highest-rated book in the dataset. |
| 2 | Harry Potter & the Sorcerer's Stone | 152 | The *Harry Potter* series dominates the top 10, showing massive fan loyalty. |
| 3 | Harry Potter & the Prisoner of Azkaban | 150 | Consistent high ratings across the franchise. |
| 4 | The Lovely Bones | 148 | Highest-rated standalone fiction (non-franchise). |

</div>

<br>

### 📉 2. Rating Distributions & Engagement

*   **The Implicit Majority:** Over **647,294** interactions were implicit (a rating of 0, meaning the user interacted with the book but didn't leave a score).
*   **Positivity Bias:** Among explicit ratings (1-10), **8 is the most common score**, indicating users are more likely to rate books they actually enjoyed.
*   **The "Wild Animus" Anomaly:** The book *Wild Animus* had the highest total occurrences (2,502) but overwhelmingly dominated the **"1-star"** rating category. High visibility clearly does not guarantee high reader satisfaction!

<br>

### ✍️ 3. Industry Insights (Authors & Publishers)

<div align="center">

| Category | Top Entity | Volume | Observation |
|---|---|---|---|
| **Top Author** | **William Shakespeare** | 495 Books | Classic literature remains highly published. |
| **Top Contemp. Author** | **Agatha Christie** | 476 Books | Closely followed by Stephen King. |
| **Top Publisher** | **Harlequin** | 7,499 Books | Romance dominates mass-market publishing, far exceeding others. |
| **Peak Publishing Era** | **1999 - 2002** | ~17k/year | The "Golden Era" of book releases within this specific dataset timeline. |

</div>

<br>

### 🌍 4. User Demographics
*   **Age:** The dominant peak for readers lies between **19 and 35 years old**. 
*   **Location:** The dataset is heavily skewed toward North America. **Toronto (13.3%)**, Seattle (11.7%), and Portland (11.2%) are the top user locations. London (8.7%) is the only non-North American city in the top 10.

<br>

---

## 🛠️ Installation & Usage

### Prerequisites

- Python 3.10+
- pip package manager

### Local Setup Instructions

1. **Clone the repository**
   ```bash
   git clone https://github.com/MarpakaPradeepSai/Book-Recommendation-System.git
   cd Book-Recommendation-System
   ```

2. **Create and activate a virtual environment (Recommended)**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install required dependencies**
   ```bash
   pip install streamlit pandas numpy scikit-learn huggingface_hub
   ```

4. **Run the Streamlit Application**
   ```bash
   streamlit run app.py
   ```

> **Note on Data Loading:** The application is configured to automatically download the pre-processed data structures directly from Hugging Face (`IamPradeep/BRS_DATA`). You do not need to download the CSV files manually to run the interface!

<br>

---

## 🙏 Thank You

<div align="center">
  <img src="https://github.com/MarpakaPradeepSai/Employee-Churn-Prediction/blob/main/Data/Images%20&%20GIFs/thank-you-33.gif?raw=true" alt="Thank You" width="300">
  
  If you found this project interesting or helpful, please consider giving the repository a ⭐!
</div>
