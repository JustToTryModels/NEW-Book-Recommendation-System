<div align="center"> 
  
# 📚 Book Recommendation System 
  
[![Python](https://img.shields.io/badge/Python-3.10%2B-blue?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-App-FF4B4B?style=for-the-badge&logo=Streamlit&logoColor=white)](https://new-book-recommendation-system.streamlit.app/)
[![Scikit-Learn](https://img.shields.io/badge/scikit--learn-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white)](https://scikit-learn.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg?style=for-the-badge)](LICENSE)

</div>
  
A machine learning project designed to predict user preferences and recommend books using **Item-Based Collaborative Filtering**. By analyzing reading patterns, user ratings, and item similarities, this system helps readers discover their next favorite book while providing insights into global reading trends, popular authors, and demographic distributions.

<br>

---

## 📋 Table of Contents

- [What is a Recommendation System?](#-what-is-a-recommendation-system)
- [Project Overview](#-project-overview)
- [Live Demo](#-live-demo)
- [Project Structure](#-project-structure)
- [Dataset Overview](#-dataset-overview)
- [Methodology](#-methodology)
- [Model Comparison & Selection](#-model-comparison--selection)
- [Recommendation Example](#-recommendation-example)
- [Key Exploratory Findings](#-key-exploratory-findings)
- [Strategic Business Insights](#-strategic-business-insights)
- [Installation & Usage](#-installation--usage)

<br>

---

## ❓ What is a Recommendation System?

<div align="center">
  <img src="https://miro.medium.com/v2/resize:fit:828/format:webp/1*3ALliiz9hG79_2xopzgyrQ.png" alt="Collaborative Filtering Concept" width="600"/>
</div>

A recommendation system is an information filtering system that predicts the "rating" or "preference" a user would give to an item. This project focuses on **Collaborative Filtering**, specifically the **Item-Based** approach:
* **Core Idea:** "Users who liked this item also liked these other items."
* **Advantage:** It requires no metadata about the books themselves (like genre or page count), relying purely on the collective behavior and implicit/explicit feedback of the user base.

<br>

---

## 🎯 Project Overview

### Objective

To build a robust recommendation engine capable of suggesting top 10 highly relevant books based on a user's specific book interest, utilizing sparse matrix computations and distance metrics.

<div align="center">

### 🛣️ Approach

| Component | Description |
|-----------|-------------|
| **Evaluation Scope** | Item-Based Collaborative Filtering (Cosine, kNN, K-Means) |
| **Selected Model** | k-Nearest Neighbors (kNN) & Cosine Similarity |
| **Sparsity Handling** | Filtered active users (≥100 ratings) & popular books (≥50 ratings) |
| **Distance Metric** | Cosine Similarity (Brute-force algorithm) |
| **Evaluation Metrics** | RMSE, MAE, Precision, Recall, F1-Score |
| **Deployment** | Streamlit Web Application |

</div>

<br>

---

## 🚀 Live Demo

Try the live recommendation model here:

[![Open Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://new-book-recommendation-system.streamlit.app/) 

> **How it works:** Simply select or type the name of a book you enjoyed, and the system will instantly query the sparse matrix to return the top 10 most mathematically similar books based on global reader behavior!

<br>

---

## 📁 Project Structure

```text
book-recommendation-system/
├── Data/
│   ├── Books.csv                           # Book metadata (ISBN, Title, Author)
│   ├── Users.csv                           # User demographics (ID, Location, Age)
│   └── Ratings.csv                         # User-Book interaction scores
├── Notebook/
│   └── Book_Recommendation_System.ipynb    # EDA, Cleaning, and Model Training
├── app.py                                  # Streamlit web application
├── requirements.txt                        # Python dependencies
├── README.md                               # Project documentation
└── models/                                 # Exported pickle files (pivot tables, similarity matrices)
```

<br>

---

## 📊 Dataset Overview

The project utilizes a comprehensive dataset containing over 1.1 million ratings.

<div align="center">

### Raw Data Dimensions

| Dataset | Rows | Key Features |
|--------|-------|--------------|
| **Books** | 271,360 | `ISBN`, `Book-Title`, `Book-Author`, `Year-Of-Publication`, `Publisher` |
| **Users** | 278,858 | `User-ID`, `Location`, `Age` |
| **Ratings** | 1,149,780 | `User-ID`, `ISBN`, `Book-Rating` (0-10) |

</div>

<br>

<div align="center">

### Data Preparation & Sparsity Reduction

To prevent the **"Curse of Dimensionality"** and ensure meaningful recommendations, the data was rigorously filtered:

| Filter Step | Criteria | Result / Rationale |
|---------|-------------|--------------------|
| **Active Users** | Users with **≥ 100 ratings** | Ensures sufficient user history to establish patterns. |
| **Popular Books** | Books with **≥ 50 ratings** | Removes obscure books to prevent noisy, inaccurate nearest neighbors. |
| **Final Matrix** | Pivot Table | Resulted in a dense, highly optimized User-Book interaction matrix. |

</div>

<br>

---

## 🔬 Methodology

### 📊 1. Data Cleaning & Preprocessing
- **Type Casting:** Converted `Year-Of-Publication` to numeric, handling invalid string entries.
- **Outlier Removal:** Identified and removed impossible `Age` values (e.g., users claiming to be >100 years old) using IQR bounds.
- **Merging:** Joined Users, Books, and Ratings on `ISBN` and `User-ID` to create a unified working dataframe.

### 📐 2. Building the Interaction Matrix
- Created a pivot table where **Rows = Book Titles**, **Columns = User IDs**, and **Values = Ratings**.
- Missing interactions were filled with `0.0` to create a workable sparse matrix.

### ⚙️ 3. Algorithm Implementation
- **Cosine Similarity Matrix:** Computed the pairwise cosine similarity between all books.
- **k-Nearest Neighbors (kNN):** Fitted a `NearestNeighbors` model using the `brute` algorithm and `cosine` metric to find the closest vectors in multi-dimensional space.
- **K-Means Clustering:** Experimented with unsupervised clustering (Optimal $k=2$ via Silhouette Score) to group similar items.

<br>

---

## ⚔️ Model Comparison & Selection

Three primary approaches were tested for the Item-Based Collaborative Filtering system. 

<div align="center">

| Model Approach | MAE | RMSE | Notes |
|:---|:---:|:---:|:---|
| **Cosine Similarity / kNN** 🏆 | **2.145** | **3.958** | **Best performance. Directly calculates angular distance between item vectors.** |
| K-Means Clustering | N/A | 5.915 | Grouped books into broad clusters; less precise for 1-to-1 top-N recommendations. |

</div>

<br>

### 🌳 Why kNN & Cosine Similarity?

**kNN with Cosine Metric** was selected for deployment based on the following:
1. **Handles Sparsity:** Cosine similarity measures the angle between vectors, ignoring the magnitude. This is perfect for sparse matrices where `0` means "unrated" rather than "disliked".
2. **Lower Error Rate:** Achieved an RMSE of 3.958, significantly outperforming K-Means (5.915).
3. **High Recall:** The system achieved a **Recall of 0.6701**, meaning it successfully retrieves a large portion of relevant books a user would actually engage with.

<br>

---

## 🔍 Recommendation Example

To validate the model, we can input a highly recognizable book and observe the semantic relevance of the outputs.

**Input:** 🪄 *Harry Potter and the Sorcerer's Stone (Book 1)*

<div align="center">

| Rank | Recommended Book | Relevance Indication |
|:---:|------------------|----------------------|
| **1** | *Harry Potter and the Chamber of Secrets (Book 2)* | Direct Sequel (Perfect match) |
| **2** | *Harry Potter and the Prisoner of Azkaban (Book 3)* | Series Continuation |
| **3** | *Harry Potter and the Goblet of Fire (Book 4)* | Series Continuation |
| **4** | *Harry Potter and the Order of the Phoenix (Book 5)* | Series Continuation |
| **5** | *The Two Towers (The Lord of the Rings, Part 2)* | High Fantasy Genre Match |
| **6** | *The Bonesetter's Daughter* | Fiction / Popular Literature |
| **7** | *A Knight in Shining Armor* | Fantasy / Romance |
| **8** | *Charlie and the Chocolate Factory* | Classic Children's Fantasy |
| **9** | *The Mists of Avalon* | Arthurian Fantasy |

</div>

> **Conclusion:** The model successfully captures **series continuity** and **genre affinity** (Fantasy/Magic) without ever being explicitly programmed with genre tags!

<br>

---

## 💡 Key Exploratory Findings

Extensive EDA revealed fascinating trends about reading habits and the dataset structure:

### 1. The "Implicit Rating" Skew
* **647,294 interactions** are marked as `0` (Implicit feedback - the user interacted with the book but didn't leave a 1-10 rating).
* Among explicit ratings (1-10), **8 is the most common**, showing that users generally only rate books they enjoy.

### 2. The Golden Era of Publishing
* The years **1999–2002** represent a massive peak in unique book releases within this dataset.
* Releases grew by **48%** throughout the 1990s.

### 3. Top Performers (The "Long Tail" Pattern)
<div align="center">

| Category | Top Entity | Metric / Insight |
|----------|------------|------------------|
| **Most Read Book** | *Wild Animus* | 2,502 occurrences (nearly double the 2nd place book, *The Lovely Bones*). |
| **Highest Rated Book** | *The Da Vinci Code* | Received the most perfect "10" ratings (160). |
| **Top Author** | *William Shakespeare* | 495 unique books/editions. Followed closely by *Agatha Christie* (476). |
| **Top Publisher** | *Harlequin* | 7,499 unique books. Mass-market romance publishers dominate the volume. |

</div>

### 4. Demographic Skew
* The user base is heavily North American. **Toronto (13.3%)**, **Seattle (11.7%)**, and **Portland (11.2%)** are the top locations. London (8.7%) is the only non-North American city in the top 10.

<br>

---

## 🎯 Strategic Business Insights

Based on the data and model behavior, here are strategic recommendations for deploying this in a production environment:

1. **Address the "Cold Start" Problem:**
   * Because Collaborative Filtering requires historical data, brand new users should be recommended global blockbusters (*The Da Vinci Code*, *Harry Potter*) or top books from the 1999-2002 "Golden Era" until they rate their first 5 books.
2. **Leverage Implicit Feedback:**
   * Since the vast majority of data is `0` (implicit), the platform should track clicks, time spent on page, and purchase history as proxies for explicit 1-10 ratings.
3. **Promote the "Long Tail":**
   * While *Wild Animus* and *Harry Potter* dominate, the kNN algorithm excels at finding niche connections. The UI should feature a "Hidden Gems" section based on the lower-bound nearest neighbors to diversify reading habits.

<br>

---

## 🛠️ Installation & Usage

### Prerequisites

- Python 3.10 or higher
- `pip` package manager

### Setup Instructions

1. **Clone the repository**
   ```bash
   git clone https://github.com/YourUsername/Book-Recommendation-System.git
   cd Book-Recommendation-System
   ```

2. **Create a virtual environment (Recommended)**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Run the Streamlit App Locally**
   ```bash
   streamlit run app.py
   ```
   *The application will launch in your default web browser at `http://localhost:8501`.*

<br>

---

## 🙏 Thank You

<div align="center">
  
  Thank you for checking out this Book Recommendation System! If you found this project interesting or helpful, please consider giving it a ⭐ on GitHub.
  
  [🌐 Try the Live App Here](https://new-book-recommendation-system.streamlit.app/)
  
</div>
