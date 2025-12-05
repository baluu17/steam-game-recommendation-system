# Steam Game Recommendation System 🎮

This project is a **content-based recommendation system** for Steam games, built using **Python, scikit-learn (TF-IDF + cosine similarity), and Streamlit**.

## 🔍 Features

- Search for similar games using **AppID**
- Filter by **genre**, **maximum price**, and **minimum positive ratings**
- Combined filters (e.g. “Action games under $10 with at least 500 positive ratings”)
- Built-in **dataset explorer** to browse games, AppIDs and genres

## 🧠 Tech stack

- Python
- pandas, numpy
- scikit-learn (TfidfVectorizer, cosine similarity)
- Streamlit

## 🚀 How to run locally

```bash
git clone https://github.com/YOUR_USERNAME/steam-game-recommendation-system.git
cd steam-game-recommendation-system

pip install -r requirements.txt

python -m streamlit run app.py
