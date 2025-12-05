# Steam - Game Recommendation System 🎮

This project is a **content-based recommendation system** for Steam games, built using **Python, scikit-learn (TF-IDF + cosine similarity), and Streamlit**.

## 🔍 Features

- Search for similar games using **AppID**
- Filter by **genre**, **maximum price**, and **minimum positive ratings**
- Combined filters (e.g. “Action games under $10 with at least 500 positive ratings”)
- Built-in **dataset explorer** to browse games, AppIDs and genres

## 📂 Dataset Information

This project originally uses the full Steam games dataset, which is **large (100MB+)** and not suitable for hosting directly on GitHub or deploying on Streamlit Cloud.
To ensure smooth performance and easy reproducibility, this repository includes a **5,000-row sample** of the dataset.
### Why a sample dataset?

- GitHub does not allow files larger than 100MB  
- Streamlit Cloud fails or times out with very large datasets  
- TF-IDF vectorisation and similarity calculations become slow  
- A smaller dataset keeps the app fast, portable, and deployable

If you have the full dataset, you can use it by changing:

python
load_and_preprocess_data("sample_steam.csv")
to
load_and_preprocess_data("steam.csv")

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
