# 📧 Spam Classifier App (Streamlit + ML)

A simple machine learning web application to classify email messages as **spam** or **not spam (ham)** using NLP techniques and Logistic Regression.

##  Features

- Preprocesses and cleans message text
- Converts text to features using TF-IDF
- Trained with Logistic Regression
- Deployed as an interactive web app using **Streamlit**

##  Model Performance

- Accuracy: ~97%
- Evaluated using confusion matrix & classification report

## 🧰 Tech Stack

- Python
- Scikit-learn
- Pandas, NumPy
- Streamlit
- NLTK

## 📦 Installation

```bash
git clone https://github.com/AkesTechSE/spam-classifier-streamlit.git
cd spam-classifier-streamlit
pip install -r requirements.txt
streamlit run spam_detection.py
