import streamlit as st
import pandas as pd
import numpy as np
import re
import joblib
import nltk
import matplotlib.pyplot as plt

from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

from sklearn.metrics import confusion_matrix, roc_curve, auc
from sklearn.calibration import CalibratedClassifierCV

from docx import Document
from PyPDF2 import PdfReader

# --------------------------------------------------
# PAGE CONFIG
# --------------------------------------------------
st.set_page_config(
    page_title="AI Powered Fake News Detection",
    layout="wide"
)

# --------------------------------------------------
# NLTK
# --------------------------------------------------
nltk.download("stopwords")
nltk.download("wordnet")

stop_words = set(stopwords.words("english"))
lemmatizer = WordNetLemmatizer()

# --------------------------------------------------
# STYLES (NEWSPAPER THEME)
# --------------------------------------------------
st.markdown("""
<style>
body {
    background-color: #f4f1ec;
}
.main-title {
    font-family: Georgia, serif;
    font-size: 40px;
    font-weight: bold;
    text-align: center;
}
.section {
    background-color: #ffffff;
    padding: 20px;
    border-radius: 10px;
}
</style>
""", unsafe_allow_html=True)

st.markdown('<div class="main-title">📰 AI Powered Fake News Detection</div>', unsafe_allow_html=True)
st.write("")

# --------------------------------------------------
# SIDEBAR MENU
# --------------------------------------------------
menu = st.sidebar.radio(
    "📂 Menu",
    ["Analyze News", "Upload Document", "About Developer", "Feedback"]
)

algorithm = st.sidebar.selectbox(
    "⚙️ Algorithm",
    ["Linear SVM", "Calibrated SVM"]
)

# --------------------------------------------------
# LOAD MODEL
# --------------------------------------------------
@st.cache_resource
def load_model():
    model = joblib.load("svm_fake_news_model.pkl")
    vectorizer = joblib.load("svm_tfidf_vectorizer.pkl")
    return model, vectorizer

model, tfidf = load_model()

# --------------------------------------------------
# TEXT CLEANING
# --------------------------------------------------
def clean_text(text):
    text = text.lower()
    text = re.sub(r"[^a-z\s]", "", text)
    words = text.split()
    words = [
        lemmatizer.lemmatize(w)
        for w in words
        if w not in stop_words and len(w) > 2
    ]
    return " ".join(words)

# --------------------------------------------------
# ANALYZE NEWS
# --------------------------------------------------
if menu == "Analyze News":
    st.markdown('<div class="section">', unsafe_allow_html=True)

    news = st.text_area("📰 Enter News Text", height=200)

    if st.button("Analyze"):
        if news.strip() == "":
            st.warning("Please enter some text")
        else:
            clean = clean_text(news)
            vec = tfidf.transform([clean])

            pred = model.predict(vec)[0]
            prob = model.predict_proba(vec)[0][1]

            if pred == 1:
                st.success(f"✅ REAL NEWS ({prob*100:.2f}%)")
            else:
                st.error(f"❌ FAKE NEWS ({(1-prob)*100:.2f}%)")

    st.markdown('</div>', unsafe_allow_html=True)

# --------------------------------------------------
# DOCUMENT UPLOAD
# --------------------------------------------------
if menu == "Upload Document":
    st.markdown('<div class="section">', unsafe_allow_html=True)

    file = st.file_uploader("Upload PDF or DOCX", type=["pdf", "docx"])

    text = ""

    if file:
        if file.type == "application/pdf":
            reader = PdfReader(file)
            for page in reader.pages:
                text += page.extract_text()

        if file.type == "application/vnd.openxmlformats-officedocument.wordprocessingml.document":
            doc = Document(file)
            for p in doc.paragraphs:
                text += p.text

        clean = clean_text(text)
        vec = tfidf.transform([clean])
        pred = model.predict(vec)[0]
        prob = model.predict_proba(vec)[0][1]

        if pred == 1:
            st.success(f"✅ REAL NEWS ({prob*100:.2f}%)")
        else:
            st.error(f"❌ FAKE NEWS ({(1-prob)*100:.2f}%)")

    st.markdown('</div>', unsafe_allow_html=True)

# --------------------------------------------------
# ABOUT DEVELOPER
# --------------------------------------------------
if menu == "About Developer":
    st.markdown('<div class="section">', unsafe_allow_html=True)

    st.subheader("👨‍💻 Developer Details")
    st.markdown("""
**Name:** K Mohanraj  
**Email:** mohanrajk137@gmail.com  

**Project:** AI Powered Fake News Detection  
**Algorithms Used:**  
- TF-IDF Vectorizer  
- Linear SVM  
- CalibratedClassifierCV  

**Framework:** Streamlit
""")

    st.markdown('</div>', unsafe_allow_html=True)

# --------------------------------------------------
# FEEDBACK
# --------------------------------------------------
if menu == "Feedback":
    st.markdown('<div class="section">', unsafe_allow_html=True)

    name = st.text_input("Name")
    email = st.text_input("Email")
    message = st.text_area("Feedback")

    if st.button("Submit"):
        if name and email and message:
            df = pd.DataFrame([[name, email, message]],
                              columns=["Name", "Email", "Message"])
            try:
                old = pd.read_csv("feedback.csv")
                df = pd.concat([old, df], ignore_index=True)
            except:
                pass

            df.to_csv("feedback.csv", index=False)
            st.success("Thank you for your feedback 🙏")
        else:
            st.warning("Please fill all fields")

    st.markdown('</div>', unsafe_allow_html=True)
