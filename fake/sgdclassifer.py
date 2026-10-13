import pandas as pd
import re
import nltk

from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# ---------------------------------------------------
# Download required NLTK resources (run once)
# ---------------------------------------------------
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('omw-1.4')

stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

# ---------------------------------------------------
# Load Dataset
# ---------------------------------------------------
df = pd.read_csv("fixed.csv", low_memory=False)

# Remove unwanted index columns
df = df.loc[:, ~df.columns.str.contains("^Unnamed")]

# Keep required columns
df = df[['text', 'label']]

# Fill missing text
df['text'] = df['text'].fillna("")

# ---------------------------------------------------
# Fix label column
# ---------------------------------------------------
df['label'] = df['label'].astype(str).str.lower().str.strip()

label_map = {
    'fake': 0, '0': 0, 'false': 0,
    'real': 1, '1': 1, 'true': 1
}

df['label'] = df['label'].map(label_map)
df = df.dropna(subset=['label'])
df['label'] = df['label'].astype(int)

print("✅ Dataset loaded successfully")
print("Label distribution:")
print(df['label'].value_counts())

# ---------------------------------------------------
# Text Cleaning + Lemmatization
# ---------------------------------------------------
def clean_text(text):
    text = text.lower()
    text = re.sub(r'[^a-z\s]', '', text)
    words = text.split()
    words = [
        lemmatizer.lemmatize(w)
        for w in words
        if w not in stop_words and len(w) > 2
    ]
    return " ".join(words)

df['clean_text'] = df['text'].apply(clean_text)
print("✅ Text preprocessing completed")

# ---------------------------------------------------
# Features and Labels
# ---------------------------------------------------
X = df['clean_text']
y = df['label']

# ---------------------------------------------------
# Train-Test Split
# ---------------------------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# ---------------------------------------------------
# TF-IDF Vectorization
# ---------------------------------------------------
tfidf = TfidfVectorizer(
    max_features=20000,
    ngram_range=(1, 3),
    min_df=2,
    max_df=0.9,
    sublinear_tf=True
)

X_train_tfidf = tfidf.fit_transform(X_train)
X_test_tfidf = tfidf.transform(X_test)

print("✅ TF-IDF vectorization completed")

# ---------------------------------------------------
# 🔥 SGD CLASSIFIER (MAIN CHANGE)
# ---------------------------------------------------
model = SGDClassifier(
    loss="hinge",          # linear SVM
    penalty="l2",
    alpha=1e-4,
    max_iter=2000,
    tol=1e-3,
    class_weight="balanced",
    random_state=42
)

model.fit(X_train_tfidf, y_train)
print("✅ SGD Classifier trained")

# ---------------------------------------------------
# Model Evaluation
# ---------------------------------------------------
y_pred = model.predict(X_test_tfidf)

print("\n🎯 Accuracy:", round(accuracy_score(y_test, y_pred) * 100, 2), "%")
print("\n📌 Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))
print("\n📄 Classification Report:")
print(classification_report(y_test, y_pred))

# ---------------------------------------------------
# Check News: REAL or FAKE
# ---------------------------------------------------
while True:
    news = input("\n📰 Enter news text (type 'exit' to quit): ")

    if news.lower() == 'exit':
        break

    news_clean = clean_text(news)
    news_tfidf = tfidf.transform([news_clean])
    prediction = model.predict(news_tfidf)

    if prediction[0] == 1:
        print("✅ REAL NEWS")
    else:
        print("❌ FAKE NEWS")
