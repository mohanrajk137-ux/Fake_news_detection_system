import pandas as pd
import re
import nltk
import matplotlib.pyplot as plt
import joblib

from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    classification_report,
    roc_curve,
    auc
)

# ---------------------------------------------------
# NLTK DOWNLOADS
# ---------------------------------------------------
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('omw-1.4')

stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

# ---------------------------------------------------
# LOAD DATASET
# ---------------------------------------------------
df = pd.read_csv("fixed.csv", low_memory=False)
df = df.loc[:, ~df.columns.str.contains("^Unnamed")]
df = df[['text', 'label']]
df['text'] = df['text'].fillna("")

# ---------------------------------------------------
# FIX LABEL COLUMN (SAFE)
# ---------------------------------------------------
df['label'] = df['label'].astype(str).str.lower().str.strip()

label_map = {
    'fake': 0, 'false': 0, '0': 0,
    'real': 1, 'true': 1, '1': 1
}

df['label'] = df['label'].map(label_map)
df = df.dropna(subset=['label'])
df['label'] = df['label'].astype(int)

print("✅ Dataset Loaded")
print(df['label'].value_counts())

# ---------------------------------------------------
# TEXT CLEANING FUNCTION
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
print("✅ Text Preprocessing Completed")

# ---------------------------------------------------
# FEATURES & LABELS
# ---------------------------------------------------
X = df['clean_text']
y = df['label']

# ---------------------------------------------------
# TRAIN-TEST SPLIT
# ---------------------------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# ---------------------------------------------------
# TF-IDF VECTORISATION
# ---------------------------------------------------
tfidf = TfidfVectorizer(
    max_features=25000,
    ngram_range=(1, 3),
    min_df=2,
    max_df=0.9,
    sublinear_tf=True
)

X_train_tfidf = tfidf.fit_transform(X_train)
X_test_tfidf = tfidf.transform(X_test)

print("✅ TF-IDF Vectorization Completed")

# ---------------------------------------------------
# ADVANCED SVM (CALIBRATED)
# ---------------------------------------------------
base_svm = LinearSVC(
    C=2.0,
    class_weight='balanced',
    max_iter=5000
)

model = CalibratedClassifierCV(
    base_svm,
    method='sigmoid',
    cv=5
)

model.fit(X_train_tfidf, y_train)
print("✅ Advanced SVM Model Trained")

# ---------------------------------------------------
# SAVE MODEL & VECTORIZER
# ---------------------------------------------------
joblib.dump(model, "svm_fake_news_model.pkl")
joblib.dump(tfidf, "svm_tfidf_vectorizer.pkl")
print("✅ Model & Vectorizer Saved")

# ---------------------------------------------------
# MODEL EVALUATION
# ---------------------------------------------------
y_pred = model.predict(X_test_tfidf)
y_prob = model.predict_proba(X_test_tfidf)[:, 1]

print("\n🎯 Accuracy:", round(accuracy_score(y_test, y_pred) * 100, 2), "%")

print("\n📌 Confusion Matrix:")
cm = confusion_matrix(y_test, y_pred)
print(cm)

print("\n📄 Classification Report:")
print(classification_report(y_test, y_pred))

# ---------------------------------------------------
# ROC CURVE
# ---------------------------------------------------
fpr, tpr, _ = roc_curve(y_test, y_prob)
roc_auc = auc(fpr, tpr)

plt.figure()
plt.plot(fpr, tpr, label=f"SVM (AUC = {roc_auc:.3f})")
plt.plot([0, 1], [0, 1], linestyle="--")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve - Fake News Detection")
plt.legend(loc="lower right")
plt.show()

# ---------------------------------------------------
# NEWS CHECKING (REAL / FAKE)
# ---------------------------------------------------
print("\n📰 FAKE NEWS CHECKER (Type 'exit' to quit)")

while True:
    news = input("\nEnter news text: ")

    if news.lower() == "exit":
        print("👋 Exiting News Checker")
        break

    news_clean = clean_text(news)
    news_tfidf = tfidf.transform([news_clean])

    prediction = model.predict(news_tfidf)[0]
    probability = model.predict_proba(news_tfidf)[0][1]

    confidence = round(probability * 100, 2)

    if prediction == 1:
        print(f"✅ REAL NEWS ({confidence}% confidence)")
    else:
        print(f"❌ FAKE NEWS ({confidence}% confidence)")
