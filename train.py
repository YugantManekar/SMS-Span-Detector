import pandas as pd
import pickle
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB

df = pd.read_csv("spam.csv", encoding="latin-1")[["v1", "v2"]]
df.columns = ["label", "text"]

df["label"] = df["label"].map({"ham": 0, "spam": 1})

X = df["text"]
y = df["label"]

tfidf = TfidfVectorizer()
X_tfidf = tfidf.fit_transform(X)

model = MultinomialNB()
model.fit(X_tfidf, y)

pickle.dump(tfidf, open("vectorizer.pkl", "wb"))
pickle.dump(model, open("model.pkl", "wb"))

print("Training complete. Files saved.")
