import pandas as pd
import random
from flask import Flask, render_template, request
import re

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
# from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC

app = Flask(__name__)

def remove_punct(text):
    res = re.sub(r'[^\w\s]', '', text)

    return res

df_train = pd.read_csv("train.csv")
df_train["text"] = df_train["text"].apply(lambda x: remove_punct(x))
df_response = pd.read_csv("response.csv")

X_train, X_test, y_train, y_test = train_test_split(df_train['text'], df_train['label'], random_state = 0)
count_vect = CountVectorizer()
X_train_counts = count_vect.fit_transform(X_train)
tfidf_transformer = TfidfTransformer()
X_train_tfidf = tfidf_transformer.fit_transform(X_train_counts)
clf = LinearSVC().fit(X_train_tfidf, y_train)

def predict(text):
    text = remove_punct(text)
    label_prediction = clf.predict(count_vect.transform([text]))[0]
    # proba_prediction = max(clf.predict_proba(count_vect.transform([text]))[0])

    response = random.choice(list(df_response.query(f"label == '{label_prediction}'")["response"]))

    print(f"predicted {label_prediction}, response {response}")

    return response

@app.route("/")
def home():
    return render_template("home.html")

@app.route("/get")
def get_bot_response():
    userText = request.args.get('msg')
    return predict(userText)

if __name__ == "__main__":
    app.run()