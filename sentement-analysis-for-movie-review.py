import nltk
import numpy as np
import pandas as pd
from nltk.corpus import movie_reviews
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report

def load_movie_reviews():
    nltk.download('movie_reviews')
    documents = [(movie_reviews.raw(fileid), category)
                 for category in movie_reviews.categories()
                 for fileid in movie_reviews.fileids(category)]
    return documents

def preprocess_text(text):
    lemmatizer = WordNetLemmatizer()
    return " ".join([lemmatizer.lemmatize(word.lower()) for word in word_tokenize(text)])

def extract_features(documents):
    featuresets = [(preprocess_text(d), c) for (d, c) in documents]
    X = [d for (d, c) in featuresets]
    y = [c for (d, c) in featuresets]
    return X, y

def train_classifier(X_train, y_train):
    vectorizer = TfidfVectorizer(max_features=3000)
    X_train_tfidf = vectorizer.fit_transform(X_train)
    classifier = MultinomialNB()
    classifier.fit(X_train_tfidf, y_train)
    return classifier, vectorizer

def predict_sentiment(classifier, vectorizer, review):
    features = preprocess_text(review)
    features_vector = vectorizer.transform([features])
    sentiment = classifier.predict(features_vector)
    return "Positive" if sentiment[0] == 'pos' else "Negative"

def main():
    documents = load_movie_reviews()
    X, y = extract_features(documents)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    classifier, vectorizer = train_classifier(X_train, y_train)

    y_pred = classifier.predict(vectorizer.transform(X_test))

    print("Accuracy:", accuracy_score(y_test, y_pred))
    print(classification_report(y_test, y_pred))

    review1 = "This product is amazing! I love it!"
    sentiment = predict_sentiment(classifier, vectorizer, review1)
    print("Review:", review1)
    print("Sentiment:", sentiment)

if __name__ == "__main__":
    main()
