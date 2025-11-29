import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import LabelEncoder
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
import re
import string

#download required NLTK data with error handling
def download_nltk_resources():
    resources = ['punkt', 'stopwords', 'wordnet', 'omw-1.4', 'punkt_tab']
    for resource in resources:
        try:
            nltk.download(resource, quiet=True)
            print(f"Successfully downloaded/loaded {resource}")
        except Exception as e:
            print(f"Error downloading {resource}: {e}")

download_nltk_resources()

class SentimentAnalyzer:
    def __init__(self):
        self.vectorizer = TfidfVectorizer(max_features=5000, stop_words='english', ngram_range=(1, 2))
        self.classifier = MultinomialNB()
        self.lemmatizer = WordNetLemmatizer()
        try:
            self.stop_words = set(stopwords.words('english'))
        except:
            print("Stopwords not available, using empty set")
            self.stop_words = set()
        self.label_encoder = LabelEncoder()
    
    def preprocess_text(self, text):
        """Preprocess the text data with lemmatization"""
        if not isinstance(text, str):
            return ""
            
        text = text.lower()
        
        #remove punctuation and numbers
        text = re.sub(f'[{re.escape(string.punctuation)}]', '', text)
        text = re.sub(r'\d+', '', text)
        
        try:
            #tokenize
            tokens = word_tokenize(text)
            
            #remove stopwords and lemmatize
            tokens = [self.lemmatizer.lemmatize(token) for token in tokens 
                     if token not in self.stop_words and len(token) > 2]
            
            return ' '.join(tokens)
        except:
            #fallback simple tokenization
            tokens = text.split()
            tokens = [self.lemmatizer.lemmatize(token) for token in tokens 
                     if token not in self.stop_words and len(token) > 2]
            return ' '.join(tokens)
    
    def preprocess_data(self, texts):
        """Preprocess all texts in the dataset"""
        return [self.preprocess_text(text) for text in texts]
    
    def train(self, X_train, y_train):
        """Train the sentiment analyzer"""
        #preprocess training data
        print("Preprocessing training data...")
        X_train_processed = self.preprocess_data(X_train)
        
        #vectorize the text data
        print("Vectorizing text data...")
        X_train_vectorized = self.vectorizer.fit_transform(X_train_processed)
        
        #encode labels
        y_train_encoded = self.label_encoder.fit_transform(y_train)
        
        #train the classifier
        print("Training Multinomial Naive Bayes classifier...")
        self.classifier.fit(X_train_vectorized, y_train_encoded)
        
        #calculate training accuracy
        train_accuracy = self.classifier.score(X_train_vectorized, y_train_encoded)
        print(f"Training Accuracy: {train_accuracy:.4f}")
    
    def predict(self, X_test):
        """Make predictions on new data"""
        #preprocess test data
        X_test_processed = self.preprocess_data(X_test)
        
        #vectorize test data
        X_test_vectorized = self.vectorizer.transform(X_test_processed)
        
        #make predictions
        predictions_encoded = self.classifier.predict(X_test_vectorized)
        
        #decode predictions
        return self.label_encoder.inverse_transform(predictions_encoded)
    
    def predict_proba(self, X_test):
        """Get prediction probabilities"""
        X_test_processed = self.preprocess_data(X_test)
        X_test_vectorized = self.vectorizer.transform(X_test_processed)
        return self.classifier.predict_proba(X_test_vectorized)
    
    def evaluate(self, X_test, y_test):
        """Evaluate the model performance"""
        predictions = self.predict(X_test)
        
        accuracy = accuracy_score(y_test, predictions)
        print(f"Test Accuracy: {accuracy:.4f}")
        print("\nClassification Report:")
        print(classification_report(y_test, predictions))
        print("\nConfusion Matrix:")
        print(confusion_matrix(y_test, predictions))
        
        return accuracy

def create_sample_review_data():
    """Create sample product review data for demonstration"""
    positive_reviews = [
        "This product is amazing! It works perfectly and exceeded my expectations.",
        "Great quality and fast delivery. I'm very satisfied with my purchase.",
        "Excellent product! Would definitely recommend to others.",
        "Love this! The quality is outstanding and it arrived quickly.",
        "Perfect fit and great value for money. Very happy with this purchase.",
        "Outstanding performance and build quality. Highly recommended!",
        "This is exactly what I needed. Works flawlessly and looks great.",
        "Impressive product with excellent features. Worth every penny.",
        "Superb quality and fantastic customer service. 5 stars!",
        "Best purchase I've made this year. Exceeded all expectations."
    ]
    
    negative_reviews = [
        "Terrible product! Stopped working after just one day.",
        "Poor quality and not worth the money. Very disappointed.",
        "This is awful. Doesn't work as described. Avoid this product.",
        "Worst purchase ever. Complete waste of money.",
        "Broken upon arrival. Poor packaging and terrible quality.",
        "Extremely disappointed. The product is nothing like described.",
        "Defective item. Customer service was unhelpful.",
        "Low quality materials. Fell apart after minimal use.",
        "Not recommended. Poor performance and unreliable.",
        "Complete garbage. Save your money and look elsewhere."
    ]
    
    neutral_reviews = [
        "The product is okay. It works but nothing special.",
        "Average quality. Does the job but could be better.",
        "It's fine for the price. Nothing extraordinary.",
        "Product works as expected. No major issues but no surprises either.",
        "Decent product. Does what it's supposed to do.",
        "Average performance. Meets basic requirements.",
        "The product is acceptable. Neither good nor bad.",
        "Standard quality. Gets the job done adequately.",
        "It's alright. Not amazing but not terrible either.",
        "Functional product. No complaints but no excitement either."
    ]
    
    reviews = positive_reviews + negative_reviews + neutral_reviews
    sentiments = ['positive'] * len(positive_reviews) + ['negative'] * len(negative_reviews) + ['neutral'] * len(neutral_reviews)
    
    return reviews, sentiments

# Main execution for Task 2
print("\n" + "=" * 60)
print("TASK 2: SENTIMENT ANALYSIS FOR PRODUCT REVIEWS")
print("=" * 60)

# Create sample data
reviews, sentiments = create_sample_review_data()

# Split the data
X_train, X_test, y_train, y_test = train_test_split(
    reviews, sentiments, test_size=0.3, random_state=42, stratify=sentiments
)

print(f"Training set size: {len(X_train)}")
print(f"Test set size: {len(X_test)}")
print(f"Positive reviews in training: {y_train.count('positive')}")
print(f"Negative reviews in training: {y_train.count('negative')}")
print(f"Neutral reviews in training: {y_train.count('neutral')}")

# Initialize and train the sentiment analyzer
sentiment_analyzer = SentimentAnalyzer()
sentiment_analyzer.train(X_train, y_train)

# Evaluate the model
print("\n" + "=" * 40)
print("MODEL EVALUATION")
print("=" * 40)
sentiment_analyzer.evaluate(X_test, y_test)

# Test with new reviews
print("\n" + "=" * 40)
print("PREDICTION ON NEW REVIEWS")
print("=" * 40)
new_reviews = [
    "This product is absolutely fantastic! Love it!",
    "Terrible quality, very disappointed with this purchase.",
    "The product works fine, but it's nothing special.",
    "Amazing value for money, highly recommended!",
    "It's okay, does the job but could be improved."
]

predictions = sentiment_analyzer.predict(new_reviews)
probabilities = sentiment_analyzer.predict_proba(new_reviews)

for review, pred, prob in zip(new_reviews, predictions, probabilities):
    print(f"Review: {review[:40]}...")
    print(f"Prediction: {pred}")
    print(f"Probabilities: Positive: {prob[0]:.3f}, Negative: {prob[1]:.3f}, Neutral: {prob[2]:.3f}")
    print("-" * 50)

# Feature importance analysis
print("\n" + "=" * 40)
print("FEATURE ANALYSIS")
print("=" * 40)
feature_names = sentiment_analyzer.vectorizer.get_feature_names_out()
class_probabilities = sentiment_analyzer.classifier.feature_log_prob_

# Get top features for each class
classes = sentiment_analyzer.label_encoder.classes_
for i, class_name in enumerate(classes):
    print(f"\nTop 10 features for '{class_name}':")
    top_indices = class_probabilities[i].argsort()[-10:][::-1]
    top_features = [(feature_names[idx], class_probabilities[i][idx]) for idx in top_indices]
    for feature, score in top_features:
        print(f"  {feature}: {score:.3f}")
