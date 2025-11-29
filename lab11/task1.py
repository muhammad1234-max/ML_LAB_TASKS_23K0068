import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize
import re
import string

# Download required NLTK data with error handling
def download_nltk_resources():
    resources = ['punkt', 'stopwords', 'punkt_tab']
    for resource in resources:
        try:
            nltk.download(resource, quiet=True)
            print(f"Successfully downloaded/loaded {resource}")
        except Exception as e:
            print(f"Error downloading {resource}: {e}")

download_nltk_resources()

class SpamClassifier:
    def __init__(self):
        self.vectorizer = CountVectorizer(max_features=5000, stop_words='english')
        self.classifier = MultinomialNB()
        self.stemmer = PorterStemmer()
        try:
            self.stop_words = set(stopwords.words('english'))
        except:
            print("Stopwords not available, using empty set")
            self.stop_words = set()
    
    def preprocess_text(self, text):
        """Preprocess the text data"""
        if not isinstance(text, str):
            return ""
            
        # Convert to lowercase
        text = text.lower()
        
        # Remove punctuation and numbers
        text = re.sub(f'[{re.escape(string.punctuation)}]', '', text)
        text = re.sub(r'\d+', '', text)
        
        try:
            # Tokenize
            tokens = word_tokenize(text)
            
            # Remove stopwords and stem
            tokens = [self.stemmer.stem(token) for token in tokens 
                     if token not in self.stop_words and len(token) > 2]
            
            return ' '.join(tokens)
        except:
            # Fallback simple tokenization
            tokens = text.split()
            tokens = [self.stemmer.stem(token) for token in tokens 
                     if token not in self.stop_words and len(token) > 2]
            return ' '.join(tokens)
    
    def preprocess_data(self, texts):
        """Preprocess all texts in the dataset"""
        return [self.preprocess_text(text) for text in texts]
    
    def train(self, X_train, y_train):
        """Train the spam classifier"""
        # Preprocess training data
        print("Preprocessing training data...")
        X_train_processed = self.preprocess_data(X_train)
        
        # Vectorize the text data
        print("Vectorizing text data...")
        X_train_vectorized = self.vectorizer.fit_transform(X_train_processed)
        
        # Train the classifier
        print("Training Multinomial Naive Bayes classifier...")
        self.classifier.fit(X_train_vectorized, y_train)
        
        # Calculate training accuracy
        train_accuracy = self.classifier.score(X_train_vectorized, y_train)
        print(f"Training Accuracy: {train_accuracy:.4f}")
    
    def predict(self, X_test):
        """Make predictions on new data"""
        # Preprocess test data
        X_test_processed = self.preprocess_data(X_test)
        
        # Vectorize test data
        X_test_vectorized = self.vectorizer.transform(X_test_processed)
        
        # Make predictions
        return self.classifier.predict(X_test_vectorized)
    
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

# Example usage with sample data
def create_sample_email_data():
    """Create sample email data for demonstration"""
    spam_emails = [
        "WINNER!! You have won a $1000 gift card! Click here to claim your prize!",
        "URGENT: Your account needs verification. Please confirm your details immediately.",
        "Get rich quick! Make money fast with this amazing opportunity!",
        "Free iPhone! Limited time offer. Claim now before it's too late!",
        "Your package delivery failed. Please update your shipping information.",
        "Exclusive deal just for you! 90% off all products today only!",
        "Your bank account has been compromised. Verify your identity now!",
        "Work from home and earn $5000 per month. No experience required!",
        "You have inherited $1,000,000 from a distant relative. Contact us!",
        "Hot singles in your area are waiting to meet you! Join now!"
    ]
    
    ham_emails = [
        "Hi Muhammad, just checking in to see how you're doing. Let's catch up soon.",
        "Meeting reminder: Project review at 3 PM tomorrow in conference room B.",
        "Thanks for your email. I'll get back to you with the information soon.",
        "The quarterly report is attached. Please review and provide feedback.",
        "Lunch tomorrow? Let me know what time works for you.",
        "Your order #12345 has been shipped and will arrive by Friday.",
        "Team building event this Friday at 5 PM. Don't forget to RSVP.",
        "Please find the document you requested attached to this email.",
        "Happy birthday! Hope you have a wonderful day filled with joy.",
        "The software update has been completed successfully."
    ]
    
    emails = spam_emails + ham_emails
    labels = ['spam'] * len(spam_emails) + ['ham'] * len(ham_emails)
    
    return emails, labels

# Main execution for Task 1
print("=" * 60)
print("TASK 1: SPAM/HAM EMAIL CLASSIFICATION")
print("=" * 60)

# Create sample data
emails, labels = create_sample_email_data()

# Split the data
X_train, X_test, y_train, y_test = train_test_split(
    emails, labels, test_size=0.3, random_state=42, stratify=labels
)

print(f"Training set size: {len(X_train)}")
print(f"Test set size: {len(X_test)}")
print(f"Spam emails in training: {y_train.count('spam')}")
print(f"Ham emails in training: {y_train.count('ham')}")

# Initialize and train the classifier
spam_classifier = SpamClassifier()
spam_classifier.train(X_train, y_train)


print("MODEL EVALUATION")
spam_classifier.evaluate(X_test, y_test)

# Test with new emails
print("\n" + "=" * 40)
print("PREDICTION ON NEW EMAILS")
print("=" * 40)
new_emails = [
    "Congratulations! You won a free vacation to Hawaii!",
    "Hi team, the meeting is scheduled for 2 PM tomorrow.",
    "Get your free trial now! Limited time offer!",
    "Please review the attached document for the project."
]

predictions = spam_classifier.predict(new_emails)
for email, pred in zip(new_emails, predictions):
    print(f"Email: {email[:50]}... -> Prediction: {pred}")
