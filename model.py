# Step 1: Data Collection and Preparation
import textblob as TextBlob
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import csv


def analyze_sentiment(sentences):
    sentiments = []

    for sentence in sentences:
        # Create a TextBlob object for sentiment analysis
        blob = TextBlob.TextBlob(sentence)

        # Get the sentiment polarity (ranges from -1 to 1)
        sentiment_polarity = blob.sentiment.polarity

        # Classify sentiment based on polarity
        if sentiment_polarity > 0:
            sentiments.append('Positive')
        elif sentiment_polarity < 0:
            sentiments.append('Negative')
        else:
            sentiments.append('Neutral')

    return sentiments

def read_csv(file_path):
    with open(file_path, 'r') as csvfile:
        # Create a CSV reader object
        csvreader = csv.reader(csvfile)

        # Read and print each line in a readable format
        # for row in csvreader:
        #     print(', '.join(row))
        #
        #

        rows = []
        for row in csvreader:
            x=', '.join(row)
            print(x)
            rows.append(x)

    return rows


# Load your dataset with student responses
data = pd.read_csv('large_student_responses.csv')

# Split the data into training and testing sets
X = data['response']
y = data['happiness']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 2: Model Selection and Training
# Use bag-of-words representation for text data and logistic regression model
vectorizer = CountVectorizer(stop_words='english')
X_train_bow = vectorizer.fit_transform(X_train)
X_test_bow = vectorizer.transform(X_test)

text_model = LogisticRegression()
text_model.fit(X_train_bow, y_train)

# Step 3: Model Evaluation
# Make predictions on the test set
y_pred_text = text_model.predict(X_test_bow)

# Evaluate the text model
accuracy_text = accuracy_score(y_test, y_pred_text)

print(f'Text Model Accuracy: {accuracy_text}')

# Step 4: Model Deployment and Interpretation
# ... (Add deployment and interpretation steps here)
# Replace 'your_data.csv' with the actual file path
file_path = 'input.csv'
print(analyze_sentiment(read_csv(file_path)))








