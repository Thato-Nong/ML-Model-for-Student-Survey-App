from textblob import TextBlob
from flask import Flask, request, jsonify
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix

# Create a Flask web application
app = Flask(__name__)

# Load your dataset with student responses
data = pd.read_csv('large_student_responses.csv')

# Extract sentiment-related features from the text using TextBlob
def extract_sentiment(text):
    analysis = TextBlob(text)
    polarity = analysis.sentiment.polarity
    subjectivity = analysis.sentiment.subjectivity
    return polarity, subjectivity

# Apply TextBlob sentiment analysis to the response column
data['polarity'], data['subjectivity'] = zip(*data['response'].apply(extract_sentiment))

# Split the data into training and testing sets
X = data[['polarity', 'subjectivity']]
y = data['happiness']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=10)

# Evaluate TextBlob sentiment analysis on the test set
# Note: This is a basic evaluation; you might want to use a more sophisticated model for better accuracy
# For demonstration purposes, let's use a logistic regression model
from sklearn.linear_model import LogisticRegression
model = LogisticRegression()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

accuracy_textblob = accuracy_score(y_test, y_pred)
confusion_mat_textblob = confusion_matrix(y_test, y_pred)

print(f"Accuracy using TextBlob: {accuracy_textblob}")
print(f"Confusion Matrix using TextBlob:\n{confusion_mat_textblob}")

# Define a route for sentiment prediction using TextBlob
@app.route('/predict_sentiment_textblob', methods=['POST'])
def predict_sentiment_textblob():
    try:
        # Get the input sentence from the request
        input_sentence = request.json['sentence']

        # Extract sentiment from the input sentence using TextBlob
        polarity, subjectivity = extract_sentiment(input_sentence)

        # Return the prediction as JSON
        return jsonify({'polarity': polarity, 'subjectivity': subjectivity})
    
    except Exception as e:
        return jsonify({'error': str(e)})

# Run the Flask application
if __name__ == '__main__':
    app.run(debug=True)
