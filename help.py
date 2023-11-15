import requests

# URL for the Flask application with TextBlob sentiment prediction
url_textblob = 'http://127.0.0.1:5000/predict_sentiment_textblob'  # Update with the correct URL if needed

# Example sentence for prediction
input_sentence =input()

# Send a POST request to the Flask application with TextBlob route
response_textblob = requests.post(url_textblob, json={'sentence': input_sentence})

# Check the response
if response_textblob.status_code == 200:
    result_textblob = response_textblob.json()
    # Extract polarity and subjectivity from the response
    polarity = result_textblob.get('polarity', 0)
    subjectivity = result_textblob.get('subjectivity', 0)
    
    # Determine sentiment based on polarity and subjectivity
    predicted_sentiment = 1 if polarity > 0 else 0
    print(f"{predicted_sentiment}")
else:
    print(f"Error: {response_textblob.status_code}, {response_textblob.json()}")
