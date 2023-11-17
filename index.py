from fastapi import FastAPI
from textblob import TextBlob
from pydantic import BaseModel

app = FastAPI()

@app.get("/")
def root():
    return {"status": "running"}

class Data(BaseModel):
    sentence: str

@app.post("/predict_sentiment_textblob")
def predict_sentiment_textblob(data: Data):
    try:
        input_sentence = data.sentence

        analysis = TextBlob(input_sentence)
        polarity = analysis.sentiment.polarity
        subjectivity = analysis.sentiment.subjectivity

        predicted_sentiment = 2 if polarity > 0 else 1 if polarity < 0 else 0

        return {'polarity': polarity, 'subjectivity': subjectivity, 'predicted_sentiment': predicted_sentiment}
    
    except Exception as e:
        return {'error': str(e)}

