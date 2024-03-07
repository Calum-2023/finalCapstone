import spacy
import pandas as pd
from textblob import TextBlob

amazon_data = "amazon_product_reviews.csv"

df = pd.read_csv(amazon_data)

nlp = spacy.load("en_core_web_sm")

# Remove lowercase text and removes stops 
def clean_text(text):
  doc = nlp(text)
  tokens = [token.text.lower() for token in doc if not token.is_stop] 
  clean_text = " ".join(tokens)
  return clean_text

df["reviews.text"] = df["reviews.text"].fillna("").apply(clean_text)

# Function to predict sentiment using TextBlob
def predict_sentiment(text):
  blob = TextBlob(text)
  sentiment = blob.sentiment  
  
# Measure of the reviews in quesion 
  if sentiment.polarity > 0.2:
    return "POSITIVE"
  elif sentiment.polarity < -0.2:
    return "NEGATIVE"
  else:
    return "NEUTRAL"

df["sentiment"] = df["reviews.text"].apply(predict_sentiment)

# Print the first 5 rows so I can check the results 
test_row = df.head(5)
print(test_row)
