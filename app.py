from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from transformers import BertTokenizer, BertForSequenceClassification
import torch

# Initialize FastAPI
app = FastAPI()

# Load the fine-tuned model and tokenizer
model = BertForSequenceClassification.from_pretrained("./fine_tuned_model")
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

# Define the input structure for the review
class Review(BaseModel):
    review: str

# Route to predict sentiment
@app.post("/predict")
async def predict_sentiment(review: Review):
    # Validate if the review is empty or contains only spaces
    if not review.review.strip():
        raise HTTPException(status_code=400, detail="The review text cannot be empty.")
    
    try:
        # Preprocess the review text for BERT
        inputs = tokenizer(review.review, return_tensors="pt", truncation=True, padding="max_length", max_length=128)
        
        # Get predictions from the model
        with torch.no_grad():
            outputs = model(**inputs)
        
        # Classify sentiment (positive or negative)
        logits = outputs.logits
        prediction = torch.argmax(logits, dim=1).item()
        sentiment = "positive" if prediction == 1 else "negative"
        
        # Return the result as JSON
        return {"review": review.review, "sentiment": sentiment}
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error during prediction: {str(e)}")
