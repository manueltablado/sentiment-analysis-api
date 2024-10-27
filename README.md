
# Sentiment Analysis with BERT

This project performs sentiment analysis using a fine-tuned BERT model on a review dataset. It includes data preprocessing, model training, and an API deployment to predict the sentiment (positive or negative) of new reviews.

## Project Structure

- `data_preparation.py`: Loads and prepares the dataset for training. Tokenizes the reviews and converts them into a format suitable for PyTorch.
- `train_model.py`: Trains a BERT model for sentiment classification. Saves the fine-tuned model at the end of training.
- `app.py`: API built with FastAPI that uses the fine-tuned model to predict the sentiment of new reviews.
- `setup.sh`: Configuration script that creates a virtual environment, installs necessary dependencies, and saves the installed packages in `requirements.txt`.
- `requirements.txt`: Dependency file generated by `setup.sh`.

## Prerequisites

1. **Python**: Ensure you have Python 3 installed on your system.
2. **Dependencies**: All dependencies are automatically installed using the `setup.sh` file.

## Environment Setup

1. **Run the Setup Script**:  
   Run the `setup.sh` file to create a virtual environment, install dependencies, and activate the environment.

   ```bash
   bash setup.sh
   ```

2. **Run Data Preprocessing**:  
   Execute `data_preparation.py` to load and preprocess the dataset if it's the first time you're running the project or if you’ve modified this file.

   ```bash
   python3 data_preparation.py
   ```

3. **Model Training**:  
   To train the BERT model on the review dataset, run `train_model.py`:

   ```bash
   python3 train_model.py
   ```

4. **API Deployment**:  
   Once the model is trained, you can run `app.py` to launch the API and predict the sentiment of new reviews.

   **Start the API Server**  
   Run the FastAPI server with uvicorn:

   ```bash
   uvicorn app:app --host 0.0.0.0 --port 8000
   ```

   **Prediction Endpoint**  
   POST /predict: Allows you to send a review in JSON format and receive a sentiment prediction (positive or negative).

   Example request:

   ```json
   {
       "review": "This product is fantastic!"
   }
   ```
