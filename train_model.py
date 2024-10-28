from transformers import BertForSequenceClassification, Trainer, TrainingArguments
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
import torch
from data_preparation import prepare_data
import numpy as np

# Prepare the datasets
train_dataset, val_dataset, val_labels = prepare_data("data/reviews.tsv")

# Load the BERT model for sequence classification
model = BertForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=2)

# Define the metrics function
def compute_metrics(pred):
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)
    precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average='binary')
    acc = accuracy_score(labels, preds)
    return {"accuracy": acc, "f1": f1, "precision": precision, "recall": recall}

# Training arguments with adjusted learning rate and number of epochs
training_args = TrainingArguments(
    output_dir="./results",
    eval_strategy="epoch",
    save_strategy="epoch",
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    num_train_epochs=5,
    learning_rate=1e-5,
    logging_dir="./logs",
    load_best_model_at_end=True,
    weight_decay=0.01
)

# Initialize the Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    compute_metrics=compute_metrics
)

# Train the model
trainer.train()

# Save the fine-tuned model
trainer.save_model("./fine_tuned_model")

# Evaluate the model
results = trainer.evaluate()
print("Validation Results:", results)

# Calculate accuracy on the validation set
predictions = trainer.predict(val_dataset)
preds = torch.argmax(torch.tensor(predictions.predictions), axis=1)
accuracy = accuracy_score(val_labels, preds)
print(f"Validation Accuracy: {accuracy:.2%}")

# Report
with open("metrics_report.txt", "w") as report_file:
    report_file.write(f"Accuracy: {results['eval_accuracy']}\n")
    report_file.write(f"Precision: {results['eval_precision']}\n")
    report_file.write(f"Recall: {results['eval_recall']}\n")
    report_file.write(f"F1 Score: {results['eval_f1']}\n")
