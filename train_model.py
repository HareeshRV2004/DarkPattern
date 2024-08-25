from transformers import DistilBertTokenizer, DistilBertForSequenceClassification, Trainer, TrainingArguments
from sklearn.model_selection import train_test_split
from datasets import Dataset
import pandas as pd
import torch
import os

# Define paths
MODEL_PATH = './distilbert-sentiment-model'

# Load dataset
df = pd.read_csv('dataset.csv')
df.columns = ['polarity', 'text']  # Rename columns

# Map polarities
df['labels'] = df['polarity'].map({1: 0, 2: 1})

# Split dataset
train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)

# Prepare datasets
train_dataset = Dataset.from_pandas(train_df[['text', 'labels']])
test_dataset = Dataset.from_pandas(test_df[['text', 'labels']])

# Load tokenizer and model
if not os.path.exists(MODEL_PATH):
    # Download model and tokenizer if they are not available locally
    tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
    model = DistilBertForSequenceClassification.from_pretrained('distilbert-base-uncased')
    # Save the model and tokenizer
    tokenizer.save_pretrained(MODEL_PATH)
    model.save_pretrained(MODEL_PATH)
else:
    tokenizer = DistilBertTokenizer.from_pretrained(MODEL_PATH)
    model = DistilBertForSequenceClassification.from_pretrained(MODEL_PATH)

# Tokenize datasets
def tokenize_function(examples):
    return tokenizer(examples['text'], padding="max_length", truncation=True)

train_dataset = train_dataset.map(tokenize_function, batched=True)
test_dataset = test_dataset.map(tokenize_function, batched=True)

# Define training arguments
training_args = TrainingArguments(
    output_dir='./results',
    evaluation_strategy="epoch",
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    num_train_epochs=3,
    weight_decay=0.01,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=test_dataset,
)

# Train the model
trainer.train()

# Save the model
model.save_pretrained(MODEL_PATH)
tokenizer.save_pretrained(MODEL_PATH)
