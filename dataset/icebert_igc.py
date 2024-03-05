import numpy as np
import torch
from datasets import load_dataset, DatasetDict
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

# Check CUDA
# print(torch.cuda.is_available())

# Load and prepare dataset
dataset_path = r'C:\Users\lenax\PycharmProjects\gigaword_scripts\dataset_structured.csv'
full_dataset = load_dataset('csv', data_files={'full': dataset_path}, split='full')

# Split the dataset into training, validation, and test
train_testvalid = full_dataset.train_test_split(test_size=0.2)  # Splitting 80% training, 20% for test+validation
test_valid = train_testvalid['test'].train_test_split(test_size=0.5)  # Splitting the 20% into 10% test, 10% validation

dataset = DatasetDict({
    'train': train_testvalid['train'],
    'validation': test_valid['test'],
    'test': test_valid['train']
})

# Tokenize the dataset
model_checkpoint = "mideind/IceBERT-igc"
tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)

def tokenize_function(examples):
    contents = examples["content"]
    # Ensure all contents are indeed strings
    for i, content in enumerate(contents):
        if not isinstance(content, str):
            print(f"Non-string content at index {i}: {type(content)}")
            contents[i] = str(content)  # Convert to string as a fallback
    tokenized_outputs = tokenizer(contents, padding="max_length", truncation=True, max_length=128)
    tokenized_outputs['labels'] = examples['label']
    return tokenized_outputs

tokenized_datasets = dataset.map(tokenize_function, batched=True, remove_columns=dataset["train"].column_names)

# Define compute_metrics function for evaluation
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=1)
    precision, recall, f1, _ = precision_recall_fscore_support(labels, predictions, average='weighted')
    acc = accuracy_score(labels, predictions)
    return {
        'accuracy': acc,
        'f1': f1,
        'precision': precision,
        'recall': recall
    }

# Load the model for sequence classification
model = AutoModelForSequenceClassification.from_pretrained(model_checkpoint, num_labels=4)  # Adjust `num_labels` as per your dataset

# Set training arguments
training_args = TrainingArguments(
    output_dir="./results",  # Directory where the results are saved
    evaluation_strategy="steps",  # Evaluate and Save at each 'eval_steps'
    eval_steps=500,  # Evaluate and save the model every 500 steps
    save_strategy="steps",
    save_steps=500,  # Save a model checkpoint every 500 steps
    learning_rate=2e-5,
    per_device_train_batch_size=16,   # Adjust
    per_device_eval_batch_size=64,   # Adjust
    num_train_epochs=3,
    weight_decay=0.01,
    #logging_dir='./logs',  # Directory for logs
    #logging_steps=10,  # Log metrics every 10 steps
)

# Initialize the Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_datasets["train"],
    eval_dataset=tokenized_datasets["validation"],
    compute_metrics=compute_metrics,
)

# Start fine-tuning
trainer.train()

eval_results = trainer.evaluate(tokenized_datasets["validation"])
print(f"Validation Results: {eval_results}")

# Evaluate the model on the test set
test_results = trainer.evaluate(tokenized_datasets["test"])
print(f"Test Set Evaluation Results: {test_results}")

# Save the fine-tuned model and tokenizer
model_path = r'C:\Users\lenax\PycharmProjects\gigaword_scripts\models\IceBert'
trainer.model.save_pretrained(model_path)
tokenizer.save_pretrained(model_path)