import numpy as np
import pandas as pd
from transformers import BertTokenizerFast, BertForTokenClassification, DataCollatorForTokenClassification, Trainer, TrainingArguments, pipeline
from datasets import load_from_disk

tokenized_datasets = load_from_disk("../data/tokenized_dataset")

print(tokenized_datasets)

tokenizer = BertTokenizerFast.from_pretrained("distilbert-base-uncased")
model = BertForTokenClassification.from_pretrained("distilbert-base-uncased", num_labels=3)

data_collator = DataCollatorForTokenClassification(tokenizer=tokenizer)

# Налаштування тренування
training_args = TrainingArguments(
    evaluation_strategy="epoch",
    save_strategy='epoch',
    learning_rate=2e-5,
    num_train_epochs=3,  # Зменшено кількість епох
    weight_decay=0.01,
    output_dir="../results",
    logging_dir="../logs",
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_datasets['train'],
    eval_dataset=tokenized_datasets['val'],
    tokenizer=tokenizer,
    data_collator=data_collator
)

# Навчання
trainer.train()

# Збереження моделі
model.save_pretrained("../models/fine_tuned_ner_mountains_model")
tokenizer.save_pretrained("../models/fine_tuned_ner_mountains_model")
