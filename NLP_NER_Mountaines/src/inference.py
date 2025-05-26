from sklearn.metrics import classification_report, accuracy_score
import numpy as np
from datasets import load_from_disk
from transformers import BertTokenizerFast, BertForTokenClassification

# Завантаження токенізованого датасету
tokenized_datasets = load_from_disk("../data/tokenized_dataset")
tokenized_test_dataset = tokenized_datasets['test']

# Завантаження моделі та токенізатора
tokenizer = BertTokenizerFast.from_pretrained("../models/fine_tuned_ner_mountains_model")
model = BertForTokenClassification.from_pretrained("../models/fine_tuned_ner_mountains_model")

# Перетворення тестового датасету на об'єкт DataLoader
import torch
from torch.utils.data import DataLoader
from transformers import DataCollatorForTokenClassification

data_collator = DataCollatorForTokenClassification(tokenizer=tokenizer)

test_dataloader = DataLoader(
    tokenized_test_dataset,
    batch_size=16,
    collate_fn=data_collator
)



# Функція для отримання предиктів
def get_predictions_and_labels(model, dataloader):
    model.eval()
    all_predictions = []
    all_labels = []

    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch['input_ids']
            attention_mask = batch['attention_mask']
            labels = batch['labels']

            outputs = model(input_ids, attention_mask=attention_mask)
            logits = outputs.logits

            # Отримання предиктів
            predictions = torch.argmax(logits, dim=-1)

            # Зберігаємо предикти та мітки
            all_predictions.append(predictions.cpu().numpy())
            all_labels.append(labels.cpu().numpy())

    # Перетворюємо в масиви NumPy
    all_predictions = np.concatenate(all_predictions, axis=0)
    all_labels = np.concatenate(all_labels, axis=0)

    return all_predictions, all_labels


# Отримання предиктів і міток
predictions, labels = get_predictions_and_labels(model, test_dataloader)

# Перетворення у формат для обчислення метрик (з фільтрацією "-100")
true_labels = []
predicted_labels = []

for true_seq, pred_seq in zip(labels, predictions):
    for true_label, pred_label in zip(true_seq, pred_seq):
        if true_label != -100:  # Ігноруємо токени, що не використовуються
            true_labels.append(true_label)
            predicted_labels.append(pred_label)

# Оцінювання метрик
print("Classification Report:")
print(classification_report(true_labels, predicted_labels))

print("Accuracy:")
print(accuracy_score(true_labels, predicted_labels))

