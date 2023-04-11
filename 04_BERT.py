import pickle
import numpy as np
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score
import torch
from transformers import TrainingArguments, Trainer
from transformers import BertTokenizer, BertForSequenceClassification
import contextlib
import random

random.seed(5)


# Create torch dataset
class Dataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels=None):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        if self.labels:
            item["labels"] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.encodings["input_ids"])


def model_init():
    model_name = "KB/bert-base-swedish-cased"
    return BertForSequenceClassification.from_pretrained(model_name, num_labels=2)


def compute_metrics(p):
    pred, labels = p
    pred = np.argmax(pred, axis=1)

    accuracy = accuracy_score(y_true=labels, y_pred=pred)
    recall = recall_score(y_true=labels, y_pred=pred)
    precision = precision_score(y_true=labels, y_pred=pred)
    f1 = f1_score(y_true=labels, y_pred=pred)
    print(
        "accuracy: " + str(accuracy) + " precision: " + str(precision) + " recall: " + str(recall) + " f1: " + str(f1))

    return {"accuracy": accuracy, "precision": precision, "recall": recall, "f1": f1}


def main():
    # Read data
    with open("data/cased_data_rs_50_m_s.pkl", "rb") as f:
        x_train, x_test, _, y_train, y_test, _ = pickle.load(f)

    p_dict = {'s': 0, 'm': 1}

    # Define pretrained tokenizer and model
    model_name = "KB/bert-base-swedish-cased"
    tokenizer = BertTokenizer.from_pretrained(model_name)

    x_train_tokenized = tokenizer(list(x_train), padding=True, truncation=True, max_length=512)
    y_train = [p_dict[i] for i in y_train]
    x_test_tokenized = tokenizer(list(x_test), padding=True, truncation=True, max_length=512)
    y_test = [p_dict[i] for i in y_test]

    train_dataset = Dataset(x_train_tokenized, y_train)
    test_dataset = Dataset(x_test_tokenized, y_test)

    # Define Trainer
    args = TrainingArguments(
        output_dir="cased_output_50",
        evaluation_strategy="steps",
        eval_steps=1000,
        save_steps=1000,
        save_total_limit=1,
        learning_rate=5e-6,
        per_device_train_batch_size=48,
        per_device_eval_batch_size=48,
        num_train_epochs=5,
        seed=5,
        load_best_model_at_end=True,
    )
    trainer = Trainer(
        model_init=model_init,
        args=args,
        train_dataset=train_dataset,
        eval_dataset=test_dataset,
        compute_metrics=compute_metrics,
        # callbacks=[EarlyStoppingCallback(early_stopping_patience=3)],
    )

    path = 'data/log_file.txt'
    with open(path, 'w') as f:
        with contextlib.redirect_stdout(f):
            # Train pre-trained model
            trainer.train()


if __name__ == "__main__":
    main()
