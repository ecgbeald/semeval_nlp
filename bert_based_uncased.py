from transformers import BertTokenizer, DataCollatorWithPadding, TrainingArguments, BertForSequenceClassification, Trainer
import numpy as np
import evaluate
from sklearn.metrics import confusion_matrix

class BertBasedTrainer:
    batch_size = 4
    def __init__(self, train_dataset, valid_dataset, dev_dataset):
        self.train_dataset = train_dataset
        self.valid_dataset = valid_dataset
        self.dev_dataset = dev_dataset
        self.training_args = TrainingArguments(
            output_dir="test",
            learning_rate=2e-5,
            per_device_train_batch_size=self.batch_size,
            per_device_eval_batch_size=self.batch_size,
            num_train_epochs=2,
            weight_decay=0.01,
            eval_strategy="epoch",
            save_strategy="epoch",
            load_best_model_at_end=True,
            push_to_hub=False,
            report_to="none",
        )
        model_name = "bert-base-uncased"
        self.model = BertForSequenceClassification.from_pretrained(model_name, num_labels=2, output_attentions=False, output_hidden_states=False)
        self.model.cuda()
        

    def preprocess_function(examples):
        examples["text"] = [str(text) for text in examples["text"]]
        tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        return tokenizer(examples["text"], truncation=True, padding="max_length", add_special_tokens=True)
    
    def compute_metrics(eval_pred):
        predictions, labels = eval_pred
        predictions = np.argmax(predictions, axis=-1)
        f1_metric = evaluate.load("f1")
        acc_metric = evaluate.load("accuracy")
        f1 = f1_metric.compute(predictions=predictions, references=labels)
        acc = acc_metric.compute(predictions=predictions, references=labels)
        return {"f1": f1, "accuracy": acc}
    
    def train(self):
        tokenized_train = self.train_dataset.map(BertBasedTrainer.preprocess_function, batched=True)
        tokenized_valid = self.valid_dataset.map(BertBasedTrainer.preprocess_function, batched=True)
        tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        self.trainer = Trainer(
            data_collator=DataCollatorWithPadding(tokenizer=tokenizer),
            model=self.model,
            args=self.training_args,
            train_dataset=tokenized_train,
            eval_dataset=tokenized_valid,
            processing_class=tokenizer,
            compute_metrics=BertBasedTrainer.compute_metrics,
        )
        self.trainer.train()

    def predict(self):
        tokenized_test = self.dev_dataset.map(BertBasedTrainer.preprocess_function, batched=True)
        predictions = self.trainer.predict(tokenized_test)

        y_pred = np.argmax(predictions.predictions, axis=1)
        y_true = tokenized_test['label']
        print(confusion_matrix(y_true, y_pred))
        f1_metric = evaluate.load("f1")
        results = f1_metric.compute(predictions=y_pred, references=y_true)
        print(results)