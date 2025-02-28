from transformers import DebertaV2Tokenizer, DebertaV2ForSequenceClassification
import evaluate
import torch
from torch.utils.data import DataLoader, TensorDataset

class DebertaUntuned:
    def __init__(self, dev_dataset):
        self.batch_size = 6
        self.model_name = "/vol/bitbucket/rm521/cw/models/deberta-v3-large"
        self.tokenizer = DebertaV2Tokenizer.from_pretrained(self.model_name)
        self.dev_dataset = dev_dataset
        self.model = DebertaV2ForSequenceClassification.from_pretrained(
                self.model_name, 
                num_labels=2, 
                output_attentions=False, 
                output_hidden_states=False,
        )
        tokenized_valid = self.dev_dataset.map(self.preprocess_function, batched=True)
        input_ids = torch.tensor(tokenized_valid["input_ids"])
        attention_mask = torch.tensor(tokenized_valid["attention_mask"])
        labels = torch.tensor(tokenized_valid["label"])
        self.tensor_dataset = TensorDataset(input_ids, attention_mask, labels)
        self.dataloader = DataLoader(self.tensor_dataset, batch_size=self.batch_size, shuffle=False)
        self.model.cuda()
        
    def compute_metrics(predictions, labels):
        f1_metric = evaluate.load("f1")
        acc_metric = evaluate.load("accuracy")
        confusion_metric = evaluate.load("confusion_matrix")
        f1 = f1_metric.compute(predictions=predictions, references=labels)['f1']
        acc = acc_metric.compute(predictions=predictions, references=labels)['accuracy']
        cm = confusion_metric.compute(predictions=predictions, references=labels)['confusion_matrix']
        return {"f1": f1, "accuracy": acc, "confusion_matrix": cm}
    
    def preprocess_function(self, examples):    
        examples["text"] = [str(text) for text in examples["text"]]
        return self.tokenizer(examples["text"], truncation=True, padding="max_length", max_length=512, return_tensors="pt")
    
    def eval(self):
        self.model.eval()
        predictions = []
        with torch.no_grad():
            for batch in self.dataloader:
                batch = [tensor.cuda() for tensor in batch]
                outputs = self.model(input_ids=batch[0], attention_mask=batch[1])
                logits = outputs.logits
                preds = torch.argmax(logits, dim=-1)
                predictions.extend(preds.cpu().numpy())
        print(DebertaUntuned.compute_metrics(predictions, self.dev_dataset['label']))
        
