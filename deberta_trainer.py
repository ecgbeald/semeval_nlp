from transformers import DebertaV2Tokenizer, DistilBertTokenizer, DataCollatorWithPadding, TrainingArguments, DistilBertForSequenceClassification, DebertaV2ForSequenceClassification, Trainer
import numpy as np
import evaluate
import optuna
import copy
from transformers.trainer_utils import EvalPrediction, get_last_checkpoint
from sklearn.metrics import confusion_matrix
import time
import pickle

class DebertaTrainer:
    
    def __init__(self, train_dataset, valid_dataset, dev_dataset, test_dataset):
        self.batch_size = 6
        self.model_name = "/vol/bitbucket/rm521/cw/models/deberta-v3-large"
        self.tokenizer = DebertaV2Tokenizer.from_pretrained(self.model_name)
        self.train_dataset = train_dataset
        self.valid_dataset = valid_dataset
        self.dev_dataset = dev_dataset
        self.test_dataset = test_dataset
        self.timestr = time.strftime("%d_%m-%H%M%S")
        self.training_args = TrainingArguments(
            output_dir=f"checkpoints/{self.timestr}",
            learning_rate=2e-05,
            per_device_train_batch_size=self.batch_size,
            per_device_eval_batch_size=self.batch_size,
            num_train_epochs=4,
            bf16=True,
            weight_decay=0.003,
            eval_strategy="epoch",
            save_strategy="epoch",
            load_best_model_at_end=True,
            max_grad_norm=1.0,
            push_to_hub=False,
            report_to="none",
            seed=42,
            data_seed=42,
        )

    
    def get_tokenizer(self):
        return self.tokenizer
        
    def model_init(self):
        self.model = DebertaV2ForSequenceClassification.from_pretrained(
                self.model_name, 
                num_labels=2, 
                output_attentions=False, 
                output_hidden_states=False,
        )
        self.model.cuda()
        return self.model

    def preprocess_function(self, examples):    
        examples["text"] = [str(text) for text in examples["text"]]
        return self.tokenizer(examples["text"], truncation=True, padding="max_length", max_length=512, return_tensors="pt")
    
    def compute_metrics(eval_pred):
        predictions, labels = eval_pred
        predictions = np.argmax(predictions, axis=-1)
        f1_metric = evaluate.load("f1")
        acc_metric = evaluate.load("accuracy")
        f1 = f1_metric.compute(predictions=predictions, references=labels)['f1']
        acc = acc_metric.compute(predictions=predictions, references=labels)['accuracy']
        return {"f1": f1, "accuracy": acc}
    
    def set_up_trainer(self):
        tokenized_train = self.train_dataset.map(self.preprocess_function, batched=True)
        tokenized_valid = self.valid_dataset.map(self.preprocess_function, batched=True)
        self.model = DebertaV2ForSequenceClassification.from_pretrained(self.model_name, num_labels=2, output_attentions=False, output_hidden_states=False)
        self.model.cuda()
        self.trainer = Trainer(
            model_init=self.model_init,
            data_collator=DataCollatorWithPadding(tokenizer=self.tokenizer),
            # model=self.model,
            args=self.training_args,
            train_dataset=tokenized_train,
            eval_dataset=tokenized_valid,
            processing_class=self.tokenizer,
            compute_metrics=DebertaTrainer.compute_metrics,
        )

    def train(self):
        self.set_up_trainer()
        self.trainer.model_init
        self.trainer.train()
    
    # doesn't work
    # def apply_best_hyperparameters(self, best_run):
    #     self.trainer.apply_hyperparameters(best_run, final_model=True)
    #     self.trainer.train()
    #     tokenized_test = self.dev_dataset.map(self.preprocess_function, batched=True)
    #     metric = self.trainer.evaluate(tokenized_test)
    #     print(metric)
    #     self.trainer.model.save_pretrained("saved_model")

    def load_best(self):
        best_path = "/vol/bitbucket/rm521/cw/checkpoints/22_02-111214/run-4/checkpoint-1908"
        self.model = DebertaV2ForSequenceClassification.from_pretrained(best_path, num_labels=2, output_attentions=False, output_hidden_states=False)
        self.model.cuda()
        tokenized_train = self.train_dataset.map(self.preprocess_function, batched=True)
        tokenized_valid = self.valid_dataset.map(self.preprocess_function, batched=True)
        self.trainer = Trainer(
            # model_init=self.model_init,
            data_collator=DataCollatorWithPadding(tokenizer=self.tokenizer),
            model=self.model,
            args=self.training_args,
            train_dataset=tokenized_train,
            eval_dataset=tokenized_valid,
            processing_class=self.tokenizer,
            compute_metrics=DebertaTrainer.compute_metrics,
        )
        self.trainer.train(resume_from_checkpoint=best_path)

    def predict(self, save_model=True):
        tokenized_test = self.dev_dataset.map(self.preprocess_function, batched=True)
        eval_res = self.trainer.evaluate(tokenized_test)
        print(eval_res)
        if (eval_res['eval_f1'] > 0.55):
            predictions = self.trainer.predict(tokenized_test)
            
            y_pred = np.argmax(predictions.predictions, axis=1)
            y_true = tokenized_test['label']
            print(confusion_matrix(y_true, y_pred))
            f1_metric = evaluate.load("f1")
            results = f1_metric.compute(predictions=y_pred, references=y_true)
            print(results)
            
            if save_model:
                timestr = time.strftime('%d_%m-%H%M%S')
                self.trainer.save_model(f"good_runs/{timestr}")

                fname = f"predictions_{timestr}.pkl"
                
                tokenized_final = self.test_dataset.map(self.preprocess_function, batched=True)
                pred_final = self.trainer.predict(tokenized_final)

                results = {
                    "results": results,
                    "valid_predictions": predictions,
                    "final_predictions": pred_final
                }
            
                with open(fname, "wb") as f:
                    pickle.dump(results, f)

        # use this to get pure labels
        # predictions = self.trainer.predict(tokenized_test)

    def hyperparameter_search(self):
        self.set_up_trainer()
        return self.trainer.hyperparameter_search(
            direction="maximize",
            backend="optuna",
            hp_space=self.optuna_hp_space,
            n_trials=25,
            compute_objective=self.compute_objective,
        )

    def optuna_hp_space(self, trial: optuna.Trial):
        return {
            "learning_rate": trial.suggest_float("learning_rate", 1e-5, 4e-5, log=True),
            "per_device_train_batch_size": trial.suggest_categorical("per_device_train_batch_size", [6]),
            "num_train_epochs": trial.suggest_categorical("num_train_epochs", [2, 3, 4, 6]),
        }
    
    def compute_objective(self, metrics) -> float:
        metrics = copy.deepcopy(metrics)
        print("Metrics received:")
        print(metrics)
        print("Evaluate eval set:")
        self.predict(save_model=True)
        return metrics["eval_f1"]
