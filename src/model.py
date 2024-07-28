from transformers import AutoModelForSequenceClassification, AutoConfig, Trainer, TrainingArguments
from transformers.integrations import NeptuneCallback
from scipy.special import softmax
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, roc_auc_score, roc_curve
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, mean_squared_error, mean_absolute_error
import torch
import torch.nn as nn
from typing import Optional
from tqdm.auto import tqdm
import numpy as np
import matplotlib.pyplot as plt
from type import Model
import os
from dotenv import load_dotenv


# Load environment variables from .env file
load_dotenv()

# Access the base address
base_address = os.getenv("BASE_ADDRESS")

class CryptoBERT(Model):
    def __init__(self, model_addr="ElKulako/cryptobert", save_path=f'./artifacts/fine_tuned_model.pth', load_path=None, load_state_dict=False, input_task='classification'):
        super().__init__("huggingface ElKulako/cryptobert")
        self.model_addr = model_addr
        self.save_path = save_path
        self.load_path = load_path
        self.input_task = input_task
        self.metrics = {}  # Initialize the metrics dictionary
        self.labels = {}  # Initialize the labels dictionary
        self.preds = {}  # Initialize the predictions dictionary
        self.probs = {}  # Initialize the probabilities dictionary
        
        # Load configuration
        config = AutoConfig.from_pretrained(model_addr)
        
        # Adjust configuration for regression task
        if input_task == "regression":
            config.num_labels = 1  # Adjust for regression task
        elif input_task == "classification":
            config.num_labels = 3  # Adjust for classification task
        
        # Load model with modified configuration
        if load_state_dict:
            self.model = AutoModelForSequenceClassification.from_pretrained(self.model_addr, config=config)
            self.model.load_state_dict(torch.load(self.load_path))
        else:
            self.model = AutoModelForSequenceClassification.from_pretrained(self.model_addr, config=config, ignore_mismatched_sizes=True)

    def train(self, dataloader, device, learning_rate=1e-5, epochs=5, neptune_run=None):
        """
        Train the model on the given data and labels.

        Args:
        data (any): The data to train the model on.
        labels (any): The labels for the data.
        """
        results = {}
        # Move the model to the device
        self.model.to(device)
        # Set up the optimizer
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=learning_rate)
        for epoch in tqdm(range(epochs)):  # Number of epochs
            all_labels = []
            all_preds = []
            all_probs = []  # For storing probabilities
            losses = []
            for batch in tqdm(dataloader):
                optimizer.zero_grad()
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                if self.input_task == "classification":
                    labels = batch['labels'].to(device)
                    outputs = self.model(input_ids, attention_mask=attention_mask, labels=labels)
                    loss = outputs.loss
                elif self.input_task == "regression":
                    labels = batch['labels'].to(device)  # Assuming true_range is provided in the batch
                    outputs = self.model(input_ids, attention_mask=attention_mask)
                    # Modify the loss function for regression task
                    loss = nn.MSELoss()(outputs.logits.squeeze(), labels.float())
                loss.backward()
                optimizer.step()

                # Store labels, predictions and probabilities for metrics calculation
                preds = torch.nn.functional.softmax(outputs.logits, dim=-1)
                losses.append(loss.item())
                all_probs.append(preds.detach().cpu().numpy())  # Store probabilities
                if self.input_task == "classification":
                    class_preds = torch.argmax(preds, dim=-1)
                elif self.input_task == "regression":
                    class_preds = outputs.logits.squeeze()  # For regression, use logits directly
                all_preds.append(class_preds.cpu().detach().numpy())
                all_labels.append(labels.cpu().detach().numpy())

            # Calculate and log metrics after each epoch
            all_labels = np.concatenate(all_labels)
            all_preds = np.concatenate(all_preds)
            all_probs = np.concatenate(all_probs)  # Concatenate probabilities
            self.labels[epoch] = all_labels
            self.preds[epoch] = all_preds
            self.probs[epoch] = all_probs
            if self.input_task == "classification":
                results[epoch] = self.compute_metrics_classification(all_labels, all_preds, all_probs, neptune_run)
            elif self.input_task == "regression":
                results[epoch] = self.compute_metrics_regression(all_labels, all_preds)

            # Log metrics to Neptune
            if neptune_run:
                for key, value in results[epoch].items():
                    neptune_run[f"train/{key}"].log(value)

            # Save the model after each epoch
            # torch.save(self.model.state_dict(), self.save_path)

        # metrics for each epoch
        return results

    def predict(self, data):
        raise NotImplementedError("Subclasses must implement this method.")
    
    def save_model(self, path):
        torch.save(self.model.state_dict(), path)
    
    def load_model(self, path):
        self.model = torch.load(path)

    def evaluate(self, dataloader, device, neptune_run=None):
        """
        Evaluate the model on the given data and labels.

        Args:
        data (any): The data to evaluate the model on.
        labels (any): The labels for the data.

        Returns:
        any: The evaluation results.
        """
        # Evaluation loop
        results = {}
        self.model.to(device)
        eval_loss = 0
        all_labels = []
        all_preds = []
        all_probs = []  # For storing probabilities
        for batch in tqdm(dataloader):
            with torch.no_grad():
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                if self.input_task == "classification":
                    labels = batch['labels'].to(device)
                    outputs = self.model(input_ids, attention_mask=attention_mask, labels=labels)
                    eval_loss += outputs.loss.item()
                    # Get the predicted probabilities from the model's outputs
                    preds = torch.nn.functional.softmax(outputs.logits, dim=-1)
                    # Convert the probabilities to class labels
                    class_preds = torch.argmax(preds, dim=-1)
                    all_probs.append(preds.cpu().numpy())  # Store probabilities
                elif self.input_task == "regression":
                    labels = batch['labels'].to(device)
                    outputs = self.model(input_ids, attention_mask=attention_mask)
                    preds = outputs.logits.squeeze()  # For regression, use logits directly
                all_preds.append(class_preds.cpu().numpy())
                all_labels.append(labels.cpu().numpy())

        # Calculate metrics
        all_labels = np.concatenate(all_labels)
        all_preds = np.concatenate(all_preds)
        if self.input_task == "classification":
            all_probs = np.concatenate(all_probs)  # Concatenate probabilities
            results = self.compute_metrics_classification(all_labels, all_preds, all_probs, neptune_run)
        elif self.input_task == "regression":
            results = self.compute_metrics_regression(all_labels, all_preds)

        # Log metrics to Neptune
        if neptune_run:
            for key, value in results.items():
                neptune_run[f"eval/{key}"].log(value)

        return results

    @staticmethod
    def compute_metrics_classification(labels, preds, probs, neptune_run=None):
        """
        Compute classification metrics based on the model's predictions and the true labels.

        Args:
        labels (any): The true labels.
        preds (any): The model's predictions.
        probs (any): The model's probabilities

        Returns:
        dict: The computed classification metrics.
        """
        precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average='macro')
        acc = accuracy_score(labels, preds)
        roc_auc = roc_auc_score(labels, probs[:, 1], multi_class='ovr')  # Assuming binary classification
        # Compute confusion matrix
        conf_matrix = confusion_matrix(labels, preds)

        # Create a dictionary of metrics
        metrics = {
            "accuracy": acc,
            "f1": f1,
            "precision": precision,
            "recall": recall,
            "roc_auc": roc_auc,
            "confusion_matrix": conf_matrix
        }

        return metrics

    @staticmethod
    def compute_metrics_regression(labels, preds):
        """
        Compute regression metrics based on the model's predictions and the true labels.

        Args:
        labels (any): The true labels.
        preds (any): The model's predictions.

        Returns:
        dict: The computed regression metrics.
        """
        mae = mean_absolute_error(labels, preds)
        mse = mean_squared_error(labels, preds)

        # Create a dictionary of metrics
        metrics = {
            "mean_absolute_error": mae,
            "mean_squared_error": mse
        }

        return metrics


    def get_trainer(self, eval_dataset, neptune_run: Optional[object] = None, train_dataset=None):

        def compute_metrics_regression(pred):
            labels = pred.label_ids
            preds = pred.predictions.squeeze()  # For regression, use predictions directly
            mae = mean_absolute_error(labels, preds)
            mse = mean_squared_error(labels, preds)
            return {
                'mean_absolute_error': mae,
                'mean_squared_error': mse
            }

        def compute_metrics_classification(pred):
            labels = pred.label_ids
            preds = pred.predictions.argmax(-1)
            probs = softmax(pred.predictions, axis=1)
            precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average='macro')
            acc = accuracy_score(labels, preds)
            roc_auc=0
            conf_matrix=0
            try:
                roc_auc = roc_auc_score(labels, probs, multi_class='ovr')
            except:
                pass
            try:
                conf_matrix = confusion_matrix(labels, preds)
                # Plot confusion matrix
                disp = ConfusionMatrixDisplay(confusion_matrix=conf_matrix, display_labels=['Down', 'Neutral', 'Up'])
                fig, ax = plt.subplots(figsize=(10, 10))
                disp.plot(ax=ax, cmap='Blues', values_format='d')
                plt.title('Confusion Matrix')
                # Save the confusion matrix to a file
                plt.savefig("./result/exp1/base_confusion_matrix.png")
                # Log the confusion matrix image to Neptune
                if neptune_run:
                    neptune_run["confusion_matrix"].upload("./result/exp1/base_confusion_matrix.png")
            except:
                pass
            return {
                'accuracy': acc,
                'f1': f1,
                'precision': precision,
                'recall': recall,
            }

        # Choose compute_metrics function based on the task type
        if self.input_task == "classification":
            compute_metrics_func = compute_metrics_classification
        elif self.input_task == "regression":
            compute_metrics_func = compute_metrics_regression

        # Define Trainer arguments
        trainer_args = TrainingArguments(
            output_dir=self.save_path,
            report_to="none"
        )

        # Create Trainer instance
        trainer = Trainer(
            model=self.model,                 # the non-fine-tuned model
            args=trainer_args,                # training arguments, defined above
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,        # test dataset
            compute_metrics=compute_metrics_func,   # the compute_metrics function
            )

        return trainer

    def plot_confusion_matrix(self, epoch):
        labels = self.labels[epoch]
        preds = self.preds[epoch]
        conf_matrix = confusion_matrix(labels, preds)
        disp = ConfusionMatrixDisplay(confusion_matrix=conf_matrix, display_labels=['Down', 'Neutral', 'Up'])
        fig, ax = plt.subplots(figsize=(10, 10))
        disp.plot(ax=ax, cmap='Blues', values_format='d')
        plt.title('Confusion Matrix')
        plt.savefig(f"./confusion_matrix_epoch_{epoch}.png")
        plt.close()

    def plot_roc_curve(self):
        plt.figure()
        for epoch in range(len(self.labels)):
            labels = self.labels[epoch]
            probs = self.probs[epoch]
            roc_auc = roc_auc_score(labels, probs[:, 1])  # Assuming binary classification
            fpr, tpr, _ = roc_curve(labels, probs[:, 1])
            plt.plot(fpr, tpr, label='ROC curve for epoch %d (area = %0.2f)' % (epoch, roc_auc))
        plt.plot([0, 1], [0, 1], 'k--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver Operating Characteristic')
        plt.legend(loc="lower right")
        plt.savefig('roc_curves.png')
        plt.close()
