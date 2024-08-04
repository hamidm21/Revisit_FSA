from itertools import cycle
from rich.progress import Progress, TextColumn, BarColumn, TimeRemainingColumn
from sklearn.preprocessing import label_binarize
from transformers import AutoModelForSequenceClassification, AutoConfig, Trainer, TrainingArguments
from transformers.integrations import NeptuneCallback
from scipy.special import softmax
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, roc_auc_score, roc_curve, auc
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, mean_squared_error, mean_absolute_error
import torch
import torch.nn as nn
from torch.optim.lr_scheduler import LambdaLR
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
    def __init__(self, model_addr="ElKulako/cryptobert", save_path=f'./artifact/fine_tuned_model.pth', load_path=None, load_state_dict=False, input_task='classification'):
        super().__init__("huggingface ElKulako/cryptobert")
        self.model_addr = model_addr
        self.save_path = save_path
        self.load_path = load_path
        self.input_task = input_task
        self.metrics = {}  # Initialize the metrics dictionary
        self.labels = {}  # Initialize the labels dictionary
        self.preds = {}  # Initialize the predictions dictionary
        self.probs = {}  # Initialize the probabilities dictionary
        self.eval_labels = {}
        self.eval_preds = {}
        self.eval_probs = {}
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

    def predict(self, data):
        raise NotImplementedError("Subclasses must implement this method.")
    
    def save_model(self, path):
        # Create the output directory if it doesn't exist
        dir_name = os.path.dirname(path)
        if not os.path.exists(dir_name):
            os.makedirs(dir_name)
        torch.save(self.model.state_dict(), path)

    def load_model(self, path):
        model_state = torch.load(path)
        self.model.load_state_dict(model_state)
        return self.model

    def train(self, dataloader, device, optimizer, learning_rate=2e-5, model_name="train", neptune_run=None):
        """
        Train the model on the given data and labels.

        Args:
        dataloader (DataLoader): The DataLoader for the training data.
        device (torch.device): The device to train the model on.
        learning_rate (float): The learning rate for the optimizer.
        num_epochs (int): The number of epochs for training.
        num_folds (int): The number of folds for cross-validation.
        neptune_run (neptune.run.Run): The Neptune run instance.

        Returns:
        Tuple[List, List, List]: The labels, predictions, and probabilities for each batch.
        """
        all_labels = []
        all_preds = []
        all_probs = []  # For storing probabilities

        for batch in tqdm(dataloader, desc=f"Training Progress...", leave=False, dynamic_ncols=True):
            optimizer.zero_grad()
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)

            outputs = self.model(input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss
            loss.backward()
            optimizer.step()

            # Store labels, predictions and probabilities for metrics calculation
            preds = torch.nn.functional.softmax(outputs.logits, dim=-1)
            class_preds = torch.argmax(preds, dim=-1)

            all_probs.append(preds.detach().cpu().numpy())  # Store probabilities
            all_preds.append(class_preds.cpu().detach().numpy())
            all_labels.append(labels.cpu().detach().numpy())

            if neptune_run:
                # Log metrics to Neptune
                neptune_metrics = ["accuracy", "precision", "f1", "recall"]
                # Compute metrics
                metrics = self.compute_metrics_classification(np.concatenate(all_labels), np.concatenate(all_preds), np.concatenate(all_probs), neptune_metrics)
                for metric_name in neptune_metrics:
                    neptune_run[f"{model_name}/{metric_name}"].append(metrics.get(metric_name))

        return all_labels, all_preds, all_probs


    def evaluate(self, dataloader, device, model_name="base", neptune_run=None):
        """
        Evaluate the model on the given data and labels.

        Args:
        dataloader (DataLoader): The DataLoader for the evaluation data.
        device (torch.device): The device to evaluate the model on.

        Returns:
        Tuple[List, List, List]: The labels, predictions, and probabilities for each batch.
        """
        # Evaluation loop
        all_labels = []
        all_preds = []
        all_probs = []  # For storing probabilities

        for batch in tqdm(dataloader, desc="Evaluating Progress...", leave=False, dynamic_ncols=True):
            with torch.no_grad():
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                labels = batch['labels'].to(device)

                outputs = self.model(input_ids, attention_mask=attention_mask, labels=labels)

                # Get the predicted probabilities from the model's outputs
                preds = torch.nn.functional.softmax(outputs.logits, dim=-1)
                # Convert the probabilities to class labels
                class_preds = torch.argmax(preds, dim=-1)

                all_probs.append(preds.cpu().numpy())  # Store probabilities
                all_preds.append(class_preds.cpu().numpy())
                all_labels.append(labels.cpu().numpy())

            if neptune_run:
                # Log metrics to Neptune
                neptune_metrics = ["accuracy", "precision", "f1", "recall"]
                # Compute metrics
                metrics = self.compute_metrics_classification(np.concatenate(all_labels), np.concatenate(all_preds), np.concatenate(all_probs), neptune_metrics)
                for metric_name in neptune_metrics:
                    neptune_run[f"{model_name}/{metric_name}"].append(metrics.get(metric_name))


        return all_labels, all_preds, all_probs

    @staticmethod
    def compute_metrics_classification(labels, preds, probs, metrics_to_return=None):
        """
        Compute classification metrics based on the model's predictions and the true labels.

        Args:
        labels (any): The true labels.
        preds (any): The model's predictions.
        probs (any): The model's probabilities
        metrics_to_return (list): List of metric names to compute and return.

        Returns:
        dict: The computed classification metrics.
        """
        if metrics_to_return is None:
            metrics_to_return = ["accuracy", "f1", "precision", "recall", "roc_score", "confusion_matrix"]

        metrics = {}

        if "precision" in metrics_to_return or "recall" in metrics_to_return or "f1" in metrics_to_return:
            precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average='macro')
            if "precision" in metrics_to_return:
                metrics["precision"] = precision
            if "recall" in metrics_to_return:
                metrics["recall"] = recall
            if "f1" in metrics_to_return:
                metrics["f1"] = f1

        if "accuracy" in metrics_to_return:
            metrics["accuracy"] = accuracy_score(labels, preds)

        if "roc_score" in metrics_to_return:
            metrics["roc_score"] = roc_auc_score(labels, probs, multi_class='ovr')

        if "confusion_matrix" in metrics_to_return:
            metrics["confusion_matrix"] = confusion_matrix(labels, preds)

        return metrics


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

    @staticmethod
    def get_linear_schedule_with_warmup(optimizer, num_warmup_steps, num_training_steps):
        def lr_lambda(current_step):
            if current_step < num_warmup_steps:
                return float(current_step) / float(max(1, num_warmup_steps))
            return max(0.0, float(num_training_steps - current_step) / float(max(1, num_training_steps - num_warmup_steps)))
        return LambdaLR(optimizer, lr_lambda)


    @staticmethod
    def plot_confusion_matrix(path, labels, preds):
        """
        Plot the confusion matrix for the given labels and predictions.

        Args:
        output_dir (str): The directory to save the confusion matrix plot.
        labels (list): The true labels.
        preds (list): The predicted labels.

        Returns:
        None
        """
        # Create the output directory if it doesn't exist
        output_dir = os.path.dirname(path)
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        conf_matrix = confusion_matrix(labels, preds)
        disp = ConfusionMatrixDisplay(confusion_matrix=conf_matrix, display_labels=['Down', 'Neutral', 'Up'])
        fig, ax = plt.subplots(figsize=(10, 10))
        disp.plot(ax=ax, cmap='Blues', values_format='d')
        plt.title('Confusion Matrix')
        plt.savefig(path)
        plt.close()

    @staticmethod
    def plot_roc_curve(path, labels, probs):
        """
        Plot the ROC curve for the given labels and probabilities.

        Args:
        path (str): The path to save the ROC curve plots.
        labels (list): The true labels.
        probs (list): The predicted probabilities.

        Returns:
        None
        """
        # Create the output directory if it doesn't exist
        output_dir = os.path.dirname(path)
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        plt.figure()
        # Binarize the labels for multi-class ROC AUC
        labels = label_binarize(labels, classes=np.unique(labels))
        n_classes = labels.shape[1]

        # Compute ROC curve and ROC area for each class
        fpr = dict()
        tpr = dict()
        roc_auc = dict()
        for i in range(n_classes):
            fpr[i], tpr[i], _ = roc_curve(labels[:, i], probs[:, i])
            roc_auc[i] = auc(fpr[i], tpr[i])

        # Plot all ROC curves
        colors = cycle(['aqua', 'darkorange', 'cornflowerblue'])
        for i, color in zip(range(n_classes), colors):
            plt.plot(fpr[i], tpr[i], color=color, lw=2,
                     label='ROC curve of class {0} (area = {1:0.2f})'
                     ''.format(i, roc_auc[i]))

        plt.plot([0, 1], [0, 1], 'k--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver Operating Characteristic')
        plt.legend(loc="lower right")

        # Save the figure to the output directory with a unique name
        plt.savefig(path)
        plt.close()
