from itertools import cycle
from sklearn.preprocessing import label_binarize
from transformers import AutoModelForSequenceClassification, AutoConfig, Trainer, TrainingArguments
from transformers.integrations import NeptuneCallback
from scipy.special import softmax
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, roc_auc_score, roc_curve, auc
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

    def train(self, dataloader, device, optimizer, learning_rate=1e-5):
        """
        Train the model on the given data and labels.

        Args:
        dataloader (DataLoader): The DataLoader for the training data.
        device (torch.device): The device to train the model on.
        learning_rate (float): The learning rate for the optimizer.
        
        Returns:
        Tuple[List, List, List]: The labels, predictions, and probabilities for each batch.
        """
        all_labels = []
        all_preds = []
        all_probs = []  # For storing probabilities

        for batch in tqdm(dataloader, desc="Training Progress"):
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

        return all_labels, all_preds, all_probs


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

    def evaluate(self, dataloader, device):
        """
        Evaluate the model on the given data and labels.

        Args:
        dataloader (DataLoader): The DataLoader for the evaluation data.
        device (torch.device): The device to evaluate the model on.

        Returns:
        Tuple[List, List, List]: The labels, predictions, and probabilities for each batch.
        """
        # Evaluation loop
        self.model.to(device)
        all_labels = []
        all_preds = []
        all_probs = []  # For storing probabilities

        for batch in tqdm(dataloader, desc="Evaluation Progress"):
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

        return all_labels, all_preds, all_probs


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
        print(probs.shape, labels.shape)
        roc_auc = roc_auc_score(labels, probs, multi_class='ovr')  # Assuming binary classification
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

    def plot_roc_curve(self, output_dir):
        # Create the output directory if it doesn't exist
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        for epoch in range(len(self.labels)):
            plt.figure()
            # Binarize the labels for multi-class ROC AUC
            labels = label_binarize(self.labels[epoch], classes=np.unique(np.concatenate(list(self.labels.values()))))
            probs = self.probs[epoch]
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
            plt.title('Receiver Operating Characteristic for Epoch {0}'.format(epoch))
            plt.legend(loc="lower right")

            # Save the figure to the output directory with a unique name
            plt.savefig(os.path.join(output_dir, 'roc_curve_epoch_{0}.png'.format(epoch)))
            plt.close()
