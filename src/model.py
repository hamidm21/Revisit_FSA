from transformers import AutoModelForSequenceClassification, Trainer, TrainingArguments, AdamW
from scipy.special import softmax
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, roc_auc_score
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import torch
from tqdm import tqdm
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
    def __init__(self, model_addr="ElKulako/cryptobert", save_path=f'{base_address}/artifacts/fine_tuned_model.pth', load_path=None, load_state_dict=False):
        super().__init__("huggingface ElKulako/cryptobert")
        self.model_addr = model_addr
        self.save_path = save_path
        self.load_path = load_path
        
        if load_state_dict:
            self.model = AutoModelForSequenceClassification.from_pretrained(self.model_addr, num_labels=3)
            self.model.load_state_dict(torch.load(self.load_path))
        else:
            self.model = AutoModelForSequenceClassification.from_pretrained(self.model_addr, num_labels=3)

    def train(self, dataloader, device, learning_rate = 1e-5, epochs = 5):
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
        optimizer = AdamW(self.model.parameters(), lr=learning_rate)
        for epoch in tqdm(range(epochs)):  # Number of epochs
            all_labels = []
            all_preds = []
            all_probs = []  # For storing probabilities
            losses = []
            for batch in tqdm(dataloader):
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
                losses.append(loss.item())
                all_probs.append(preds.detach().cpu().numpy())  # Store probabilities
                class_preds = torch.argmax(preds, dim=-1)
                all_preds.append(class_preds.cpu().numpy())
                all_labels.append(labels.cpu().numpy())

            # Calculate and log metrics after each epoch
            all_labels = np.concatenate(all_labels)
            all_preds = np.concatenate(all_preds)
            all_probs = np.concatenate(all_probs)  # Concatenate probabilities
            results[epoch] = self.compute_metrics(all_labels, all_preds, all_probs)
            
            # Save the model after each epoch
            torch.save(self.model.state_dict(), self.save_path)

        # metrics for each epoch
        return results


    def predict(self, data):
        raise NotImplementedError("Subclasses must implement this method.")

    def evaluate(self, dataloader, device):
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
        eval_loss = 0
        all_labels = []
        all_preds = []
        all_probs = []  # For storing probabilities
        for batch in tqdm(dataloader):
            with torch.no_grad():
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                labels = batch['labels'].to(device)
                outputs = self.model(input_ids, attention_mask=attention_mask, labels=labels)
                eval_loss += outputs.loss.item()
                all_labels.append(labels.cpu().numpy())
                # Get the predicted probabilities from the model's outputs
                preds = torch.nn.functional.softmax(outputs.logits, dim=-1)
                all_probs.append(preds.cpu().numpy())  # Store probabilities
                # Convert the probabilities to class labels
                class_preds = torch.argmax(preds, dim=-1)
                all_preds.append(class_preds.cpu().numpy())
        
        # Calculate metrics
        all_labels = np.concatenate(all_labels)
        all_preds = np.concatenate(all_preds)
        all_probs = np.concatenate(all_probs)  # Concatenate probabilities
    
        results = self.compute_metrics(all_labels, all_preds, all_probs)

        return results

    @staticmethod
    def compute_metrics(labels, preds, probs):
        """
        Compute metrics based on the model's predictions and the true labels.

        Args:
        labels (any): The true labels.
        preds (any): The model's predictions.
        probs (any): The model's probabilities

        Returns:
        dict: The computed metrics.
        """
        precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average='macro')
        acc = accuracy_score(labels, preds)
        
        # Compute confusion matrix
        conf_matrix = confusion_matrix(labels, preds)

        # Create a dictionary of metrics
        metrics = {
            "accuracy": acc,
            "f1": f1,
            "precision": precision,
            "recall": recall,
            'confusion_matrix': conf_matrix
        }

        return metrics


    def get_trainer(self, eval_dataset):
        
        def compute_metrics(pred):    
            labels = pred.label_ids
            preds = pred.predictions.argmax(-1)
            probs = softmax(pred.predictions, axis=1)
            precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average='macro')
            acc = accuracy_score(labels, preds)
            roc_auc = roc_auc_score(labels, probs, multi_class='ovr')

            # Compute confusion matrix
            conf_matrix = confusion_matrix(labels, preds)

            # Plot confusion matrix
            # disp = ConfusionMatrixDisplay(confusion_matrix=conf_matrix, display_labels=['Up', 'Neutral', 'Down'])
            # disp.plot(cmap='Blues', values_format='d')
            # plt.title('Confusion Matrix')
            # plt.show()

            return {
                'accuracy': acc,
                'f1': f1,
                'precision': precision,
                'recall': recall,
                'roc_auc': roc_auc,
                'confusion_matrix': conf_matrix
            }

        trainer_args = TrainingArguments(
            output_dir=self.save_path,
        )
        trainer = Trainer(
            model=self.model,                 # the non-fine-tuned model
            args=trainer_args,                # training arguments, defined above
            eval_dataset=eval_dataset,        # test dataset
            compute_metrics=compute_metrics,   # the compute_metrics function
            callbacks=[]
        )

        return trainer