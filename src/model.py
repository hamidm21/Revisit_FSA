from itertools import cycle
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
import matplotlib.pyplot as plt
from type import Model
import os
from dotenv import load_dotenv

import numpy as np
import pandas as pd
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.regularizers import l2
from sklearn.metrics import precision_score, recall_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.utils import resample
from ta.momentum import RSIIndicator


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

    def train(self, dataloader, device, optimizer, scheduler, learning_rate=2e-5, model_name="train", neptune_run=None):
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
        Tuple[List, List, List, List]: The labels, predictions, probabilities, and losses for each batch.
        """
        all_labels = []
        all_preds = []
        all_probs = []
        all_losses = []

        for batch in tqdm(dataloader, desc=f"Training Progress...", leave=False, dynamic_ncols=True):
            optimizer.zero_grad()
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)

            outputs = self.model(input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss
            loss.backward()
            optimizer.step()
            scheduler.step()

            # Store labels, predictions and probabilities for metrics calculation
            preds = torch.nn.functional.softmax(outputs.logits, dim=-1)
            class_preds = torch.argmax(preds, dim=-1)

            all_probs.append(preds.detach().cpu().numpy())  # Store probabilities
            all_preds.append(class_preds.cpu().detach().numpy())
            all_labels.append(labels.cpu().detach().numpy())
            all_losses.append(loss.item())

            if neptune_run:
                # Log metrics to Neptune
                neptune_metrics = ["accuracy", "precision", "f1", "recall"]
                # Compute metrics
                metrics = self.compute_metrics_classification(np.concatenate(all_labels), np.concatenate(all_preds), np.concatenate(all_probs), neptune_metrics)
                for metric_name in neptune_metrics:
                    neptune_run[f"{model_name}/{metric_name}"].append(metrics.get(metric_name))
                neptune_run[f"{model_name}/loss"].append(loss.item())

        return all_labels, all_preds, all_probs, all_losses


    def evaluate(self, dataloader, device, model_name="base", neptune_run=None):
        """
        Evaluate the model on the given data and labels.

        Args:
        dataloader (DataLoader): The DataLoader for the evaluation data.
        device (torch.device): The device to evaluate the model on.

        Returns:
        Tuple[List, List, List, list]: The labels, predictions, probabilities, losses for each batch.
        """
        # Evaluation loop
        all_labels = []
        all_preds = []
        all_probs = []
        all_losses = []

        for batch in tqdm(dataloader, desc="Evaluating Progress...", leave=False, dynamic_ncols=True):
            with torch.no_grad():
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                labels = batch['labels'].to(device)

                outputs = self.model(input_ids, attention_mask=attention_mask, labels=labels)
                loss = outputs.loss

                # Get the predicted probabilities from the model's outputs
                preds = torch.nn.functional.softmax(outputs.logits, dim=-1)
                # Convert the probabilities to class labels
                class_preds = torch.argmax(preds, dim=-1)

                all_probs.append(preds.cpu().numpy())  # Store probabilities
                all_preds.append(class_preds.cpu().numpy())
                all_labels.append(labels.cpu().numpy())
                all_losses.append(loss.item())

            if neptune_run:
                # Log metrics to Neptune
                neptune_metrics = ["accuracy", "precision", "f1", "recall"]
                # Compute metrics
                metrics = self.compute_metrics_classification(np.concatenate(all_labels), np.concatenate(all_preds), np.concatenate(all_probs), neptune_metrics)
                for metric_name in neptune_metrics:
                    neptune_run[f"{model_name}/{metric_name}"].append(metrics.get(metric_name))
                neptune_run[f"{model_name}/loss"].append(loss.item())

        return all_labels, all_preds, all_probs, all_losses

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


class LSTMModel(Model):
    def __init__(self, name, average_duration, test_size=0.2, batch_size=64, epochs=200, learning_rate=0.001):
        super().__init__(name)
        self.average_duration = average_duration
        self.test_size = test_size
        self.batch_size = batch_size
        self.epochs = epochs
        self.learning_rate = learning_rate
        self.model = None
        self.scaler = StandardScaler()

    def preprocess_data(self, prices):
        """
        Preprocess the data: calculate ROC, RSI, Momentum, normalize features, and split into train/test sets.

        Args:
        prices (DataFrame): The dataframe containing price and label information.

        Returns:
        tuple: Training and test datasets along with their labels.
        """
        # Calculate ROC, RSI, and Momentum
        prices['ROC'] = prices['close'].pct_change()
        rsi_indicator = RSIIndicator(close=prices['close'], window=self.average_duration)
        prices['RSI'] = rsi_indicator.rsi()
        prices['Momentum'] = prices['close'].diff(periods=self.average_duration)

        # Shift labels to create y_true and drop NaN rows
        prices['y_true'] = prices['labels'].shift(-1).dropna()
        prices = prices.dropna()

        # Define features and labels
        features = ['ROC', 'RSI', 'Momentum', 'rise_over_trend', 'previous_window_trend']
        labels = prices['y_true'].astype(int)

        # Normalize features
        prices[features] = self.scaler.fit_transform(prices[features])

        # Split the data
        X_train, X_test, y_train, y_test = train_test_split(prices[features].values, labels.values, test_size=self.test_size, shuffle=False)

        # Reshape for LSTM
        X_train = X_train.reshape((X_train.shape[0], 1, X_train.shape[1]))
        X_test = X_test.reshape((X_test.shape[0], 1, X_test.shape[1]))

        return X_train, X_test, y_train, y_test

    def balance_data(self, X_train, y_train):
        """
        Balance the training data using oversampling.

        Args:
        X_train (array): The training feature data.
        y_train (array): The training labels.

        Returns:
        tuple: Balanced training data and labels.
        """
        # Combine features and labels into a single DataFrame for balancing
        train_data = np.hstack((X_train.reshape(X_train.shape[0], X_train.shape[2]), y_train.reshape(-1, 1)))
        train_df = pd.DataFrame(train_data, columns=['ROC', 'RSI', 'Momentum', 'rise_over_trend', 'previous_window_trend', 'y_true'])

        # Separate classes and oversample
        class_0 = train_df[train_df['y_true'] == 0]
        class_1 = train_df[train_df['y_true'] == 1]
        class_2 = train_df[train_df['y_true'] == 2]

        class_1_over = resample(class_1, replace=True, n_samples=len(class_0), random_state=42)
        class_2_over = resample(class_2, replace=True, n_samples=len(class_0), random_state=42)

        # Combine the oversampled classes
        balanced_train_df = pd.concat([class_0, class_1_over, class_2_over])

        # Sort by index to maintain order and separate features and labels
        balanced_train_df.sort_index(inplace=True)
        X_train_balanced = balanced_train_df.iloc[:, :-1].values.reshape(-1, 1, len(balanced_train_df.columns) - 1)
        y_train_balanced = to_categorical(balanced_train_df['y_true'].values, num_classes=3)

        return X_train_balanced, y_train_balanced

    def build_model(self, input_shape):
        """
        Build and compile the LSTM model.

        Args:
        input_shape (tuple): The shape of the input data.
        """
        self.model = Sequential([
            LSTM(64, input_shape=input_shape, return_sequences=True),
            Dropout(0.3),
            LSTM(32, return_sequences=False),
            Dropout(0.3),
            Dense(32, activation='relu', kernel_regularizer=l2(0.001)),
            Dense(3, activation='softmax')
        ])
        self.model.compile(loss='categorical_crossentropy',
                           optimizer=Adam(learning_rate=self.learning_rate),
                           metrics=['accuracy'])

    def train(self, X_train, y_train, X_val, y_val):
        """
        Train the LSTM model with early stopping.

        Args:
        X_train (array): The training feature data.
        y_train (array): The training labels.
        X_val (array): The validation feature data.
        y_val (array): The validation labels.
        """
        early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
        self.history = self.model.fit(X_train, y_train, epochs=self.epochs, batch_size=self.batch_size, 
                                      validation_data=(X_val, y_val), callbacks=[early_stopping], verbose=1)

    def predict(self, X_test):
        """
        Predict using the trained model.

        Args:
        X_test (array): The test feature data.

        Returns:
        array: The predicted labels.
        """
        y_pred = self.model.predict(X_test)
        return np.argmax(y_pred, axis=1)

    def evaluate(self, X_test, y_test):
        """
        Evaluate the model performance on the test data.

        Args:
        X_test (array): The test feature data.
        y_test (array): The true labels for the test data.

        Returns:
        float: The accuracy of the model.
        """
        loss, accuracy = self.model.evaluate(X_test, y_test, verbose=0)
        print(f'Test Accuracy: {accuracy:.4f}')
        return accuracy

    def compute_metrics(self, y_pred, y_true):
        """
        Compute accuracy, precision, recall, and F1 score.

        Args:
        y_pred (array): The predicted labels.
        y_true (array): The true labels.

        Returns:
        dict: The computed metrics.
        """
        accuracy = accuracy_score(y_true, y_pred)
        f1 = f1_score(y_true, y_pred, average='weighted')
        recall = recall_score(y_true, y_pred, average='weighted')
        precision = precision_score(y_true, y_pred, average='weighted')

        print(f"Accuracy: {accuracy}")
        print(f"F1 Score: {f1}")
        print(f"Recall: {recall}")
        print(f"Precision: {precision}")

        return {
            'accuracy': accuracy,
            'f1_score': f1,
            'recall': recall,
            'precision': precision
        }
