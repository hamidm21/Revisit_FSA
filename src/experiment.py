import datetime
from datetime import timedelta

from tqdm.auto import tqdm
import numpy as np
import pandas as pd
from transformers import (
    AutoTokenizer,
    AutoModel,
)
from tqdm import tqdm
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from scipy.special import softmax
import torch
from torch.utils.data import DataLoader
from datasets import Dataset, ClassLabel


# internal imports
from type import Experiment
from model import CryptoBERT
from labeler import TripleBarrierLabeler, TrueRangeLabeler
from dataset import HFDataset, TextDataset
from util import *
from functools import partial
import os
from dotenv import load_dotenv


# Load environment variables from .env file
load_dotenv()

# Access the base address
base_addr = os.getenv("BASE_ADDRESS")


class SentimentLabelingExperiment(Experiment):
    def __init__(self, logger=None, data_addr='./raw/combined_2015_to_2021.csv'):
        super().__init__(
        id=0,
        description="labeling textual data based on the base sentiment model",
        logger=logger,
        base_addr=base_addr,
        model=CryptoBERT())
        self.text_df_addr = data_addr  # Assuming data_addr is the address to the textual data
        self.tokenizer = AutoTokenizer.from_pretrained("ElKulako/cryptobert")

    def run(self):
        self.start_time = datetime.datetime.now()
        self.logger.info(f"started experiment at {self.start_time}")

        # Load the textual data
        text_df = self.load_textual_data()
        text_df = text_df

        # Initialize an empty list to store the sentiment labels
        sentiment_labels = []

        # Iterate over the tokenized inputs
        self.logger.info(f"labeling...")
        for tweet in tqdm(text_df['text']):
            inputs = self.tokenizer(tweet, return_tensors='pt', padding=True, truncation=True)
            # Get the model's prediction
            with torch.no_grad():
                outputs = self.model.model(**inputs)

            # Get the predicted class (sentiment label)
            _, predicted_class = torch.max(outputs.logits, dim=1)
            sentiment_labels.append(predicted_class.item())

        # Add the sentiment labels as a new column in the DataFrame
        text_df['sentiment_label'] = sentiment_labels

        # Save the DataFrame with the sentiment labels
        text_df.to_csv(f'./raw/labeled_tweets.csv')

        self.logger.info('Sentiment labeling completed and results saved.')

        self.end_time = datetime.datetime.now()
        return self.results

    def load_textual_data(self) -> pd.DataFrame:
        text_df = pd.read_csv(self.text_df_addr, usecols=["date", "text_split"])
        text_df.rename(columns={"text_split": "text"}, inplace=True)
        text_df.set_index("date", inplace=True)
        text_df.index = pd.to_datetime(text_df.index)
        return text_df

class DirectionSplitTBL(Experiment):
    def __init__(
        self,
        num_samples=60000,
        price_df_addr="raw/daily-2020.csv",
        text_df_addr="raw/combined_tweets_2020_labeled.csv",
        logger=None
    ):
        super().__init__(
            id=1,
            base_addr=base_addr,
            model=CryptoBERT(),
            logger=logger,
            description="""
                comparing base cryptoBERT model to finetuned cryptoBERT on impact direction labelings
                """,
        )

        self.price_df_addr = price_df_addr
        self.text_df_addr = text_df_addr
        self.num_samples = num_samples
        # hard code essentials
        self.labeler = TripleBarrierLabeler(volatility_period=10, upper_barrier_factor=1.5, lower_barrier_factor=1.2, vertical_barrier=7, min_trend_days=2, barrier_type='volatility')

        self.results = {}

    def run(self):
        # constants
        params = {
            "samples": self.num_samples,
            "SEED":42,
            "TRAIN_TEST_SPLIT":0.2,
            "TRAINING_BATCH_SIZE":5,
            "EPOCHS":5,
            "LEARNING_RATE":1e-5,
        }

        self.start_time = datetime.datetime.now()
        self.logger.info(f"started experiment at {self.start_time}")

        # loading and labeling the data
        self.logger.info(f"loading and labeling the data...")
        text_df = self.load_textual_data()
        price_df = self.load_price_data()
        self.labeler.fit(price_df)
        triple_barrier_labels = self.labeler.transform()
        triple_barrier_labels["label"] = triple_barrier_labels["label"].shift(-1)
        labeled_texts = text_df.merge(
            triple_barrier_labels[["label"]], left_index=True, right_index=True, how="left"
        )
        labeled_texts.dropna(inplace=True)

        # undersample labels
        labeled_texts = self.undersample_tweets(labeled_texts)

        # creating a huggingface dataset for base model evaluation
        self.logger.info(f"creating and tokenizing the dataset...")
        labeled_texts = HFDataset.from_pandas(labeled_texts[["text", "label"]])
        # preprocess the text column
        self.logger.info(f"preprocessing the dataset...")
        #labeled_texts = HFDataset.preprocess(labeled_texts)

        self.logger.info(f"slicing and spliting the dataset...")
        # Spliting the dataset for evaluation

        self.logger.info(f"changing the label type of the dataset...")
        labeled_texts = labeled_texts.shuffle()
        #labeled_texts = labeled_texts.select(range(self.num_samples))
        labeled_texts = labeled_texts.class_encode_column('label')
        labeled_texts = labeled_texts.train_test_split(params["TRAIN_TEST_SPLIT"], seed=42)
        self.logger.info(labeled_texts)

        # tokenizing the dataset text to be used in train and test loops
        tokenizer = AutoTokenizer.from_pretrained("ElKulako/cryptobert")
        labeled_texts = HFDataset.tokenize(
            tokenizer, labeled_texts
        )

        device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        neptune_run = self.init_neptune_run("#1.1", description="evaluating the base model without fintuning", params=params)
        trainer = self.model.get_trainer(labeled_texts["test"], neptune_run=neptune_run)
        self.logger.info(f"evaluating the base model without fintuning...")
        non_fine_tuned_eval_result = trainer.evaluate()
        # Log metrics
        self.results["base"] = {}
        for key, value in non_fine_tuned_eval_result.items():
            self.results["base"][key] = value
            neptune_run[f"eval/{key}"].append(value)

        neptune_run.stop()

        self.logger.info(f"preparing data for finetuning the model...")
        train_dataset = TextDataset(labeled_texts['train'])
        test_dataset = TextDataset(labeled_texts['test'])

        train_dataloader = DataLoader(train_dataset, batch_size=params["TRAINING_BATCH_SIZE"])
        test_dataloader = DataLoader(test_dataset, batch_size=params["TRAINING_BATCH_SIZE"])

        self.logger.info(f"training the model...")
        neptune_run = self.init_neptune_run("#1.2", description="finetuning the base model on impact labels", params=params)
        train_metrics = self.model.train(dataloader=train_dataloader, device=device, learning_rate=params["LEARNING_RATE"], epochs=params["EPOCHS"], neptune_run=neptune_run)
        self.results["train"] = train_metrics
        self.model.save_model("./trained.pth")
        neptune_run.stop()

        self.logger.info(f"evaluating the finetuned model...")
        neptune_run = self.init_neptune_run("#1.3", description="evaluating the base model without fintuning", params=params)
        eval_metrics = self.model.evaluate(dataloader=test_dataloader, device=device, neptune_run=neptune_run)
        self.results["eval"] = eval_metrics
        neptune_run.stop()

        self.end_time = datetime.datetime.now()
        return self.results

    def load_textual_data(self) -> pd.DataFrame:
        """
        returns text_df and price_df in raw format
        """
        text_df = pd.read_csv(self.text_df_addr, usecols=["date", "text_split"])
        text_df.rename(columns={"text_split": "text"}, inplace=True)
        text_df.set_index("date", inplace=True)
        text_df.index = pd.to_datetime(text_df.index)

        return text_df

    def load_price_data(self) -> pd.DataFrame:
        price_df = pd.read_csv(
            self.price_df_addr,
            usecols=["timestamp", "close", "open", "high", "low", "volume"],
        )
        price_df.set_index("timestamp", inplace=True)
        price_df.index = pd.to_datetime(price_df.index, unit='s')

        return price_df

    @staticmethod
    def undersample_tweets(df):
        # Count the number of tweets for each trend
        trend_counts = df['label'].value_counts()
        
        # Identify the minority class
        minority_class = trend_counts.idxmin()
        minority_count = trend_counts.min()
        
        # Initialize an empty DataFrame to store the undersampled data
        undersampled_df = pd.DataFrame()
        
        # Iterate over the trends
        for trend in df['label'].unique():
            # If this is the minority class, add all tweets to the undersampled data
            if trend == minority_class:
                undersampled_df = pd.concat([undersampled_df, df[df['label'] == trend]])
            else:
                # Otherwise, randomly select a subset of tweets equal to the minority count
                subset = df[df['label'] == trend].sample(minority_count)
                undersampled_df = pd.concat([undersampled_df, subset])
        
        return undersampled_df

class DirectionSplitSentiment(Experiment):
    def __init__(
        self,
        num_samples=100,
        tweets_dataset_addr = 'raw/st-data-full.csv',
        logger=None
        ):
        super().__init__(
            id=2,
            base_addr=base_addr,
            logger=logger,
            description="""
                comparing base cryptoBERT model to finetuned cryptoBERT on sentiment labels
            """
        )
        self.num_samples = num_samples
        self.tweets_dataset_addr = tweets_dataset_addr
        self.results = {}
        
    def run(self):
        self.start_time = datetime.datetime.now()
        self.logger.info(f"started experiment at {self.start_time}")
        
        # Load the data
        self.logger.info(f"loading the data...")
        tweets_df = self.load_data()
        
        # Create a HuggingFace dataset
        self.logger.info(f"creating and tokenizing the dataset...")
        tweets_dataset = HFDataset.from_pandas(tweets_df)
        
        # Tokenize the text field in the dataset
        def tokenize_function(tokenizer, examples):
            # Tokenize the text and return only the necessary fields
            encoded = tokenizer(examples["text"], padding='max_length', max_length=512)
            return {"input_ids": encoded["input_ids"], "attention_mask": encoded["attention_mask"], "label": examples["label"]}
        
        # tokenizing the dataset text to be used in train and test loops
        tokenizer = AutoTokenizer.from_pretrained("ElKulako/cryptobert")
        partial_tokenize_function = partial(tokenize_function, tokenizer)
        
        # Tokenize the text in the datasets
        tokenized_dataset = tweets_dataset.map(partial_tokenize_function, batched=True)
        
        # Load the base model
        base_model = CryptoBERT(save_path=f'{base_addr}/artifacts/base_model_DSS_eval.pth')
        
        load_path = base_addr + '/artifacts/fine_tuned_model.pth'
        fine_tuned_model = CryptoBERT(load_state_dict=True, load_path=load_path, save_path=f'{base_addr}/artifacts/fine_tuned_model_DSS_eval.pth')
        
        # Prepare the base model for evaluation
        base_model_trainer = base_model.get_trainer(tokenized_dataset)
        
        self.logger.info(f'evaluating the base model...')
        # Evaluate the base model
        base_model_eval_result = base_model_trainer.evaluate()    
        
        # Log metrics
        self.results["base_model_DSS"] = {}
        for key, value in base_model_eval_result.items():
            self.results["base_model_DSS"][key] = value
            
        # Prepare the fine-tuned model for evaluation
        fine_tuned_model_trainer = fine_tuned_model.get_trainer(tokenized_dataset)
        
        self.logger.info(f'evaluating the fine-tuned model...')
        # Evaluate the fine-tuned model
        fine_tuned_model_eval_result = fine_tuned_model_trainer.evaluate()    
        
        # Log metrics
        self.results["fine_tuned_model_DSS"] = {}
        for key, value in fine_tuned_model_eval_result.items():
            self.results["fine_tuned_model_DSS"][key] = value
            
        self.end_time = datetime.datetime.now()
        return self.results

    
    def load_data(self):
        """
        Load the data from the given address.
        """
        tweets_df = pd.read_csv(self.tweets_dataset_addr)
        tweets_df = tweets_df.sample(n=self.num_samples, random_state=42)
        tweets_df = tweets_df[["text", "label"]]
        return tweets_df

class DirectionCrossValidate(Experiment):
    def __init__(self):
        super().__init__(
            id=3,
            base_addr=base_addr,
            description="""
                crossvalidating the base and finetuned model on impact direction labelings
            """
        )

class IntensitySplit(Experiment):
    def __init__(
        self,
        num_samples=100,
        price_df_addr="raw/daily-2020.csv",
        text_df_addr="raw/combined_tweets_2020_labeled.csv",
        logger=None
        ):
        super().__init__(
            id=4,
            base_addr=base_addr,
            model=None,
            logger=logger,
            description="""
                finetuning cryptoBERT on impact intensity
            """
        )
        self.num_samples = num_samples
        self.price_df_addr = price_df_addr
        self.text_df_addr = text_df_addr
        self.results = {}
        
    def run(self):
        SEED=42
        TRAIN_TEST_SPLIT=0.2
        BATCH_SIZE=5
        EPOCHS=3
        LEARNING_RATE=1e-5
        
        self.start_time = datetime.datetime.now()
        self.logger.info(f"started experiment at {self.start_time}")
        
        # Load and label the data
        self.logger.info(f"loading the data...")
        text_df, price_df = self.load_data()
        self.labeler = TrueRangeLabeler(price_df)
        true_range_data = self.labeler.transform()
        labeled_df = text_df.merge(true_range_data[['label']], left_index=True, right_index=True, how='left')
        
        # Drop rows with NaN labels (corresponding to the last day)
        labeled_df = labeled_df.dropna()
                
        # creating a hugging face dataset for base model evaluation
        self.logger.info(f"creating and tokenizing the dataset...")

        # Split the dataset into training and testing subsets with stratification
        # train_df, test_df = train_test_split(labeled_df, test_size=TRAIN_TEST_SPLIT, random_state=SEED, stratify=labeled_df['label'])
        train_df, test_df = train_test_split(labeled_df, test_size=TRAIN_TEST_SPLIT, random_state=SEED)
        
        # Create Dataset objects from the split dataframes
        train_dataset = Dataset.from_pandas(train_df[['text', 'label']])
        test_dataset = Dataset.from_pandas(test_df[['text', 'label']])

        # Tokenize the text field in the dataset
        def tokenize_function(tokenizer, examples):
            # Tokenize the text and return only the necessary fields
            encoded = tokenizer(examples["text"], padding='max_length', max_length=512)
            return {"input_ids": encoded["input_ids"], "attention_mask": encoded["attention_mask"], "label": examples["label"]}
        
        # tokenizing the dataset text to be used in train and test loops
        tokenizer = AutoTokenizer.from_pretrained("ElKulako/cryptobert")
        partial_tokenize_function = partial(tokenize_function, tokenizer)
        
        # Tokenize the text in the datasets
        tokenized_train_dataset = train_dataset.map(partial_tokenize_function, batched=True)
        tokenized_test_dataset = test_dataset.map(partial_tokenize_function, batched=True)
        
        tokenized_train_dataset = tokenized_train_dataset.select(range(8 * self.num_samples))
        tokenized_test_dataset = tokenized_test_dataset.select(range(int(2 * self.num_samples)))
        
        # 5. Evaluation of Base CryptoBERT Model
        base_model = CryptoBERT(input_task='regression')
        
        # Remove the '__index_level_0__' column from the dataset
        if '__index_level_0__' in tokenized_train_dataset.column_names:
            tokenized_train_dataset = tokenized_train_dataset.remove_columns('__index_level_0__')
        
        # Remove the '__index_level_0__' column from the dataset
        if '__index_level_0__' in tokenized_test_dataset:
            tokenized_test_dataset = tokenized_test_dataset.remove_columns('__index_level_0__')

        train_dataset = TextDataset(tokenized_train_dataset)
        test_dataset = TextDataset(tokenized_test_dataset)
        
        device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

        # Create DataLoader
        eval_dataloader = DataLoader(test_dataset, batch_size=BATCH_SIZE)

        # Evaluate the model using the DataLoader
        self.logger.info(f"evaluating the base model...")
        base_model_eval_results = base_model.evaluate(dataloader=eval_dataloader, device=device)

        # Print evaluation results
        print(f'Base Model Evaluation Results: {base_model_eval_results}')
        
        # Log metrics
        self.results["base_model_intensity_split"] = {}
        for key, value in base_model_eval_results.items():
            self.results["base_model_intensity_split"][key] = value
            
        # Instantiate the CryptoBERT model for regression task
        fine_tuned_model = CryptoBERT(input_task='regression', save_path=f'{self.base_addr}/artifacts/exp3_fine_tuned_model.pth')
        
        # Create DataLoader
        train_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE)

        # Train the model
        self.logger.info(f"training the model on the training dataset...")
        train_results = fine_tuned_model.train(
            dataloader=train_dataloader,
            device=device,
            learning_rate=LEARNING_RATE,
            epochs=EPOCHS
        )
        
        device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

        # Create DataLoader
        test_dataloader = DataLoader(test_dataset, batch_size=BATCH_SIZE)

        # Evaluate the model using the DataLoader
        self.logger.info(f'evaluating the fine-tuned model...')
        fine_tuned_model_test_results = fine_tuned_model.evaluate(dataloader=test_dataloader, device=device)

        # Print test results
        print(f'Fine-Tuned Model Test Results: {fine_tuned_model_test_results}')
        
        # Log metrics
        self.results["fine_tuned_model_intensity_split"] = {}
        for key, value in fine_tuned_model_test_results.items():
            self.results["fine_tuned_model_intensity_split"][key] = value
            
        self.end_time = datetime.datetime.now()
        return self.results
        
    def load_data(self):
        """
        Load the data from the given address.
        """
        
        text_df = pd.read_csv(self.text_df_addr, usecols=["date", "text_split"])
        text_df.rename(columns={"text_split": "text"}, inplace=True)
        text_df.set_index('date', inplace=True)
        text_df.index = pd.to_datetime(text_df.index)
        
        price_df = pd.read_csv(self.price_df_addr, usecols=["timestamp", "close", "open", "high", "low", "volume"])
        price_df.set_index('timestamp', inplace=True)
        price_df.index = pd.to_datetime(price_df.index, unit='s')
        
        # Shift the Bitcoin price data by one day forward
        price_df = price_df.shift(-1)
        
        return text_df, price_df

class ScoreSplit(Experiment):
    def __init__(self):
        super().__init__(
            id=5,
            base_addr=base_addr,
            description="""
                finetuning cryptoBERT on impact score
            """
        )

class TextualFeatureContextAware(Experiment):
    def __init__(
        self,
        num_samples=1000,
        price_df_addr="raw/daily-2020.csv",
        text_df_addr="raw/combined_tweets_2020_labeled.csv",
        logger=None
    ):
        super().__init__(
            id=6,
            base_addr=base_addr,
            model=CryptoBERT(),
            logger=logger,
            description="""
                comparing fine-tuned model on textual dataset with fine-tuned model on context-aware dataset
                """,
        )

        self.price_df_addr = price_df_addr
        self.text_df_addr = text_df_addr
        self.num_samples = num_samples
        # hard code essentials
        self.labeler = TripleBarrierLabeler()
        self.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        self.results = {}

    def run(self):
        # constants
        params = {
            "samples": self.num_samples,
            "BATCH_SIZE":10,
            "SEED":42,
            "TRAIN_TEST_SPLIT":0.2,
            "TRAINING_BATCH_SIZE":5,
            "EPOCHS":10,
            "LEARNING_RATE":1e-5,
        }

        self.start_time = datetime.datetime.now()
        self.logger.info(f"started experiment at {self.start_time}")

        # loading and labeling the data
        self.logger.info(f"loading and labeling the data...")
        text_df, price_df = self.load_data()

        labeled_texts = self.label_data(self.labeler, price_df, text_df)
                
        # Select equal numbers of tweets from each day in the dataset
        how_many_tweets_per_day = 100
        sampled_df = self.select_equal_samples(labeled_texts, how_many_tweets_per_day)
        
        # creating a huggingface dataset for base model evaluation
        self.logger.info(f"creating and tokenizing the dataset...")
        dataset = Dataset.from_pandas(sampled_df[['text', 'context_aware', 'label']])        
        
        # preprocess the text column
        self.logger.info(f"preprocessing the dataset...")
        # labeled_texts = HFDataset.preprocess(labeled_texts)

        self.logger.info(f"slicing and spliting the dataset...")
        # Spliting the dataset for evaluation
        dataset = dataset.train_test_split(0.2, shuffle=False)

        # tokenizing the dataset text to be used in train and test loops
        # Tokenize the text field in the dataset
        def tokenize_function(tokenizer, examples, text_col="text"):
            # Tokenize the text and return only the necessary fields
            encoded = tokenizer(examples[text_col], padding='max_length', max_length=512)
            return {"input_ids": encoded["input_ids"], "attention_mask": encoded["attention_mask"], "label": examples["label"]}

        tokenizer = AutoTokenizer.from_pretrained("ElKulako/cryptobert")
        
        partial_tokenize_function_text = partial(tokenize_function, tokenizer, text_col="text")
        partial_tokenize_function_context = partial(tokenize_function, tokenizer, text_col="context_aware")
        
        # Tokenizing
        tokenized_train_text = dataset["train"].map(partial_tokenize_function_text, batched=True)
        tokenized_test_text = dataset["test"].map(partial_tokenize_function_text, batched=True)
        tokenized_train_context = dataset["train"].map(partial_tokenize_function_context, batched=True)
        tokenized_test_context = dataset["test"].map(partial_tokenize_function_context, batched=True)

        tokenized_train_text_dataset = TextDataset(tokenized_train_text)
        tokenized_test_text_dataset = TextDataset(tokenized_test_text)
        tokenized_train_context_dataset = TextDataset(tokenized_train_context)
        tokenized_test_context_dataset = TextDataset(tokenized_test_context)
        
        tokenized_train_text_dataloader = DataLoader(tokenized_train_text_dataset, batch_size=params['BATCH_SIZE'], shuffle=False)
        tokenized_test_text_dataloader = DataLoader(tokenized_test_text_dataset, batch_size=params['BATCH_SIZE'], shuffle=False)
        tokenized_train_context_dataloader = DataLoader(tokenized_train_context_dataset, batch_size=params['BATCH_SIZE'], shuffle=False)
        tokenized_test_context_dataloader = DataLoader(tokenized_test_context_dataset, batch_size=params['BATCH_SIZE'], shuffle=False)

        self.logger.info(f"Evaluating the base model on textual dataset...")        
        neptune_run = self.init_neptune_run("#6.1: base_model", description="evaluating the base model without fine-tuning", params=params)
        base_model_eval_metrics = self.model.evaluate(tokenized_test_text_dataloader, device=self.device, neptune_run=neptune_run)
        self.results["base_model_eval_metrics"] = base_model_eval_metrics
        neptune_run.stop()
        
        self.logger.info(f"Training and evaluating the model on textual dataset...")
        neptune_run = self.init_neptune_run(name="#6.2: base_text_model", description="base model fine-tuned and evaluated on textual data without temporal or market context", params=params)
        
        text_trained_model = CryptoBERT()
        train_metrics = text_trained_model.train(dataloader=tokenized_train_text_dataloader, device=self.device, learning_rate=params["LEARNING_RATE"], epochs=params["EPOCHS"], neptune_run=neptune_run)
        self.results["textual_fine_tuned_model"] = train_metrics
        text_trained_model_eval_metrics = text_trained_model.evaluate(tokenized_test_text_dataloader, device=self.device, neptune_run=neptune_run)
        self.results["textual_fine_tuned_model_eval_metrics"] = text_trained_model_eval_metrics
        neptune_run.stop()

        self.logger.info(f"training and evaluating the model on context-aware dataset...")
        neptune_run = self.init_neptune_run(name="#6.3: temporal_context_model", description="temporal context-aware model fine-tuned and evaluated on context-aware dataset with temporal or market context", params=params)

        context_trained_model = CryptoBERT()
        train_metrics = context_trained_model.train(dataloader=tokenized_train_context_dataloader, device=self.device, learning_rate=params["LEARNING_RATE"], epochs=params["EPOCHS"], neptune_run=neptune_run)
        self.results["temporal_context_fine_tuned_model"] = train_metrics
        context_trained_model_eval_metrics = context_trained_model.evaluate(tokenized_test_context_dataloader, device=self.device, neptune_run=neptune_run)
        self.results["temporal_context_fine_tuned_model_eval_metrics"] = context_trained_model_eval_metrics
        neptune_run.stop()

        self.end_time = datetime.datetime.now()
        return self.results

    def load_data(self) -> tuple:
        """
        returns text_df and price_df in raw format
        """
        text_df = pd.read_csv(self.text_df_addr, usecols=["date", "text_split"])
        text_df.rename(columns={"text_split": "text"}, inplace=True)
        text_df.set_index("date", inplace=True)
        text_df.index = pd.to_datetime(text_df.index)

        price_df = pd.read_csv(
            self.price_df_addr,
            usecols=["timestamp", "close", "open", "high", "low", "volume"],
        )
        price_df.set_index("timestamp", inplace=True)
        price_df.index = pd.to_datetime(price_df.index, unit="s")

        return text_df, price_df
    
    def label_data(self, labeler, price_df, text_df):
        labeler.fit(price_df)
        price_df = labeler.transform()
        price_df["text_label"] = price_df.label.map({0: 'bearish', 1: 'neutral', 2: 'bullish'})
        price_df["label"] = price_df.label.shift(-1)
        price_df.dropna(inplace=True)
        
        text_df = self.extract_time_string(text_df)
        
        labeled_texts = text_df.merge(
            price_df[["label", "text_label"]], left_index=True, right_index=True, how="left"
        )
        labeled_texts = self.prefix_text_column(labeled_texts, 'time', 'text_label', 'text')
        labeled_texts.dropna(inplace=True)

        # Convert labels to integers
        labeled_texts["label"] = labeled_texts["label"].astype(int)  # Ensure labels are integers
        
        return labeled_texts

class ContextFeatureContextAware(Experiment):
    def __init__(
        self,
        num_samples=1000,
        price_df_addr="raw/daily-2020.csv",
        text_df_addr="raw/combined_tweets_2020_labeled.csv",
        logger=None
    ):
        super().__init__(
            id=7,
            base_addr=base_addr,
            model=CryptoBERT(),
            logger=logger,
            description="""
                comparing fine-tuned model on textual dataset with fine-tuned model on context-aware dataset both evaluated on the context-aware dataset
                """,
        )

        self.price_df_addr = price_df_addr
        self.text_df_addr = text_df_addr
        self.num_samples = num_samples
        # hard code essentials
        self.labeler = TripleBarrierLabeler()
        self.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        self.results = {}

    def run(self):
        # constants
        params = {
            "samples": self.num_samples,
            "BATCH_SIZE":10,
            "SEED":42,
            "TRAIN_TEST_SPLIT":0.2,
            "TRAINING_BATCH_SIZE":5,
            "EPOCHS":10,
            "LEARNING_RATE":1e-5,
        }

        self.start_time = datetime.datetime.now()
        self.logger.info(f"started experiment at {self.start_time}")
        self.logger.info(f"The device is {self.device}")

        # loading and labeling the data
        self.logger.info(f"loading and labeling the data...")
        text_df, price_df = self.load_data()

        labeled_texts = self.label_data(self.labeler, price_df, text_df)
                
        # Select equal numbers of tweets from each day in the dataset
        how_many_tweets_per_day = 100
        sampled_df = self.select_equal_samples(labeled_texts, how_many_tweets_per_day)
        
        # creating a huggingface dataset for base model evaluation
        self.logger.info(f"creating and tokenizing the dataset...")
        dataset = Dataset.from_pandas(sampled_df[['text', 'context_aware', 'label']])        
        
        # preprocess the text column
        self.logger.info(f"preprocessing the dataset...")
        # labeled_texts = HFDataset.preprocess(labeled_texts)

        self.logger.info(f"slicing and spliting the dataset...")
        # Spliting the dataset for evaluation
        dataset = dataset.train_test_split(0.2, shuffle=False)

        # tokenizing the dataset text to be used in train and test loops
        # Tokenize the text field in the dataset
        def tokenize_function(tokenizer, examples, text_col="text"):
            # Tokenize the text and return only the necessary fields
            encoded = tokenizer(examples[text_col], padding='max_length', max_length=512)
            return {"input_ids": encoded["input_ids"], "attention_mask": encoded["attention_mask"], "label": examples["label"]}

        tokenizer = AutoTokenizer.from_pretrained("ElKulako/cryptobert")
        
        partial_tokenize_function_text = partial(tokenize_function, tokenizer, text_col="text")
        partial_tokenize_function_context = partial(tokenize_function, tokenizer, text_col="context_aware")
        
        # Tokenizing
        tokenized_train_text = dataset["train"].map(partial_tokenize_function_text, batched=True)
        tokenized_test_text = dataset["test"].map(partial_tokenize_function_text, batched=True)
        tokenized_train_context = dataset["train"].map(partial_tokenize_function_context, batched=True)
        tokenized_test_context = dataset["test"].map(partial_tokenize_function_context, batched=True)

        tokenized_train_text_dataset = TextDataset(tokenized_train_text)
        tokenized_test_text_dataset = TextDataset(tokenized_test_text)
        tokenized_train_context_dataset = TextDataset(tokenized_train_context)
        tokenized_test_context_dataset = TextDataset(tokenized_test_context)
        
        tokenized_train_text_dataloader = DataLoader(tokenized_train_text_dataset, batch_size=params['BATCH_SIZE'], shuffle=False)
        tokenized_test_text_dataloader = DataLoader(tokenized_test_text_dataset, batch_size=params['BATCH_SIZE'], shuffle=False)
        tokenized_train_context_dataloader = DataLoader(tokenized_train_context_dataset, batch_size=params['BATCH_SIZE'], shuffle=False)
        tokenized_test_context_dataloader = DataLoader(tokenized_test_context_dataset, batch_size=params['BATCH_SIZE'], shuffle=False)

        self.logger.info(f"Evaluating the base model on textual dataset...")        
        neptune_run = self.init_neptune_run("#6.1: base_model", description="evaluating the base model without fine-tuning", params=params)
        base_model_eval_metrics = self.model.evaluate(tokenized_test_text_dataloader, device=self.device, neptune_run=neptune_run)
        self.results["base_model_eval_metrics"] = base_model_eval_metrics
        neptune_run.stop()
        
        self.logger.info(f"Training the model on textual dataset and evaluating it on context-aware test dataset...")
        neptune_run = self.init_neptune_run(name="#6.2: base_text_model", description="base model fine-tuned on textual and evaluated on context-aware test data", params=params)
        
        text_trained_model = CryptoBERT()
        train_metrics = text_trained_model.train(dataloader=tokenized_train_text_dataloader, device=self.device, learning_rate=params["LEARNING_RATE"], epochs=params["EPOCHS"], neptune_run=neptune_run)
        self.results["textual_fine_tuned_model"] = train_metrics
        text_trained_model_eval_metrics = text_trained_model.evaluate(tokenized_test_context_dataloader, device=self.device, neptune_run=neptune_run)
        self.results["textual_fine_tuned_model_eval_metrics"] = text_trained_model_eval_metrics
        neptune_run.stop()

        self.logger.info(f"Training and evaluating the model on context-aware dataset...")
        neptune_run = self.init_neptune_run(name="#6.3: temporal_context_model", description="temporal context-aware model fine-tuned and evaluated on context-aware dataset with temporal or market context", params=params)

        context_trained_model = CryptoBERT()
        train_metrics = context_trained_model.train(dataloader=tokenized_train_context_dataloader, device=self.device, learning_rate=params["LEARNING_RATE"], epochs=params["EPOCHS"], neptune_run=neptune_run)
        self.results["temporal_context_fine_tuned_model"] = train_metrics
        context_trained_model_eval_metrics = context_trained_model.evaluate(tokenized_test_context_dataloader, device=self.device, neptune_run=neptune_run)
        self.results["temporal_context_fine_tuned_model_eval_metrics"] = context_trained_model_eval_metrics
        neptune_run.stop()

        self.end_time = datetime.datetime.now()
        return self.results

    def load_data(self) -> tuple:
        """
        returns text_df and price_df in raw format
        """
        text_df = pd.read_csv(self.text_df_addr, usecols=["date", "text_split"])
        text_df.rename(columns={"text_split": "text"}, inplace=True)
        text_df.set_index("date", inplace=True)
        text_df.index = pd.to_datetime(text_df.index)

        price_df = pd.read_csv(
            self.price_df_addr,
            usecols=["timestamp", "close", "open", "high", "low", "volume"],
        )
        price_df.set_index("timestamp", inplace=True)
        price_df.index = pd.to_datetime(price_df.index, unit="s")

        return text_df, price_df
    
    def label_data(self, labeler, price_df, text_df):
        labeler.fit(price_df)
        price_df = labeler.transform()
        price_df["text_label"] = price_df.label.map({0: 'bearish', 1: 'neutral', 2: 'bullish'})
        price_df["label"] = price_df.label.shift(-1)
        price_df.dropna(inplace=True)
        
        text_df = self.extract_time_string(text_df)
        
        labeled_texts = text_df.merge(
            price_df[["label", "text_label"]], left_index=True, right_index=True, how="left"
        )
        labeled_texts = self.prefix_text_column(labeled_texts, 'time', 'text_label', 'text')
        labeled_texts.dropna(inplace=True)

        # Convert labels to integers
        labeled_texts["label"] = labeled_texts["label"].astype(int)  # Ensure labels are integers
        
        return labeled_texts

class TemporalVsNonTemporal(Experiment):
    def __init__(
        self,
        num_samples=1000,
        price_df_addr="raw/daily-2020.csv",
        text_df_addr="raw/combined_tweets_2020_labeled.csv",
        logger=None
    ):
        super().__init__(
            id=8,
            base_addr=base_addr,
            model=CryptoBERT(),
            logger=logger,
            description="""
                comparing fine-tuned model on non-temporal context-aware dataset with fine-tuned model on temporal context-aware.
                """,
        )

        self.price_df_addr = price_df_addr
        self.text_df_addr = text_df_addr
        self.num_samples = num_samples
        # hard code essentials
        self.labeler = TripleBarrierLabeler()
        self.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        self.results = {}

    def run(self):
        # constants
        params = {
            "samples": self.num_samples,
            "BATCH_SIZE":10,
            "SEED":42,
            "TRAIN_TEST_SPLIT":0.2,
            "TRAINING_BATCH_SIZE":5,
            "EPOCHS":10,
            "LEARNING_RATE":1e-5,
        }

        self.start_time = datetime.datetime.now()
        self.logger.info(f"tarted experiment at {self.start_time}")
        self.logger.info(f"The device is {self.device}")

        # loading and labeling the data
        self.logger.info(f"loading and labeling the data...")
        text_df, price_df = self.load_data()

        non_temporal_labeled_texts = self.label_data(self.labeler, price_df, text_df, is_temporal=False)
        temporal_labeled_texts = self.label_data(self.labeler, price_df, text_df, is_temporal=True)
                
        # Select equal numbers of tweets from each day in the dataset
        how_many_tweets_per_day = 100
        non_temporal_sampled_df = self.select_equal_samples(non_temporal_labeled_texts, how_many_tweets_per_day)
        temporal_sampled_df = self.select_equal_samples(temporal_labeled_texts, how_many_tweets_per_day)
        
        # creating a huggingface dataset for base model evaluation
        self.logger.info(f"creating and tokenizing the dataset...")
        non_temporal_dataset = Dataset.from_pandas(non_temporal_sampled_df[['text', 'context_aware', 'label']])        
        temporal_dataset = Dataset.from_pandas(temporal_sampled_df[['text', 'context_aware', 'label']])        
        
        # preprocess the text column
        self.logger.info(f"preprocessing the dataset...")
        # non_temporal_dataset = HFDataset.preprocess(non_temporal_dataset)
        # temporal_dataset = HFDataset.preprocess(temporal_dataset)
        

        self.logger.info(f"slicing and spliting the dataset...")
        # Spliting the dataset for evaluation
        non_temporal_dataset = non_temporal_dataset.train_test_split(0.2, shuffle=False)
        temporal_dataset = temporal_dataset.train_test_split(0.2, shuffle=False)

        # tokenizing the dataset text to be used in train and test loops
        # Tokenize the text field in the dataset
        def tokenize_function(tokenizer, examples, text_col="text"):
            # Tokenize the text and return only the necessary fields
            encoded = tokenizer(examples[text_col], padding='max_length', max_length=512)
            return {"input_ids": encoded["input_ids"], "attention_mask": encoded["attention_mask"], "label": examples["label"]}

        tokenizer = AutoTokenizer.from_pretrained("ElKulako/cryptobert")
        
        partial_tokenize_function_context = partial(tokenize_function, tokenizer, text_col="context_aware")
        
        # Tokenizing
        tokenized_train_non_temporal = non_temporal_dataset["train"].map(partial_tokenize_function_context, batched=True)
        tokenized_test_non_temporal = non_temporal_dataset["test"].map(partial_tokenize_function_context, batched=True)
        tokenized_train_temporal = temporal_dataset["train"].map(partial_tokenize_function_context, batched=True)
        tokenized_test_temporal = temporal_dataset["test"].map(partial_tokenize_function_context, batched=True)

        tokenized_train_non_temporal_dataset = TextDataset(tokenized_train_non_temporal)
        tokenized_test_non_temporal_dataset = TextDataset(tokenized_test_non_temporal)
        tokenized_train_temporal_dataset = TextDataset(tokenized_train_temporal)
        tokenized_test_temporal_dataset = TextDataset(tokenized_test_temporal)
        
        tokenized_train_non_temporal_dataloader = DataLoader(tokenized_train_non_temporal_dataset, batch_size=params['BATCH_SIZE'], shuffle=False)
        tokenized_test_non_temporal_dataloader = DataLoader(tokenized_test_non_temporal_dataset, batch_size=params['BATCH_SIZE'], shuffle=False)
        tokenized_train_temporal_dataloader = DataLoader(tokenized_train_temporal_dataset, batch_size=params['BATCH_SIZE'], shuffle=False)
        tokenized_test_temporal_dataloader = DataLoader(tokenized_test_temporal_dataset, batch_size=params['BATCH_SIZE'], shuffle=False)
        
        self.logger.info(f"Training and evaluating the model on NON-temporal context-aware dataset...")
        neptune_run = self.init_neptune_run(name="#6.2: non_temporal_model", description="base model fine-tuned on non-temporal and evaluated on non-temporal test data", params=params)
        
        non_temporal_trained_model = CryptoBERT()
        train_metrics = non_temporal_trained_model.train(dataloader=tokenized_train_non_temporal_dataloader, device=self.device, learning_rate=params["LEARNING_RATE"], epochs=params["EPOCHS"], neptune_run=neptune_run)
        self.results["non_temporal_fine_tuned_model"] = train_metrics
        non_temporal_trained_model_eval_metrics = non_temporal_trained_model.evaluate(tokenized_test_non_temporal_dataloader, device=self.device, neptune_run=neptune_run)
        self.results["non_temporal_fine_tuned_model_eval_metrics"] = non_temporal_trained_model_eval_metrics
        neptune_run.stop()

        self.logger.info(f"Training and evaluating the model on temporal context-aware dataset...")
        neptune_run = self.init_neptune_run(name="#6.3: temporal_context_model", description="temporal context-aware model fine-tuned and evaluated on context-aware dataset with temporal or market context", params=params)

        temporal_trained_model = CryptoBERT()
        train_metrics = temporal_trained_model.train(dataloader=tokenized_train_temporal_dataloader, device=self.device, learning_rate=params["LEARNING_RATE"], epochs=params["EPOCHS"], neptune_run=neptune_run)
        self.results["temporal_context_fine_tuned_model"] = train_metrics
        temporal_trained_model_eval_metrics = temporal_trained_model.evaluate(tokenized_test_temporal_dataloader, device=self.device, neptune_run=neptune_run)
        self.results["temporal_context_fine_tuned_model_eval_metrics"] = temporal_trained_model_eval_metrics
        neptune_run.stop()

        self.end_time = datetime.datetime.now()
        return self.results

    def load_data(self) -> tuple:
        """
        returns text_df and price_df in raw format
        """
        text_df = pd.read_csv(self.text_df_addr, usecols=["date", "text_split"])
        text_df.rename(columns={"text_split": "text"}, inplace=True)
        text_df.set_index("date", inplace=True)
        text_df.index = pd.to_datetime(text_df.index)

        price_df = pd.read_csv(
            self.price_df_addr,
            usecols=["timestamp", "close", "open", "high", "low", "volume"],
        )
        price_df.set_index("timestamp", inplace=True)
        price_df.index = pd.to_datetime(price_df.index, unit="s")

        return text_df, price_df
    
    def label_data(self, labeler, price_df, text_df, is_temporal=True):
        labeler.fit(price_df)
        price_df = labeler.transform()
        price_df["text_label"] = price_df.label.map({0: 'bearish', 1: 'neutral', 2: 'bullish'})
        price_df["label"] = price_df.label.shift(-1)
        price_df.dropna(inplace=True)
        
        text_df = self.extract_time_string(text_df)
        
        labeled_texts = text_df.merge(
            price_df[["label", "text_label"]], left_index=True, right_index=True, how="left"
        )
        labeled_texts = self.prefix_text_column(labeled_texts, 'time', 'text_label', 'text', is_temporal=is_temporal)
        labeled_texts.dropna(inplace=True)

        # Convert labels to integers
        labeled_texts["label"] = labeled_texts["label"].astype(int)  # Ensure labels are integers
        
        return labeled_texts


REGISTERED_EXPERIMENTS = [
    SentimentLabelingExperiment,
    DirectionSplitTBL,
    DirectionSplitSentiment,
    DirectionCrossValidate,
    IntensitySplit,
    ScoreSplit,
    TextualFeatureContextAware,
    ContextFeatureContextAware,
    TemporalVsNonTemporal,
]
