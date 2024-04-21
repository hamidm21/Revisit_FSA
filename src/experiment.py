import datetime
from datetime import timedelta

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
from scipy.special import softmax
import torch
from torch.utils.data import DataLoader

# internal imports
from type import Experiment
from model import CryptoBERT
from labeler import TripleBarrierLabeler
from dataset import HFDataset, TextDataset
from util import *
from functools import partial
import os
from dotenv import load_dotenv


# Load environment variables from .env file
load_dotenv()

# Access the base address
base_addr = os.getenv("BASE_ADDRESS")


class DirectionSplitTBL(Experiment):
    def __init__(
        self,
        num_samples=100,
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
        self.labeler = TripleBarrierLabeler()

        self.results = {}

    def run(self):
        # constants
        SEED=42
        TRAIN_TEST_SPLIT=0.2
        TRAINING_BATCH_SIZE=5
        EPOCHS=2
        LEARNING_RATE=1e-5

        self.start_time = datetime.datetime.now()
        self.logger.info(f"started experiment at {self.start_time}")

        # loading and labeling the data
        self.logger.info(f"loading and labeling the data...")
        text_df, price_df = self.load_data()
        self.labeler.fit(price_df)
        triple_barrier_labels = self.labeler.transform()
        labeled_texts = text_df.merge(
            triple_barrier_labels[["label"]], left_index=True, right_index=True, how="left"
        )

        # creating a huggingface dataset for base model evaluation
        self.logger.info(f"creating and tokenizing the dataset...")
        labeled_texts = HFDataset.from_pandas(labeled_texts[["text", "label"]])
        # shuffeling the dataset for a more unbiased mix
        labeled_texts = labeled_texts.shuffle(seed=SEED)
        # preprocess the text column
        labeled_texts = HFDataset.preprocess(labeled_texts, labeled_texts)

        # tokenizing the dataset text to be used in train and test loops
        tokenizer = AutoTokenizer.from_pretrained("ElKulako/cryptobert")
        labeled_texts = HFDataset.tokenize(
            tokenizer, labeled_texts
        )

        self.logger.info(f"slicing and spliting the dataset...")
        # Spliting the dataset for evaluation
        labeled_texts = labeled_texts.select(range(self.num_samples))

        labeled_texts = labeled_texts.train_test_split(TRAIN_TEST_SPLIT)
        print(labeled_texts)

        trainer = self.model.get_trainer(labeled_texts["test"])

        self.logger.info(f"evaluating the base model without fintuning...")
        non_fine_tuned_eval_result = trainer.evaluate()

        # Log metrics
        self.results["base"] = {}
        for key, value in non_fine_tuned_eval_result.items():
            self.results["base"][key] = value

        self.logger.info(f"preparing data for finetuning the model...")
        train_dataset = TextDataset(labeled_texts['train'])
        test_dataset = TextDataset(labeled_texts['test'])

        train_dataloader = DataLoader(train_dataset, batch_size=TRAINING_BATCH_SIZE)
        test_dataloader = DataLoader(test_dataset, batch_size=TRAINING_BATCH_SIZE)

        self.logger.info(f"training the model...")
        device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        train_metrics = self.model.train(dataloader=train_dataloader, device=device, learning_rate=LEARNING_RATE, epochs=EPOCHS)
        self.results["train"] = train_metrics

        self.logger.info(f"evaluating the finetuned model...")
        eval_metrics = self.model.evaluate(dataloader=test_dataloader, device=device)
        self.results["eval"] = eval_metrics

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
    def __init__(self):
        super().__init__(
            id=4,
            base_addr=base_addr,
            description="""
                finetuning cryptoBERT on impact intensity
            """
        )

class ScoreSplit(Experiment):
    def __init__(self):
        super().__init__(
            id=5,
            base_addr=base_addr,
            description="""
                finetuning cryptoBERT on impact score
            """
        )

REGISTERED_EXPERIMENTS = [
    DirectionSplitTBL,
    DirectionSplitSentiment,
    DirectionCrossValidate,
    IntensitySplit,
    ScoreSplit,
]
