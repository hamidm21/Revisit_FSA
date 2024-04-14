from datasets import Dataset as HuggingfaceDataset
from torch.utils.data import DataLoader, Dataset as torchDS
from functools import partial
import torch

class HFDataset(HuggingfaceDataset):
    def preprocess(self, tokenizer):
        # TODO: add preprocessor function to the map function like this:
        # 
        self.map()
        self.tokenizer = tokenizer

    @staticmethod
    def tokenize(tokenizer, dataset):
        # Tokenize the text field in the dataset
        def tokenize_function(tokenizer, item):
            # Tokenize the text and return only the necessary fields
            encoded = tokenizer(item["text"], padding="max_length", max_length=512)
            return {"input_ids": encoded["input_ids"], "attention_mask": encoded["attention_mask"], "label": item["label"]}

        # tokenizing the dataset text to be used in train and test loops
        partial_tokenize_function = partial(tokenize_function, tokenizer)
        tokenized_datasets = dataset.map(partial_tokenize_function, batched=True)

        return tokenized_datasets

class TextDataset(torchDS):
    def __init__(self, hf_dataset):
        self.hf_dataset = hf_dataset

    def __len__(self):
        return len(self.hf_dataset)

    def __getitem__(self, idx):
        item = self.hf_dataset[idx]
        return {
            'input_ids': torch.tensor(item['input_ids']),
            'attention_mask': torch.tensor(item['attention_mask']),
            'labels': torch.tensor(item['label'])
        }