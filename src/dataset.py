from datasets import Dataset as HuggingfaceDataset
from torch.utils.data import DataLoader, Dataset as torchDS
from functools import partial
import torch
import re
import string
import emoji
import time

class HFDataset(HuggingfaceDataset):
    def preprocess(self, return_emojis = False, return_hashtags = False):
        ads_keywords = ["nft", "bonus", "campaign", "invite", "friends"]
        def run_all_preprocess_functions(item):
            text_list = item['text']
            is_ads_tweet = False
            emojis_list = []
            hashtags_list = []
            for enum,text in enumerate(text_list):
                text = HFDataset.lowercase_tweet(text)
                text = HFDataset.remove_URL(text)
                emojis_list += HFDataset.extract_emojis_and_emoticons(text)
                hashtags_list += HFDataset.extract_hashtags(text) 
                text = HFDataset.remove_user_ids(text)
                text = HFDataset.remove_punctuations(text)
                text_list[enum] = HFDataset.replace_with_BTC(text)
                if (not is_ads_tweet):
                    is_ads_tweet = HFDataset.is_ads(text, ads_keywords)
            if(return_emojis == False):
                if(return_hashtags == False):
                    return {"text" : text_list}
                else:
                    return {"text" : text_list, "hashtags" : hashtags_list}
            else:
                if(return_hashtags == False):
                    return {"text" : text_list, "emojis" : emojis_list}
                else:
                    return {"text" : text_list, "emojis" : emojis_list, "hashtags" : hashtags_list}

        preprocessed_data =  self.map(run_all_preprocess_functions, batched = True)
        return preprocessed_data
        # self.tokenizer = tokenizer

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

    @staticmethod
    def remove_URL(text):
            return re.sub(r"(?:https?://|www\.)\S+\.\S+", "", text)

    @staticmethod
    def lowercase_tweet(text):
        return text.lower()

    @staticmethod
    def remove_punctuations(text):
        exclude = set(string.punctuation)
        for char in ['!','?','%','$','&']:
            exclude.remove(char)
        text_without_punctuations = ''.join(ch for ch in text if ch not in exclude)
        return text_without_punctuations

    @staticmethod
    def extract_hashtags(text):
        hashtags_list = re.findall(r"#(\w+)", text)
        return hashtags_list

    @staticmethod
    def replace_with_BTC(text):
        return re.sub(r"Bitcoin|bitcoin|btc|BitCoin", "BTC", text)

    @staticmethod
    def remove_user_ids(text):
        return re.sub(r"@\w+", "", text)

    @staticmethod
    def extract_emojis_and_emoticons(text):
        emojis = emoji.distinct_emoji_list(text)
        emoticons = []
        patterns = [
            r":\)+",  # :) or :-) - Smiling face (happiness, amusement, friendliness)
            r":\(-",  # :( or :-( - Frowning face (sadness, disappointment, disapproval)
            r";\)",   # ;) or ;-) - Wink (joke, flirtation, secrecy)
            r":D+|:-D+",  # :D or :-D - Big smile or grin (amusement, laughter, joy)
            r":P+|:-P+",  # :P or :-P - Sticking out tongue (silliness, teasing, raspberries)
            r":=\)",    # Equals sign smile (tentative smile, unsure)
            r"/:",      # Slash frown (disappointment, annoyance)
            r"\*-*",    # Asterisk kiss (hugs and kisses)
            r":\=",     # Equals sign sad (grimace, helplessness)
            r":\"",     # Double quote (air quotes, sarcasm)
            r"\*_*",   # Asterisk happy face (big smile, eyes closed)
            r"\(/:",    # Backslash frown (extreme frustration)
            r"\||_",    # Sleeping face (tired, bored)
            r"\^_^",    # Happy face with underscore eyes (content, smug)
            r":*-^",    # Cat face (playful, mischief)
            r"^-^",    # Simple happy face
            r":*_^",    # Wink with happy face
            r"^_*",    # Happy face with wink
            r"^-*",    # Confused face
        ]
        for pattern in patterns:
            emoticons.extend(re.findall(pattern, text))
        return emojis + emoticons

    @staticmethod
    def is_ads(text, ads_keywords):
        for ads_keyword in ads_keywords:
            if ads_keyword in text :
                return True

        return False


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