import pandas as pd
import random
from tqdm import tqdm

class UndersampleTweetsMixin:
    @staticmethod
    def undersample_tweets(df, num_samples=None):
        trend_counts = df['next_day_label'].value_counts()
        minority_class = trend_counts.idxmin()
        minority_count = trend_counts.min()
        if num_samples is not None and num_samples < minority_count:
            minority_count = num_samples
        undersampled_df = pd.DataFrame()
        for trend in df['next_day_label'].unique():
            subset = df[df['next_day_label'] == trend].sample(minority_count)
            undersampled_df = pd.concat([undersampled_df, subset])
        return undersampled_df

class ExtractWindowsMixin:
    @staticmethod
    def extract_windows(df, max_windows=None):
        days = df.groupby(df.index.date).first()
        window_origins = days[days['next_day_window_start']].index
        windows = []
        for i in range(len(window_origins) - 1):
            if max_windows is not None and len(windows) >= max_windows:
                break
            start_index = window_origins[i]
            end_index = days.index[days.index.get_loc(window_origins[i + 1]) - 1]
            windows.append(days.loc[start_index:end_index])
        if max_windows is None or len(windows) < max_windows:
            windows.append(days.loc[window_origins[-1]:])
        return windows

class ExtractTweetsMixin:
    @staticmethod
    def extract_tweets(windows, df, max_tweet_packs=None):
        extracted_tweets = []
        for window in tqdm(windows, desc="Processing windows"):
            dates = window.index
            window_tweet_packs = []
            min_tweet_count = min(
                df.loc[date.strftime('%Y-%m-%d'), 'text'].size if isinstance(df.loc[date.strftime('%Y-%m-%d'), 'text'], (pd.Series, pd.DataFrame)) else 1
                for date in dates
            )
            if max_tweet_packs is not None:
                min_tweet_count = min(min_tweet_count, max_tweet_packs)
            for i in range(min_tweet_count):
                tweet_pack = []
                for date in dates:
                    tweet = df.loc[date.strftime('%Y-%m-%d'), ["text", "next_day_label"]].iloc[i]
                    tweet_pack.append(tweet)
                window_tweet_packs.append(tweet_pack)
            extracted_tweets.append(window_tweet_packs)
        return extracted_tweets

class ShuffleTweetPacksMixin:
    @staticmethod
    def shuffle_tweet_packs(tweet_packs, seed=None):
        if seed is not None:
            random.seed(seed)
        shuffled_packs = tweet_packs.copy()
        random.shuffle(shuffled_packs)
        return shuffled_packs

class StoreResultsMixin:
    @staticmethod
    def store_results(model_name, fold_epoch_addr, labels, preds, probs, neptune_run):
        self.to_pickle(f"./result/report/exp1/{fold_epoch_addr}/{model_name}.pkl", self.results["{model_name}"][f"fold_{fold_epoch_addr}"])
        self.to_pickle(f"./result/output/exp1/{fold_epoch_addr}/{model_name}.pkl", {"labels": labels, "preds": preds, "probs": probs})
        self.model.plot_roc_curve(f"./result/figure/exp1/{fold_epoch_addr}/{model_name}_roc_curve.png", np.concatenate(labels), np.concatenate(probs))
        self.model.plot_confusion_matrix(f"./result/figure/exp1/{fold_epoch_addr}/{model_name}_matrix.png", np.concatenate(labels), np.concatenate(preds))
        neptune_run[f"{model_name}/{fold_epoch_addr}/roc_curve"].upload(f"./result/figure/exp1/{fold_epoch_addr}/{model_name}_roc_curve.png") if neptune_run else None
        neptune_run[f"{model_name}/{fold_epoch_addr}/matrix"].upload(f"./result/figure/exp1/{fold_epoch_addr}/{model_name}_matrix.png") if neptune_run else None

class DataFrameReaderMixin:
    def __init__():
        # Transform col to index
        self.to_index = lambda col, df: df.set_index(col)
        # Rename text_plit to text
        self.rename = lambda original, new, df: df.rename(columns={original: new})

    @staticmethod
    def pandas_data_loader(addr: str, columns: List[str], *transforms: Callable[[pd.DataFrame], pd.DataFrame]) -> pd.DataFrame:
        # Load the data from the CSV file
        df = pd.read_csv(addr, usecols=columns)

        # Apply each transform to the DataFrame
        for transform in transforms:
            df = transform(df)

        return df

    @staticmethod
    # Transform index to datetime
    def index_to_datetime(df, unit="s"):
        df.index = pd.to_datetime(df.index, unit=unit)
        return df