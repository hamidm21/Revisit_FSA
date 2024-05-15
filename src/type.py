from logging import Logger
import neptune
import pandas as pd

class Experiment:
    def __init__(self, id: int, description: str, base_addr: str, logger: Logger=None, data_addr: str=None, model: any=None):
        """
        Initialize and validate the experiment.

        Args:
        id (int): The id of the experiment.
        description (str): The description of the experiment.
        data_addr (str): The address to the raw dataset to be used in the experiment.
        model (any): The model to be used in the experiment.
        """
        self.id = id
        self.description = description
        self.logger = logger
        self.base_addr = base_addr
        self.data_addr = data_addr
        self.model = model
        self.start_time = None
        self.end_time = None
        self.results = None

    def run(self):
        """
        Run the experiment.

        This method should be overridden by subclasses to implement
        the actual experiment logic.
        """
        raise NotImplementedError("Subclasses must implement this method.")

    def report(self):
        """
        Report the results of the experiment.

        Returns:
        dict: A dictionary containing the experiment results and metadata.
        """
        if self.start_time is None or self.end_time is None:
            raise ValueError("Experiment has not been run.")
        
        duration = self.end_time - self.start_time
        report = {
            "start_time": self.start_time,
            "end_time": self.end_time,
            "duration": duration,
            "results": self.results,
        }
        return report

    @staticmethod
    def init_neptune_run(name, description, params):
        """
        initializes and returns an instance of neptune run and sends the parameters
        """
        run = neptune.init_run(
        proxies={
            "http": "http://tracker:nlOv5rC7cL3q3bYR@95.216.41.71:3128",
            "https": "http://tracker:nlOv5rC7cL3q3bYR@95.216.41.71:3128"
        },
        project="Financial-NLP/market-aware-embedding",
        api_token="eyJhcGlfYWRkcmVzcyI6Imh0dHBzOi8vYXBwLm5lcHR1bmUuYWkiLCJhcGlfdXJsIjoiaHR0cHM6Ly9hcHAubmVwdHVuZS5haSIsImFwaV9rZXkiOiI2YWViODAxNC05MzNkLTRiZGMtOGI4My04M2U3MDViN2U3ODEifQ==",
        name=name,
        description=description
        )

        run["parameters"] = params
        return run

    def __str__(self):
        return self.description
    
    def extract_time_string(self, df):
        """
        Extract time string from date column to be used in the tweet
        """
        df['time'] = df.index.to_series().dt.strftime('%d,%b,%Y')
        return df
    
    def prefix_text_column(self, df, time_col, trend_col, text_col):
        """
        Prefix a text column with temporal and market context.

        Parameters:
        df (DataFrame): The input DataFrame.
        time_col (str): The name of the time column.
        trend_col (str): The name of the trend column.
        text_col (str): The name of the text column.

        Returns:
        DataFrame: The DataFrame with the prefixed text column.
        """
        # Create a new column by combining the time, trend, and text columns
        df["context_aware"] = "time: " + df[time_col].astype(str) + " trend: " + df[trend_col].astype(str) + " text: " + df[text_col]

        # Return the DataFrame
        return df
    
    def select_equal_samples(self, df, n_samples):
        """
        Select equal numbers of tweets from each day in the dataset.

        Parameters:
        df (DataFrame): The input DataFrame.
        n_samples (int): The number of samples to select from each day.

        Returns:
        DataFrame: The DataFrame with the selected samples.
        """
        # Get the unique dates
        unique_dates = df.index.unique()

        # Initialize an empty DataFrame to store the selected samples
        selected_samples = pd.DataFrame()

        # Iterate over each unique date
        for date in unique_dates:
            # Select n_samples from the current date
            samples = df.loc[date].sample(n_samples, replace=True)

            # Append the samples to the selected_samples DataFrame
            selected_samples = pd.concat([selected_samples, samples])

        # Return the selected_samples DataFrame
        return selected_samples

class Labeler:
    def __init__(self, name):
        """
        Initialize the labeler.

        Args:
        name (str): The name of the labeler.
        """
        self.name = name

    def fit(self, data):
        """
        Fit the labeler to the data.

        This method should be overridden by subclasses to implement
        the actual fitting logic.

        Args:
        data (any): The data to fit the labeler to.
        """
        raise NotImplementedError("Subclasses must implement this method.")

    def transform(self, data):
        """
        Transform the data into labels.

        This method should be overridden by subclasses to implement
        the actual transformation logic.

        Args:
        data (any): The data to transform into labels.

        Returns:
        any: The labels.
        """
        raise NotImplementedError("Subclasses must implement this method.")

class Model:
    def __init__(self, name):
        """
        Initialize the model.

        Args:
        name (str): The name of the model.
        """
        self.name = name

    def train(self, data, labels):
        """
        Train the model on the given data and labels.

        This method should be overridden by subclasses to implement
        the actual training logic.

        Args:
        data (any): The data to train the model on.
        labels (any): The labels for the data.
        """
        raise NotImplementedError("Subclasses must implement this method.")

    def predict(self, data):
        """
        Use the model to make predictions on the given data.

        This method should be overridden by subclasses to implement
        the actual prediction logic.

        Args:
        data (any): The data to make predictions on.

        Returns:
        any: The predictions.
        """
        raise NotImplementedError("Subclasses must implement this method.")

    def evaluate(self, data, labels):
        """
        Evaluate the model on the given data and labels.

        This method should be overridden by subclasses to implement
        the actual evaluation logic.

        Args:
        data (any): The data to evaluate the model on.
        labels (any): The labels for the data.

        Returns:
        any: The evaluation results.
        """
        raise NotImplementedError("Subclasses must implement this method.")

    def compute_metrics(self, predictions, labels):
        """
        Compute metrics based on the model's predictions and the true labels.

        This method should be overridden by subclasses to implement
        the actual metrics computation logic.

        Args:
        predictions (any): The model's predictions.
        labels (any): The true labels.

        Returns:
        any: The computed metrics.
        """
        raise NotImplementedError("Subclasses must implement this method.")