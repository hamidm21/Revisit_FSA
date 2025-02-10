import numpy as np
import pandas as pd

from type import Labeler
from util.decorator import validate_columns

class TripleBarrierLabeler(Labeler):
    def __init__(self, volatility_period=7, upper_barrier_factor=1, lower_barrier_factor=1, vertical_barrier=7, min_trend_days=2, barrier_type='volatility', touch_type="HL", up_label=2, neutral_label=1, down_label=0):
        """
        Initialize the labeler.
        """
        super().__init__(name="triple barrier labeling")
        self.volatility_period = volatility_period
        self.upper_barrier_factor = upper_barrier_factor
        self.lower_barrier_factor = lower_barrier_factor
        self.vertical_barrier = vertical_barrier
        self.min_trend_days = min_trend_days
        self.barrier_type = barrier_type
        self.touch_type = touch_type
        self.up_label = up_label
        self.down_label = down_label
        self.neutral_label = neutral_label

    def calculate_barriers(self, df, i, window):
        """calculate the barriers based on either volatility or returns of the backward window

        Args:
            df (pd.DataFrame): Data
            i (pd.index): the index of the beginning of the window
            window (int): window size

        Returns:
            df: Data including barriers for the forward window
        """
        end_window = min(i+window, len(df)-1)  # Ensure the window does not exceed the dataframe

        # Calculate the mean volatility or daily returns over the volatility_period
        if self.barrier_type == 'volatility':
            current_value = df.loc[i, 'volatility']
        elif self.barrier_type == 'returns':
            current_value = df.loc[i, 'daily_returns']
        else:
            raise ValueError("Invalid barrier_type. Choose either 'volatility' or 'returns'")

        df.loc[i:end_window, 'upper_barrier'] = df.loc[i, 'close'] + (df.loc[i, 'close'] * current_value * self.upper_barrier_factor)
        df.loc[i:end_window, 'lower_barrier'] = df.loc[i, 'close'] - (df.loc[i, 'close'] * current_value * self.lower_barrier_factor)
        return df

    def label_observations(self, df, origin, i, label):
        df.loc[origin:i+1, 'label'] = label
        return df

    def get_daily_vol(self, close, span0=30):
        """
        Calculate the daily volatility of closing prices.
        
        Parameters:
        - close: A pandas Series of closing prices.
        - span0: The span for the EWM standard deviation.
        
        Returns:
        - A pandas Series of daily volatility estimates.
        """
        daily_returns = np.log(price_df.close / price_df.close.shift(1))
        ewma_volatility = daily_returns.ewm(span=span).std()
        return daily_returns, ewma_volatility

    def fit(self, sdf):
        df = sdf.copy()
        # Calculate daily returns and volatility
        df['daily_returns'], df['volatility'] = self.get_daily_vol(df.close, self.volatility_period)

        df = df.reset_index()
        # Initialize label and window start
        df['label'] = self.neutral_label
        df['window_start'] = False

        self.data = df

    def transform(self):
        """
        Transform the data into labels.

        Returns:
        pd.DataFrame: The labels.
        """
        window = self.vertical_barrier
        origin = 0
        touch_upper = lambda high, barrier: high >= barrier
        touch_lower = lambda low, barrier: low <= barrier
        # For each observation
        for i in range(0, len(self.data)):
            # Define your barriers at the beginning of each window
            if i == origin:
                self.data = self.calculate_barriers(self.data, i, window)
                self.data.loc[i, 'window_start'] = True  # Mark the start of the window

            # one of the conditions were met
            if touch_upper(self.data.loc[i, "high" if self.touch_type == 'HL' else 'close'], self.data.loc[i, "upper_barrier"]):
                if (i - origin > self.min_trend_days):
                    # label the observations
                    self.data = self.label_observations(self.data, origin, i, self.up_label)
                    # set new origin
                    origin = i + 1 if i + 1 < len(self.data) else i  # Check if i + 1 is within the DataFrame's index
                    # reset window
                    window = self.vertical_barrier
            elif touch_lower(self.data.loc[i, "low" if self.touch_type == 'HL' else 'close'], self.data.loc[i, "lower_barrier"]):
                if (i - origin > self.min_trend_days):
                    # label the observations
                    self.data = self.label_observations(self.data, origin, i, self.down_label)
                    # set new origin
                    origin = i + 1 if i + 1 < len(self.data) else i  # Check if i + 1 is within the DataFrame's index
                    # reset window
                    window = self.vertical_barrier

            # none of the conditions were met
            else:
                if window > 0:
                    # reduce window size by one
                    window = window - 1
                else:
                    # reset window
                    window = self.vertical_barrier
                    # label neutral from origin to origin + window
                    self.data.loc[origin:min(origin+window, len(self.data)-1), 'label'] = self.neutral_label  # Ensure the window does not exceed the dataframe
                    # set origin to the next id
                    origin = i + 1 if i + 1 < len(self.data) else i  # Check if i + 1 is within the DataFrame's index

        self.data = self.data.set_index("timestamp")
        return self.data

class TrueRangeLabeler(Labeler):
    def __init__(self, data):
        """
        Initialize the labeler.
        """
        super().__init__("true range labeler")
        self.data = data.copy()
        self.fit()

    def fit(self):
        """
        Fit the labeler to the data.

        Args:
        data (pd.DataFrame): The data to fit the labeler to.
        """
        # Calculate the True Range
        self.data['high_low'] = self.data['high'] - self.data['low']
        self.data['high_prev_close'] = np.abs(self.data['high'] - self.data['close'].shift())
        self.data['low_prev_close'] = np.abs(self.data['low'] - self.data['close'].shift())

    def transform(self):
        """
        Transform the data into labels.

        Returns:
        pd.DataFrame: The labels.
        """
        self.data['true_range'] = self.data[['high_low', 'high_prev_close', 'low_prev_close']].max(axis=1)

        # Normalize the True Range to be between 0 and 1
        self.data['label'] = (self.data['true_range'] - self.data['true_range'].min()) / (self.data['true_range'].max() - self.data['true_range'].min())
        
        return self.data


class ImpactScoreLabeler(Labeler):
    def __init__(self, name, direction_labeler, intensity_labeler):
        """
        Initialize the labeler.

        Args:
        name (str): The name of the labeler.
        direction_labeler (Labeler): The labeler for the direction.
        intensity_labeler (Labeler): The labeler for the intensity.
        """
        super().__init__(name)
        self.direction_labeler = direction_labeler
        self.intensity_labeler = intensity_labeler

    def fit(self, data):
        """
        Fit the labeler to the data.

        Args:
        data (pd.DataFrame): The data to fit the labeler to.
        """
        self.direction_labeler.fit(data)
        self.intensity_labeler.fit(data)

    def transform(self):
        """
        Transform the data into labels.

        Returns:
        pd.DataFrame: The labels.
        """
        direction_labels = self.direction_labeler.transform(up_label=1, down_label=-1, neutral_label=0)
        intensity_labels = self.intensity_labeler.transform()

        # Calculate the impact score
        impact_score = direction_labels * intensity_labels

        return impact_score
