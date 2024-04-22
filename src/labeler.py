import numpy as np

from type import Labeler
from util.decorator import validate_columns

class TripleBarrierLabeler(Labeler):
    def __init__(self, volatility_period=7, vertical_barrier_timedelta=2, upper_barrier_factor=1, lower_barrier_factor=1):
        """
        Initialize the labeler.
        """
        super().__init__(name="triple barrier lableing")
        self.volatility_period = volatility_period
        self.upper_barrier_factor = upper_barrier_factor
        self.lower_barrier_factor = lower_barrier_factor
        self.vertical_barrier_timedelta = vertical_barrier_timedelta

    @validate_columns(['low', 'close', 'high'])
    def fit(self, data):
        """
        Fit the labeler to the data.

        Args:
        data (pd.DataFrame): The data to fit the labeler to.
        """
        self.data = data.copy()
        # Define your barriers
        self.data['upper_barrier'] = self.data['close'] + self.data['close'].rolling(self.volatility_period).std() * self.upper_barrier_factor
        self.data['lower_barrier'] = self.data['close'] - self.data['close'].rolling(self.volatility_period).std() * self.lower_barrier_factor
        self.data['vertical_barrier'] = self.data.index.shift(self.vertical_barrier_timedelta, freq='T')

    def transform(self, up_label = 2, down_label = 0, neutral_label = 1):
        """
        Transform the data into labels.

        Returns:
        pd.DataFrame: The labels.
        """
        # Label the observations
        self.data['label'] = np.where(self.data['high'].shift(-1) > self.data['upper_barrier'], up_label,
                                      np.where(self.data['low'].shift(-1) < self.data['lower_barrier'], neutral_label, down_label))

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
