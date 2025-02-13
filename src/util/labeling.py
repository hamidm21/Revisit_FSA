import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from datetime import timedelta

def label_ROC(df, window_size):
    """
    Label data based on the rate of change (ROC) over a fixed window.

    Args:
    df (pd.DataFrame): DataFrame containing the price data. It should have columns 'open' and 'close'.
    window_size (int): The number of rows to consider for each ROC calculation.

    Returns:
    labels (pd.Series): A series of labels ('up' or 'down') for each row in df.
    """
    # Define a custom function to apply to each window
    def roc_label(window):
        open_price = window['open'].iloc[0]
        close_price = window['close'].iloc[-1]
        return 'up' if close_price > open_price else 'down'

    # Apply the custom function to each window
    labels = df.rolling(window_size).apply(roc_label, raw=False)

    return labels

def triple_barrier_labeling(sdf, volatility_period=4, upper_barrier_factor=1, lower_barrier_factor=1, vertical_barrier_timedelta=timedelta(days=7)):
    df = sdf.copy()
    # Define your barriers
    df['upper_barrier'] = df['close'] + df['close'].rolling(volatility_period).std() * upper_barrier_factor
    df['lower_barrier'] = df['close'] - df['close'].rolling(volatility_period).std() * lower_barrier_factor
    df['vertical_barrier'] = df.index + vertical_barrier_timedelta

    # Label the observations
    df['label'] = np.where(df['high'].shift(-1) > df['upper_barrier'], 2,
                           np.where(df['low'].shift(-1) < df['lower_barrier'], 0, 1))
    
    return df

def get_label_probabilities(df, features):
    # Define and train your model
    model = RandomForestClassifier()
    model.fit(df[features], df['label'])

    # Get the predicted probabilities for each label
    probabilities = model.predict_proba(df[features])

    return probabilities