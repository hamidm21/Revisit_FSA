import os
import logging
import pandas as pd

def clean_dataset(source: pd.DataFrame):
    """
    This function cleans the input DataFrame by:
    - Converting 'user_followers' and 'user_friends' columns to integer type
    - Converting 'user_verified' column to boolean type
    - Converting 'date' column to datetime type
    - Dropping rows with any null values
    - Setting 'date' as the index of the DataFrame
    It then returns the cleaned DataFrame.
    """
    df = source.copy()
    df["user_followers"] = pd.to_numeric(df["user_followers"], errors='coerce').astype('Int64')
    df["user_friends"] = pd.to_numeric(df["user_friends"], errors='coerce').astype('Int64')
    df["user_verified"] = df["user_verified"].astype("bool")
    df["date"] = pd.to_datetime(df["date"], errors='coerce')
    df = df.dropna().set_index("date")
    return df

def handle_dataset(file_path, df=None, columns=None, index="date"):
    """
    This function handles the reading and writing of datasets.
    - If a DataFrame is provided, it writes the DataFrame to a CSV file at the given file path.
    - If no DataFrame is provided, it checks if a file exists at the given file path and reads it if it does.
    - If no file exists, it returns None.
    """
    if df is not None:
        logging.debug(f"Writing to csv at {file_path}")
        df.to_csv(file_path)
    elif os.path.isfile(file_path):
        logging.debug(f"Reading dataset from {file_path}")
        return pd.read_csv(file_path, usecols=columns).set_index(index)
    return None

def slice_dataframe(sdf, start_datetime, end_datetime):
    """
    This function slices a DataFrame based on a given date range.
    - It first sorts the DataFrame by 'date'.
    - It then slices the DataFrame based on the given start and end datetime.
    - It returns the sliced DataFrame.
    """
    sorted_df = sdf.sort_values(by='date')
    sliced_df = sorted_df.loc[start_datetime:end_datetime]
    return sliced_df

def get_dataframe(source_dataset_address, clean_dataset_address, sliced_dataset_address, start_time, end_time):
    """
    This function gets a DataFrame from a given file path.
    - It first tries to read a sliced dataset from a given file path.
    - If no sliced dataset exists, it tries to read a clean dataset from a given file path.
    - If no clean dataset exists, it generates a clean dataset from a source dataset.
    - It then slices the DataFrame and writes it to a CSV file.
    - It returns the DataFrame.
    """
    columns = ['user_followers', 'user_friends', 'user_verified', 'date', 'text']

    df = handle_dataset(sliced_dataset_address, columns=columns)
    if df is None:
        df = handle_dataset(clean_dataset_address, columns=columns)
        if df is None:
            logging.debug("Generating clean dataset from source")
            df = pd.read_csv(source_dataset_address, lineterminator='\n', usecols=columns)
            df = clean_dataset(df)
            handle_dataset(clean_dataset_address, df=df)
        df = slice_dataframe(df, start_time, end_time)
        handle_dataset(sliced_dataset_address, df=df)

    return df