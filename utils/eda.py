import pandas as pd

def generate_report(df: pd.DataFrame):
    # Initialize an empty dictionary to hold the report
    report = {}

    # Basic information
    report['basic_info'] = pd.DataFrame({
        'number_of_rows': [df.shape[0]],
        'number_of_columns': [df.shape[1]],
        'column_names': [df.columns.tolist()]
    })

    # Data types of each column
    report['data_types'] = pd.DataFrame(df.dtypes, columns=['data_type'])

    # Summary statistics for numeric columns
    report['summary_statistics'] = df.describe().transpose()

    # Count of null values in each column
    report['null_counts'] = pd.DataFrame(df.isnull().sum(), columns=['null_count'])

    return report
