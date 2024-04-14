def validate_columns(required_columns):
    def decorator(func):
        def wrapper(self, data):
            if not all(column in data.columns for column in required_columns):
                missing_columns = [column for column in required_columns if column not in data.columns]
                raise ValueError(f"The input DataFrame is missing the following required columns: {', '.join(missing_columns)}")
            return func(self, data)
        return wrapper
    return decorator