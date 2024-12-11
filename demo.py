import pandas as pd
import requests
from sqlalchemy import create_engine
from spellchecker import SpellChecker
from sklearn.preprocessing import StandardScaler, OneHotEncoder
import re

def load_csv(file_path):
    return pd.read_csv(file_path)

def load_excel(file_path):
    return pd.read_excel(file_path)

def load_api(api_url):
    response = requests.get(api_url)
    data = response.json()
    return pd.DataFrame(data)

def load_database(connection_string, query):
    engine = create_engine(connection_string)
    return pd.read_sql(query, engine)

def clean_data(data):
    # Remove duplicates
    data = data.drop_duplicates()

    # Remove null values
    data = data.dropna()

    # Correct spelling mistakes
    spell = SpellChecker()
    for col in data.select_dtypes(include=[object]).columns:
        data[col] = data[col].apply(lambda x: ' '.join([spell.correction(word) for word in x.split()]))

    # Remove unwanted characters
    for col in data.select_dtypes(include=[object]).columns:
        data[col] = data[col].apply(lambda x: re.sub(r'[^A-Za-z0-9\s]+', '', x))

    return data

def transform_data(data):
    # Example transformation: Standard scaling of numerical columns
    scaler = StandardScaler()
    numerical_cols = data.select_dtypes(include=[float, int]).columns
    data[numerical_cols] = scaler.fit_transform(data[numerical_cols])

    # Example transformation: One-hot encoding of categorical columns
    encoder = OneHotEncoder(sparse=False)
    categorical_cols = data.select_dtypes(include=[object]).columns
    encoded_data = pd.DataFrame(encoder.fit_transform(data[categorical_cols]), columns=encoder.get_feature_names_out(categorical_cols))
    data = data.drop(categorical_cols, axis=1)
    data = pd.concat([data, encoded_data], axis=1)

    return data

def sort_data(data, column_name, ascending=True):
    return data.sort_values(by=column_name, ascending=ascending)

def check_domain_constraints(data, column_name, valid_values):
    invalid_values = data[~data[column_name].isin(valid_values)]
    if not invalid_values.empty:
        print(f"Invalid values found in column '{column_name}':")
        print(invalid_values)
    else:
        print(f"No domain constraint problems found in column '{column_name}'.")

def perform_analysis(data):
    # Clean the data
    cleaned_data = clean_data(data)

    # Example: Sort data by a specific column (replace 'column_name' with your actual column name)
    cleaned_data = sort_data(cleaned_data, 'column_name')

    # Example: Check for domain constraint problems (replace 'column_name' and 'valid_values' with your actual values)
    valid_values = ['value1', 'value2', 'value3']
    check_domain_constraints(cleaned_data, 'column_name', valid_values)

    # Transform the data
    transformed_data = transform_data(cleaned_data)

    return cleaned_data, transformed_data

if __name__ == "__main__":
    import sys

    if len(sys.argv) < 3:
        print("Usage: python file.py <data_source> <source_path_or_url>")
        sys.exit(1)

    data_source = sys.argv[1]
    source_path_or_url = sys.argv[2]

    if data_source == 'csv':
        data = load_csv(source_path_or_url)
    elif data_source == 'excel':
        data = load_excel(source_path_or_url)
    elif data_source == 'api':
        data = load_api(source_path_or_url)
    elif data_source == 'database':
        if len(sys.argv) < 4:
            print("Usage for database: python file.py database <connection_string> <query>")
            sys.exit(1)
        query = sys.argv[3]
        data = load_database(source_path_or_url, query)
    else:
        print("Unsupported data source")
        sys.exit(1)

    cleaned_data, transformed_data = perform_analysis(data)

    # Save the cleaned data to a new file
    cleaned_data.to_csv('cleaned_data.csv', index=False)

    # Save the transformed data to a new file
    transformed_data.to_csv('transformed_data.csv', index=False)