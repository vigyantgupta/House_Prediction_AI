import pandas as pd

def load_data(path):
    df = pd.read_csv(path, sep="\t")
    return df

def handle_missing_values(df):
    # Drop columns with too many missing values
    df = df.drop(columns=["PoolQC", "Alley", "Fence", "MiscFeature"], errors="ignore")
    
    # Fill numerical missing values with median
    num_cols = df.select_dtypes(include=["int64", "float64"]).columns
    df[num_cols] = df[num_cols].fillna(df[num_cols].median())
    
    # Fill categorical missing values with "Unknown"
    cat_cols = df.select_dtypes(include=["object"]).columns
    df[cat_cols] = df[cat_cols].fillna("Unknown")
    
    return df

def encode_categorical(df):
    df = pd.get_dummies(df, drop_first=True)
    return df

def preprocess_data(path):
    df = load_data(path)
    df = handle_missing_values(df)
    df = encode_categorical(df)
    return df
