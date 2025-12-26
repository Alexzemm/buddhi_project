"""
Data loading and cleaning for Titanic dataset
"""
import pandas as pd
import numpy as np
from typing import Tuple

def load_data(path: str) -> pd.DataFrame:
    """Load Titanic dataset from CSV file."""
    return pd.read_csv(path)

def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """Clean Titanic dataset: handle missing values, drop irrelevant columns, create HasCabin."""
    df = df.copy()
    # Fill missing Age with median
    df['Age'] = df['Age'].fillna(df['Age'].median())
    # Fill missing Embarked with mode
    df['Embarked'] = df['Embarked'].fillna(df['Embarked'].mode()[0])
    # Create HasCabin feature
    df['HasCabin'] = df['Cabin'].notnull().astype(int)
    # Fill missing Cabin as 'Unknown'
    df['Cabin'] = df['Cabin'].fillna('Unknown')
    # Fix Fare inconsistencies
    df['Fare'] = df['Fare'].replace(0, np.nan)
    df['Fare'] = df['Fare'].fillna(df['Fare'].median())
    # Drop irrelevant columns
    df = df.drop(['Ticket', 'PassengerId'], axis=1)
    return df
