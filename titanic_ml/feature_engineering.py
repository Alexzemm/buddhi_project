"""
Feature engineering for Titanic dataset
"""
import pandas as pd
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler
import numpy as np
import re

def add_family_size(df: pd.DataFrame) -> pd.DataFrame:
    """Add FamilySize feature."""
    df['FamilySize'] = df['SibSp'] + df['Parch'] + 1
    return df

def extract_title(df: pd.DataFrame) -> pd.DataFrame:
    """Extract Title from Name and group rare titles."""
    df = df.copy()
    df['Title'] = df['Name'].apply(lambda x: re.search(r',\s*([^\.]+)\.', x).group(1))
    # Group rare titles
    rare_titles = ['Lady', 'Countess', 'Capt', 'Col', 'Don', 'Dr', 'Major', 'Rev', 'Sir', 'Jonkheer', 'Dona']
    df['Title'] = df['Title'].replace(rare_titles, 'Rare')
    df['Title'] = df['Title'].replace({'Mlle': 'Miss', 'Ms': 'Miss', 'Mme': 'Mrs'})
    return df

def encode_features(df: pd.DataFrame) -> pd.DataFrame:
    """Encode categorical features and scale numerical features."""
    df = df.copy()
    # Label Encoding for Sex, Title, Embarked
    for col in ['Sex', 'Title', 'Embarked']:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col])
    # One-Hot Encoding for Pclass
    df = pd.get_dummies(df, columns=['Pclass'], drop_first=True)
    # Scale Age, Fare, FamilySize
    scaler = StandardScaler()
    for col in ['Age', 'Fare', 'FamilySize']:
        df[col] = scaler.fit_transform(df[[col]])
    return df
