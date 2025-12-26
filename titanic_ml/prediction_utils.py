"""
Prediction filtering and sorting utilities for Titanic dataset
"""
import pandas as pd

def filter_by_pclass(df: pd.DataFrame, pclass: int) -> pd.DataFrame:
    """Filter predictions by passenger class (handles one-hot encoding)."""
    # If original Pclass column exists (not encoded)
    if 'Pclass' in df.columns:
        return df[df['Pclass'] == pclass]
    # If one-hot encoded columns exist (with _enc suffix)
    p2_col = 'Pclass_2_enc' if 'Pclass_2_enc' in df.columns else 'Pclass_2'
    p3_col = 'Pclass_3_enc' if 'Pclass_3_enc' in df.columns else 'Pclass_3'
    if pclass == 1:
        # Pclass_2 == 0 and Pclass_3 == 0
        return df[(df.get(p2_col, 0) == 0) & (df.get(p3_col, 0) == 0)]
    elif pclass == 2:
        return df[df.get(p2_col, 0) == 1]
    elif pclass == 3:
        return df[df.get(p3_col, 0) == 1]
    else:
        return df

def filter_by_age_range(df: pd.DataFrame, min_age: float, max_age: float) -> pd.DataFrame:
    """Filter predictions by age range."""
    return df[(df['Age'] >= min_age) & (df['Age'] <= max_age)]

def filter_by_gender(df: pd.DataFrame, gender: str) -> pd.DataFrame:
    """Filter predictions by gender ('male' or 'female')."""
    return df[df['Sex'] == gender]

def sort_by_survival_probability(df: pd.DataFrame, prob_col: str = 'Survival_Prob', ascending: bool = False) -> pd.DataFrame:
    """Sort predictions by survival probability."""
    return df.sort_values(by=prob_col, ascending=ascending)
