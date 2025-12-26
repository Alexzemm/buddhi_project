"""
Exploratory Data Analysis for Titanic dataset
"""
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def plot_survival_by_gender(df: pd.DataFrame):
    """Bar plot of survival by gender."""
    sns.barplot(x='Sex', y='Survived', data=df)
    plt.title('Survival Rate by Gender')
    plt.show()

def plot_survival_by_pclass(df: pd.DataFrame):
    """Bar plot of survival by passenger class."""
    sns.barplot(x='Pclass', y='Survived', data=df)
    plt.title('Survival Rate by Passenger Class')
    plt.show()

def plot_age_histogram(df: pd.DataFrame):
    """Histogram of age distribution."""
    df['Age'].hist(bins=20)
    plt.title('Age Distribution')
    plt.xlabel('Age')
    plt.ylabel('Count')
    plt.show()

def plot_survival_by_embarked(df: pd.DataFrame):
    """Bar plot of survival by embarkation port."""
    sns.barplot(x='Embarked', y='Survived', data=df)
    plt.title('Survival Rate by Embarked Port')
    plt.show()

def plot_pclass_vs_survival(df: pd.DataFrame):
    """Box plot of Age vs Survival by Pclass."""
    sns.boxplot(x='Pclass', y='Age', hue='Survived', data=df)
    plt.title('Age vs Survival by Pclass')
    plt.show()
