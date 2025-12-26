"""
Main script to run Titanic Survival Prediction pipeline
"""
import os
import pandas as pd
from data_loader import load_data, clean_data
from feature_engineering import add_family_size, extract_title, encode_features
from eda import plot_survival_by_gender, plot_survival_by_pclass, plot_age_histogram, plot_survival_by_embarked, plot_pclass_vs_survival
from modeling import split_data, train_models, evaluate_models, cross_validate_model, hyperparameter_tuning
from prediction_utils import filter_by_pclass, filter_by_age_range, filter_by_gender, sort_by_survival_probability

# Path to Titanic CSV (update as needed)
data_path = os.path.join('titanic', 'train.csv')

def main():
    # Load and clean train and test data
    train_df = load_data(os.path.join('titanic', 'train.csv'))
    test_df = load_data(os.path.join('titanic', 'test.csv'))
    train_df = clean_data(train_df)
    test_df = clean_data(test_df)
    # Feature engineering
    train_df = add_family_size(train_df)
    train_df = extract_title(train_df)
    train_df = encode_features(train_df)
    test_df = add_family_size(test_df)
    test_df = extract_title(test_df)
    test_df = encode_features(test_df)
    # Save cleaned and preprocessed dataset
    train_df.to_csv('titanic_ml/titanic_cleaned_train.csv', index=False)
    test_df.to_csv('titanic_ml/titanic_cleaned_test.csv', index=False)
    # EDA (use original train_df with columns before encoding for plots)
    plot_survival_by_gender(load_data(os.path.join('titanic', 'train.csv')))
    plot_survival_by_pclass(load_data(os.path.join('titanic', 'train.csv')))
    plot_age_histogram(load_data(os.path.join('titanic', 'train.csv')))
    plot_survival_by_embarked(load_data(os.path.join('titanic', 'train.csv')))
    plot_pclass_vs_survival(load_data(os.path.join('titanic', 'train.csv')))
    # Modeling
    drop_cols = ['Name', 'Cabin', 'Ticket']
    X_train = train_df.drop(['Survived'] + drop_cols, axis=1, errors='ignore')
    y_train = train_df['Survived']
    X_test = test_df.drop(drop_cols, axis=1, errors='ignore')
    models = train_models(X_train, y_train)
    # Evaluate on train set (since test set has no Survived column)
    evaluate_models(models, X_train, y_train)
    # Cross-validation and hyperparameter tuning (example for Random Forest)
    cross_validate_model(models['RandomForest'], X_train, y_train)
    param_grid = {'n_estimators': [100, 200], 'max_depth': [5, 10, None]}
    best_rf = hyperparameter_tuning(models['RandomForest'], param_grid, X_train, y_train)
    # Prediction on test set
    y_prob = best_rf.predict_proba(X_test)[:,1]
    # Merge original values for Age, Fare, FamilySize, and Sex from the raw test set
    raw_test = load_data(os.path.join('titanic', 'test.csv'))
    raw_test['FamilySize'] = raw_test['SibSp'] + raw_test['Parch'] + 1
    # Add predictions and keep original columns for display
    results = test_df.copy()
    results['Survival_Prob'] = y_prob
    # Overwrite with original values for display (keep original names)
    for col in ['Age', 'Fare', 'FamilySize', 'Sex']:
        if col in raw_test.columns:
            results[col] = raw_test[col]
        else:
            results[col] = raw_test[col]
    # Ensure original Pclass is present for filtering in Streamlit
    if 'Pclass' in raw_test.columns:
        results['Pclass'] = raw_test['Pclass']
    # Rename encoded columns with _enc suffix
    encoded_cols = [c for c in results.columns if c not in raw_test.columns and c not in ['Survival_Prob']]
    results.rename(columns={c: c + '_enc' for c in encoded_cols}, inplace=True)
    # Example usage (no Survived column in test set)
    print(filter_by_pclass(results, 1))
    print(filter_by_age_range(results, 20, 30))
    print(filter_by_gender(results, 'male'))
    print(sort_by_survival_probability(results))
    # Save results for Streamlit
    results.to_csv('titanic_ml/results.csv', index=False)

if __name__ == "__main__":
    main()
