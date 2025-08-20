import pandas as pd
import numpy as np
import xgboost as xgb
import optuna
import os

from sklearn.model_selection import StratifiedKFold
from sklearn.linear_model import LassoCV
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error

# --- Configuration ---

# 1. Define the file path to the dataset
FILE_PATH = '../data/preprocessed/final_nfi_ch2018_merged/NFI_with_Climate_Averages_RCP45.csv'
# 2. Define features and targets
# List of features to be used for training
FEATURES = [
    'mean_dry_days_count', 'mean_frost_days_count', 'mean_gdd_sum',
    'mean_pr_sum', 'mean_pr_variance', 'mean_tas_mean', 'mean_tas_variance',
    'mean_tasmax_mean', 'mean_tasmax_variance', 'mean_tasmin_mean',
    'mean_tasmin_variance', 'HWSW_prop', 'INVYR', 'BASFPH_squared',
    'Time_Diff_years', 'BASFPH', 'BEWIRTINT1', 'ASPECT25',
    'SLOPE25', 'PH', 'Z25', 'NAISHSTKOMB'
]

# List of categorical features that need one-hot encoding
CATEGORICAL_FEATURES = ['BEWIRTINT1', 'NAISHSTKOMB']

# List of response variables (targets) to model
TARGETS = ['BASFPH_next_INVNR', 'HWSW_prop_next_INVNR']

# Stratification column
STRATA_COL = 'INVNR'

# Cross-validation settings
N_SPLITS = 10
RANDOM_STATE = 35

# Number of alpha steps for LassoCV
ALPHA_NUM_STEPS = 10000
ALPHA_LOGSPACE_START = -6  # from 1e-6
ALPHA_LOGSPACE_END = 2 

# Maximum iterations for LassoCV
LASSO_MAX_ITER = 1000

# Optuna settings for XGBoost
N_TRIALS_OPTUNA = 1000 

# --- 1. Data Loading and Preprocessing  ---

def load_and_preprocess_data(filepath):
    """
    Loads data from a CSV, drops rows with missing values in features or targets,
    performs one-hot encoding, and separates features and targets.
    """
    print(f"Loading data from {filepath}...")
    try:
        df = pd.read_csv(filepath) # Try to load the CSV file
    except FileNotFoundError:
        print(f"Error: File not found at {filepath}")
        print("Please update the 'FILE_PATH' variable with the correct path to your CSV file.")
        return None, None, None

    # Ensure all specified feature columns exist before proceeding
    required_cols = FEATURES + TARGETS + [STRATA_COL]
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Required columns not found in the DataFrame: {missing_cols}")


    initial_rows = len(df)
    print(f"Initial number of rows: {initial_rows}")
    
    # Drop rows where 'BASFPH' is greater than 100
    df = df[(df['BASFPH'] <= 100) & (df['BASFPH_next_INVNR'] <= 100)].copy()

    # drop all rows that are not INVNR 150, 250 or 350 or 450
    df = df[df['INVNR'].isin([150, 250, 350, 450])].copy()

    # Define all columns to check for missing values
    columns_to_check_for_na = FEATURES + TARGETS
    
    # Drop any rows that have missing values in the specified feature or target columns
    df_clean = df.dropna(subset=columns_to_check_for_na).copy()
    
    rows_after_drop = len(df_clean)
    print(f"Number of rows after dropping NAs in features/targets: {rows_after_drop} ({initial_rows - rows_after_drop} rows removed).")

    # One-Hot Encode Categorical Features
    print(f"Performing one-hot encoding for: {CATEGORICAL_FEATURES}")
    df_processed = pd.get_dummies(df_clean, columns=CATEGORICAL_FEATURES, drop_first=True)

    # Get the list of original numeric features
    numeric_features = [f for f in FEATURES if f not in CATEGORICAL_FEATURES]
    # Get the list of new one-hot encoded column names
    ohe_features = [col for col in df_processed.columns if col.startswith(tuple(cat + '_' for cat in CATEGORICAL_FEATURES))]
    # Combine them to get the final list of features for the model
    final_features = numeric_features + ohe_features
    
    # Separate features (X), targets (y), and stratification column
    X = df_processed[final_features]
    y = df_processed[TARGETS]
    strata_series = df_processed[STRATA_COL]
    
    print("Data loading and preprocessing complete.")
    print(f"Shape of feature matrix X: {X.shape}")
    
    return X, y, strata_series


# --- 2. Model Tuning Functions ---

def tune_lasso_cv(X, y_series, strata_col):
    """
    Tunes Lasso using LassoCV with stratified K-fold cross-validation.
    """
    print(f"\n--- Tuning LassoCV for target: {y_series.name} ---")
    
    # Define the cross-validation strategy
    cv = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE)
    
    # Define a continuous-like range of alphas for LassoCV to test
    alphas = np.logspace(ALPHA_LOGSPACE_START, ALPHA_LOGSPACE_END, ALPHA_NUM_STEPS)
    
    # Create a pipeline with a scaler and the LassoCV model
    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('lasso', LassoCV(alphas=alphas, cv=cv.split(X, strata_col), random_state=RANDOM_STATE, max_iter=LASSO_MAX_ITER, n_jobs=-1))
    ])
    
    # Fit the model to the data
    pipeline.fit(X, y_series)
    
    best_alpha = pipeline.named_steps['lasso'].alpha_
    
    print(f"Best alpha found: {best_alpha:.6f}")
    
    return {'best_alpha': best_alpha}


def tune_xgboost_optuna(X, y_series, strata_col, target_name): # No scaler needed for tree-based models
    """
    Tunes XGBoost using Optuna with stratified K-fold cross-validation.
    """
    print(f"\n--- Tuning XGBoost with Optuna for target: {target_name} ---")
    
    cv = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE)

    def objective(trial):
        """
        The objective function for Optuna to minimize.
        """
        
        params = {
            'objective': 'reg:squarederror',  # Squared error objective for regression
            'eval_metric': 'rmse',  # Evaluation metric as root mean squared error
            'n_estimators': trial.suggest_int('n_estimators', 200, 2000),  # Number of boosting rounds
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),  # Learning rate
            'max_depth': trial.suggest_int('max_depth', 4, 12),  # Max depth of tree
            'min_child_weight': trial.suggest_int('min_child_weight', 1, 10),  # Minimum child weight
            'subsample': trial.suggest_float('subsample', 0.5, 1.0),  # Fraction of samples to train on
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),  # Fraction of features to use
            'gamma': trial.suggest_float('gamma', 1e-8, 1.0, log=True),  # Regularization parameter
            'reg_lambda': trial.suggest_float('lambda', 1e-8, 1.0, log=True),  # L2 regularization
            'reg_alpha': trial.suggest_float('alpha', 1e-8, 1.0, log=True),  # L1 regularization
            
            'random_state': RANDOM_STATE,  # Ensures reproducibility
            'n_jobs': -1  # Use all available cores for parallelism
        }
        
        model = xgb.XGBRegressor(**params)
        
        rmse_scores = [] # List to store RMSE scores for each fold
        
        # Perform cross-validation for model evaluation
        for fold, (train_idx, val_idx) in enumerate(cv.split(X, strata_col)):
            X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
            y_train, y_val = y_series.iloc[train_idx], y_series.iloc[val_idx]
            
            model.fit(X_train, y_train)  # Fit the model to the training data
            
            preds = model.predict(X_val)  # Make predictions on validation data
            rmse = np.sqrt(mean_squared_error(y_val, preds))  # Calculate RMSE
            rmse_scores.append(rmse)  # Store RMSE score

        return np.mean(rmse_scores)  # Return the average RMSE across all folds

    study = optuna.create_study(direction='minimize')  # Create Optuna study to minimize RMSE
    study.optimize(objective, n_trials=N_TRIALS_OPTUNA, show_progress_bar=True)  # Start the optimization
    
    # Print the best trial and hyperparameters found by Optuna
    print(f"Best trial completed with RMSE: {study.best_value:.4f}")
    print("Best hyperparameters found:")
    best_params = study.best_params
    for param, value in best_params.items():
        print(f"  - {param}: {value}")
        
    return best_params


# --- 3. Save Results Function ---

def save_results_to_csv(results_dict, script_path):
    """
    Saves the nested hyperparameter dictionary to a CSV file in the same
    directory as the script.
    """
    records = []
    # Flatten the nested dictionary for easier storage in a dataframe
    for target, models in results_dict.items():
        for model_name, params in models.items():
            for param, value in params.items():
                records.append({
                    'target': target,
                    'model': model_name,
                    'parameter': param,
                    'value': value
                })

    # Check if there are no results to save
    if not records:
        print("\nNo results to save.")
        return

    results_df = pd.DataFrame(records)  # Convert to DataFrame

    script_dir = os.path.dirname(script_path)  # Get the directory of the current script
    output_filename = 'hyperparameter_tuning_results.csv'  # Define the output filename
    output_path = os.path.join(script_dir, output_filename)  # Define the output path

    try:
        results_df.to_csv(output_path, index=False)  # Save the results to CSV
        print(f"\nSuccessfully saved hyperparameter results to:\n{output_path}")
    except Exception as e:
        print(f"\nAn error occurred while saving the results to CSV: {e}")


# --- 4. Main Execution Block ---

def main():
    """
    Main function to run the entire tuning pipeline.
    """
    X, y, strata_series = load_and_preprocess_data(FILE_PATH)
    
    if X is None:
        return

    all_best_params = {}  # Dictionary to store best parameters for each target

    # Loop over all targets and tune models
    for target in TARGETS:
        print("\n" + "="*50)
        print(f"Processing Target Variable: {target}")
        print("="*50)
        
        y_series = y[target]  # Get the target series
        all_best_params[target] = {}  # Initialize dictionary for this target
        
        lasso_params = tune_lasso_cv(X, y_series, strata_series)  # Tune LassoCV
        all_best_params[target]['Lasso'] = lasso_params  # Save the best alpha for Lasso
        
        xgboost_params = tune_xgboost_optuna(X, y_series, strata_series, target_name=target)  # Tune XGBoost
        all_best_params[target]['XGBoost'] = xgboost_params  # Save the best params for XGBoost

    try:
        # Get the path of the current script to save the CSV in the same directory
        script_path = os.path.realpath(__file__)
        save_results_to_csv(all_best_params, script_path)  # Save results to CSV
    except NameError:
        # __file__ is not defined in some environments (like Jupyter notebooks)
        print("\nCould not determine script path automatically.")
        print("Saving results to the current working directory.")
        save_results_to_csv(all_best_params, os.getcwd())  # Save results to current working directory


if __name__ == '__main__':
    main()  # Run the main function
