import pandas as pd
import numpy as np
import xgboost as xgb
from tqdm import tqdm
from pathlib import Path
import warnings
from joblib import Parallel, delayed
import traceback

# suppress FutureWarnings and chained assignment warnings
warnings.filterwarnings('ignore', category=FutureWarning) 
pd.options.mode.chained_assignment = None 

# --- Configuration ---

# Define the RCP scenario and other parameters
RCP_SCENARIO = "RCP26"
N_SEEDS = 100
RANDOMSEED = 35
N_JOBS = 16 # Number of cores to use for parallel processing

# File Paths
base_dir = Path("../data/preprocessed/final_nfi_ch2018_merged") 
input_file = base_dir / f"NFI_with_Climate_Averages_{RCP_SCENARIO}.csv"
output_dir = Path("../data/predictions") / f"Predictions_{RCP_SCENARIO}"
output_filename_template = f"NFI_{RCP_SCENARIO}_prediction_seed_{{seed}}.csv"

# Feature and Target Definitions
FEATURES = [
    'mean_dry_days_count', 'mean_frost_days_count', 'mean_gdd_sum',
    'mean_pr_sum', 'mean_pr_variance', 'mean_tas_mean', 'mean_tas_variance',
    'mean_tasmax_mean', 'mean_tasmax_variance', 'mean_tasmin_mean',
    'mean_tasmin_variance', 'HWSW_prop', 'INVYR', 'BASFPH_squared',
    'Time_Diff_years', 'BASFPH', 'BEWIRTINT1', 'ASPECT25',
    'SLOPE25', 'PH', 'Z25', 'NAISHSTKOMB'
]
CATEGORICAL_FEATURES = ['BEWIRTINT1', 'NAISHSTKOMB']
TARGETS = ['BASFPH_next_INVNR', 'HWSW_prop_next_INVNR']
NUMERIC_FEATURES = [f for f in FEATURES if f not in CATEGORICAL_FEATURES]

# XGBoost Hyperparameter
HYPERPARAMETERS = {
    'BASFPH_next_INVNR': {
        'n_estimators': 645,
        'learning_rate': 0.011181740295145053,
        'max_depth': 5,
        'min_child_weight': 10,
        'subsample': 0.9889076467082099,
        'colsample_bytree': 0.5273417972412774,
        'gamma': 0.020892473680481548,
        'reg_lambda': 4.768498243134519e-08,   
        'reg_alpha': 0.991303765218906,        
        'objective': 'reg:squarederror'
    },
    'HWSW_prop_next_INVNR': {
        'n_estimators': 679,
        'learning_rate': 0.010009114988628689,
        'max_depth': 4,
        'min_child_weight': 6,
        'subsample': 0.9114788633891485,
        'colsample_bytree': 0.991491129101056,
        'gamma': 4.939930865269561e-05,
        'reg_lambda': 0.0022122803936371204,   
        'reg_alpha': 1.6908226810244194e-07,  
        'objective': 'reg:squarederror'
    }
}


def get_model(target_name, seed):
    """Initializes an XGBoost model for a specific target with a given seed."""
    params = HYPERPARAMETERS[target_name].copy()
    params['random_state'] = seed
    params['n_jobs'] = 1  # Each model runs in a single thread within the parallel job
    return xgb.XGBRegressor(**params)

def process_single_seed(seed, train_data, prediction_data, final_feature_cols, original_data_for_saving):
    """
    Worker function for a single seed run. Trains models and performs iterative prediction.
    Receives pre-processed dataframes and the original, unmodified dataframe for final saving.
    """
    try:
        # --- Prepare Training Data ---
        X_train = train_data[final_feature_cols]
        y_train = train_data[TARGETS]

        # --- Train Two Independent Models ---
        model_basfph = get_model('BASFPH_next_INVNR', seed)
        model_basfph.fit(X_train, y_train['BASFPH_next_INVNR'])

        model_hwsw = get_model('HWSW_prop_next_INVNR', seed)
        model_hwsw.fit(X_train, y_train['HWSW_prop_next_INVNR'])

        # --- Prepare Full Data for Iterative Prediction ---
        prediction_df_processed = prediction_data.copy()

        # --- Run Iterative Prediction Loop ---
        for clnr, plot_data in prediction_df_processed.groupby('CLNR'):
            # Find the row where INVNR == 550 and BASFPH is not NaN
            start_row = plot_data[(plot_data['INVNR'] == 550) & plot_data['BASFPH'].notna()]
            if start_row.empty:
                continue  # Skip if no valid starting point

            # Start from the first valid row
            plot_indices = plot_data.index.to_list()
            start_idx = plot_indices.index(start_row.index[0])

            # Iterate forward and predict missing values
            for i in range(start_idx, len(plot_indices) - 1):
                current_idx = plot_indices[i]
                next_idx = plot_indices[i + 1]

                current_features = prediction_df_processed.loc[[current_idx], final_feature_cols]

                predicted_basfph = np.clip(model_basfph.predict(current_features)[0], 0, 100)
                predicted_hwsw_prop = np.clip(model_hwsw.predict(current_features)[0], 0, 1)

                prediction_df_processed.at[next_idx, 'BASFPH'] = predicted_basfph
                prediction_df_processed.at[next_idx, 'HWSW_prop'] = predicted_hwsw_prop
                prediction_df_processed.at[next_idx, 'BASFPH_squared'] = predicted_basfph ** 2



        # --- Finalize and Save Results for this Seed ---
        # Extract only the keys and updated columns from our prediction results.
        predicted_values_df = prediction_df_processed[['CLNR', 'INVYR', 'BASFPH', 'HWSW_prop', 'BASFPH_squared']].copy()

        # Use the original dataframe. Drop old columns that will be replaced.
        output_df = original_data_for_saving.drop(columns=['BASFPH', 'HWSW_prop' , 'BASFPH_squared'], errors='ignore')

        # Merge the original data structure with our new predicted values.
        final_df = pd.merge(output_df, predicted_values_df, on=['CLNR', 'INVYR'], how='left')
        
        
        # Save to CSV
        current_output_path = output_dir / output_filename_template.format(seed=seed)
        final_df.to_csv(current_output_path, index=False, float_format='%.4f')
        return True
    
    except Exception as e:
        print(f"Error in seed {seed}: {e}")
        traceback.print_exc()
        return False


if __name__ == "__main__":
    print("--- Running Parallel Forest Prediction Script ---")

    master_rng = np.random.RandomState(RANDOMSEED)
    SEEDS = master_rng.randint(0, 100000, N_SEEDS)
    output_dir.mkdir(parents=True, exist_ok=True)

    # --- Load and Prepare Data Globally  ---
    try:
        full_data = pd.read_csv(input_file)
        print(f"Data loaded successfully. Shape: {full_data.shape}")
    except FileNotFoundError:
        print(f"FATAL: The file was not found at {input_file}.")
        exit()

    # Apply initial filters
    # Keep rows where value <= 100 OR value is NaN (future rows)
    full_data = full_data[(full_data['BASFPH'].isna()) | (full_data['BASFPH'] <= 100)].copy()
    full_data = full_data[(full_data['BASFPH_next_INVNR'].isna()) | (full_data['BASFPH_next_INVNR'] <= 100)].copy()


    # Create a safe copy of the original data before any processing for the final save step
    original_data_for_saving = full_data.copy()

    # Perform one-hot encoding on the entire dataset.
    full_data_processed = pd.get_dummies(full_data, columns=CATEGORICAL_FEATURES, drop_first=True)
    print("One-hot encoding performed on the entire dataset.")

    # Ensure all numeric features are present and handle NaNs
    ohe_features = [col for col in full_data_processed.columns if col.startswith(tuple(cat + '_' for cat in CATEGORICAL_FEATURES))]
    FINAL_FEATURE_COLS = NUMERIC_FEATURES + ohe_features
    print(f"Total features after one-hot encoding: {len(FINAL_FEATURE_COLS)}")
    
    # Create the training set by filtering the fully processed data.
    train_data_processed = full_data_processed.dropna(subset=TARGETS).copy()
    train_data_processed.dropna(subset=FINAL_FEATURE_COLS, inplace=True) # Ensure no NaNs in features
    print(f"Prepared {len(train_data_processed)} historical records for training.")

    # Prepare the full dataset for prediction (sorting)
    # The processed data contains the future rows with NaNs to be filled
    prediction_data_processed = full_data_processed.sort_values(by=['CLNR', 'INVYR'], inplace=False, ignore_index=True)

    # --- Run Parallel Predictions ---
    print(f"\nStarting parallel execution on {N_JOBS} cores for {N_SEEDS} seeds...")
    results = Parallel(n_jobs=N_JOBS)(
        delayed(process_single_seed)(
            seed, 
            train_data_processed, 
            prediction_data_processed, 
            FINAL_FEATURE_COLS, 
            original_data_for_saving # Pass the original, unprocessed data
        ) 
        for seed in tqdm(SEEDS, desc="Overall Progress")
    )

    successful_runs = sum(results)
    print(f"\n{'='*25} PREDICTION RUNS COMPLETED {'='*25}")
    print(f"Total Successful Runs: {successful_runs} / {N_SEEDS}")
    if successful_runs < N_SEEDS:
        print("Some runs failed. Please check logs for errors.")
    
    print(f"Output files saved in: {output_dir}")
    print("\n--- Script Finished ---")