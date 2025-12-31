import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

import sys
import os
import re
import json

import numpy as np
import pandas as pd
import random
from matplotlib import pyplot as plt

from tqdm import trange

# Import ALS and rank estimate
from sequential_python_als import SequentialPythonALS, rank_estimate_mice_svd

print(f"System version: {sys.version}")

COL_USER = "UserId"
COL_ITEM = "MovieId"
COL_RATING = "Rating"
# Use the directory where this script is located
PATH = os.path.dirname(os.path.abspath(__file__)) + "/"
DATA_DIR = os.path.join(PATH, "data")
RESULTS_DIR = os.path.join(PATH, "results")
LOGS_DIR = os.path.join(PATH, "log")

# Create directories if they don't exist
os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(LOGS_DIR, exist_ok=True)

header = {
    "userCol": COL_USER,
    "itemCol": COL_ITEM,
    "ratingCol": COL_RATING,
}


regrets_separate = [0]
regrets_random = [0]



# ========================================================================
# CONFIGURATIONS
# ========================================================================
num_machines = 50
intended_rank = 15
num_pars = 50
missing_rate = 80
missing_N = int(num_pars * missing_rate/100)
M_budget = 15 #missing_N
UseIntendedBool = True
lambda_val = 0.001
ALSInitStrategy = 'conservative' # choices: conservative / data_driven
small_matrix_rows = 5 # m1

# SWITCH: Control whether to save selection logs
SAVE_SELECTION_LOGS = False  # Set to True to save selection logs to JSON files
directory = '.'

# Enable detailed logging for first iteration
enable_detailed_logging = False

# Load the test matrix file
data_np = np.genfromtxt(os.path.join(DATA_DIR, f'FPy_random_matrix_{num_machines}_{num_pars}_{intended_rank}.csv'), delimiter=',')

# Load row mapping to track which U_all rows correspond to each data_np row
row_mapping_file = os.path.join(DATA_DIR, f'FPy_random_matrix_{num_machines}_{num_pars}_{intended_rank}_row_mapping.json')
with open(row_mapping_file, 'r') as f:
    data_np_to_U_all_mapping = json.load(f)

print(f"\n{'='*70}")
print(f"Loaded row mapping: {row_mapping_file}")
print(f"  data_np has {data_np.shape[0]} rows")
print(f"{'='*70}\n")

# Find j_k^{true} = argmax_j U_{k,j} for each machine k BEFORE masking
# This identifies the TRUE OPTIMAL item for each machine based on the fully observed matrix
true_max_indices = np.argmax(data_np, axis=1)

# ========================================================================
# DATA PREPARATION
# ========================================================================

row_size, col_size = data_np.shape[0], data_np.shape[1]

# Pre-allocate arrays for better performance
train_data_list = []
test_data_list = []
# Store solo data in memory (no file I/O)
train_solo_dict = {}
test_solo_dict = {}
masked_result_matrix = np.full((row_size, col_size), np.nan)

# Calculate uniform separation between unmasked elements
unmasked_ratio = (100 - missing_rate) / 100
num_unmasked = int(col_size * unmasked_ratio)


# ========================================================================
# LOAD MASKING METADATA
# ========================================================================
# Load pre-computed masking metadata based on missing_rate
print(f"\n{'='*70}")

# Load masking metadata from the new format (single JSON file)
metadata_filename = os.path.join(DATA_DIR, f"FPy_masking_metadata_{missing_rate}pct.json")
with open(metadata_filename, 'r') as f:
    masking_metadata = json.load(f)

# Extract components from the metadata
metadata_train_cols = masking_metadata['train_cols_per_row']
metadata_masked_cols = masking_metadata['masked_cols_per_row']
metadata_true_optimal_dict = masking_metadata['true_optimal_indices']

# Convert true_optimal_indices dict to array for compatibility
max_idx = max(int(k) for k in metadata_true_optimal_dict.keys())
metadata_true_optimal = np.zeros(max_idx + 1, dtype=int)
for idx_str, opt_col in metadata_true_optimal_dict.items():
    metadata_true_optimal[int(idx_str)] = opt_col

print(f"Loaded masking metadata from: {metadata_filename}")
print(f"  - Total machines: {len(metadata_train_cols)}")
print(f"  - Missing rate: {missing_rate}%")
print(f"{'='*70}\n")

# Verify metadata has correct number of rows for current data
# This should always match since masking metadata is generated for the same U_extended matrix
assert len(metadata_train_cols) >= row_size, \
    (f"Metadata has insufficient rows: {len(metadata_train_cols)} < {row_size}. "
     f"This indicates the masking metadata was generated for a different matrix. "
     f"Please regenerate masking metadata using generate_random_matrix_c.py with the correct U_extended matrix.")



# Masking using METADATA
# Track which columns have at least one unmasked element
column_coverage = np.zeros(col_size, dtype=bool)

# Initialize row-wise rank estimates storage
rowwise_r_est = np.zeros(row_size, dtype=int)

for row in trange(row_size, desc='Masking (using metadata)'):
    full_col_set = np.arange(col_size)

    # USE ROW MAPPING: Map data_np row index to U_all row index
    # This ensures we get the correct metadata for this specific row
    u_all_row_idx = data_np_to_U_all_mapping[str(row)]

    # Verify that row mapping is valid (should never be -1 with updated code)
    assert u_all_row_idx != -1, \
        (f"Row {row} has invalid mapping (-1). This indicates legacy data format. "
         f"Please regenerate data using updated generate_random_matrix_c.py which properly "
         f"handles synthetic rows in the row_mapping file.")

    # USE METADATA: Get train and test columns from pre-computed metadata
    # This works for both real rows (index < 200) and synthetic rows (index >= 200)
    train_cols = np.array(metadata_train_cols[str(u_all_row_idx)])
    test_cols = np.array(metadata_masked_cols[str(u_all_row_idx)])

    # Verify metadata consistency
    assert metadata_true_optimal[u_all_row_idx] in test_cols, \
        f"Row {row} (U_all row {u_all_row_idx}): true optimal not in masked cols!"
    assert len(set(train_cols) & set(test_cols)) == 0, \
        f"Row {row} (U_all row {u_all_row_idx}): overlap between train and test!"
    assert len(train_cols) + len(test_cols) == col_size, \
        f"Row {row} (U_all row {u_all_row_idx}): train+test != col_size!"

    # Mark these columns as covered
    column_coverage[train_cols] = True

    # Create masks for vectorized operations
    train_mask = np.isin(full_col_set, train_cols)
    test_mask = ~train_mask

    masked_result_matrix[row, train_cols] = data_np[row, train_cols]


    # Build train data for global model
    train_rows = np.full(len(train_cols), row)
    train_ratings = data_np[row, train_cols]
    train_batch = np.column_stack([train_rows, train_cols, train_ratings])
    train_data_list.append(train_batch)

    # Build test data for global model
    test_rows = np.full(len(test_cols), row)
    test_ratings = data_np[row, test_cols]
    test_batch = np.column_stack([test_rows, test_cols, test_ratings])
    test_data_list.append(test_batch)

    # Build solo train/test data (transformed coordinates)
    train_solo_user = train_cols // small_matrix_rows
    train_solo_item = train_cols % small_matrix_rows
    train_solo_data = np.column_stack([train_solo_user, train_solo_item, train_ratings])

    test_solo_user = test_cols // small_matrix_rows
    test_solo_item = test_cols % small_matrix_rows
    test_solo_data = np.column_stack([test_solo_user, test_solo_item, test_ratings])

    # Store solo data in memory for later use (no file I/O)
    train_solo_dict[row] = train_solo_data
    test_solo_dict[row] = test_solo_data




# Concatenate all batches at once (much faster than incremental DataFrame operations)
train_np = np.vstack(train_data_list)
test_np = np.vstack(test_data_list)

# Create Spark DataFrames directly from numpy arrays (NO FILE I/O)
# Method: Use pandas as intermediate
train_pd = pd.DataFrame(train_np, columns=['UserId', 'MovieId', 'Rating'])
test_pd = pd.DataFrame(test_np, columns=['UserId', 'MovieId', 'Rating'])

# Ensure correct data types for ALS requirements
train_pd['UserId'] = train_pd['UserId'].astype('int32')
train_pd['MovieId'] = train_pd['MovieId'].astype('int32')
train_pd['Rating'] = train_pd['Rating'].astype('float32')

test_pd['UserId'] = test_pd['UserId'].astype('int32')
test_pd['MovieId'] = test_pd['MovieId'].astype('int32')
test_pd['Rating'] = test_pd['Rating'].astype('float32')


cumu_regrets_largest_colab = np.zeros(shape=(row_size, M_budget))
cumu_regrets_largest_solo = np.zeros(shape=(row_size, M_budget))
first_find_minimal_tab = M_budget*np.ones(shape=(num_machines, 4))
result_tab = np.zeros(shape=(num_machines, 4))


# Rank estimation
print(f"\n{'='*70}")
estimated_rank, masked_imputed = rank_estimate_mice_svd(masked_result_matrix)
print(f"Using estimated rank: {estimated_rank} for matrix completion (intended: {intended_rank})")
print("="*70 + "\n")


# Convert pandas DataFrames to numpy arrays directly
train_np_current = train_pd.to_numpy()
test_np_current = test_pd.to_numpy()

if UseIntendedBool:
    r_exp = intended_rank
else:
    r_exp = estimated_rank

pbar = trange(M_budget, desc='Collab. (Python ALS)')


# Compute regret_max_dict ONCE before the loop starts
# This should be the maximum rating among ALL test items for each user
# NOT recomputed every iteration (which would shrink as items are selected)
regret_max_dict = {}
for row in test_np_current:
    user_id = int(row[0])
    rating = row[2]
    if user_id not in regret_max_dict:
        regret_max_dict[user_id] = rating
    else:
        regret_max_dict[user_id] = max(regret_max_dict[user_id], rating)


# LOGGING: Track all selections for each machine at every trial
# Format: {machine_id: [{'trial': t, 'item': item_idx, 'rating': rating, 'regret': regret, 'is_optimal': bool}, ...]}
selection_log_colab = {i: [] for i in range(num_machines)}

for iter_i in range(M_budget):
    model = SequentialPythonALS(
        rank=r_exp,
        max_iter=15,
        reg_param=lambda_val,
        random_state=42,
        init_strategy=ALSInitStrategy
    )
    # CRITICAL: Pass explicit matrix dimensions to ensure full matrix structure
    # This ensures ALS always works with num_machines × num_pars matrix (50×50)
    # even if some entries are not yet observed in training data
    model.fit(train_np_current, n_users=num_machines, n_items=num_pars)

    if enable_detailed_logging and iter_i == 0:
        print(f"\n{'='*70}")
        print(f"COLLABORATIVE METHOD - Iteration {iter_i} - Detailed Logging")
        print(f"{'='*70}")
        print(f"\nMatrix Dimensions:")
        print(f"  Expected: {num_machines}×{num_pars} (machines × parameters)")
        print(f"  ALS factors: user_factors={model.user_factors.shape}, item_factors={model.item_factors.shape}")
        print(f"  Training entries: {len(train_np_current)}")
        print(f"  Test entries: {len(test_np_current)}")

    # Generate predictions for test data
    predictions = model.predict(test_np_current)

    if len(predictions) == 0:
        break

    # Extract factors for matrix operations
    matrix_X = model.user_factors  # (n_users, rank)
    matrix_Y = model.item_factors  # (n_items, rank)

    if enable_detailed_logging and iter_i == 0:
        print(f"\nFactor Matrix Statistics:")
        print(f"  matrix_X shape: {matrix_X.shape}")
        print(f"  matrix_Y shape: {matrix_Y.shape}")
        print(f"  matrix_X[0] L2 norm: {np.linalg.norm(matrix_X[0]):.6f}")
        print(f"  matrix_Y[0] L2 norm: {np.linalg.norm(matrix_Y[0]):.6f}")
        print(f"  matrix_X[0, :5]: {matrix_X[0, :5]}")
        print(f"  matrix_Y[0, :5]: {matrix_Y[0, :5]}")

    # Build test_dict: mapping from user_id to list of (item_id, rating) tuples
    test_dict = {}
    for row in test_np_current:
        user_id = int(row[0])
        item_id = int(row[1])
        rating = row[2]
        if user_id not in test_dict:
            test_dict[user_id] = []
        test_dict[user_id].append((item_id, rating))


    candidates = []

    avg_r = 0
    avg_t = 0

    for printer_i in range(row_size):
        # coldStartStrategy='drop': skip if user not in test data
        if printer_i not in test_dict:
            continue

        # coldStartStrategy='drop': skip if user is unknown (beyond learned factors)
        if printer_i >= matrix_X.shape[0]:
            continue

        # Vectorized utility computation for all test items of this user
        test_items = np.array([item for item, _ in test_dict[printer_i]])

        # Filter out items beyond learned factors (coldStartStrategy='drop')
        valid_mask = test_items < matrix_Y.shape[0]
        if not np.any(valid_mask):
            continue

        test_items_valid = test_items[valid_mask]
        user_vec = matrix_X[printer_i]
        item_vecs = matrix_Y[test_items_valid]

        # Compute predictions (explicit feedback: dot product)
        utilities = user_vec @ item_vecs.T

        argmax_idx = np.argmax(utilities)
        predicted_utility = utilities[argmax_idx]

        # Map back to original test_dict index
        valid_indices = np.where(valid_mask)[0]
        original_idx = valid_indices[argmax_idx]

        user = printer_i
        movie = test_dict[printer_i][original_idx][0]
        rating = test_dict[printer_i][original_idx][1]

        regret_max = regret_max_dict.get(printer_i, 0)

        # Detailed logging for first printer in first iteration
        if enable_detailed_logging and iter_i == 0 and printer_i == 0:
            print(f"\nPrinter {printer_i} Selection Details:")
            print(f"  Number of test items: {len(test_dict[printer_i])}")
            print(f"  Test items (first 5): {test_items[:5]}")
            print(f"  User vector norm: {np.linalg.norm(user_vec):.6f}")
            print(f"  Computed utilities (first 5): {utilities[:5]}")
            #print(f"  Max utility: {utilities[argmax_idx]:.6f} at index {argmax_idx}")
            print(f"  Selected movie: {movie}, rating: {rating:.4f}")
            print(f"  Regret baseline (min rating): {regret_max:.4f}")
            print(f"  Regret: {abs(rating - regret_max):.4f}")

        candidates.append((user, movie))

        # LOGGING: Record this selection for collaborative method
        is_optimal = (movie == true_max_indices[printer_i])
        selection_log_colab[printer_i].append({
            'trial': iter_i + 1,  # 1-indexed trial number
            'item': int(movie),
            'rating': float(rating),
            'regret': float(abs(rating - regret_max)),
            'is_optimal': bool(is_optimal)
        })

        # Calculate and store regret (matches PySpark structure)
        if iter_i == 0:
            cumu_regrets_largest_colab[printer_i, iter_i] = abs(rating - regret_max)
            if abs(rating - regret_max)==0:
                first_find_minimal_tab[printer_i, 0] = min(iter_i, first_find_minimal_tab[printer_i, 0])
        else:
            cumu_regrets_largest_colab[printer_i, iter_i] = abs(rating - regret_max)
            if abs(rating - regret_max)==0:
                first_find_minimal_tab[printer_i, 0] = min(iter_i, first_find_minimal_tab[printer_i, 0])

        # Terminal cumulative regret:
        # regret^(t)_k = |U^(t)_{k,j*} - U_{k,j_true}| where j_true = argmin U_{k,j}
        result_tab[printer_i, 0] += abs(rating - data_np[printer_i, true_max_indices[printer_i]])
        
        # Track t_k^*: first trial where machine k selects its true optimal configuration
        # Definition: t_k^* = min{t : x^(t)_{j_k^*} = x_k^*}
        # This checks if selected movie index equals true optimal index for this machine
        # Store as (iter_i + 1) so that: 0=not found, >0=found at trial
        if abs(rating - regret_max)==0:
            if result_tab[printer_i, 2] == 0:  # Only update if not found yet
                result_tab[printer_i, 2] = iter_i + 1  # 1-indexed: iteration 0 becomes 1, etc.

        
        avg_r += result_tab[printer_i, 0]
        avg_t += result_tab[printer_i, 2]

    # Batch update train/test splits in numpy
    if len(candidates) > 0:
        # Find rows to move from test to train
        move_mask = np.zeros(len(test_np_current), dtype=bool)
        for user_i, movie_i in candidates:
            move_mask |= (test_np_current[:, 0] == user_i) & (test_np_current[:, 1] == movie_i)

        rows_to_move = test_np_current[move_mask]

        if len(rows_to_move) > 0:
            # Update train and test arrays
            train_np_current = np.vstack([train_np_current, rows_to_move])
            test_np_current = test_np_current[~move_mask]

    # Summary logging for first iteration
    if enable_detailed_logging and iter_i == 0:
        print(f"\n{'='*70}")
        print(f"Iteration {iter_i} Summary:")
        print(f"  Total candidates selected: {len(candidates)}")
        print(f"  First 3 candidates (user, movie): {candidates[:3]}")
        print(f"  Average cumulative regret: {avg_r/row_size:.4f}")
        print(f"{'='*70}\n")

    pbar.set_postfix({'avg: CR/t_bar': f"{(avg_r/row_size):.2f}/{(avg_t/row_size):.2f}"})
    pbar.update(1)

pbar.close()


def train_solo_printer_sequential(printer_i, train_data_init, test_data_init, missing_N_val,
                                   rank, max_iter, reg_param, true_max_index, data_row, small_matrix_rows_val, first_it_log):
    """
    Train solo model for a single printer using sequential pure Python ALS.

    Args:
        printer_i: printer index
        train_data_init: initial training data
        test_data_init: initial test data
        missing_N_val: number of missing/test items
        rank: ALS rank
        max_iter: maximum ALS iterations
        reg_param: regularization parameter
        true_max_index: true maximum index in global coordinates
        data_row: full data row for this printer
        small_matrix_rows_val: number of rows in small matrix

    Returns:
        tuple of (printer_i, result_metrics, cumu_regrets, first_find_iteration)
    """
    # Make copies to avoid modifying shared data
    train_data = train_data_init.copy()
    test_data = test_data_init.copy()

    # Initialize results
    cumu_regrets = np.zeros(missing_N_val)
    first_find = missing_N_val  # Default: not found
    cumulative_regret = 0.0
    time_found = 0

    # LOGGING: Track all selections for this machine
    selection_log = []

    # Calculate matrix dimensions for the reshaped row
    # Each row has 50 columns, reshaped as (50//small_matrix_rows_val) × small_matrix_rows_val
    # With small_matrix_rows_val=5: 50//5 = 10 users, 5 items per user -> 10×5 matrix
    n_users_solo = 50 // small_matrix_rows_val  # Should be 10
    n_items_solo = small_matrix_rows_val  # Should be 5

    # Iterative training loop
    for iter_i in range(missing_N_val):
        if len(test_data) == 0:
            break

        # Fit ALS model with sequential implementation
        # Pass explicit matrix dimensions (10×5) to ensure full matrix structure
        model = SequentialPythonALS(rank=rank, max_iter=max_iter, reg_param=reg_param, random_state=42, init_strategy=ALSInitStrategy)
        model.fit(train_data, n_users=n_users_solo, n_items=n_items_solo)

        # Generate predictions for test data
        predictions = model.predict(test_data)

        if len(predictions) == 0:
            break

        # Find minimum prediction and minimum rating
        max_pred_idx = np.argmax(predictions[:, 3])
        max_rating = np.max(predictions[:, 2]) 
        predicted_utility = predictions[max_pred_idx, 3]

        # Extract selected item
        selected_user = int(predictions[max_pred_idx, 0])
        selected_item = int(predictions[max_pred_idx, 1])
        selected_rating = predictions[max_pred_idx, 2]

        # Convert from solo coordinates to global coordinates
        global_col = selected_user * small_matrix_rows_val + selected_item

        # LOGGING: Record this selection for sequential method
        is_optimal = (global_col == true_max_index)
        regret_value = abs(selected_rating - max_rating)
        selection_log.append({
            'trial': iter_i + 1,  # 1-indexed trial number
            'item': int(global_col),
            'rating': float(selected_rating),
            'regret': float(regret_value),
            'is_optimal': bool(is_optimal)
        })

        # Detailed logging for first printer in first iteration
        if printer_i == 0 and iter_i == 0 and first_it_log:
            print(f"\n{'='*70}")
            print(f"SOLO METHOD - Printer {printer_i}, Iteration {iter_i} - Detailed Logging")
            print(f"{'='*70}")
            print(f"\nMatrix Dimensions:")
            print(f"  Reshaped matrix: {n_users_solo}×{n_items_solo} (users × items)")
            print(f"  ALS factors: user_factors={model.user_factors.shape}, item_factors={model.item_factors.shape}")
            print(f"  Training entries: {len(train_data)}")
            print(f"\nPrediction Statistics:")
            print(f"  Total predictions: {len(predictions)}")
            print(f"  Predictions (first 5):")
            for idx in range(min(5, len(predictions))):
                print(f"    [{idx}] user={int(predictions[idx,0])}, item={int(predictions[idx,1])}, rating={predictions[idx,2]:.4f}, pred={predictions[idx,3]:.4f}")
            print(f"\nSelection Details:")
            print(f"  Max prediction value: {predictions[max_pred_idx, 3]:.6f} at index {max_pred_idx}")
            print(f"  Max rating (regret baseline): {max_rating:.6f}")
            print(f"  Selected: user={selected_user}, item={selected_item}, rating={selected_rating:.4f}")
            print(f"  Global column: {global_col}")
            print(f"  True max index: {true_max_index}")
            print(f"  Regret: {abs(selected_rating - max_rating):.6f}")

        # Alternative regret metric (relative to test set minimum)
        regret = abs(selected_rating - max_rating)
        cumu_regrets[iter_i] = regret

        # Terminal cumulative regret:
        # regret^(t)_k = |U^(t)_{k,j*} - U_{k,j_true}| where j_true = argmin U_{k,j}
        # This accumulates: sum_{t=1}^{M} regret^(t)_k based on FULLY OBSERVED matrix
        cumulative_regret += abs(selected_rating - data_row[true_max_index])

        # Check if we found the minimal rating (matches PySpark line 405)
        if regret == 0:
            first_find = min(iter_i, first_find)

        # Track t_k^*: first trial where machine k selects its true optimal configuration
        # Definition: t_k^* = min{t : x^(t)_{j_k^*} = x_k^*}
        # This checks if selected item (in global coordinates) equals true optimal index
        # Store as (iter_i + 1) so that: 0=not found, >0=found at trial
        if abs(selected_rating - max_rating) == 0: #global_col == true_max_index: # --- CHECK
            if time_found == 0:  # Only update if not found yet
                time_found = iter_i + 1  # 1-indexed: iteration 0 becomes 1, etc.

        # Move selected item from test to train
        test_mask = ~((test_data[:, 0] == selected_user) & (test_data[:, 1] == selected_item))
        selected_row = test_data[~test_mask]

        if len(selected_row) > 0:
            train_data = np.vstack([train_data, selected_row])
            test_data = test_data[test_mask]

    # Prepare result metrics
    result_metrics = {
        'cumulative_regret': cumulative_regret,
        'time_found': time_found,
        'selection_log': selection_log  # Add selection log to results
    }

    return (printer_i, result_metrics, cumu_regrets, first_find)


# ========================================================================
# SEQUENTIAL SOLO MODEL TRAINING USING PURE PYTHON ALS
# ========================================================================
print(f"\n{'='*70}")
print("Starting SEQUENTIAL solo model training using Pure Python ALS")
print(f"{'='*70}\n")

# Initialize results storage
avg_cr = 0
avg_t = 0

# LOGGING: Track all selections for each machine in sequential method
selection_log_solo = {}

# Progress tracking
pbar = trange(row_size, desc='Solo (Sequential)')

# Sequential processing loop
for printer_i in range(row_size):
    train_data = train_solo_dict[printer_i]
    test_data = test_solo_dict[printer_i]

    # Use row-wise estimated rank for this printer
    printer_rank = 2

    # Train solo model for this printer
    printer_i_result, result_metrics, cumu_regrets, first_find = train_solo_printer_sequential(
        printer_i=printer_i,
        train_data_init=train_data,
        test_data_init=test_data,
        missing_N_val=M_budget,
        rank=printer_rank,
        max_iter=15,
        reg_param=lambda_val,
        true_max_index=true_max_indices[printer_i],
        data_row=data_np[printer_i, :],
        small_matrix_rows_val=small_matrix_rows,
        first_it_log=enable_detailed_logging
    )

    # Update result table
    result_tab[printer_i, 1] = result_metrics['cumulative_regret']
    result_tab[printer_i, 3] = result_metrics['time_found']

    # LOGGING: Store selection log for this machine
    selection_log_solo[printer_i] = result_metrics['selection_log']

    # Update cumulative regrets
    cumu_regrets_largest_solo[printer_i, :] = cumu_regrets
    first_find_minimal_tab[printer_i, 2] = first_find

    # Update averages
    avg_cr += result_tab[printer_i, 1]
    avg_t += result_tab[printer_i, 3]

    # Update progress bar with running averages
    pbar.set_postfix({
        'avg: CR/SR/t': f"{(avg_cr/(printer_i+1)):.2f}/{(avg_t/(printer_i+1)):.2f}"
    })
    pbar.update(1)

pbar.close()

print(f"\nFinal averages:")
print(f"  Cumulative Regret: {avg_cr / row_size:.2f}")
print(f"  Average Time Found: {avg_t / row_size:.2f}")
print(f"  Sequential solo model training completed for all {row_size} printers")
print(f"{'='*70}\n")

print(result_tab)

result_tab_filename = os.path.join(RESULTS_DIR, "FRPy_result_tab_" + str(int(num_machines)) + '_' + str(int(num_pars)) + '_' + str(int(intended_rank)) + '_' + str(int(r_exp)) + "_" + str(int(missing_rate)) + "_" + str(int(M_budget)) + ".csv")
np.savetxt(result_tab_filename, result_tab, delimiter=",")

# LOGGING: Save selection logs to JSON files
# These logs contain all selections made by each machine at every trial
if SAVE_SELECTION_LOGS:
    collab_log_filename = os.path.join(LOGS_DIR, "selection_log_collaborative_" + str(int(num_machines)) + '_' + str(int(num_pars)) + '_' + str(int(intended_rank)) + '_' + str(int(r_exp)) + "_" + str(int(missing_rate)) + "_" + str(int(M_budget)) + ".json")
    solo_log_filename = os.path.join(LOGS_DIR, "selection_log_sequential_" + str(int(num_machines)) + '_' + str(int(num_pars)) + '_' + str(int(intended_rank)) + '_' + str(int(r_exp)) + "_" + str(int(missing_rate)) + "_" + str(int(M_budget)) + ".json")

    # Convert integer keys to strings for JSON serialization
    selection_log_colab_str = {str(k): v for k, v in selection_log_colab.items()}
    selection_log_solo_str = {str(k): v for k, v in selection_log_solo.items()}

    with open(collab_log_filename, 'w') as f:
        json.dump(selection_log_colab_str, f, indent=2)

    with open(solo_log_filename, 'w') as f:
        json.dump(selection_log_solo_str, f, indent=2)

    print(f"\n{'='*70}")
    print("SELECTION LOGS SAVED")
    print(f"{'='*70}")
    print(f"  Collaborative method log: {collab_log_filename}")
    print(f"  Sequential method log: {solo_log_filename}")
    print(f"\nLog format (per machine):")
    print(f"  [{{'trial': 1, 'item': 23, 'rating': 0.85, 'regret': 0.02, 'is_optimal': False}}, ...]")
    print(f"{'='*70}\n")
else:
    print(f"\n{'='*70}")
    print("SELECTION LOGS SKIPPED (SAVE_SELECTION_LOGS = False)")
    print(f"{'='*70}\n")
