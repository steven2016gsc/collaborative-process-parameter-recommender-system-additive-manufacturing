import os
import csv
import json
import numpy as np


def generate_random_matrix():
    """
    Generate a 200x50 matrix with local shared structure and uniform spacing.

    Structure:
    1. Generate 50 linearly independent base rows (50-dimensional vectors) using QR decomposition
    2. Partition the 200 machines into 10 groups of 20 rows each
    3. Each group is assigned 5 group-specific base rows from the 50 base rows
    4. Within each group, base rows and linear combinations are interleaved uniformly:
       Pattern: B, L, L, L, B, L, L, L, B, L, L, L, B, L, L, L, B, L, L, L
       (B = base row at positions 0, 4, 8, 12, 16; L = linear combination)
    5. The 15 linear combination rows are random convex combinations of the group's 5 base rows
    6. All entries are normalized to [0, 1]

    Returns:
        np.ndarray: A 200x50 matrix with the specified structure
    """
    # Step 1: Generate 50 linearly independent base rows using QR decomposition
    print("Step 1: Generating 50 linearly independent base rows using QR decomposition...")

    # Generate a random 50x50 matrix and perform QR decomposition
    random_matrix = np.random.rand(50, 50)
    Q, _ = np.linalg.qr(random_matrix)

    # Q is now a 50x50 orthonormal matrix, each row is a linearly independent vector
    base_rows = Q  # 50 base rows, each is a 50-dimensional vector

    # Scale each base row to [0, 1] range
    for i in range(50):
        base_rows[i] = (base_rows[i] - base_rows[i].min()) / (base_rows[i].max() - base_rows[i].min())

    print(f"  Generated {base_rows.shape[0]} linearly independent base rows")
    print(f"  Base rows rank: {np.linalg.matrix_rank(base_rows)}")

    # Step 2: Initialize the 200x50 matrix
    matrix = np.zeros((200, 50))

    # Step 3: Partition into 10 groups and populate each group
    num_groups = 10
    rows_per_group = 20
    base_rows_per_group = 5

    print(f"\nStep 2: Partitioning 200 machines into {num_groups} groups of {rows_per_group} rows each...")

    for group_idx in range(num_groups):
        group_start = group_idx * rows_per_group
        group_end = group_start + rows_per_group

        # Select 5 base rows for this group from the 50 base rows
        # Use deterministic selection: group i gets base rows [5*i, 5*i+1, 5*i+2, 5*i+3, 5*i+4]
        group_base_indices = [5 * group_idx + j for j in range(base_rows_per_group)]
        group_base_rows = base_rows[group_base_indices]

        # Interleave base rows and linear combinations in pattern: 1 base, 3 linear combs
        # Pattern for 20 rows: B, L, L, L, B, L, L, L, B, L, L, L, B, L, L, L, B, L, L, L
        # Where B = base row, L = linear combination
        position = 0
        base_idx = 0

        while position < rows_per_group:
            # Place one base row
            if base_idx < base_rows_per_group:
                matrix[group_start + position] = group_base_rows[base_idx]
                base_idx += 1
                position += 1

            # Place three linear combinations (or as many as needed to fill the group)
            for _ in range(3):
                if position >= rows_per_group:
                    break

                # Generate random coefficients (ensure they're not all zero)
                max_attempts = 1000
                for attempt in range(max_attempts):
                    coeffs = np.random.rand(base_rows_per_group)

                    # Check if coefficients are all effectively zero
                    if np.sum(np.abs(coeffs)) > 1e-10:
                        break

                    if attempt == max_attempts - 1:
                        # Fallback: use uniform weights if random generation keeps failing
                        coeffs = np.ones(base_rows_per_group) / base_rows_per_group

                # Normalize coefficients to sum to 1 (convex combination to keep values in [0, 1])
                coeffs = coeffs / np.sum(coeffs)

                # Compute linear combination
                combination = np.zeros(50)
                for j in range(base_rows_per_group):
                    combination += coeffs[j] * group_base_rows[j]

                matrix[group_start + position] = combination
                position += 1

        print(f"  Group {group_idx}: rows {group_start}-{group_end-1}, using base rows {group_base_indices}")

    # Final verification: Check the rank of the generated matrix
    matrix_rank = np.linalg.matrix_rank(matrix)
    expected_rank = 50  # Should be 50 since we have 50 linearly independent base rows
    print(f"\nMatrix rank verification: {matrix_rank} (expected: {expected_rank})")

    if matrix_rank != expected_rank:
        print(f"WARNING: Generated matrix has rank {matrix_rank}, expected {expected_rank}.")

    return matrix


def apply_masking_to_U_all(U_all, missing_rate, random_seed=128):
    """
    Apply masking strategy to U_all matrix, generating consistent masking patterns per machine ID.

    This function ensures:
    1. Each machine has its true optimal configuration masked
    2. Every configuration retains at least one observation across all machines
    3. The masking pattern for machine ID i is ALWAYS the same (tied to machine index in U_all)

    Parameters:
    -----------
    U_all : np.ndarray
        Full U_all matrix (200 x 50) where row index = machine ID
    missing_rate : int
        Percentage of entries to mask (e.g., 60 means 60% masked)
    random_seed : int
        Base random seed (default: 128)

    Returns:
    --------
    masking_metadata : dict
        Dictionary containing:
        - 'masked_cols_per_row': dict mapping machine_id -> list of masked column indices
        - 'train_cols_per_row': dict mapping machine_id -> list of observed column indices
        - 'true_optimal_indices': dict mapping machine_id -> true optimal column index
    """
    num_machines, num_configs = U_all.shape
    num_unmasked = num_configs - int(num_configs * missing_rate / 100)

    print(f"\nGenerating masking patterns for {num_machines} machines...")
    print(f"  Missing rate: {missing_rate}%")
    print(f"  Observed configs per machine: {num_unmasked}/{num_configs}")

    # Storage for masking patterns (keyed by machine ID)
    masked_cols_per_row = {}
    train_cols_per_row = {}
    true_optimal_indices = {}
    column_coverage = np.zeros(num_configs, dtype=bool)

    # Generate masking pattern for each machine ID
    for machine_id in range(num_machines):
        # Set seed based on machine ID to ensure consistency
        np.random.seed(random_seed + machine_id)

        # Find true optimal configuration for this machine
        true_optimal_col = np.argmax(U_all[machine_id])
        true_optimal_indices[machine_id] = int(true_optimal_col)

        # Create candidate pool: all configs EXCEPT the true optimal
        all_configs = np.arange(num_configs)
        candidate_configs = np.setdiff1d(all_configs, [true_optimal_col])

        # Randomly select which configs to keep unmasked
        shuffled_candidates = np.random.permutation(candidate_configs)
        observed_configs = np.sort(shuffled_candidates[:num_unmasked])

        # Track column coverage
        column_coverage[observed_configs] = True

        # Store the masking pattern for this machine ID
        masked_configs = np.setdiff1d(all_configs, observed_configs)
        masked_cols_per_row[machine_id] = masked_configs.tolist()
        train_cols_per_row[machine_id] = observed_configs.tolist()

    # Ensure every configuration has at least one observation
    uncovered_configs = np.where(~column_coverage)[0]
    if len(uncovered_configs) > 0:
        print(f"  Ensuring coverage for {len(uncovered_configs)} uncovered configurations...")

        for config in uncovered_configs:
            # Find machines where this config is NOT the true optimal
            valid_machine_ids = [mid for mid in range(num_machines)
                                if true_optimal_indices[mid] != config]

            if len(valid_machine_ids) == 0:
                print(f"    WARNING: Config {config} is optimal for all machines, skipping")
                continue

            # Use deterministic selection for consistency
            np.random.seed(random_seed + config)
            selected_machine = np.random.choice(valid_machine_ids)

            # Add this config to the observed set for the selected machine
            if config in masked_cols_per_row[selected_machine]:
                masked_cols_per_row[selected_machine].remove(config)
                train_cols_per_row[selected_machine].append(config)
                train_cols_per_row[selected_machine].sort()
                print(f"    Unmasked config {config} for machine {selected_machine}")

    print(f"  Masking patterns generated for all {num_machines} machines")

    return {
        'masked_cols_per_row': masked_cols_per_row,
        'train_cols_per_row': train_cols_per_row,
        'true_optimal_indices': true_optimal_indices
    }


def save_matrix_to_csv(matrix, filename):
    """save matrix as CSV file."""
    with open(filename, 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerows(matrix)


def save_test_matrix(U_clustered, num_clusters, missing_rate, row_mapping=None):
    U_clustered_masked = U_clustered.copy()
    rows, cols = U_clustered.shape[0], U_clustered.shape[1]
    np.random.seed(128)
    for i in range(rows):
        # Randomly mask 60% of entries
        mask_idx = np.random.choice(cols, size=int(cols * missing_rate/100), replace=False)
        U_clustered_masked[i, mask_idx] = np.nan

    row_size, col_size = U_clustered.shape[0], U_clustered.shape[1]
    small_matrix_rows = 5 # how many rows in the small matrix
    missing_N = int(col_size * missing_rate/100) # missing rate is 80%

    # Create masked version of result_matrix (80x50) with NaN for missing values
    # Initialize with NaN values
    masked_result_matrix = np.full((row_size, col_size), np.nan)

    np.random.seed(128)  # Reset seed to match the masking process
    for row_idx in range(row_size):
        full_col_set = np.arange(col_size)
        diag_set = [i*(col_size//small_matrix_rows)+i for i in range(5)]
        diag_set.append(5)
        diag_set.append(13)
        set_to_be_missed = np.setdiff1d(full_col_set, diag_set)

        # Generate test indices (same random process as before)
        test_id = np.random.choice(set_to_be_missed, missing_N, replace=False)

        # Create train mask and get training columns
        train_mask = ~np.isin(full_col_set, test_id)
        train_cols = full_col_set[train_mask]

        # Fill observed values, leave NaN for missing entries
        masked_result_matrix[row_idx, train_cols] = U_clustered[row_idx, train_cols]
    

    csv_filename = "FPy_random_matrix_" + str(int(rows)) + "_" + str(int(cols)) + "_" + str(int(num_clusters)) + ".csv"
    save_matrix_to_csv(U_clustered, csv_filename)

    # Save row mapping if provided
    if row_mapping is not None:
        mapping_filename = "FPy_random_matrix_" + str(int(rows)) + "_" + str(int(cols)) + "_" + str(int(num_clusters)) + "_row_mapping.json"
        with open(mapping_filename, 'w') as f:
            json.dump(row_mapping, f, indent=2)
        print(f"Row mapping saved to {mapping_filename}")

def extract_test_matrix(U_all, K, l, rank, synthetic_start_idx=200):
    """
    Extract test matrix U_test from U_all by subsampling with controlled rank.

    Strategy (Example: K=50, rank=15):
    1. Select 'rank' base machines -> ceil(15/5) = 3 groups needed
    2. Each group: 5 base + 15 linear combinations = 20 machines (interleaved: B,L,L,L pattern)
    3. Extract: 15 base machines + 35 linear combinations (distributed 11, 12, 12)
    4. If still need more -> generate synthetic linear combinations of base machines

    Parameters:
    -----------
    U_all : np.ndarray
        Source matrix (200 x 50), organized as 10 groups of 20 machines
        Each group: base machines at positions 0, 4, 8, 12, 16 (interleaved with linear combinations)
    K : int
        Number of machines needed in test matrix
    l : int
        Number of configurations (always 50)
    rank : int
        Number of base machines to select (controls latent rank)
    synthetic_start_idx : int
        Starting ID for synthetic machines (default: 200)

    Returns:
    --------
    U_test : np.ndarray
        Extracted test matrix (K x l)
    row_mapping : dict
        Maps U_test row index -> U_all machine ID (≥200 for synthetic)
    next_synthetic_idx : int
        Next available synthetic machine ID
    """
    BASE_PER_GROUP = 5  # Each group has 5 base machines
    MACHINES_PER_GROUP = 20  # Total machines per group

    U_test = np.zeros((K, l))
    row_mapping = {}
    filled_count = 0

    # Step 1: Determine which groups to use
    num_groups_needed = int(np.ceil(rank / BASE_PER_GROUP))
    selected_groups = list(range(num_groups_needed))

    # Step 2: Extract 'rank' base machines (interleaved pattern: positions 0, 4, 8, 12, 16 in each group)
    base_machine_ids = []
    for base_idx in range(rank):
        group_idx = base_idx // BASE_PER_GROUP  # Which group
        local_base_idx = base_idx % BASE_PER_GROUP  # Which base within group (0-4)
        # Base machines are at positions 0, 4, 8, 12, 16 within each group (interleaved pattern)
        local_position = local_base_idx * 4  # 0*4=0, 1*4=4, 2*4=8, 3*4=12, 4*4=16
        machine_id = group_idx * MACHINES_PER_GROUP + local_position
        base_machine_ids.append(machine_id)

        U_test[filled_count] = U_all[machine_id]
        row_mapping[filled_count] = machine_id
        filled_count += 1

    # Step 3: Extract linear combinations from selected groups
    if filled_count < K:
        # Calculate how many more machines we need
        needed = K - filled_count

        # Collect all linear combination machines from selected groups
        # Linear combinations are at positions 1,2,3, 5,6,7, 9,10,11, 13,14,15, 17,18,19 in each group
        linear_comb_machines = []
        for group_idx in selected_groups:
            group_start = group_idx * MACHINES_PER_GROUP
            for local_idx in range(MACHINES_PER_GROUP):
                # Skip base machine positions (0, 4, 8, 12, 16)
                if local_idx % 4 != 0:
                    machine_id = group_start + local_idx
                    linear_comb_machines.append(machine_id)

        # Distribute evenly across groups
        machines_to_extract = min(needed, len(linear_comb_machines))

        for i in range(machines_to_extract):
            machine_id = linear_comb_machines[i]
            U_test[filled_count] = U_all[machine_id]
            row_mapping[filled_count] = machine_id
            filled_count += 1

    # Step 4: Generate synthetic linear combinations if still need more
    synthetic_counter = synthetic_start_idx

    if filled_count < K:
        # Get base machines for generating synthetics
        base_machines = [U_all[base_id] for base_id in base_machine_ids]
        generated_synthetics = []  # Track generated machines to avoid duplicates

        while filled_count < K:
            max_attempts = 10000
            unique_found = False

            for _ in range(max_attempts):
                # Generate convex combination (coefficients sum to 1 -> values stay in [0,1])
                coeffs = np.random.rand(len(base_machines))

                # Ensure coefficients are not all zero
                if np.sum(np.abs(coeffs)) < 1e-10:
                    continue  # Skip this iteration and try again

                coeffs = coeffs / np.sum(coeffs)

                # Compute synthetic machine
                synthetic_machine = np.zeros(l)
                for j, base_vec in enumerate(base_machines):
                    synthetic_machine += coeffs[j] * base_vec

                # Check uniqueness (ensure not already generated)
                is_duplicate = False
                for existing_machine in generated_synthetics:
                    if np.allclose(synthetic_machine, existing_machine, atol=1e-10):
                        is_duplicate = True
                        break

                if not is_duplicate:
                    unique_found = True
                    break

            if not unique_found:
                # Fallback: if can't find unique after max attempts, use the last generated one
                # This is extremely unlikely with continuous random coefficients
                pass

            U_test[filled_count] = synthetic_machine
            row_mapping[filled_count] = synthetic_counter
            generated_synthetics.append(synthetic_machine.copy())
            synthetic_counter += 1
            filled_count += 1

    return U_test, row_mapping, synthetic_counter

# Extract, save, and collect synthetic rows
def extract_and_save(K, l, rank, next_idx, synth_dict, data_dir=""):
    """Extract test matrix, save it, and collect synthetic rows."""
    U_test, row_mapping, next_idx = extract_test_matrix(
        U_all, K, l, rank, synthetic_start_idx=next_idx
    )

    # Save the test matrix
    csv_filename = os.path.join(data_dir, f"FPy_random_matrix_{K}_{l}_{rank}.csv")
    save_matrix_to_csv(U_test, csv_filename)

    # Save the row mapping
    row_mapping_filename = os.path.join(data_dir, f"FPy_random_matrix_{K}_{l}_{rank}_row_mapping.json")
    with open(row_mapping_filename, 'w') as f:
        json.dump(row_mapping, f, indent=2)

    # Collect synthetic rows (indices >= 200)
    for u_test_idx, u_all_idx in row_mapping.items():
        if u_all_idx >= 200:
            synth_dict[u_all_idx] = U_test[u_test_idx].copy()

    return next_idx


if __name__ == "__main__":
    # ========================================================================
    # CREATE DATA DIRECTORY
    # ========================================================================
    DATA_DIR = "data"
    os.makedirs(DATA_DIR, exist_ok=True)
    print(f"Data directory: {DATA_DIR}/")
    print()

    # ========================================================================
    # SWITCH: Choose generation method
    # ========================================================================
    # GENERATION_MODE options:
    #   'random'    - Use random linear combination generation
    #   'load'      - Load existing U_all from file
    GENERATION_MODE = 'load'  # Change this to switch generation methods

    LOAD_EXISTING_U_ALL = (GENERATION_MODE == 'load')
    EXISTING_U_ALL_FILE = os.path.join(DATA_DIR, "FPy_random_matrix_200_50_50.csv")

    if LOAD_EXISTING_U_ALL:
        print("="*80)
        print("LOADING EXISTING U_all FROM FILE")
        print("="*80)
        print(f"Loading from: {EXISTING_U_ALL_FILE}")

        if not os.path.exists(EXISTING_U_ALL_FILE):
            raise FileNotFoundError(f"U_all file not found: {EXISTING_U_ALL_FILE}\n"
                                    f"Please run generate_random_matrix_with_masking.py first, "
                                    f"or set LOAD_EXISTING_U_ALL = False to generate a new matrix.")

        U_all = np.genfromtxt(EXISTING_U_ALL_FILE, delimiter=',')

        print(f"Loaded U_all with shape: {U_all.shape}")
        print("="*80 + "\n")

        # Verify matrix properties
        matrix_rank = np.linalg.matrix_rank(U_all)
        print(f"U_all properties:")
        print(f"  Shape: {U_all.shape}")
        print(f"  Rank: {matrix_rank}")


    elif GENERATION_MODE == 'random':
        print("="*80)
        print("GENERATING RANDOM LINEAR COMBINATION MATRIX")
        print("="*80 + "\n")

        U_all = generate_random_matrix()
        csv_filename = os.path.join(DATA_DIR, "FPy_random_matrix_" + str(int(200)) + "_" + str(int(50)) + "_" + str(int(50)) + ".csv")
        save_matrix_to_csv(U_all, csv_filename)

    else:
        raise ValueError(f"Invalid GENERATION_MODE: {GENERATION_MODE}. Must be 'random' or 'load'.")

    # ========================================================================
    # EXTRACT TEST MATRICES (Method depends on generation mode)
    # ========================================================================

    
    print("="*80)
    print("EXTRACTING TEST MATRICES")
    print("="*80)
    print()

    

    # Track synthetic row indices and storage
    next_synthetic_idx = 200
    synthetic_rows_dict = {}

    # Extract and save all test matrices
    next_synthetic_idx = extract_and_save(50, 50, 5, next_synthetic_idx, synthetic_rows_dict, DATA_DIR)
    next_synthetic_idx = extract_and_save(50, 50, 10, next_synthetic_idx, synthetic_rows_dict, DATA_DIR)
    next_synthetic_idx = extract_and_save(50, 50, 15, next_synthetic_idx, synthetic_rows_dict, DATA_DIR)
    next_synthetic_idx = extract_and_save(50, 50, 20, next_synthetic_idx, synthetic_rows_dict, DATA_DIR)
    next_synthetic_idx = extract_and_save(30, 50, 10, next_synthetic_idx, synthetic_rows_dict, DATA_DIR)
    next_synthetic_idx = extract_and_save(100, 50, 10, next_synthetic_idx, synthetic_rows_dict, DATA_DIR)

    # Append synthetic rows to U_all to create extended matrix
    if len(synthetic_rows_dict) > 0:
        print(f"\n{'='*80}")
        print(f"APPENDING SYNTHETIC ROWS TO U_all")
        print(f"{'='*80}")
        print(f"  Original U_all shape: {U_all.shape}")
        print(f"  Generated {len(synthetic_rows_dict)} synthetic rows (indices 200-{next_synthetic_idx-1})")

        # Create extended matrix
        max_synthetic_idx = max(synthetic_rows_dict.keys())
        total_rows = max_synthetic_idx + 1
        U_extended = np.zeros((total_rows, 50))

        # Copy original U_all rows (0-199)
        U_extended[:200, :] = U_all

        # Append synthetic rows (200+)
        for synthetic_idx, row_vec in synthetic_rows_dict.items():
            U_extended[synthetic_idx, :] = row_vec

        print(f"  Extended U_all shape: {U_extended.shape}")

        # Save extended matrix
        extended_csv_filename = os.path.join(DATA_DIR, "FPy_random_matrix_extended_" + str(int(total_rows)) + "_" + str(int(50)) + "_" + str(int(50)) + ".csv")
        save_matrix_to_csv(U_extended, extended_csv_filename)
        print(f"  Saved extended U_all to: {extended_csv_filename}")

        # Verify rank of extended matrix
        extended_rank = np.linalg.matrix_rank(U_extended)
        print(f"  Extended matrix rank: {extended_rank}")
        print(f"{'='*80}\n")

        # Use extended matrix for masking pattern generation
        U_for_masking = U_extended
    else:
        print("\n  No synthetic rows generated, U_all remains unchanged")
        # Use original U_all for masking pattern generation
        U_for_masking = U_all

    # ========================================================================
    # GENERATE MASKING PATTERNS FOR MULTIPLE MISSING RATES
    # ========================================================================
    print(f"{'='*80}")
    print("GENERATING MASKING PATTERNS FOR MULTIPLE MISSING RATES")
    print(f"{'='*80}")
    print(f"Matrix for masking: shape {U_for_masking.shape}")
    print()

    missing_rates = [40, 50, 60, 70, 80]
    random_seed = 128

    for missing_rate in missing_rates:
        print(f"\n{'-'*80}")
        print(f"Generating masking pattern for missing rate: {missing_rate}%")
        print(f"{'-'*80}")

        # Generate masking metadata
        masking_metadata = apply_masking_to_U_all(U_for_masking, missing_rate, random_seed)

        # Save masking metadata to JSON file
        metadata_filename = os.path.join(DATA_DIR, f"FPy_masking_metadata_{missing_rate}pct.json")
        with open(metadata_filename, 'w') as f:
            json.dump(masking_metadata, f, indent=2)

        print(f"  Masking metadata saved to: {metadata_filename}")
        print(f"  - Total machines: {len(masking_metadata['train_cols_per_row'])}")
        print(f"  - Missing rate: {missing_rate}%")
        print(f"  - Observed configs per machine: {len(masking_metadata['train_cols_per_row'][0])}/50")

    print(f"\n{'='*80}")
    print("MASKING PATTERN GENERATION COMPLETE")
    print(f"{'='*80}")
    print(f"Generated {len(missing_rates)} masking metadata files in {DATA_DIR}/:")
    for missing_rate in missing_rates:
        print(f"  - FPy_masking_metadata_{missing_rate}pct.json")
    print(f"{'='*80}\n")
