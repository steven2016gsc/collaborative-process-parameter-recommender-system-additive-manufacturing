"""
Sequential Pure Python ALS Implementation with Non-Negative Constraints.

This module provides a pure Python implementation of Alternating Least Squares (ALS)
for collaborative filtering using Cholesky decomposition.
"""

import numpy as np
from sklearn.metrics.pairwise import pairwise_distances

# Imports for rank estimation
from sklearn.metrics.pairwise import rbf_kernel
from sklearn.experimental import enable_iterative_imputer  # Enable MICE
from sklearn.impute import IterativeImputer
from scipy.linalg import svd


def rank_estimate_mice_svd(U_sparse, energy_threshold=0.95, k_neighbors=7):
    """
    Estimates latent rank r using MICE imputation and SVD Energy variance,
    with adaptive Sigma calibration as described.

    Parameters:
    -----------
    U_sparse : np.ndarray
        The input utility matrix with NaN indicating missing entries.
        Shape: (K_machines, N_parameters)
    energy_threshold : float
        The cumulative energy threshold (0.0 to 1.0) to determine rank.
        Default is 0.95 (95% variance explained).
    k_neighbors : int
        The k-th nearest neighbor index for sigma calibration. Default 7.

    Returns:
    --------
    estimated_rank : int
        The estimated rank r.
    U_imputed : np.ndarray
        The dense matrix after MICE imputation.
    sigma : float
        The calibrated kernel bandwidth.
    """

    # ---------------------------------------------------------
    # 1. MICE Iterative Imputation
    # ---------------------------------------------------------
    # Imputes missing values by modeling columns as functions of others.
    # 'ascending' order matches the description (least to most missing).
    imputer = IterativeImputer(
        max_iter=10,
        random_state=42,
        imputation_order='ascending'
    )
    U_imputed = imputer.fit_transform(U_sparse)

    # ---------------------------------------------------------
    # 2. Adaptive Kernel Calibration (Sigma Selection)
    # ---------------------------------------------------------
    # Compute pairwise Euclidean distances between Machine vectors (Rows)
    # Shape: (K, K)
    dist_matrix = pairwise_distances(U_imputed, metric='euclidean')

    # Sort distances to find neighbors
    # axis=1 sorts each row. Index 0 is the machine itself (dist=0).
    sorted_dists = np.sort(dist_matrix, axis=1)

    # Safety check: ensure we have enough machines for k=7
    n_machines = U_imputed.shape[0]
    safe_k = min(k_neighbors, n_machines - 1)

    # Extract the k-th nearest neighbor distance for each machine
    # d_k_vec represents {d_k^(7)} for all k
    d_k_vec = sorted_dists[:, safe_k]

    # Define Sigma as the median of these distances
    sigma = np.median(d_k_vec)

    # Construct Affinity Matrix (Visual check / spectral prep)
    # S_kk' = exp( - ||U_k - U_k'||^2 / (2*sigma^2) )
    # Gamma for rbf_kernel is 1 / (2*sigma^2)
    if sigma > 0:
        gamma_val = 1.0 / (2.0 * sigma**2)
    else:
        gamma_val = 1.0 # Fallback if all machines are identical

    affinity_matrix = rbf_kernel(U_imputed, gamma=gamma_val)

    # ---------------------------------------------------------
    # 3. Robust Rank Estimation via SVD Energy
    # ---------------------------------------------------------
    # Center the data before SVD (Standard practice for PCA/Variance analysis)
    from scipy.linalg import svd
    U_centered = U_imputed - np.mean(U_imputed, axis=0)

    # Perform SVD: U = U_s * Sigma * Vt
    # We only need singular values (S)
    _, singular_values, _ = svd(U_centered, full_matrices=False)

    # Compute Energy (Squared Singular Values)
    squared_singular_values = singular_values**2
    total_energy = np.sum(squared_singular_values)

    # Cumulative Density Function (CDF) of energy
    cumulative_energy_ratio = np.cumsum(squared_singular_values) / total_energy

    # Find the first rank where cumulative energy >= threshold
    # np.searchsorted finds the first index where condition fits
    # We add 1 because indices are 0-based, but rank is 1-based
    estimated_rank = np.searchsorted(cumulative_energy_ratio, energy_threshold) + 1

    print(f"--- Results ---")
    #print(f"Calibrated Sigma: {sigma:.4f}")
    #print(f"Total Energy (Variance): {total_energy:.2f}")
    print(f"Estimated Rank (at {energy_threshold*100}% energy): {estimated_rank}")

    return estimated_rank, U_imputed


class SequentialPythonALS:
    """
    Sequential Pure Python ALS with non-negative constraints.
    Uses Cholesky decomposition.
    """
    def __init__(self, rank=3, max_iter=15, reg_param=0.05, random_state=42, init_strategy='conservative'):
        """
        Initialize ALS model.

        Args:
            rank: Number of latent factors
            max_iter: Maximum number of ALS iterations
            reg_param: Regularization parameter (lambda)
            random_state: Random seed for reproducibility
            init_strategy: Factor initialization strategy:
                - 'conservative': Rank-adaptive with floor (max(0.15, sqrt(0.5/rank)))
                  Works well for ranks 5-25, accounts for regularization shrinkage
                - 'data_driven': Based on sqrt(mean_rating / rank) (fully adaptive to data)
                - 'deterministic': Constant value of sqrt(0.5/rank), resembling a target
                  at a mean utility of 0.5 (no randomness)
        """
        self.rank = rank
        self.max_iter = max_iter
        self.reg_param = reg_param
        self.random_state = random_state
        self.init_strategy = init_strategy
        self.user_factors = None
        self.item_factors = None
        self.n_users = 0
        self.n_items = 0

    def _solve_cholesky(self, A, b):
        """
        Solve normal equations (A^T A + λI) x = A^T b using Cholesky decomposition.
        This matches Spark's ALS implementation.

        Args:
            A: Feature matrix (n_observations, rank)
            b: Target vector (n_observations,)

        Returns:
            x: Solution vector (rank,) with non-negative values
        """
        # Compute normal equations: (A^T A + λI) x = A^T b
        AtA = A.T @ A  # (rank, rank)
        Atb = A.T @ b  # (rank,)

        # Add regularization to diagonal
        AtA_reg = AtA + self.reg_param * np.eye(self.rank)

        try:
            # Cholesky decomposition: AtA_reg = L L^T
            L = np.linalg.cholesky(AtA_reg)

            # Solve L y = Atb (forward substitution)
            y = np.linalg.solve(L, Atb)

            # Solve L^T x = y (backward substitution)
            x = np.linalg.solve(L.T, y)
        except np.linalg.LinAlgError:
            # If Cholesky fails (matrix not positive definite), use lstsq as fallback
            x = np.linalg.lstsq(AtA_reg, Atb, rcond=None)[0]

        return x

    def fit(self, train_data, n_users=None, n_items=None):
        """
        Fit ALS model on training data using Cholesky decomposition.

        Args:
            train_data: numpy array of shape (n_samples, 3) with columns [user_id, item_id, rating]
            n_users: Fixed number of users (if None, inferred from max user_id + 1)
            n_items: Fixed number of items (if None, inferred from max item_id + 1)
        """
        # Build rating matrix from sparse data
        user_ids = train_data[:, 0].astype(int)
        item_ids = train_data[:, 1].astype(int)
        ratings = train_data[:, 2]

        # Use explicit matrix dimensions if provided
        # For sequential (solo) method: should always be 10-by-5 (from 50-element row reshaped)
        if n_users is not None:
            self.n_users = n_users
        else:
            self.n_users = int(user_ids.max()) + 1

        if n_items is not None:
            self.n_items = n_items
        else:
            self.n_items = int(item_ids.max()) + 1

        # Create sparse observation mask and rating matrix
        rating_matrix = np.zeros((self.n_users, self.n_items))
        mask = np.zeros((self.n_users, self.n_items), dtype=bool)

        for user_id, item_id, rating in zip(user_ids, item_ids, ratings):
            rating_matrix[user_id, item_id] = rating
            mask[user_id, item_id] = True

        # Initialize factors based on selected strategy
        if self.init_strategy == 'conservative':
            # Option: Conservative rank-adaptive initialization WITH NOISE
            #init_mean = max(0.1, np.sqrt(0.2 / self.rank))
            # Pessimistic / Zero-centered
            init_mean = 0.01

            # Use a slightly wider scale to break symmetry faster
            scale_val = 0.001

            # Set random seed for reproducibility
            np.random.seed(self.random_state)

            # Use Random Uniform or Normal distribution instead of np.full
            # We add small noise centered around init_mean
            self.user_factors = np.random.normal(loc=init_mean, scale=scale_val, size=(self.n_users, self.rank))
            self.item_factors = np.random.normal(loc=init_mean, scale=scale_val, size=(self.n_items, self.rank))

            # Debug: Print initialization info (only for first call)
            if not hasattr(SequentialPythonALS, '_init_printed'):
                print(f"\n[ALS Init] Using CONSERVATIVE initialization (randomized)")
                print(f"[ALS Init] Rank: {self.rank}, init_val: {init_mean:.3f}")
                SequentialPythonALS._init_printed = True

        elif self.init_strategy == 'data_driven':
            # Option: Data-driven initialization based on mean rating
            # Most principled: targets mean rating via sqrt(mean_rating / rank)
            mean_rating = np.mean(ratings)
            init_mean = np.sqrt(mean_rating / self.rank)

            scale_val = 0.1

            # Set random seed for reproducibility
            np.random.seed(self.random_state)

            # FIX: Use Randomness
            self.user_factors = np.random.normal(loc=init_mean, scale=scale_val, size=(self.n_users, self.rank))
            self.item_factors = np.random.normal(loc=init_mean, scale=scale_val, size=(self.n_items, self.rank))


            # Debug: Print initialization info (only for first call)
            if not hasattr(SequentialPythonALS, '_init_printed'):
                print(f"\n[ALS Init] Using DATA-DRIVEN initialization based on mean rating")
                print(f"[ALS Init] Mean rating: {mean_rating:.3f}")
                print(f"[ALS Init] Computed init value: sqrt({mean_rating:.3f} / {self.rank}) = {init_mean:.3f}")
                SequentialPythonALS._init_printed = True

        elif self.init_strategy == 'deterministic':
            # Option: Deterministic initialization with constant value
            # Initialize A and B deterministically with sqrt(0.5/r), targeting mean utility of 0.5
            init_val = np.sqrt(0.5 / self.rank)

            # Initialize all factors with the same constant value (fully deterministic)
            self.user_factors = np.full((self.n_users, self.rank), init_val)
            self.item_factors = np.full((self.n_items, self.rank), init_val)

            # Debug: Print initialization info (only for first call)
            if not hasattr(SequentialPythonALS, '_init_printed'):
                print(f"\n[ALS Init] Using DETERMINISTIC initialization")
                print(f"[ALS Init] Rank: {self.rank}, init_val: {init_val:.3f}")
                SequentialPythonALS._init_printed = True

        else:
            raise ValueError(f"Unknown init_strategy: {self.init_strategy}. Use 'conservative', 'data_driven', or 'deterministic'.")

        # Alternating Least Squares iterations (matching Spark ALS)
        for iteration in range(self.max_iter):
            # Fix item factors, update user factors
            for u in range(self.n_users):
                observed_items = mask[u, :]
                if np.sum(observed_items) == 0:
                    continue

                # Get observed items and ratings for this user
                Y_u = self.item_factors[observed_items, :]  # (n_observed_items, rank)
                r_u = rating_matrix[u, observed_items]  # (n_observed_items,)

                # Solve using Cholesky decomposition (matches Spark)
                self.user_factors[u, :] = self._solve_cholesky(Y_u, r_u)

            # Fix user factors, update item factors
            for i in range(self.n_items):
                observed_users = mask[:, i]
                if np.sum(observed_users) == 0:
                    continue

                # Get observed users and ratings for this item
                X_i = self.user_factors[observed_users, :]  # (n_observed_users, rank)
                r_i = rating_matrix[observed_users, i]  # (n_observed_users,)

                # Solve using Cholesky decomposition (matches Spark)
                self.item_factors[i, :] = self._solve_cholesky(X_i, r_i)

        return self

    def predict(self, test_data):
        """
        Generate predictions for test data.
        Implements coldStartStrategy='drop' by only returning predictions for known users/items.

        Args:
            test_data: numpy array of shape (n_samples, 3) with columns [user_id, item_id, rating]

        Returns:
            predictions: numpy array of shape (n_samples, 4) with columns [user_id, item_id, rating, prediction]
        """
        user_ids = test_data[:, 0].astype(int)
        item_ids = test_data[:, 1].astype(int)
        ratings = test_data[:, 2]

        predictions = []

        for idx, (user_id, item_id, rating) in enumerate(zip(user_ids, item_ids, ratings)):
            # coldStartStrategy='drop': Check if user and item are within bounds
            if user_id < self.n_users and item_id < self.n_items:
                # Compute prediction: user_factor @ item_factor (explicit feedback)
                pred = np.dot(self.user_factors[user_id], self.item_factors[item_id])
                predictions.append([user_id, item_id, rating, pred])

        # Return only valid predictions (drops unknown users/items)
        return np.array(predictions) if len(predictions) > 0 else np.array([]).reshape(0, 4)
