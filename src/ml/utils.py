"""Machine learning utilities for GPU acceleration."""
from typing import Optional, Tuple, Union
import numpy as np
import cuml


def train_test_split_gpu(
    X: Union[np.ndarray, cuml.array],
    y: Optional[Union[np.ndarray, cuml.array]] = None,
    test_size: float = 0.2,
    random_state: Optional[int] = None
) -> Tuple:
    """Split arrays into random train and test subsets using GPU acceleration.
    
    Args:
        X: Features array
        y: Target array
        test_size: Proportion of the dataset to include in the test split
        random_state: Controls the shuffling applied to the data
        
    Returns:
        tuple: (X_train, X_test, y_train, y_test) if y is not None,
               (X_train, X_test) otherwise
    """
    rng = np.random.default_rng(random_state)
    n_samples = X.shape[0]
    
    # Generate indices for train/test split
    indices = np.arange(n_samples)
    rng.shuffle(indices)
    
    split_idx = int(n_samples * (1 - test_size))
    train_idx = indices[:split_idx]
    test_idx = indices[split_idx:]
    
    # Convert to GPU arrays if needed
    if not isinstance(X, cuml.array):
        X = cuml.array(X)
    
    X_train = X[train_idx]
    X_test = X[test_idx]
    
    if y is None:
        return X_train, X_test
    
    if not isinstance(y, cuml.array):
        y = cuml.array(y)
    
    y_train = y[train_idx]
    y_test = y[test_idx]
    
    return X_train, X_test, y_train, y_test


def scale_features_gpu(
    X_train: Union[np.ndarray, cuml.array],
    X_test: Optional[Union[np.ndarray, cuml.array]] = None
) -> Tuple:
    """Scale features using GPU acceleration.
    
    Args:
        X_train: Training data
        X_test: Test data (optional)
        
    Returns:
        tuple: (X_train_scaled, X_test_scaled) if X_test is not None,
               X_train_scaled otherwise
    """
    scaler = cuml.preprocessing.StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    
    if X_test is not None:
        X_test_scaled = scaler.transform(X_test)
        return X_train_scaled, X_test_scaled
    
    return X_train_scaled
