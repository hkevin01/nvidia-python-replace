"""Tests for GPU acceleration utilities."""
import numpy as np
import pytest
from src.data.utils import generate_sample_data, benchmark_operation
from src.ml.utils import train_test_split_gpu, scale_features_gpu


def test_generate_sample_data():
    """Test sample data generation."""
    n_samples, n_features = 1000, 10
    X, y = generate_sample_data(n_samples, n_features)
    
    assert X.shape == (n_samples, n_features)
    assert y.shape == (n_samples,)
    assert np.unique(y).tolist() == [0, 1]


def test_benchmark_operation():
    """Test benchmarking utility."""
    def dummy_func():
        return np.sum(np.random.randn(1000, 1000))
    
    avg_time = benchmark_operation(dummy_func, n_runs=3)
    assert isinstance(avg_time, float)
    assert avg_time > 0


@pytest.mark.gpu
def test_train_test_split_gpu():
    """Test GPU-accelerated train-test split."""
    X, y = generate_sample_data(1000, 10)
    X_train, X_test, y_train, y_test = train_test_split_gpu(X, y, test_size=0.2)
    
    assert X_train.shape[0] == 800
    assert X_test.shape[0] == 200
    assert y_train.shape[0] == 800
    assert y_test.shape[0] == 200


@pytest.mark.gpu
def test_scale_features_gpu():
    """Test GPU-accelerated feature scaling."""
    X, _ = generate_sample_data(1000, 10)
    X_train, X_test = X[:800], X[800:]
    
    X_train_scaled, X_test_scaled = scale_features_gpu(X_train, X_test)
    
    assert X_train_scaled.shape == X_train.shape
    assert X_test_scaled.shape == X_test.shape
    
    # Check scaling properties
    assert np.abs(X_train_scaled.mean()) < 1e-6
    assert np.abs(X_train_scaled.std() - 1) < 1e-6
