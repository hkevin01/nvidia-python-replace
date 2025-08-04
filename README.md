# GPU-Accelerated Data Science with Python

This repository contains examples of how to accelerate common Python data science libraries using NVIDIA GPUs. Each notebook demonstrates a different library and shows how to enable GPU acceleration with minimal code changes.

## Prerequisites

- NVIDIA GPU with CUDA support
- Python 3.8+
- CUDA Toolkit 11.x or later
- Required Python packages (see `requirements.txt`)

## Installation

1. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

2. Install dependencies:
```bash
pip install -r requirements.txt --extra-index-url https://pypi.nvidia.com
```

## Examples

1. [pandas_acceleration.ipynb](notebooks/pandas_acceleration.ipynb) - Accelerate pandas with cuDF
2. [polars_acceleration.ipynb](notebooks/polars_acceleration.ipynb) - GPU acceleration in Polars
3. [sklearn_acceleration.ipynb](notebooks/sklearn_acceleration.ipynb) - scikit-learn acceleration with cuML
4. [xgboost_acceleration.ipynb](notebooks/xgboost_acceleration.ipynb) - XGBoost on GPU
5. [umap_acceleration.ipynb](notebooks/umap_acceleration.ipynb) - Fast UMAP with cuML
6. [hdbscan_acceleration.ipynb](notebooks/hdbscan_acceleration.ipynb) - HDBSCAN clustering on GPU
7. [networkx_acceleration.ipynb](notebooks/networkx_acceleration.ipynb) - NetworkX with cuGraph

Each notebook contains:
- Setup instructions
- Example code
- Performance comparisons
- Sample datasets

## Usage

Launch Jupyter Lab to explore the notebooks:
```bash
jupyter lab
```
