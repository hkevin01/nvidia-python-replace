"""Visualization utilities for GPU-accelerated data science."""
from typing import Optional, List
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


def plot_performance_comparison(
    cpu_times: List[float],
    gpu_times: List[float],
    operations: List[str],
    title: str = "Performance Comparison: CPU vs GPU",
    figsize: tuple = (10, 6)
) -> None:
    """Plot performance comparison between CPU and GPU operations.
    
    Args:
        cpu_times: List of execution times for CPU operations
        gpu_times: List of execution times for GPU operations
        operations: List of operation names
        title: Plot title
        figsize: Figure size
    """
    x = np.arange(len(operations))
    width = 0.35
    
    fig, ax = plt.subplots(figsize=figsize)
    cpu_bars = ax.bar(x - width/2, cpu_times, width, 
                      label='CPU', color='blue', alpha=0.6)
    gpu_bars = ax.bar(x + width/2, gpu_times, width,
                      label='GPU', color='green', alpha=0.6)
    
    ax.set_ylabel('Time (seconds)')
    ax.set_title(title)
    ax.set_xticks(x)
    ax.set_xticklabels(operations)
    ax.legend()
    
    # Add speedup annotations
    for i, (cpu_time, gpu_time) in enumerate(zip(cpu_times, gpu_times)):
        speedup = cpu_time / gpu_time
        ax.text(i, max(cpu_time, gpu_time), 
                f'{speedup:.1f}x speedup',
                ha='center', va='bottom')
    
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()


def plot_umap_comparison(
    embeddings_cpu: np.ndarray,
    embeddings_gpu: np.ndarray,
    labels: Optional[np.ndarray] = None,
    title: str = "UMAP Projections: CPU vs GPU",
    figsize: tuple = (15, 6)
) -> None:
    """Plot UMAP embeddings comparison between CPU and GPU.
    
    Args:
        embeddings_cpu: 2D embeddings from CPU UMAP
        embeddings_gpu: 2D embeddings from GPU UMAP
        labels: Optional labels for coloring points
        title: Plot title
        figsize: Figure size
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
    
    scatter_kwargs = dict(alpha=0.5, s=5)
    if labels is not None:
        scatter_kwargs['c'] = labels
    
    ax1.scatter(embeddings_cpu[:, 0], embeddings_cpu[:, 1], 
                **scatter_kwargs)
    ax1.set_title('CPU UMAP')
    
    ax2.scatter(embeddings_gpu[:, 0], embeddings_gpu[:, 1],
                **scatter_kwargs)
    ax2.set_title('GPU UMAP')
    
    fig.suptitle(title)
    plt.tight_layout()
    plt.show()
