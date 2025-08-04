"""
Main Streamlit application for demonstrating GPU acceleration examples.
"""
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from pathlib import Path
import time
import sys
from typing import Tuple, Optional

# Add project root to Python path
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.append(str(PROJECT_ROOT))

from src.data.utils import generate_sample_data, benchmark_operation
from src.ml.utils import train_test_split_gpu, scale_features_gpu
from src.viz.utils import plot_performance_comparison

# Page config
st.set_page_config(
    page_title="GPU-Accelerated Data Science",
    page_icon="🚀",
    layout="wide"
)

# Sidebar
st.sidebar.title("Navigation")
page = st.sidebar.selectbox(
    "Select Example",
    ["Home", "Pandas Acceleration", "UMAP & HDBSCAN", "NetworkX GPU"]
)

def home_page():
    """Render the home page."""
    st.title("🚀 GPU-Accelerated Data Science")
    st.markdown("""
    Welcome to the GPU acceleration examples! This application demonstrates how to accelerate
    common data science workflows using NVIDIA's RAPIDS ecosystem.
    
    ### Available Examples:
    
    1. **Pandas Acceleration** 📊
       - Load and process large datasets faster
       - GPU-accelerated groupby and aggregations
       - Performance comparisons
    
    2. **UMAP & HDBSCAN** 🎯
       - Fast dimensionality reduction
       - GPU-accelerated clustering
       - Visual comparisons
    
    3. **NetworkX GPU** 🕸️
       - Scale graph analytics
       - Accelerated graph algorithms
       - Real-world examples
    
    ### How to Use
    
    1. Select an example from the sidebar
    2. Follow the interactive demonstrations
    3. Compare CPU vs GPU performance
    4. View and copy code examples
    
    ### System Requirements
    
    - NVIDIA GPU with CUDA support
    - CUDA Toolkit 11.0+
    - Required Python packages (see requirements.txt)
    """)
    
    # Show system info
    st.subheader("System Information")
    import cudf
    st.code(f"""
    RAPIDS version: {cudf.__version__}
    Python version: {sys.version.split()[0]}
    """)

def pandas_acceleration_page():
    """Render the pandas acceleration demo page."""
    st.title("📊 Pandas Acceleration with cuDF")
    
    st.markdown("""
    This example demonstrates how to accelerate pandas operations using NVIDIA cuDF.
    We'll generate sample data and compare performance between CPU and GPU implementations.
    """)
    
    # Data generation parameters
    n_rows = st.slider("Number of rows", 1000, 10_000_000, 1_000_000)
    n_cols = st.slider("Number of columns", 2, 100, 10)
    
    if st.button("Generate Data & Run Comparison"):
        with st.spinner("Generating sample data..."):
            X, y = generate_sample_data(n_rows, n_cols)
            df = pd.DataFrame(X, columns=[f"feature_{i}" for i in range(n_cols)])
            df["target"] = y
            
            # Show sample of the data
            st.subheader("Sample Data")
            st.dataframe(df.head())
            
            # Run performance comparison
            st.subheader("Performance Comparison")
            
            # CPU timing
            start = time.time()
            cpu_result = df.groupby("target").agg(["mean", "std"]).compute()
            cpu_time = time.time() - start
            
            # GPU timing
            import cudf
            gdf = cudf.DataFrame.from_pandas(df)
            start = time.time()
            gpu_result = gdf.groupby("target").agg(["mean", "std"]).compute()
            gpu_time = time.time() - start
            
            # Plot results
            times = pd.DataFrame({
                "Platform": ["CPU", "GPU"],
                "Time (seconds)": [cpu_time, gpu_time]
            })
            fig = px.bar(times, x="Platform", y="Time (seconds)",
                        title="GroupBy Aggregation Performance")
            st.plotly_chart(fig)
            
            # Show speedup
            speedup = cpu_time / gpu_time
            st.success(f"🚀 GPU acceleration achieved {speedup:.1f}x speedup!")

def umap_hdbscan_page():
    """Render the UMAP and HDBSCAN demo page."""
    st.title("🎯 UMAP & HDBSCAN Acceleration")
    
    st.markdown("""
    This example shows how to accelerate UMAP dimensionality reduction and HDBSCAN
    clustering using GPU acceleration.
    """)
    
    # Parameters
    n_samples = st.slider("Number of samples", 1000, 50000, 10000)
    n_features = st.slider("Number of features", 10, 1000, 100)
    
    if st.button("Run UMAP & HDBSCAN"):
        with st.spinner("Generating high-dimensional data..."):
            X, y = generate_sample_data(n_samples, n_features)
            
            # CPU UMAP
            st.subheader("UMAP Dimensionality Reduction")
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("### CPU UMAP")
                start = time.time()
                umap_cpu = umap.UMAP(random_state=42)
                X_umap_cpu = umap_cpu.fit_transform(X)
                cpu_time = time.time() - start
                
                fig = px.scatter(x=X_umap_cpu[:, 0], y=X_umap_cpu[:, 1],
                               color=y, title=f"CPU Time: {cpu_time:.2f}s")
                st.plotly_chart(fig)
            
            with col2:
                st.markdown("### GPU UMAP")
                start = time.time()
                import cuml
                umap_gpu = cuml.UMAP(random_state=42)
                X_umap_gpu = umap_gpu.fit_transform(X)
                gpu_time = time.time() - start
                
                fig = px.scatter(x=X_umap_gpu[:, 0], y=X_umap_gpu[:, 1],
                               color=y, title=f"GPU Time: {gpu_time:.2f}s")
                st.plotly_chart(fig)
            
            speedup = cpu_time / gpu_time
            st.success(f"🚀 GPU acceleration achieved {speedup:.1f}x speedup for UMAP!")

def networkx_page():
    """Render the NetworkX acceleration demo page."""
    st.title("🕸️ NetworkX GPU Acceleration")
    
    st.markdown("""
    This example demonstrates how to accelerate NetworkX graph algorithms using
    NVIDIA's cuGraph backend.
    """)
    
    # Parameters
    n_nodes = st.slider("Number of nodes", 100, 10000, 1000)
    edge_probability = st.slider("Edge probability", 0.001, 0.1, 0.01)
    
    if st.button("Generate Graph & Compare"):
        with st.spinner("Generating random graph..."):
            import networkx as nx
            
            # Generate random graph
            G = nx.erdos_renyi_graph(n_nodes, edge_probability, seed=42)
            
            # CPU timing
            start = time.time()
            cpu_centrality = nx.betweenness_centrality(G)
            cpu_time = time.time() - start
            
            # GPU timing
            import cugraph
            df = nx.to_pandas_edgelist(G)
            G_gpu = cugraph.Graph()
            G_gpu.from_pandas_edgelist(df, "source", "target")
            
            start = time.time()
            gpu_centrality = cugraph.betweenness_centrality(G_gpu)
            gpu_time = time.time() - start
            
            # Plot results
            times = pd.DataFrame({
                "Platform": ["CPU", "GPU"],
                "Time (seconds)": [cpu_time, gpu_time]
            })
            fig = px.bar(times, x="Platform", y="Time (seconds)",
                        title="Betweenness Centrality Computation Time")
            st.plotly_chart(fig)
            
            speedup = cpu_time / gpu_time
            st.success(f"🚀 GPU acceleration achieved {speedup:.1f}x speedup!")

# Route to the correct page
if page == "Home":
    home_page()
elif page == "Pandas Acceleration":
    pandas_acceleration_page()
elif page == "UMAP & HDBSCAN":
    umap_hdbscan_page()
else:
    networkx_page()
