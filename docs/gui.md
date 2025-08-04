# Interactive GUI Documentation

The GUI interface provides an interactive way to explore and understand GPU acceleration examples. This document explains how to use the interface and what each section demonstrates.

## Running the GUI

1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

2. Launch the GUI:
   ```bash
   ./run_gui.sh
   ```
   Or run directly with Streamlit:
   ```bash
   streamlit run src/gui/app.py
   ```

## Available Examples

### 1. Pandas Acceleration
- Demonstrates how to accelerate pandas operations using cuDF
- Interactive data generation with configurable size
- Real-time performance comparison between CPU and GPU
- Visualization of speedup achieved

### 2. UMAP & HDBSCAN
- Shows acceleration of dimensionality reduction and clustering
- Interactive parameter configuration
- Side-by-side comparison of CPU and GPU implementations
- Visual representation of the results

### 3. NetworkX GPU
- Demonstrates graph analytics acceleration
- Interactive graph generation
- Performance comparison of common graph algorithms
- Visual representation of speedup

## Features

- **Interactive Controls**: Adjust parameters and see results in real-time
- **Visual Comparisons**: Clear visualizations of performance differences
- **Code Examples**: View and copy example code for each acceleration technique
- **Performance Metrics**: Real-time computation of speedup factors

## Tips for Best Results

1. **Hardware Requirements**:
   - NVIDIA GPU with CUDA support
   - CUDA Toolkit 11.0 or later
   - Sufficient GPU memory for larger datasets

2. **Performance Optimization**:
   - Start with smaller datasets to understand the behavior
   - Gradually increase data size to see scaling effects
   - Monitor GPU memory usage for large operations

3. **Common Issues**:
   - If you see "CUDA out of memory" errors, reduce the data size
   - For best performance, ensure no other GPU-intensive tasks are running
   - Some operations may require specific GPU architectures for optimal performance

## Contributing

Feel free to contribute to the GUI by:
1. Adding new examples
2. Improving visualizations
3. Optimizing performance
4. Adding new features

Submit pull requests to the GitHub repository.
