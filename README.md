# Botnet Detection with Autoencoders

This project implements various autoencoder architectures for anomaly detection in network traffic data, specifically targeting botnet detection. The system trains multiple autoencoder models, evaluates their performance, and selects the best model for deployment.

## Project Structure

- **train_eval_autoencoder.py**: Main script for training and evaluating autoencoder models
- **analyze_dataset.py**: Script for data preprocessing and analysis
- **requirements.txt**: List of Python dependencies
- **dataset/**: Directory containing network traffic datasets
- **models/**: Directory where trained models are saved
- **results/**: Directory where evaluation results are stored

## Dependencies

The project requires the following Python packages:
```
pandas>=1.0.0
numpy>=1.18.0
scikit-learn>=0.22.0
tensorflow>=2.9.0
joblib>=0.14.0
tensorflow_addons
seaborn
matplotlib
```

## Data Processing

The `analyze_dataset.py` script performs:
- Data cleaning and normalization
- Feature extraction and selection
- Statistical analysis of the dataset

## Training Results

The latest training run produced the following results:

### Model Performance Summary

| Model | Test MAE | Test RMSE | Best Epoch | Notes |
|-------|----------|-----------|------------|-------|
| **autoencoder_conv_swish** | **0.002656** | **0.012965** | 71 | Best overall model |
| autoencoder_dense_leaky | 0.005669 | 0.012965 | 101 | Early stopping triggered |

### Cross-Validation Results

| Fold | Validation Loss |
|------|----------------|
| 1    | 0.000298       |
| 2    | N/A            |
| 3    | 0.000342       |

### Test Metrics
- MSE: 0.000168
- MAE: 0.005669
- RMSE: 0.012965
- Unscaled MAE: 0.861188
- Unscaled RMSE: 4.913602

### Feature-specific Reconstruction Error (MAE)
| Feature | MAE |
|---------|-----|
| HH_L0.01_covariance | 16.303163 |
| HH_L0.1_covariance | 10.538815 |
| HH_jit_L0.01_variance | 4.524730 |
| HH_L3_magnitude | 3.694796 |
| HH_L1_magnitude | 3.678633 |

### Model Artifacts
- **Best Model Path**: `./models/best_autoencoder_20250407-184642.h5`
- **Scaler Path**: `./models/best_autoencoder_20250407-184642_scaler.pkl`
- **Results Directory**: `./results/analysis_20250407-184642`

## Model Architectures

The project implements several autoencoder architectures:
1. **Dense Autoencoder**: Traditional fully-connected layers with various activation functions
2. **Convolutional Autoencoder**: Using 1D convolutions for feature extraction
3. **LSTM Autoencoder**: Sequence-based model for temporal patterns

## Usage

To train and evaluate the models:

```bash
python train_eval_autoencoder.py
```

The script will:
1. Look for a dataset in the specified path
2. Train multiple autoencoder architectures
3. Evaluate models using cross-validation
4. Select the best model based on validation loss
5. Save the best model and evaluation results

## Results Analysis

Training results are saved to the `results/` directory with timestamps. Each result folder contains:
- Error statistics for each feature
- Visualization of reconstruction errors
- Model performance metrics

## Environment

The project uses a Python virtual environment (`botnet_env`) with TensorFlow and other dependencies installed. GPU acceleration is supported if available.
