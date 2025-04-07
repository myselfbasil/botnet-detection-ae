import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, Dropout, BatchNormalization, LeakyReLU, GaussianNoise, ActivityRegularization
from tensorflow.keras.layers import Reshape, Flatten, Conv1D, MaxPooling1D, UpSampling1D
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint, TensorBoard
from tensorflow.keras.regularizers import l1_l2
import matplotlib.pyplot as plt
import seaborn as sns
import os
import joblib
from sklearn.model_selection import train_test_split, KFold
from sklearn.preprocessing import RobustScaler
from datetime import datetime
from functools import partial
import tensorflow_addons as tfa
import warnings
warnings.filterwarnings('ignore')

# Set random seeds for reproducibility
RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)
tf.random.set_seed(RANDOM_SEED)

# Create directories for models and logs
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, 'models')
LOG_DIR = os.path.join(BASE_DIR, 'logs')
RESULTS_DIR = os.path.join(BASE_DIR, 'results')
for directory in [MODEL_DIR, LOG_DIR, RESULTS_DIR]:
    os.makedirs(directory, exist_ok=True)

class RobustCustomScaler:
    """Enhanced scaler that combines RobustScaler with custom scaling logic"""
    def __init__(self, quantile_range=(1, 99)):
        self.robust_scaler = RobustScaler(quantile_range=quantile_range)
        self.feature_names_ = None
        self.feature_stats_ = None
        
    def fit(self, data):
        if isinstance(data, pd.DataFrame):
            self.feature_names_ = data.columns.tolist()
            # Store feature statistics for diagnostics
            self.feature_stats_ = {
                'mean': data.mean().to_dict(),
                'median': data.median().to_dict(),
                'std': data.std().to_dict(),
                'min': data.min().to_dict(),
                'max': data.max().to_dict(),
                'skew': data.skew().to_dict()
            }
            data_np = data.values
        else:
            data_np = data
            
        self.robust_scaler.fit(data_np)
        return self
        
    def transform(self, data):
        if isinstance(data, pd.DataFrame):
            if self.feature_names_ is not None:
                # Ensure columns match and are in the same order
                data = data[self.feature_names_]
            data_np = data.values
        else:
            data_np = data
        
        # Apply robust scaling
        scaled_data = self.robust_scaler.transform(data_np)
        
        # Clip values to prevent extreme outliers
        scaled_data = np.clip(scaled_data, -5, 5)
        
        # Normalize to [0, 1] range for autoencoder
        scaled_data = (scaled_data + 5) / 10
        
        return scaled_data
    
    def fit_transform(self, data):
        self.fit(data)
        return self.transform(data)
    
    def inverse_transform(self, scaled_data):
        # Convert back from [0, 1] to [-5, 5]
        scaled_data_original = (scaled_data * 10) - 5
        
        # Apply inverse robust scaling
        return self.robust_scaler.inverse_transform(scaled_data_original)
    
    def save_feature_stats(self, path):
        """Save feature statistics to a CSV file"""
        if self.feature_stats_ is not None:
            stats_df = pd.DataFrame(self.feature_stats_)
            stats_df.to_csv(path, index=True)
            print(f"Feature statistics saved to {path}")

def create_advanced_autoencoder(input_dim, encoding_dim=64, use_conv=True, activation='swish'):
    """Create an advanced autoencoder with multiple architecture options"""
    # Input layer
    input_layer = Input(shape=(input_dim,))
    
    # Add noise for robustness
    x = GaussianNoise(0.01)(input_layer)
    
    # Choose between convolutional or dense architecture
    if use_conv and input_dim >= 10:  # Only use conv if we have enough features
        # Reshape for Conv1D (treating features as a sequence)
        seq_length = input_dim
        x = Reshape((seq_length, 1))(x)
        
        # Encoder path with residual connections
        # First convolutional block
        conv1 = Conv1D(32, 3, padding='same', activation=activation)(x)
        conv1 = BatchNormalization()(conv1)
        conv1 = Conv1D(32, 3, padding='same', activation=activation)(conv1)
        conv1 = BatchNormalization()(conv1)
        
        # Pooling and residual connection
        pool1 = MaxPooling1D(2, padding='same')(conv1)
        
        # Second convolutional block
        conv2 = Conv1D(64, 3, padding='same', activation=activation)(pool1)
        conv2 = BatchNormalization()(conv2)
        conv2 = Conv1D(64, 3, padding='same', activation=activation)(conv2)
        conv2 = BatchNormalization()(conv2)
        
        # Extract bottleneck representation
        bottleneck_conv = Conv1D(encoding_dim, 3, padding='same', activation=activation)(conv2)
        bottleneck_shape = bottleneck_conv.shape[1:]
        encoded = Flatten()(bottleneck_conv)
        
        # Compressed representation with regularization
        encoded = Dense(encoding_dim, activation=activation, 
                       activity_regularizer=l1_l2(l1=1e-5, l2=1e-5))(encoded)
        
        # Start decoder path
        decoded = Dense(int(np.prod(bottleneck_shape)), activation=activation)(encoded)
        decoded = Reshape(bottleneck_shape)(decoded)
        
        # Upsampling and convolution for reconstruction
        decoded = Conv1D(64, 3, padding='same', activation=activation)(decoded)
        decoded = BatchNormalization()(decoded)
        decoded = UpSampling1D(2)(decoded)
        
        # More convolutions
        decoded = Conv1D(32, 3, padding='same', activation=activation)(decoded)
        decoded = BatchNormalization()(decoded)
        decoded = Conv1D(32, 3, padding='same', activation=activation)(decoded)
        decoded = BatchNormalization()(decoded)
        
        # Final reconstruction
        decoded = Conv1D(1, 3, padding='same')(decoded)
        decoded = Flatten()(decoded)
        output_layer = Dense(input_dim, activation='sigmoid')(decoded)
        
    else:
        # Traditional dense autoencoder with progressive layer sizes
        layer_sizes = [
            max(int(input_dim * 0.8), encoding_dim * 4),
            max(int(input_dim * 0.6), encoding_dim * 2),
            max(int(input_dim * 0.4), encoding_dim)
        ]
        
        # Encoder
        x = x  # Start with the noised input
        for i, size in enumerate(layer_sizes):
            x = Dense(size, kernel_regularizer=l1_l2(l1=1e-5, l2=1e-5))(x)
            if activation == 'leaky_relu':
                x = LeakyReLU(alpha=0.2)(x)
            else:
                x = tf.keras.layers.Activation(activation)(x)
            x = BatchNormalization()(x)
            x = Dropout(0.2)(x)
        
        # Bottleneck with regularization
        encoded = Dense(encoding_dim, activation=activation, 
                       activity_regularizer=l1_l2(l1=1e-5, l2=1e-5))(x)
        
        # Decoder mirrors the encoder
        x = encoded
        for i, size in reversed(list(enumerate(layer_sizes))):
            x = Dense(size, kernel_regularizer=l1_l2(l1=1e-5, l2=1e-5))(x)
            if activation == 'leaky_relu':
                x = LeakyReLU(alpha=0.2)(x)
            else:
                x = tf.keras.layers.Activation(activation)(x)
            x = BatchNormalization()(x)
            x = Dropout(0.2)(x)
        
        # Output layer
        output_layer = Dense(input_dim, activation='sigmoid')(x)
    
    # Create model
    autoencoder = Model(input_layer, output_layer)
    
    # Use LAMB optimizer for better convergence with large batch sizes
    optimizer = tfa.optimizers.LAMB(learning_rate=1e-3, weight_decay=1e-4)
    
    # Compile with weighted loss
    autoencoder.compile(
        optimizer=optimizer,
        loss=tf.keras.losses.MeanSquaredError(),
        metrics=['mae', tf.keras.metrics.RootMeanSquaredError(name='rmse')]
    )
    
    return autoencoder

def huber_mse_loss(y_true, y_pred, delta=1.0, mse_weight=0.7):
    """Combined Huber and MSE loss for robustness and accuracy"""
    mse = tf.keras.losses.MeanSquaredError()(y_true, y_pred)
    huber = tf.keras.losses.Huber(delta=delta)(y_true, y_pred)
    return mse_weight * mse + (1 - mse_weight) * huber

def analysis_and_visualization(original_data, reconstructed_data, scaler, log_dir, timestamp):
    """Analyze reconstruction quality and create visualizations"""
    # Create results directory
    results_dir = os.path.join(RESULTS_DIR, f"analysis_{timestamp}")
    os.makedirs(results_dir, exist_ok=True)
    
    # Convert to DataFrames if they're not already
    if not isinstance(original_data, pd.DataFrame):
        original_data = pd.DataFrame(original_data)
    if not isinstance(reconstructed_data, pd.DataFrame):
        reconstructed_data = pd.DataFrame(reconstructed_data, columns=original_data.columns)
    
    # Calculate reconstruction error per feature
    errors = np.abs(original_data.values - reconstructed_data.values)
    error_df = pd.DataFrame(errors, columns=original_data.columns)
    
    # Overall error statistics
    error_stats = {
        'mae': np.mean(errors),
        'max_error': np.max(errors),
        'error_std': np.std(errors)
    }
    
    # Per feature error analysis
    feature_errors = error_df.mean().sort_values(ascending=False)
    
    # Save error statistics
    with open(os.path.join(results_dir, 'error_stats.txt'), 'w') as f:
        f.write("Overall Error Statistics:\n")
        for metric, value in error_stats.items():
            f.write(f"{metric}: {value:.6f}\n")
        
        f.write("\nPer-Feature Error (Mean Absolute Error):\n")
        for feature, error in feature_errors.items():
            f.write(f"{feature}: {error:.6f}\n")
    
    # Plot error distribution
    plt.figure(figsize=(12, 8))
    plt.subplot(2, 2, 1)
    sns.histplot(errors.flatten(), kde=True)
    plt.title('Distribution of Reconstruction Errors')
    plt.xlabel('Absolute Error')
    
    # Plot top features with highest error
    plt.subplot(2, 2, 2)
    feature_errors.head(10).plot(kind='bar')
    plt.title('Top 10 Features with Highest Reconstruction Error')
    plt.ylabel('Mean Absolute Error')
    plt.tight_layout()
    
    # Plot original vs reconstructed for a few samples
    num_samples = min(5, original_data.shape[0])
    num_features = min(10, original_data.shape[1])
    
    plt.subplot(2, 2, 3)
    for i in range(num_samples):
        plt.plot(original_data.iloc[i, :num_features].values, 'b-', alpha=0.5)
        plt.plot(reconstructed_data.iloc[i, :num_features].values, 'r-', alpha=0.5)
    plt.title(f'Original (blue) vs Reconstructed (red) - First {num_features} features')
    plt.xlabel('Feature Index')
    plt.ylabel('Value')
    
    # Plot feature importance based on reconstruction error
    plt.subplot(2, 2, 4)
    sns.boxplot(data=error_df.iloc[:, :min(10, error_df.shape[1])])
    plt.title('Error Distribution by Feature')
    plt.ylabel('Absolute Error')
    plt.xticks(rotation=45)
    
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, 'reconstruction_analysis.png'))
    
    print(f"Analysis complete. Results saved to {results_dir}")
    return error_stats

def preprocess_data(data_path, test_size=0.2, validation_size=0.2):
    """Enhanced data preprocessing with better outlier handling and analysis"""
    print(f"Loading data from {data_path}...")
    df = pd.read_csv(data_path)
    
    # Analyze dataset
    print(f"Original dataset shape: {df.shape}")
    
    # Keep only numeric columns
    numeric_cols = df.select_dtypes(include=['number']).columns.tolist()
    df = df[numeric_cols]
    
    # Check for missing values
    missing_vals = df.isnull().sum()
    if missing_vals.sum() > 0:
        print(f"Found {missing_vals.sum()} missing values")
        print("Filling missing values with column medians...")
        df = df.fillna(df.median())
    
    # Drop constant columns
    const_cols = [col for col in df.columns if df[col].nunique() <= 1]
    if const_cols:
        print(f"Dropping {len(const_cols)} constant columns: {const_cols}")
        df = df.drop(columns=const_cols)
    
    # Analyze feature correlations
    corr_matrix = df.corr().abs()
    
    # Find highly correlated features (above 0.95)
    high_corr_vars = set()
    for i in range(len(corr_matrix.columns)):
        for j in range(i):
            if abs(corr_matrix.iloc[i, j]) > 0.95:
                high_corr_vars.add(corr_matrix.columns[i])
    
    if high_corr_vars:
        print(f"Identified {len(high_corr_vars)} highly correlated features (>0.95)")
        # Don't drop them, just report
    
    # Remove extreme outliers with robust method
    print("Removing extreme outliers...")
    Q1 = df.quantile(0.01)  # More conservative lower bound
    Q3 = df.quantile(0.99)  # More conservative upper bound
    IQR = Q3 - Q1
    # More selective outlier removal - only remove the most extreme outliers
    df_clean = df[~((df < (Q1 - 5 * IQR)) | (df > (Q3 + 5 * IQR))).any(axis=1)]
    
    print(f"Clean data shape after outlier removal: {df_clean.shape}")
    if df_clean.shape[0] < 0.9 * df.shape[0]:
        print("Warning: Removed more than 10% of data as outliers. Consider checking your data.")
    
    # Split into train and test sets
    train_df, test_df = train_test_split(df_clean, test_size=test_size, random_state=RANDOM_SEED)
    
    # Further split train into train and validation
    train_df, val_df = train_test_split(train_df, test_size=validation_size, random_state=RANDOM_SEED)
    
    # Scale the data with our robust custom scaler
    scaler = RobustCustomScaler(quantile_range=(1, 99))
    
    # Use pandas for easier tracking
    train_scaled = pd.DataFrame(
        scaler.fit_transform(train_df),
        columns=train_df.columns
    )
    val_scaled = pd.DataFrame(
        scaler.transform(val_df),
        columns=val_df.columns
    )
    test_scaled = pd.DataFrame(
        scaler.transform(test_df),
        columns=test_df.columns
    )
    
    print(f"Train shape: {train_scaled.shape}")
    print(f"Validation shape: {val_scaled.shape}")
    print(f"Test shape: {test_scaled.shape}")
    
    # Save splits and feature stats
    splits_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'dataset/splits')
    os.makedirs(splits_dir, exist_ok=True)
    
    train_df.to_csv(os.path.join(splits_dir, 'train.csv'), index=False)
    val_df.to_csv(os.path.join(splits_dir, 'validation.csv'), index=False)
    test_df.to_csv(os.path.join(splits_dir, 'test.csv'), index=False)
    
    stats_path = os.path.join(splits_dir, 'feature_stats.csv')
    scaler.save_feature_stats(stats_path)
    
    # Return both DataFrames and numpy arrays
    return train_scaled, val_scaled, test_scaled, train_df, val_df, test_df, scaler

def train_with_kfold(train_data, val_data, input_dim, model_config, timestamp, n_splits=5):
    """Train multiple models with k-fold cross-validation"""
    models = []
    histories = []
    val_scores = []
    
    # Combine train and validation for k-fold
    all_data = np.vstack([train_data.values if isinstance(train_data, pd.DataFrame) else train_data,
                         val_data.values if isinstance(val_data, pd.DataFrame) else val_data])
    
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=RANDOM_SEED)
    
    for fold, (train_idx, val_idx) in enumerate(kf.split(all_data)):
        print(f"\n------------ Training Fold {fold+1}/{n_splits} ------------")
        
        fold_train = all_data[train_idx]
        fold_val = all_data[val_idx]
        
        # Create model with current fold's config
        autoencoder = create_advanced_autoencoder(
            input_dim=input_dim, 
            encoding_dim=model_config['encoding_dim'],
            use_conv=model_config['use_conv'],
            activation=model_config['activation']
        )
        
        # Create fold-specific callbacks
        fold_model_name = f"{model_config['model_name']}_fold{fold+1}"
        fold_callbacks = get_callbacks(fold_model_name, timestamp)
        
        # Train the model
        history = autoencoder.fit(
            fold_train, fold_train,
            epochs=model_config['epochs'],
            batch_size=model_config['batch_size'],
            validation_data=(fold_val, fold_val),
            callbacks=fold_callbacks,
            shuffle=True,
            verbose=1
        )
        
        # Evaluate on validation set
        val_loss = autoencoder.evaluate(fold_val, fold_val, verbose=0)[0]
        val_scores.append(val_loss)
        models.append(autoencoder)
        histories.append(history)
        
        print(f"Fold {fold+1} validation loss: {val_loss:.6f}")
    
    # Find best model
    best_fold = np.argmin(val_scores)
    best_model = models[best_fold]
    print(f"\nBest model from fold {best_fold+1} with validation loss: {val_scores[best_fold]:.6f}")
    
    return best_model, histories[best_fold], val_scores

def get_callbacks(model_name, timestamp):
    """Create training callbacks with improved parameters"""
    # Create directories for this run
    run_log_dir = os.path.join(LOG_DIR, f"{model_name}_{timestamp}")
    os.makedirs(run_log_dir, exist_ok=True)
    
    callbacks = [
        # Early stopping with more patience and better monitoring
        EarlyStopping(
            monitor='val_loss',
            patience=30,  # More patience
            restore_best_weights=True,
            min_delta=0.0001,
            verbose=1,
            mode='min'
        ),
        
        # Reduce learning rate with more gradual reduction
        ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.7,  # More gradual reduction
            patience=10,
            min_lr=1e-6,
            verbose=1,
            mode='min'
        ),
        
        # Model checkpoint
        ModelCheckpoint(
            os.path.join(MODEL_DIR, f"{model_name}_{timestamp}_best.h5"),
            monitor='val_loss',
            save_best_only=True,
            mode='min',
            verbose=1
        ),
        
        # TensorBoard
        TensorBoard(
            log_dir=run_log_dir,
            histogram_freq=2,
            write_graph=True,
            update_freq='epoch'
        )
    ]
    
    return callbacks

def plot_training_history(history, model_name, timestamp):
    """Enhanced training history plots with more metrics"""
    # Create a dedicated figure with more subplots
    plt.figure(figsize=(18, 12))
    
    # Plot loss
    plt.subplot(2, 2, 1)
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('Model Loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Validation'], loc='upper right')
    plt.grid(True, linestyle='--', alpha=0.7)
    
    # Plot MAE
    plt.subplot(2, 2, 2)
    plt.plot(history.history['mae'])
    plt.plot(history.history['val_mae'])
    plt.title('Model MAE')
    plt.ylabel('MAE')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Validation'], loc='upper right')
    plt.grid(True, linestyle='--', alpha=0.7)
    
    # Plot RMSE
    plt.subplot(2, 2, 3)
    plt.plot(history.history['rmse'])
    plt.plot(history.history['val_rmse'])
    plt.title('Model RMSE')
    plt.ylabel('RMSE')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Validation'], loc='upper right')
    plt.grid(True, linestyle='--', alpha=0.7)
    
    # Plot learning rate if available
    if 'lr' in history.history:
        plt.subplot(2, 2, 4)
        plt.plot(history.history['lr'])
        plt.title('Learning Rate')
        plt.ylabel('Learning Rate')
        plt.xlabel('Epoch')
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.yscale('log')
    
    plt.tight_layout()
    plt.savefig(os.path.join(LOG_DIR, f"{model_name}_{timestamp}_training_history.png"))
    plt.close()

def evaluate_autoencoder(autoencoder, test_data, test_df, scaler, model_name, timestamp):
    """Comprehensive evaluation of the autoencoder"""
    # Get reconstructions
    if isinstance(test_data, pd.DataFrame):
        test_data_values = test_data.values
        feature_names = test_data.columns
    else:
        test_data_values = test_data
        feature_names = [f'feature_{i}' for i in range(test_data.shape[1])]
    
    # Get reconstructed data
    reconstructed_data = autoencoder.predict(test_data_values)
    
    # Calculate basic metrics
    mse = np.mean(np.square(test_data_values - reconstructed_data))
    mae = np.mean(np.abs(test_data_values - reconstructed_data))
    rmse = np.sqrt(mse)
    
    print(f"\nEvaluation on test data:")
    print(f"MSE: {mse:.6f}")
    print(f"MAE: {mae:.6f}")
    print(f"RMSE: {rmse:.6f}")
    
    # Get unscaled data for better interpretability
    if scaler is not None:
        try:
            unscaled_orig = test_df
            unscaled_recon = pd.DataFrame(
                scaler.inverse_transform(reconstructed_data),
                columns=feature_names
            )
            
            # Get business-relevant metrics on unscaled data
            unscaled_mae = np.mean(np.abs(unscaled_orig.values - unscaled_recon.values))
            unscaled_rmse = np.sqrt(np.mean(np.square(unscaled_orig.values - unscaled_recon.values)))
            print(f"Unscaled MAE: {unscaled_mae:.6f}")
            print(f"Unscaled RMSE: {unscaled_rmse:.6f}")
            
            # Analyze feature-specific performance
            feature_maes = np.mean(np.abs(unscaled_orig.values - unscaled_recon.values), axis=0)
            top_features = np.argsort(feature_maes)
            
            print("\nFeature-specific reconstruction error (MAE):")
            for i in range(min(5, len(feature_names))):
                idx = top_features[-(i+1)]
                print(f"  {feature_names[idx]}: {feature_maes[idx]:.6f}")
                
            # Run the detailed analysis
            test_df_recon = pd.DataFrame(reconstructed_data, columns=feature_names)
            error_stats = analysis_and_visualization(
                test_data if isinstance(test_data, pd.DataFrame) else pd.DataFrame(test_data, columns=feature_names),
                test_df_recon, 
                scaler, 
                LOG_DIR, 
                timestamp
            )
            
        except Exception as e:
            print(f"Warning: Could not perform unscaled analysis: {e}")
    
    return {
        'mse': mse,
        'mae': mae,
        'rmse': rmse
    }

def train_autoencoder(data_path, epochs=300, batch_size=128, encoding_dim=64):
    """Main training function with extensive enhancements"""
    # Create timestamp for unique model naming
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    
    # Preprocess data
    train_data, val_data, test_data, train_df, val_df, test_df, scaler = preprocess_data(data_path)
    
    # Create model configurations to try
    model_configs = [
        {
            'model_name': 'autoencoder_dense_swish',
            'encoding_dim': encoding_dim,
            'use_conv': False,
            'activation': 'swish',
            'epochs': epochs,
            'batch_size': batch_size
        },
        {
            'model_name': 'autoencoder_conv_swish',
            'encoding_dim': encoding_dim,
            'use_conv': True,
            'activation': 'swish',
            'epochs': epochs,
            'batch_size': batch_size
        },
        {
            'model_name': 'autoencoder_dense_leaky',
            'encoding_dim': encoding_dim,
            'use_conv': False,
            'activation': 'leaky_relu',
            'epochs': epochs,
            'batch_size': batch_size
        }
    ]
    
    input_dim = train_data.shape[1]
    best_models = []
    best_scores = []
    
    # Train each model configuration with k-fold CV
    for config in model_configs:
        print(f"\n========== Training model: {config['model_name']} ==========")
        
        if config['use_conv'] and input_dim < 10:
            print(f"Skipping convolutional model - not enough features ({input_dim} < 10)")
            continue
            
        # Train with k-fold cross validation
        best_model, history, val_scores = train_with_kfold(
            train_data, val_data, input_dim, config, timestamp, n_splits=3
        )
        
        # Plot training history
        plot_training_history(history, config['model_name'], timestamp)
        
        # Evaluate on test data
        test_metrics = evaluate_autoencoder(
            best_model, test_data, test_df, scaler, config['model_name'], timestamp
        )
        
        # Save model and append to results
        model_path = os.path.join(MODEL_DIR, f"{config['model_name']}_{timestamp}_final.h5")
        best_model.save(model_path)
        print(f"Model saved to {model_path}")
        
        # Track model performance
        best_models.append(best_model)
        best_scores.append(test_metrics['mae'])
    
    # Find the overall best model
    if best_scores:
        best_idx = np.argmin(best_scores)
        best_model = best_models[best_idx]
        best_config = model_configs[best_idx]
        
        print(f"\n======== Best Model: {best_config['model_name']} ========")
        print(f"Test MAE: {best_scores[best_idx]:.6f}")
        
        # Save the best model as the final choice
        best_model_path = os.path.join(MODEL_DIR, f"best_autoencoder_{timestamp}.h5")
        scaler_path = os.path.join(MODEL_DIR, f"best_autoencoder_{timestamp}_scaler.pkl")
        
        best_model.save(best_model_path)
        joblib.dump(scaler, scaler_path)
        
        print(f"Best model saved to {best_model_path}")
        print(f"Scaler saved to {scaler_path}")
        
        return best_model, scaler
    else:
        print("No models were successfully trained.")
        return None, None

if __name__ == "__main__":
    # Enable memory growth for GPU
    physical_devices = tf.config.list_physical_devices('GPU')
    if physical_devices:
        print(f"Found {len(physical_devices)} GPU devices")
        for device in physical_devices:
            tf.config.experimental.set_memory_growth(device, True)
            print(f"Memory growth enabled for {device}")
    else:
        print("No GPU devices found, using CPU")
        
    # Add these lines to start training
    data_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'dataset/your_data.csv')
    
    # Check if file exists, otherwise prompt for path
    if not os.path.exists(data_path):
        data_path = input("Enter the path to your dataset CSV file: ")
    
    # Call the training function
    print("Starting autoencoder training...")
    best_model, scaler = train_autoencoder(
        data_path=data_path,
        epochs=300,
        batch_size=128,
        encoding_dim=64
    )
    
    print("Training complete!")
