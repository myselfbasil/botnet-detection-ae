import pandas as pd
import numpy as np

# Load the original dataset
file_path = '/Users/basilshaji/Desktop/Projects/data-science/dataset/original.csv'
print(f"Loading data from {file_path}")

try:
    df = pd.read_csv(file_path)
    
    # Basic dataset information
    print("\n=== Dataset Overview ===")
    print(f"Shape: {df.shape}")
    print(f"Columns: {df.columns.tolist()}")
    print("\n=== Data Types ===")
    print(df.dtypes)
    
    # Check for missing values
    print("\n=== Missing Values ===")
    missing = df.isnull().sum()
    print(missing[missing > 0])
    
    # Basic statistics
    print("\n=== Basic Statistics ===")
    desc = df.describe()
    print(f"Min values: {desc.loc['min'].min()}")
    print(f"Max values: {desc.loc['max'].max()}")
    print(f"Mean values range: {desc.loc['mean'].min()} to {desc.loc['mean'].max()}")
    
    # Identify columns with extreme values
    print("\n=== Columns with Extreme Values ===")
    for col in df.columns:
        if df[col].dtype in [np.float64, np.int64]:
            col_max = df[col].max()
            col_min = df[col].min()
            col_mean = df[col].mean()
            col_std = df[col].std()
            
            # Check if column has extreme values
            if col_max > 1e6 or abs(col_min) > 1e6 or col_std > 1e6:
                print(f"Column '{col}': min={col_min}, max={col_max}, mean={col_mean}, std={col_std}")
    
    # Create cleaned version
    print("\n=== Creating Cleaned Dataset ===")
    df_cleaned = df.copy()
    
    # 1. Handle missing values
    for col in df_cleaned.columns:
        if df_cleaned[col].isnull().sum() > 0:
            if df_cleaned[col].dtype in [np.float64, np.int64]:
                # Fill numeric columns with median
                df_cleaned[col] = df_cleaned[col].fillna(df_cleaned[col].median())
            else:
                # Fill categorical columns with mode
                df_cleaned[col] = df_cleaned[col].fillna(df_cleaned[col].mode()[0])
    
    # 2. Handle extreme values
    for col in df_cleaned.columns:
        if df_cleaned[col].dtype in [np.float64, np.int64]:
            # Calculate column statistics
            col_mean = df_cleaned[col].mean()
            col_std = df_cleaned[col].std()
            
            # Cap values at 5 standard deviations
            lower_bound = col_mean - 5 * col_std
            upper_bound = col_mean + 5 * col_std
            
            # Apply capping
            df_cleaned[col] = df_cleaned[col].clip(lower=lower_bound, upper=upper_bound)
            
            # Apply log transformation for columns with large values
            if df_cleaned[col].max() > 1000:
                # Shift to make all values positive
                min_val = df_cleaned[col].min()
                if min_val < 0:
                    df_cleaned[col] = df_cleaned[col] - min_val + 1
                
                # Apply log transform
                df_cleaned[col] = np.log1p(df_cleaned[col])
    
    # Save cleaned dataset
    cleaned_path = '/Users/basilshaji/Desktop/Projects/data-science/dataset/cleaned_dataset.csv'
    df_cleaned.to_csv(cleaned_path, index=False)
    print(f"Cleaned dataset saved to {cleaned_path}")
    
    # Print statistics of cleaned dataset
    print("\n=== Cleaned Dataset Statistics ===")
    cleaned_desc = df_cleaned.describe()
    print(f"Min values: {cleaned_desc.loc['min'].min()}")
    print(f"Max values: {cleaned_desc.loc['max'].max()}")
    print(f"Mean values range: {cleaned_desc.loc['mean'].min()} to {cleaned_desc.loc['mean'].max()}")
    
except Exception as e:
    print(f"Error processing the dataset: {str(e)}")
