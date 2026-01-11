import os
import pickle
import shap
from tqdm import tqdm
import numpy as np
import pandas as pd

SHAP_SAVE_DIR = 'shap_values'
os.makedirs(SHAP_SAVE_DIR, exist_ok=True)

def compute_and_save_shap_full_dataset(rf_model, X, save_dir=SHAP_SAVE_DIR):
    """
    Compute SHAP values for the FULL dataset and save to disk.
    
    Parameters:
    -----------
    rf_model : RandomForestRegressor
        Trained Random Forest model
    X : pd.DataFrame
        Full feature dataset
    save_dir : str
        Directory to save SHAP values
    
    Returns:
    --------
    dict : Dictionary containing SHAP values and metadata
    """
    print("="*70)
    print("COMPUTING SHAP VALUES FOR FULL DATASET")
    print("="*70)
    
    dataset_size = len(X)
    print(f"\nDataset size: {dataset_size} samples")
    print("This may take several minutes for the full dataset...")
    
    # Create SHAP explainer
    print("\n[1/3] Creating SHAP explainer...")
    explainer = shap.TreeExplainer(rf_model)
    print(" Explainer created")
    
    # Compute SHAP values in batches with progress bar
    print(f"\n[2/3] Computing SHAP values for {dataset_size} samples...")
    batch_size = 100  # Adjust based on your memory
    
    shap_values_list = []
    
    with tqdm(total=dataset_size, desc="SHAP Computation", unit="samples") as pbar:
        for i in range(0, dataset_size, batch_size):
            batch_end = min(i + batch_size, dataset_size)
            batch = X.iloc[i:batch_end]
            
            # Compute SHAP for this batch
            batch_shap = explainer.shap_values(batch, check_additivity=False)
            shap_values_list.append(batch_shap)
            
            # Update progress bar
            pbar.update(batch_end - i)
    
    # Combine all batches
    shap_values = np.vstack(shap_values_list)
    
    
    # Prepare data to save
    print("\n[3/3] Saving SHAP values to disk...")
    
    shap_data = {
        'shap_values': shap_values,
        'expected_value': explainer.expected_value,
        'feature_names': X.columns.tolist(),
        'X_data': X.values,  # Save the actual feature values too
        'X_index': X.index.tolist(),
        'dataset_size': dataset_size,
        'model_params': rf_model.get_params()
    }
    
    # Save to pickle file
    filename = f"shap_values_full_dataset.pkl"
    filepath = os.path.join(save_dir, filename)
    
    with open(filepath, 'wb') as f:
        pickle.dump(shap_data, f, protocol=pickle.HIGHEST_PROTOCOL)
    
    # Also save a "latest" version for easy loading
    latest_filepath = os.path.join(save_dir, "shap_values_latest.pkl")
    with open(latest_filepath, 'wb') as f:
        pickle.dump(shap_data, f, protocol=pickle.HIGHEST_PROTOCOL)
    
    file_size_mb = os.path.getsize(filepath) / (1024 * 1024)
    print(f" Saved to: {filepath}")
    print(f" Also saved to: {latest_filepath} (for easy loading)")
    print(f" File size: {file_size_mb:.2f} MB")
    
    print("\n" + "="*70)
    print("SHAP VALUES SAVED SUCCESSFULLY!")
    print("="*70)
    
    return shap_data


def load_shap_values(filepath=None, save_dir=SHAP_SAVE_DIR):
    """
    Load previously computed SHAP values from disk.
    
    Parameters:
    -----------
    filepath : str, optional
        Specific file to load. If None, loads the latest.
    save_dir : str
        Directory where SHAP values are saved
    
    Returns:
    --------
    dict : Dictionary containing SHAP values and metadata
    """
    if filepath is None:
        filepath = os.path.join(save_dir, "shap_values_latest.pkl")
    
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"SHAP values file not found: {filepath}")
    
    print(f"Loading SHAP values from: {filepath}")
    
    with open(filepath, 'rb') as f:
        shap_data = pickle.load(f)
    
    print(f" Loaded SHAP values for {shap_data['dataset_size']} samples")
    
    return shap_data


def get_sample_for_plotting(shap_data, sample_size=8760, random_state=42):
    """
    Get a random sample from the full SHAP data for plotting.
    
    Parameters:
    -----------
    shap_data : dict
        Full SHAP data dictionary
    sample_size : int
        Number of samples to return
    random_state : int
        Random seed for reproducibility
    
    Returns:
    --------
    tuple : (shap_values_sample, X_sample)
    """
    np.random.seed(random_state)
    
    total_size = shap_data['dataset_size']
    sample_size = min(sample_size, total_size)
    
    if sample_size < total_size:
        # Random sample indices
        indices = np.random.choice(total_size, size=sample_size, replace=False)
        
        # Extract sample
        shap_values_sample = shap_data['shap_values'][indices]
        X_sample = pd.DataFrame(
            shap_data['X_data'][indices],
            columns=shap_data['feature_names']
        )
    else:
        shap_values_sample = shap_data['shap_values']
        X_sample = pd.DataFrame(
            shap_data['X_data'],
            columns=shap_data['feature_names']
        )
    
    return shap_values_sample, X_sample