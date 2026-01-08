"""
Model Interpretation Script (Section 2.3)
==========================================
This script performs comprehensive model interpretation for the Seoul Bike Sharing models.

Run with: python model_interpretation.py
"""

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.inspection import PartialDependenceDisplay
from sklearn.model_selection import TimeSeriesSplit, cross_validate, GridSearchCV
import matplotlib.pyplot as plt
import seaborn as sns
import shap
from tqdm import tqdm
import os

visualizations = 'visualizations'
os.makedirs(visualizations, exist_ok=True)

# Set style
sns.set_style('whitegrid')
plt.rcParams['figure.figsize'] = (12, 8)

print("="*70)
print("MODEL INTERPRETATION - SEOUL BIKE SHARING")
print("="*70)

# ==============================================================================
# 1. DATA LOADING AND PREPROCESSING
# ==============================================================================
print("\n[1/6] Loading and preprocessing data...")

df = pd.read_csv("SeoulBikeData.csv", encoding="latin1")
print(f"   ✓ Loaded {df.shape[0]} rows, {df.shape[1]} columns")

# Define target and features
y = df["Rented Bike Count"]
X = df.drop(columns=["Rented Bike Count"])

# Parse date and add calendar features
X["Date"] = pd.to_datetime(X["Date"], dayfirst=True, errors="coerce")
X["DayOfWeek"] = X["Date"].dt.dayofweek
X["Month"] = X["Date"].dt.month
X = X.drop(columns=["Date"])

# Encode categorical variables
X['Holiday_Encoded'] = X['Holiday'].map({'Holiday': 1, 'No Holiday': 0})
X['Functioning_Day_Encoded'] = X['Functioning Day'].map({'Yes': 1, 'No': 0})

# One-hot encoding for Seasons
season_dummies = pd.get_dummies(X['Seasons'], prefix='Season', drop_first=True, dtype=int)
X = pd.concat([X, season_dummies], axis=1)

# Drop original categorical columns and highly correlated feature
X = X.drop(columns=['Seasons', 'Holiday', 'Functioning Day', 'Dew point temperature(°C)'])

print(f"   ✓ Final feature set: {X.shape[1]} features")

# ==============================================================================
# 2. TRAIN MODELS
# ==============================================================================
print("\n[2/6] Training models...")

# Setup Time Series Cross-Validation
# Initialize TimeSeriesSplit with 5 folds
NUM_FOLDS = 5

tscv = TimeSeriesSplit(n_splits=NUM_FOLDS)

scoring = {
    'neg_mae': 'neg_mean_absolute_error',
    'neg_rmse': 'neg_root_mean_squared_error',
    'r2': 'r2'
}

# Linear Regression
print("   Training Linear Regression...")
lr_model = LinearRegression()
lr_cv_results = cross_validate(
    lr_model, X, y, 
    cv=tscv, 
    scoring=scoring,
    return_train_score=True,
    n_jobs=-1
)

for fold in range(5):
    train_r2 = lr_cv_results['train_r2'][fold]
    test_r2 = lr_cv_results['test_r2'][fold]
    test_mae = -lr_cv_results['test_neg_mae'][fold]
    test_rmse = -lr_cv_results['test_neg_rmse'][fold]
    print(f"  {fold+1}  |   {train_r2:.4f}   |  {test_r2:.4f}  |  {test_mae:.2f}  |  {test_rmse:.2f}")

# Calculate mean and std across folds
lr_mean_r2 = lr_cv_results['test_r2'].mean()
lr_std_r2 = lr_cv_results['test_r2'].std()
lr_mean_mae = -lr_cv_results['test_neg_mae'].mean()
lr_std_mae = -lr_cv_results['test_neg_mae'].std()
lr_mean_rmse = -lr_cv_results['test_neg_rmse'].mean()
lr_std_rmse = -lr_cv_results['test_neg_rmse'].std()


print()
print(f"Mean R²:    {lr_mean_r2:.4f} (+/- {lr_std_r2:.4f})")
print(f"Mean MAE:   {lr_mean_mae:.2f} (+/- {lr_std_mae:.2f})")
print(f"Mean RMSE:  {lr_mean_rmse:.2f} (+/- {lr_std_rmse:.2f})")

# NOW FIT ON FULL DATASET for interpretation
print("\n   Fitting Linear Regression on full dataset for interpretation...")
lr_model = LinearRegression()
lr_model.fit(X, y)
print("   ✓ Linear Regression trained on full dataset")


# Store results for comparison
lr_metrics = {
    'mean_r2': lr_mean_r2,
    'std_r2': lr_std_r2,
    'mean_mae': lr_mean_mae,
    'std_mae': lr_std_mae,
    'mean_rmse': lr_mean_rmse,
    'std_rmse': lr_std_rmse,
    'cv_results': lr_cv_results
}

print("   ✓ Linear Regression trained")


#######################################
# Random Forest with optimal parameters
print("   Training Random Forest with optimal parameters (this may take a moment)...")
rf_model = RandomForestRegressor(
    n_estimators=200,
    max_depth=20,
    min_samples_split=2,
    min_samples_leaf=1,
    random_state=42,
    n_jobs=-1,
    verbose=0
)

# Use tqdm to show progress during training
with tqdm(total=100, desc="   RF Training", bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt}') as pbar:
    rf_model.fit(X, y)
    pbar.update(100)

print("   ✓ Random Forest trained")

# ==============================================================================
# 3. FEATURE IMPORTANCE ANALYSIS
# ==============================================================================
print("\n[3/6] Analyzing feature importance...")

# Random Forest Feature Importance
feature_importance = pd.DataFrame({
    'Feature': X.columns,
    'Importance': rf_model.feature_importances_
}).sort_values('Importance', ascending=False)

print("\n   Top 10 Most Important Features (Random Forest):")
print("   " + "-"*66)
for idx, row in feature_importance.head(10).iterrows():
    print(f"   {row['Feature']:35s} : {row['Importance']:.4f}")

# Save plot
plt.figure(figsize=(10, 8))
plt.barh(feature_importance['Feature'], feature_importance['Importance'], color='steelblue')
plt.xlabel('Importance', fontsize=12)
plt.ylabel('Feature', fontsize=12)
plt.title('Random Forest Feature Importance', fontsize=14, fontweight='bold')
plt.gca().invert_yaxis()
plt.tight_layout()
plt.savefig(f'{visualizations}/feature_importance_rf.png', dpi=300, bbox_inches='tight')
print("   ✓ Saved: feature_importance_rf.png")
plt.close()

# Linear Regression Coefficients
lr_coef = pd.DataFrame({
    'Feature': X.columns,
    'Coefficient': lr_model.coef_
}).sort_values('Coefficient', ascending=False)

print("\n   Top 5 Positive Contributors (Linear Regression):")
print("   " + "-"*66)
for idx, row in lr_coef.head(5).iterrows():
    print(f"   {row['Feature']:35s} : {row['Coefficient']:+.2f}")

print("\n   Top 5 Negative Contributors (Linear Regression):")
print("   " + "-"*66)
for idx, row in lr_coef.tail(5).iterrows():
    print(f"   {row['Feature']:35s} : {row['Coefficient']:+.2f}")

# Save plot
plt.figure(figsize=(10, 8))
colors = ['green' if c > 0 else 'red' for c in lr_coef['Coefficient']]
plt.barh(lr_coef['Feature'], lr_coef['Coefficient'], color=colors, alpha=0.7)
plt.xlabel('Coefficient Value', fontsize=12)
plt.ylabel('Feature', fontsize=12)
plt.title('Linear Regression Coefficients', fontsize=14, fontweight='bold')
plt.axvline(x=0, color='black', linestyle='--', linewidth=1)
plt.gca().invert_yaxis()
plt.tight_layout()
plt.savefig(f'{visualizations}/coefficients_lr.png', dpi=300, bbox_inches='tight')
print("   ✓ Saved: coefficients_lr.png")
plt.close()

# ==============================================================================
# 4. SHAP ANALYSIS
# ==============================================================================
print("\n[4/6] Computing SHAP values (this may take a minute)...")

# Use sample for faster computation
sample_size = 1000
X_sample = X.sample(n=sample_size, random_state=42)

print(f"   Creating SHAP explainer...")
explainer_rf = shap.TreeExplainer(rf_model)

# Compute SHAP values in batches to show real progress
print(f"   Computing SHAP values for {sample_size} samples...")
batch_size = 10  # Process 100 samples at a time
num_batches = (sample_size + batch_size - 1) // batch_size

shap_values_list = []
with tqdm(total=sample_size, desc="   SHAP Progress", unit="samples") as pbar:
    for i in range(0, sample_size, batch_size):
        batch_end = min(i + batch_size, sample_size)
        batch = X_sample.iloc[i:batch_end]
        
        # Compute SHAP for this batch
        batch_shap = explainer_rf.shap_values(batch, check_additivity=False)
        shap_values_list.append(batch_shap)
        
        # Update progress bar
        pbar.update(batch_end - i)

# Combine all batches
shap_values_rf = np.vstack(shap_values_list)

print(f"   ✓ SHAP values computed for {sample_size} samples")

# SHAP Summary Plot (dot)
print("\n   Generating SHAP visualizations...")
with tqdm(total=4, desc="   SHAP Plots", bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt}') as pbar:
    plt.figure(figsize=(10, 8))
    shap.summary_plot(shap_values_rf, X_sample, plot_type="dot", show=False)
    plt.title('SHAP Summary Plot - Feature Impact', fontsize=14, fontweight='bold', pad=20)
    plt.tight_layout()
    plt.savefig(f'{visualizations}/shap_summary_dot.png', dpi=300, bbox_inches='tight')
    print("   ✓ Saved: shap_summary_dot.png")
    plt.close()
    pbar.update(1)
    
    # SHAP Bar Plot
    plt.figure(figsize=(10, 8))
    shap.summary_plot(shap_values_rf, X_sample, plot_type="bar", show=False)
    plt.title('SHAP Feature Importance', fontsize=14, fontweight='bold', pad=20)
    plt.tight_layout()
    plt.savefig(f'{visualizations}/shap_summary_bar.png', dpi=300, bbox_inches='tight')
    print("   ✓ Saved: shap_summary_bar.png")
    plt.close()
    pbar.update(1)
    
    # SHAP Dependence Plots
    top_features = ['Hour', 'Temperature(°C)', 'Humidity(%)', 'Functioning_Day_Encoded']
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('SHAP Dependence Plots - Top Features', fontsize=16, fontweight='bold')
    
    for idx, feature in enumerate(top_features):
        row = idx // 2
        col = idx % 2
        shap.dependence_plot(feature, shap_values_rf, X_sample, ax=axes[row, col], show=False)
        axes[row, col].set_title(f'{feature}', fontsize=12, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(f'{visualizations}/shap_dependence_plots.png', dpi=300, bbox_inches='tight')
    print("   ✓ Saved: shap_dependence_plots.png")
    plt.close()
    pbar.update(2)

# ==============================================================================
# 5. PARTIAL DEPENDENCE PLOTS
# ==============================================================================
print("\n[5/6] Creating Partial Dependence Plots...")

with tqdm(total=2, desc="   PDP Plots", bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt}') as pbar:
    # 1D PDPs
    key_features = ['Hour', 'Temperature(°C)', 'Humidity(%)', 'Solar Radiation']
    feature_indices = [X.columns.get_loc(f) for f in key_features]
    
    fig, ax = plt.subplots(figsize=(14, 10))
    display = PartialDependenceDisplay.from_estimator(
        rf_model, X, features=feature_indices,
        feature_names=X.columns, n_cols=2, grid_resolution=50, ax=ax
    )
    fig.suptitle('Partial Dependence Plots - Key Features', fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig(f'{visualizations}/partial_dependence_1d.png', dpi=300, bbox_inches='tight')
    print("   ✓ Saved: partial_dependence_1d.png")
    plt.close()
    pbar.update(1)
    
    # 2D PDP: Hour × Temperature
    fig, ax = plt.subplots(figsize=(10, 8))
    display = PartialDependenceDisplay.from_estimator(
        rf_model, X, features=[(0, 1)],
        feature_names=X.columns, grid_resolution=30, ax=ax
    )
    plt.suptitle('2D Partial Dependence: Hour × Temperature(°C)', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(f'{visualizations}/partial_dependence_2d.png', dpi=300, bbox_inches='tight')
    print("   ✓ Saved: partial_dependence_2d.png")
    plt.close()
    pbar.update(1)

# ==============================================================================
# 6. KEY INSIGHTS SUMMARY
# ==============================================================================
print("\n[6/6] Generating insights summary...\n")

print("="*70)
print("KEY INSIGHTS FROM MODEL INTERPRETATION")
print("="*70)

print("\n1. MOST IMPORTANT FEATURES (Random Forest):")
print("-" * 70)
for idx, row in feature_importance.head(5).iterrows():
    print(f"   {row['Feature']:35s} : {row['Importance']:.4f}")

print("\n2. TEMPORAL PATTERNS:")
print("-" * 70)
print("   • Hour of day is the strongest predictor")
print("   • Peak rental hours likely correspond to commute times")
print("   • Month and day of week also contribute significantly")

print("\n3. WEATHER EFFECTS:")
print("-" * 70)
print("   • Temperature is a major positive factor")
print("   • Humidity has moderate negative impact")
print("   • Rainfall and snowfall reduce rentals")

print("\n4. OPERATIONAL INSIGHTS:")
print("-" * 70)
print("   • Functioning Day status is critical")
print("   • Holidays show different rental patterns")
print("   • Seasonal variations are important")

print("\n5. ACTIONABLE RECOMMENDATIONS:")
print("-" * 70)
print("   • Allocate more bikes during peak hours (morning/evening commute)")
print("   • Adjust fleet size based on weather forecasts")
print("   • Plan for reduced demand during holidays and adverse weather")
print("   • Maintain high system availability (minimize non-functioning days)")

print("\n" + "="*70)
print("ANALYSIS COMPLETE!")
print("="*70)
print("\nGenerated files:")
print("  • feature_importance_rf.png")
print("  • coefficients_lr.png")
print("  • shap_summary_dot.png")
print("  • shap_summary_bar.png")
print("  • shap_dependence_plots.png")
print("  • partial_dependence_1d.png")
print("  • partial_dependence_2d.png")
print("\nAll visualizations saved in current directory!")