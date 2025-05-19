import pandas as pd
import numpy as np
from scipy.stats import zscore
from imblearn.over_sampling import SMOTE
from statsmodels.stats.outliers_influence import variance_inflation_factor
from sklearn.model_selection import train_test_split

target_var = 'SeriousDlqin2yrs'

def train_preprocess_GMSC(df): #preprocessing for give me some credit dataset
    df.sort_index(inplace=True)
    # Scheid de target kolom; zodat hier geen preprocessing op wordt toegepast
    target = df[target_var]
    features = df.drop(columns=[target_var])

    # Missing values
        # For numerical columns
    numeric_cols = features.select_dtypes(include=[np.number]).columns
    features_num_mean = features[numeric_cols].mean()
    features[numeric_cols] = features[numeric_cols].fillna(features_num_mean) 
        # For categorical columns
    categorical_cols = features.select_dtypes(exclude=[np.number]).columns
    if not categorical_cols.empty:
        features_cat_mode = features[categorical_cols].mode().iloc[0]
        features[categorical_cols] = features[categorical_cols].fillna(features_cat_mode)
    else:
        features_cat_mode = None

    # Outliers verwijderen of cappen op zscore 3 / -3?
    numeric_columns = features.select_dtypes(include=['number'])
    z_scores = numeric_columns.apply(zscore)
    threshold = 3
        # Verwijderen
    # Outliers = (z_scores > threshold) | (z_scores < -threshold)
    # Features = features[~outliers.any(axis=1)]
        # Cappen
    z_scores = z_scores.clip(lower=-threshold, upper=threshold)
    train_outlier_mean = features[numeric_columns.columns].mean()
    train_outlier_std = features[numeric_columns.columns].std()
    features[numeric_columns.columns] = z_scores * train_outlier_std + train_outlier_mean
   
    # Voeg de target kolom weer toe
    df = pd.concat([features, target], axis=1)
    # Duplicates
    df.drop_duplicates(inplace=True)
  
    # Balancing or oversampling? SMOTE?
    #target = df[target_var]
    #features = df.drop(columns=[target_var])
    #smote = SMOTE(sampling_strategy='auto', random_state=42)
    #features_resampled, target_resampled = smote.fit_resample(features, target)
    #df = pd.concat([features_resampled, target_resampled], axis=1)
    
    # Standardization
    target = df[target_var]
    features = df.drop(columns=[target_var])
    features_num = features.select_dtypes(include=[np.number])
    features_cat = features.select_dtypes(exclude=np.number)
    features_num_mean2 = features_num.mean()
    features_num_std_dev = features_num.std()
    features_num_scaled = (features_num - features_num_mean2) / features_num_std_dev
    features = pd.concat([features_num_scaled, features_cat], axis=1)
    df = pd.concat([features, target], axis=1)

    # Dummification / WoE encoding
    target = df[target_var]
    features = df.drop(columns=[target_var])
    df = pd.concat([features, target], axis=1)

    # Multicollinearity
    target = df[target_var]
    features = df.drop(columns=[target_var])
    def calculate_vif(df):
        vif = pd.DataFrame()
        vif["variables"] = df.columns
        vif["VIF"] = [variance_inflation_factor(df.values, i) for i in range(df.shape[1])]
        return vif
    
        # Calculate VIF for numerical features
    vif = calculate_vif(features.select_dtypes(include=[np.number]))

        # Drop features with high VIF (e.g., VIF > 10)
    high_vif_features = vif[vif["VIF"] > 10]["variables"]
    features = features.drop(columns=high_vif_features)
    df = pd.concat([features, target], axis=1)
    #if high_vif_features.empty:
    #    print("No multicollinearity detected.")
    #else:
    #    print("High multicollinearity detected. Dropping features: ", high_vif_features)
    
    return df, features_num_mean, features_cat_mode, features_num_mean2, features_num_std_dev, train_outlier_mean, train_outlier_std

def test_preprocess_GMSC(df, train_num_mean, train_cat_mode, train_mean, train_stdev, train_outlier_mean, train_outlier_std): #preprocessing for give me some credit dataset
    df.sort_index(inplace=True)
    # Scheid de target kolom; zodat hier geen preprocessing op wordt toegepast
    target = df[target_var]
    features = df.drop(columns=[target_var])

    # Missing values
        # For numerical columns
    numeric_cols = features.select_dtypes(include=[np.number]).columns
    features[numeric_cols] = features[numeric_cols].fillna(train_num_mean) 
    
        # For categorical columns
    categorical_cols = features.select_dtypes(exclude=[np.number]).columns
    if not categorical_cols.empty:
        if train_cat_mode is not None:
            features[categorical_cols] = features[categorical_cols].fillna(train_cat_mode)
        else:
            features = features.dropna(subset=categorical_cols)    
    df = pd.concat([features, target], axis=1)
    

    # Outliers verwijderen of cappen op zscore 3 / -3?
    numeric_columns = features.select_dtypes(include=['number'])
    z_scores = numeric_columns.apply(zscore)
    threshold = 3
        # Verwijderen
    # Outliers = (z_scores > threshold) | (z_scores < -threshold)
    # Features = features[~outliers.any(axis=1)]
        # Cappen
    z_scores = z_scores.clip(lower=-threshold, upper=threshold)
    features[numeric_columns.columns] = z_scores * train_outlier_std + train_outlier_mean
   
    # Voeg de target kolom weer toe
    df = pd.concat([features, target], axis=1)
    
    # Standardization
    target = df[target_var]
    features = df.drop(columns=[target_var])
    features_num = features.select_dtypes(include=[np.number])
    features_cat = features.select_dtypes(exclude=np.number)
    features_num_scaled = (features_num - train_mean) / train_stdev
    features = pd.concat([features_num_scaled, features_cat], axis=1)
    df = pd.concat([features, target], axis=1)

    # Dummification
    target = df[target_var]
    features = df.drop(columns=[target_var])
    features = pd.get_dummies(features, drop_first=True)
    df = pd.concat([features, target], axis=1)

    return df