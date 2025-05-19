import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeRegressor, plot_tree
import matplotlib.pyplot as plt
from pd_calib import HD_calib, pd_calibration, df_prediction
from EL_UL_K_Calc import EL_Calc, UL_Calc


def f_regression_tree(test_pred, df_pred, target_var, max_leaf_nodes, calib_method):
    # Train regressieboom en ontvang input X + PD-range annotaties
    reg, pd_range_per_leaf, X = train_regression_tree(test_pred, max_leaf_nodes)

    # Visualisatie
    # plot_regression_tree(reg, pd_range_per_leaf, X)

    # Pas predictie toe op test-pred en volledige dataset
    rating_df = predict_with_regression_tree(test_pred.copy(), reg)
    df_new = predict_with_regression_tree(df_pred.copy(), reg) # hier geen assign_thresholds want getrainde regression tree van test_pred wordt meegegeven (reg)

    # Analyse
    rating_counts = rating_df['Rating'].value_counts().sort_index()
    #print(rating_counts)

    # PD Calibration: 
    calibrated_pd = HD_calib(df_new, rating_df, target_var)
    calibrated_pd_method = calib_method(df_new, rating_df, target_var)
    calibrated_pd['method Cal_PD'] = calibrated_pd_method['method Cal_PD']
    # Print the average Cal_PD per rating
    avg_cal_pd_per_rating = calibrated_pd.groupby('Rating')['HD_Cal_PD'].mean()
    avg_cal_pd_per_rating_method = calibrated_pd_method.groupby('Rating')['method Cal_PD'].mean()
    #print(avg_cal_pd_per_rating)
    # Map rating counts to avg_cal_pd_per_rating
    avg_pd_rating = pd_calibration(calibrated_pd, 'Rating', 'Prediction')
    avg_pd_rating_df = pd_calibration(df_new, 'Rating', 'Prediction')
    df_rating_counts = df_new['Rating'].value_counts()
    df_rating_counts = df_rating_counts.sort_index()
    df_default_counts = df_new.groupby('Rating')[target_var].sum()
    rating_summary = pd.DataFrame({
        'test counts': rating_counts,
        'avg PD': avg_pd_rating,
        'df counts': df_rating_counts,
        'df avg PD': avg_pd_rating_df,
        'df defaults': df_default_counts,
        'HD_Cal_PD': avg_cal_pd_per_rating,
        'method Cal_PD': avg_cal_pd_per_rating_method
    }).reset_index()

    # Bereken EL & UL
    calibrated_pd_copy = calibrated_pd.copy()
    calibrated_pd_EL, total_EL = EL_Calc(calibrated_pd)
    calibrated_pd_UL, total_UL = UL_Calc(calibrated_pd_copy)
    calibrated_pd = pd.concat([calibrated_pd_EL, calibrated_pd_UL["UL"]], axis=1)

    return calibrated_pd, total_EL, total_UL, rating_summary


def train_regression_tree(df, max_leaf_nodes):
    """
    Train regressieboom op PD <= 1.0
    Geeft ook X terug voor plotting.
    """

    X = df[['Prediction']].values
    y = df['Prediction'].values

    reg = DecisionTreeRegressor(random_state=42, max_leaf_nodes=max_leaf_nodes)
    reg.fit(X, y)

    # Bepaal leaf nodes en de PD-range per leaf
    leaf_ids = reg.apply(X)
    pd_ranges = pd.DataFrame({'Leaf': leaf_ids, 'PD': y})
    pd_range_per_leaf = pd_ranges.groupby('Leaf')['PD'].agg(['min', 'max']).reset_index()

    return reg, pd_range_per_leaf, X


def predict_with_regression_tree(df, reg):
    """
    Voorspel met regressieboom + wijs ratings toe op basis van voorspelde groeps-PD's.
    Geen aparte defaultklasse — alles gaat door dezelfde boom.
    """
    df = df.copy()

    # Voorspel voor alle rijen
    X_all = df[['Prediction']]
    df['Group_PD'] = reg.predict(X_all)

    # Rangschik unieke predicties tot discrete ratings
    unique_pds = sorted(df['Group_PD'].unique())
    mapping = {pd: i for i, pd in enumerate(unique_pds)}
    df['Rating'] = df['Group_PD'].map(mapping)
    df['Rating'] = df['Rating'].astype(int)

    return df



def plot_regression_tree(regressor_model, pd_range_per_leaf, X):
    """
    Plot de regressieboomstructuur + toon PD-range bij leaves.
    """
    fig, ax = plt.subplots(figsize=(14, 6))
    plot_tree(
        regressor_model,
        feature_names=['PD'],
        filled=True,
        rounded=True,
        precision=5,
        ax=ax
    )

    # Bepaal leaf nodes waar observaties naartoe gaan
    sample_leaves = regressor_model.apply(X)
    leaf_nodes = np.unique(sample_leaves)

    #print("\n📌 Leaf PD-ranges:")
    for leaf in leaf_nodes:
        pd_range = pd_range_per_leaf[pd_range_per_leaf['Leaf'] == leaf]
        if not pd_range.empty:
            min_pd = pd_range['min'].values[0]
            max_pd = pd_range['max'].values[0]
            #print(f"Leaf {leaf}: {min_pd:.4f} – {max_pd:.4f}")

    plt.title("Regressieboom voor PD-clustering met leaf PD-ranges")
    plt.tight_layout()
    plt.show()
