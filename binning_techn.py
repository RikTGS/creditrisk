import pandas as pd
import numpy as np
from pd_calib import HD_calib, pd_calibration, df_prediction
from EL_UL_K_Calc import EL_Calc, UL_Calc
from threshold_assignment import assign_thresholds


def f_binning_methods(test_pred, df_pred, target_var, num_bins, binning_method, calib_method):
    rating_df, bins = binning_method(test_pred, num_bins)
    df_new = assign_thresholds(df_pred,'Prediction',bins)
    rating_counts = rating_df['Rating'].value_counts()
    rating_counts = rating_counts.sort_index()
    #print('Test df rating_counts: \n',rating_counts)
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

    # Calc EL and UL
    calibrated_pd_copy = calibrated_pd.copy()
    calibrated_pd_EL, total_EL = EL_Calc(calibrated_pd)
    calibrated_pd_UL, total_UL = UL_Calc(calibrated_pd_copy)
    calibrated_pd = pd.concat([calibrated_pd_EL, calibrated_pd_UL["UL"]], axis=1)
    
    # Plot the 'Prediction' column as a histogram
    #plt.hist(calibrated_pd['Prediction'], bins=50, edgecolor='k')
    #plt.xlabel('Prediction')
    #plt.ylabel('Frequency')
    #plt.title('Histogram of Predictions')
    #plt.show()
    # Plot the 'Prediction' column as a histogram for values between 0 and 0.2
    #plt.hist(calibrated_pd['Prediction'], bins=50, range=(0, 0.2), edgecolor='k')
    #plt.xlabel('Prediction')
    #plt.ylabel('Frequency')
    #plt.title('Histogram of Predictions (0 to 0.2)')
    #plt.show()

    return calibrated_pd, total_EL, total_UL, rating_summary

def bin_by_equal_width(df, num_bins):
    """
    Verdeel op basis van gelijke intervalgroottes (equal width).
    """
    col = 'Prediction'
    df = df.copy()
    df['Rating'], bins = pd.cut(df[col], bins=num_bins, labels=False, retbins=True)
    
    bins = bins[1:-1]  # Exclude the last bin edge
    bins = bins.astype(float)

    return df, bins

def bin_by_equal_count(df, num_bins):
    """
    Verdeel op basis van gelijke aantallen observaties per klasse (equal frequency).
    """
    col = 'Prediction'
    df = df.copy()
    df['Rating'], bins = pd.qcut(df[col], q=num_bins, labels=False, retbins=True, duplicates='drop')

    bins = bins[1:-1]
    bins = bins.astype(float)
    return df, bins

def bin_by_equal_pd(df, num_bins):
    """
    Verdeel op basis van gelijke som van PD's per klasse.
    Dit zorgt ervoor dat elke klasse ongeveer evenveel totale risico bevat.
    """
    col = 'Prediction'
    df = df.copy().sort_values(by=col).reset_index(drop=True)
    total_pd = df[col].sum()
    target_pd_per_bin = total_pd / num_bins

    cumulative_pd = df[col].cumsum()
    bin_edges = [0]
    for i in range(1, num_bins):
        idx = (cumulative_pd - i * target_pd_per_bin).abs().idxmin()
        bin_edges.append(idx)
    bin_edges.append(len(df))

    ratings = np.zeros(len(df), dtype=int)
    for i in range(num_bins):
        start, end = bin_edges[i], bin_edges[i + 1]
        ratings[start:end] = i
    df['Rating'] = ratings

    bins = [df[col].iloc[bin_edges[i]] for i in range(1, len(bin_edges) - 1)]
    bins = np.array(bins, dtype=float)
    return df, bins
