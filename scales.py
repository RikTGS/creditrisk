from pd_calib import HD_calib, df_prediction, pd_calibration, TTC_calib, QMM_calibration, SLR_calib, IR_calib
from own_scale import create_rating_scale
from EL_UL_K_Calc import EL_Calc, UL_Calc
from threshold_assignment import assign_thresholds
import pandas as pd

def f_ERR(test_pred, df_pred, target_var, number_of_ratings, x_min, x_max, calib_method):
    # Apply rating binning to the predictions
    pred_col = 'Prediction'
    thresholds = create_rating_scale(number_of_ratings, x_min, x_max)
    rating_df = assign_thresholds(test_pred, pred_col, thresholds)
    df_new = assign_thresholds(df_pred, pred_col, thresholds)
    rating_counts = rating_df['Rating'].value_counts()
    rating_counts = rating_counts.sort_index()
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

    return calibrated_pd, total_EL, total_UL, rating_summary, thresholds
