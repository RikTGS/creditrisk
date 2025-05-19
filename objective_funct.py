from pd_calib import HD_calib
from threshold_assignment import assign_thresholds
from EL_UL_K_Calc import EL_Calc, UL_Calc
import numpy as np
import pandas as pd


def f_objective_funct_2(test_pred, df_pred, test, target_var, num_ratings, calib_method):
    pred_col = 'Prediction'
    optimal_thresholds = min_MSE_stepwise(test_pred, df_pred, target_var, num_ratings, calib_method)
    # print("Optimal thresholds:", optimal_thresholds)
    rating_df = assign_thresholds(test_pred.copy(), pred_col, optimal_thresholds)
    df_new = assign_thresholds(df_pred.copy(), pred_col, optimal_thresholds)

    calibrated_pd = HD_calib(df_new, rating_df, target_var)
    calibrated_pd_method = calib_method(df_new, rating_df, target_var)
    calibrated_pd['method Cal_PD'] = calibrated_pd_method['method Cal_PD']
    # Samenvattende statistieken
    rating_counts = rating_df['Rating'].value_counts().sort_index()
    avg_pred_per_rating = rating_df.groupby("Rating")[pred_col].mean()
    avg_cal_pd = calibrated_pd.groupby('Rating')['HD_Cal_PD'].mean()
    avg_method_cal_pd = calibrated_pd_method.groupby('Rating')['method Cal_PD'].mean()
    df_rating_counts = df_new['Rating'].value_counts().sort_index()
    df_avg_pd = df_new.groupby('Rating')[pred_col].mean()
    df_defaults = df_new.groupby('Rating')[target_var].sum()

    rating_summary = pd.DataFrame({
        'test counts': rating_counts,
        'avg PD': avg_pred_per_rating,
        'df counts': df_rating_counts,
        'df avg PD': df_avg_pd,
        'df defaults': df_defaults,
        'HD_Cal_PD': avg_cal_pd,
        'method Cal_PD': avg_method_cal_pd
    }).reset_index()

    # EL en UL berekenen
    el_pd, total_EL = EL_Calc(calibrated_pd)
    ul_pd, total_UL = UL_Calc(calibrated_pd.copy())
    calibrated_pd = pd.concat([el_pd, ul_pd["UL"]], axis=1)

    return calibrated_pd, total_EL, total_UL, rating_summary

def min_MSE_stepwise(test_pred, df_pred, target_var, num_ratings, calib_method):
    # thresholds = [0.02, 0.0205, 0.048, 0.0767, 0.1126, 0.1375]
    # num_ratings = len(thresholds)
    thresholds = np.linspace(0.001, 0.2, num_ratings + 1)[1:-1]
    num_ratings = len(thresholds)
    
    max_iterations = 1000
    min_step_size = 1e-6
    decay_factor = 0.9
    initial_step = 0.001
    min_distance = 0.01  # minimale afstand tussen opeenvolgende thresholds

    def get_mse(thresh_list):
        try:
            rating_df = assign_thresholds(test_pred.copy(), 'Prediction', thresh_list)
            df_new = assign_thresholds(df_pred.copy(), 'Prediction', thresh_list)
            calibrated_df = calib_method(df_new, rating_df, target_var)

            preds = rating_df['Prediction'].values
            cal_pds = calibrated_df["method Cal_PD"].values

            if len(preds) != len(cal_pds):
                return 1e6

            mse = np.mean((preds - cal_pds) ** 2)
            return mse
        except Exception as e:
            print(f"⚠️ Fout in get_mse(): {e}")
            return 1e6

    print("\n🚀 Start threshold-optimalisatie")
    for idx in range(num_ratings - 1, -1, -1):
        print(f"\n🔧 Optimaliseer threshold {idx}")
        print(f"📌 Startwaarde: {thresholds[idx]:.6f}")
        step_size = initial_step
        prev_direction = None
        iteration = 0
        oscillation_count = 0
        best_mse = get_mse(thresholds)

        lower = thresholds[idx - 1] + min_distance if idx > 0 else 0.001
        upper = thresholds[idx + 1] - min_distance if idx < len(thresholds) - 1 else 0.999
        print(f"   ➤ Geldig bereik: {lower:.5f} – {upper:.5f}")
        print(f"   ➤ Start MSE: {best_mse:.8f}")

        while step_size >= min_step_size and iteration < max_iterations:
            iteration += 1

            thresholds_up = thresholds.copy()
            thresholds_up[idx] = min(thresholds_up[idx] + step_size, upper)
            thresholds_up = sorted(thresholds_up)
            mse_up = get_mse(thresholds_up)

            thresholds_down = thresholds.copy()
            thresholds_down[idx] = max(thresholds_down[idx] - step_size, lower)
            thresholds_down = sorted(thresholds_down)
            mse_down = get_mse(thresholds_down)

            direction = None
            if mse_up < best_mse:
                thresholds = thresholds_up
                best_mse = mse_up
                direction = "up"
                print(f"   ✅ Iteratie {iteration}: MSE verbeterd (up) → {best_mse:.8f}, stap {step_size:.6f}")
            elif mse_down < best_mse:
                thresholds = thresholds_down
                best_mse = mse_down
                direction = "down"
                print(f"   ✅ Iteratie {iteration}: MSE verbeterd (down) → {best_mse:.8f}, stap {step_size:.6f}")
            else:
                step_size *= decay_factor
                print(f"   ⏸ Geen verbetering → stap verkleind naar {step_size:.6f}")
                continue

            if prev_direction and direction != prev_direction:
                oscillation_count += 1
                step_size *= decay_factor ** oscillation_count
                print(f"   🔄 Richting veranderd → nieuwe stap: {step_size:.8f}")

            prev_direction = direction

        print(f"🔚 Klaar met threshold {idx}: eindwaarde = {thresholds[idx]:.6f}, eind MSE = {best_mse:.8f}")

    print("\n✅ Alle thresholds geoptimaliseerd.")
    print("🔢 Final thresholds:", [round(t, 6) for t in thresholds])
    return thresholds
