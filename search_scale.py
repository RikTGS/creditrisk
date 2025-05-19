import numpy as np
from logreg import evaluate_model   
from gmsc_preprocessing import test_preprocess_GMSC
from EL_UL_K_Calc import EL_Calc, UL_Calc
import pandas as pd
from threshold_assignment import assign_thresholds
from pd_calib import HD_calib, pd_calibration
import matplotlib.pyplot as plt

def f_search_scale(test_pred, df_pred, target_var, number_of_ratings, calib_method):
    # Apply rating binning to the predictions
    thresholds = new_search_thresholds(test_pred, df_pred, target_var, calib_method)
    pred_col = 'Prediction'
    rating_df = assign_thresholds(test_pred, pred_col, thresholds)
    df_new = assign_thresholds(df_pred, pred_col, thresholds)
    # Print the number of rows that each rating contains
    rating_counts = rating_df['Rating'].value_counts()
    rating_counts = rating_counts.sort_index()
    #print(rating_counts)
    # PD Calibration: taking the avg of the predictions per rating
    #calibrated_pd = PD_cal(df, test_preprocess, logreg_model, rating_df, train_num_mean, train_cat_mode, train_mean, train_stdev, target_var, method, thresholds)
    calibrated_pd = HD_calib(df_new, rating_df, target_var)
    calibrated_pd_method = calib_method(df_new, rating_df, target_var)
    calibrated_pd['method Cal_PD'] = calibrated_pd_method['method Cal_PD']
    # Print the average Cal_PD per rating
    avg_cal_pd_per_rating = calibrated_pd.groupby('Rating')['HD_Cal_PD'].mean()
    avg_cal_pd_per_rating_method = calibrated_pd_method.groupby('Rating')['method Cal_PD'].mean()
    #print(avg_cal_pd_per_rating)
    # Map rating counts to avg_cal_pd_per_rating
    avg_pd_rating = pd_calibration(calibrated_pd, 'Rating', 'Prediction')
    df_rating_counts = df_new['Rating'].value_counts()
    df_rating_counts = df_rating_counts.sort_index()
    df_default_counts = df_new.groupby('Rating')[target_var].sum()
    rating_summary = pd.DataFrame({
        'test counts': rating_counts,
        'avg PD': avg_pd_rating,
        'df counts': df_rating_counts,
        'df defaults': df_default_counts,
        'HD_Cal_PD': avg_cal_pd_per_rating,
        'method Cal_PD': avg_cal_pd_per_rating_method
    }).reset_index()

    # Calc EL and UL
    calibrated_pd_copy = calibrated_pd.copy()
    calibrated_pd_EL, total_EL = EL_Calc(calibrated_pd)
    calibrated_pd_UL, total_UL = UL_Calc(calibrated_pd_copy)
    calibrated_pd = pd.concat([calibrated_pd_EL, calibrated_pd_UL["UL"]], axis=1)
    return calibrated_pd, total_EL, total_UL, rating_summary


def calculate_PD(test_pred, thresholds, thresh, df_pred, target_var, calib_method):
    # Apply rating binning to the predictions
    rating_df = assign_thresholds(test_pred, 'Prediction', thresholds)
    df_new = assign_thresholds(df_pred, 'Prediction', thresholds)
    #print(rating_df.head())
    # Print the number of rows that each rating contains
    rating_counts = rating_df['Rating'].value_counts()
    # print(rating_counts)
    # PD Calibration: taking the avg of the predictions per rating
    calibrated_pd = calib_method(df_new, rating_df, target_var)

    # Stap 1: extract unieke (bestaande) rating-Cal_PD paren
    unique_ratings = calibrated_pd.drop_duplicates(subset='Rating')[['Rating', 'method Cal_PD']]

    # Stap 2: zet als index en converteer naar dict
    rating_to_calpd = unique_ratings.set_index('Rating')['method Cal_PD'].to_dict()

    # Stap 3: bouw lijst van lengte 7, met fallback 0.0 voor ontbrekende ratings
    unique_ratings_list = [rating_to_calpd.get(i, 0.0) for i in range(7)]

    # print("Cal_PD:", unique_ratings_list)

    return unique_ratings_list[thresh]

def new_search_thresholds(test_pred, df_pred, target_var, calib_method):
    
    thresholds = [0.0174, 0.0205, 0.048, 0.0767, 0.1126, 0.1375]
    num_ratings = len(thresholds)
    agency_PDs = [0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.4]
    max_diff = [0.0001, 0.0005, 0.0001, 0.0005, 0.0005, 0.0005, 0.0005]

    try:
        for thresh in range(num_ratings, 0, -1):
            not_close_enough = True
            PD_agency = agency_PDs[thresh]
            step_size = 0.0001  # Beginstapgrootte
            prev_direction = None
            iteration_count = 0
            max_iterations = 100000  # Limiet om oneindige loops te voorkomen

            threshold_history = []
            PD_history = []

            print(f"\n=== Start threshold aanpassing voor rating {thresh} ===")

            oscillation_count = 0
            min_step_size = 1e-6
            decay_factor = 0.9
            delta_history = []

            while not_close_enough and iteration_count < max_iterations:
                iteration_count += 1
                print(f"\nIteratie {iteration_count}:")
                print("Thresholds:", thresholds)

                current_PD = calculate_PD(test_pred, thresholds, thresh, df_pred, target_var, calib_method)

                print(f"Huidige threshold: {thresholds[thresh-1]}")
                print(f"Huidige PD: {current_PD}")
                print(f"Agency PD: {PD_agency}")

                difference_PD = PD_agency - current_PD
                print(f"Verschil PD: {difference_PD}")
                print(f"Max verschil toegestaan: {max_diff[thresh]}")

                threshold_history.append(thresholds[thresh-1])
                PD_history.append(current_PD)
                delta_history.append(abs(difference_PD))
                if len(delta_history) > 1000:
                    delta_history.pop(0)

                # Check convergence
                if abs(difference_PD) < max_diff[thresh]:
                    not_close_enough = False
                    print("✅ PD is binnen de limiet, threshold wordt behouden.")
                    break

                # Check lack of improvement
                if len(delta_history) == 1000 and max(delta_history) - min(delta_history) < 0.00001:
                    print("📉 Geen significante verandering in PD. Stoppen.")
                    break

                # Determine direction
                direction = 1 if difference_PD > 0 else -1

                # Oscillation check
                if prev_direction is not None and prev_direction != direction:
                    oscillation_count += 1
                    step_size *= decay_factor ** oscillation_count
                    print(f"🔄 Richting veranderd! Nieuwe stapgrootte: {step_size:.8f}")

                if step_size < min_step_size:
                    print("🚨 Stapgrootte te klein. Stoppen om oneindige loop te voorkomen.")
                    break

                thresholds[thresh-1] += direction * step_size
                prev_direction = direction

                if direction == -1:
                    print("Threshold verlaagd.")
                    None
                else:
                    print("Threshold verhoogd.")
                    None
                
                # Pause the code for 20 seconds
                # time.sleep(20)

            if iteration_count >= max_iterations:
                print(f"Max iteraties bereikt bij threshold {thresh-1}, mogelijke oscillatie.")
                None
            """
            plt.figure(figsize=(8, 5))
            plt.plot(threshold_history, PD_history, marker='o', linestyle='-', color='b', label="Current PD")
            plt.axhline(y=PD_agency, color='r', linestyle='--', label="Agency PD")
            plt.xlabel("Threshold")
            plt.ylabel("Current PD")
            plt.title(f"Convergentie van PD bij threshold {thresh-1}")
            plt.legend()
            plt.grid(True)
            plt.show()
            """

    except KeyboardInterrupt:
        print(f"\n🚨 Script onderbroken! We plotten de laatste verzamelde data voor threshold {thresh-1}.")

        # Controleer of er al data is verzameld voordat we de grafiek plotten
        if threshold_history and PD_history:
            plt.figure(figsize=(8, 5))
            plt.plot(threshold_history, PD_history, marker='o', linestyle='-', color='b', label="Current PD")
            plt.axhline(y=PD_agency, color='r', linestyle='--', label="Agency PD")
            plt.xlabel("Threshold")
            plt.ylabel("Current PD")
            plt.title(f"Laatste status voor threshold {thresh-1} (voortijdig gestopt)")
            plt.legend()
            plt.grid(True)
            plt.show()
        else:
            print("⚠️ Geen data verzameld om te plotten!")

    return thresholds