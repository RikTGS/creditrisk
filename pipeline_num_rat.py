import pandas as pd
import numpy as np
import os
import winsound
from scales import f_ERR
from clustering import f_clustering
from regression_tree import f_regression_tree
from search_scale import f_search_scale
from logreg import logreg_model  
from gmsc_preprocessing import train_preprocess_GMSC, test_preprocess_GMSC
from binning_techn import f_binning_methods, bin_by_equal_count, bin_by_equal_width, bin_by_equal_pd
from objective_funct import f_objective_funct_2
from pd_calib import HD_calib, TTC_calib, QMM_calibration, SLR_calib, IR_calib


# GIVE ME SOME CREDIT DATASET
# inladen dataset + target_var definieren
df = pd.read_csv('C:/Users/rikte/VS Code Python Projects/thesis_riktgs/data/givemesomecredit dataset/gmsc_dataset.csv')
# Insight: Sommige variabelen zoals RevolvingUtilizationOfUnsecuredLines en DebtRatio kunnen extreem hoge waarden bevatten.
df_describe = df.describe()
# df_describe.to_csv('C:/Users/rikte/VS Code Python Projects/thesis_riktgs/data/givemesomecredit dataset/gmsc_dataset_describe.csv', sep=';', decimal=',', index=False)
target_var = 'SeriousDlqin2yrs'
# trainen logreg + predictions uithalen
test_pred, df_pred, train2_pred, test = logreg_model(df, target_var, train_preprocess_GMSC, test_preprocess_GMSC)
df_name = "gmsc"
# Summary statistics for test_pred
summary_df = pd.DataFrame({
    "Mean": test_pred.mean(),
    "Standard Deviation": test_pred.std(),
    "Minimum": test_pred.min(),
    "Maximum": test_pred.max(),
    "25th Percentile": test_pred.quantile(0.25),
    "Median (50th Percentile)": test_pred.median(),
    "75th Percentile": test_pred.quantile(0.75)
})
summary_df = summary_df.T
x_min = test_pred['Prediction'].quantile(0.05)   # bv. 5e percentiel
x_max = test_pred['Prediction'].quantile(0.95)   # bv. 95e percentiel

number_ratings = [2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14]
# number_ratings = [7]
for num_ratings in number_ratings:
    print(num_ratings)

    # pipeline
    if num_ratings == 7:
        HDcalib_df_ERR, HDcalib_TEL_ERR, HDcalib_TUL_ERR, HDcalib_summary_ERR, HDcalib_thresholds_ERR = f_ERR(test_pred, train2_pred, target_var, num_ratings, x_min, x_max, HD_calib)
        HDcalib_df_clustering, HDcalib_TEL_clustering, HDcalib_TUL_clustering, HDcalib_summary_clustering = f_clustering(test_pred, train2_pred, target_var, num_ratings, HD_calib)
        HDcalib_df_RT, HDcalib_TEL_RT, HDcalib_TUL_RT, HDcalib_summary_RT = f_regression_tree(test_pred, train2_pred, target_var, num_ratings, HD_calib)
        HDcalib_df_SS, HDcalib_TEL_SS, HDcalib_TUL_SS, HDcalib_summary_SS = f_search_scale(test_pred, train2_pred, target_var, num_ratings, HD_calib)
        HDcalib_df_equalcount, HDcalib_TEL_equalcount, HDcalib_TUL_equalcount, HDcalib_summary_equalcount = f_binning_methods(test_pred, train2_pred, target_var, num_ratings, bin_by_equal_count, HD_calib)
        HDcalib_df_equalwidth, HDcalib_TEL_equalwidth, HDcalib_TUL_equalwidth, HDcalib_summary_equalwidth = f_binning_methods(test_pred, train2_pred, target_var, num_ratings, bin_by_equal_width, HD_calib)
        HDcalib_df_equalpd, HDcalib_TEL_equalpd, HDcalib_TUL_equalpd, HDcalib_summary_equalpd = f_binning_methods(test_pred, train2_pred, target_var, num_ratings, bin_by_equal_pd, HD_calib)
        HDcalib_df_objective, HDcalib_TEL_objective, HDcalib_TUL_objective, HDcalib_summary_objective = f_objective_funct_2(test_pred, train2_pred, test, target_var, num_ratings, HD_calib)
        
        TTCcalib_df_ERR, TTCcalib_TEL_ERR, TTCcalib_TUL_ERR, TTCcalib_summary_ERR, thresholds = f_ERR(test_pred, train2_pred, target_var, num_ratings, x_min, x_max, TTC_calib)
        TTCcalib_df_clustering, TTCcalib_TEL_clustering, TTCcalib_TUL_clustering, TTCcalib_summary_clustering = f_clustering(test_pred, train2_pred, target_var, num_ratings, TTC_calib)
        TTCcalib_df_RT, TTCcalib_TEL_RT, TTCcalib_TUL_RT, TTCcalib_summary_RT = f_regression_tree(test_pred, train2_pred, target_var, num_ratings, TTC_calib)
        # TTCcalib_df_SS, TTCcalib_TEL_SS, TTCcalib_TUL_SS, TTCcalib_summary_SS = f_search_scale(test_pred, train2_pred, target_var, num_ratings, TTC_calib)
        TTCcalib_df_equalcount, TTCcalib_TEL_equalcount, TTCcalib_TUL_equalcount, TTCcalib_summary_equalcount = f_binning_methods(test_pred, train2_pred, target_var, num_ratings, bin_by_equal_count, TTC_calib)
        TTCcalib_df_equalwidth, TTCcalib_TEL_equalwidth, TTCcalib_TUL_equalwidth, TTCcalib_summary_equalwidth = f_binning_methods(test_pred, train2_pred, target_var, num_ratings, bin_by_equal_width, TTC_calib)
        TTCcalib_df_equalpd, TTCcalib_TEL_equalpd, TTCcalib_TUL_equalpd, TTCcalib_summary_equalpd = f_binning_methods(test_pred, train2_pred, target_var, num_ratings, bin_by_equal_pd, TTC_calib)
        TTCcalib_df_objective, TTCcalib_TEL_objective, TTCcalib_TUL_objective, TTCcalib_summary_objective = f_objective_funct_2(test_pred, train2_pred, test, target_var, num_ratings, TTC_calib)
    
        QMMcalib_df_ERR, QMMcalib_TEL_ERR, QMMcalib_TUL_ERR, QMMcalib_summary_ERR, thresholds = f_ERR(test_pred, train2_pred, target_var, num_ratings, x_min, x_max, QMM_calibration)
        QMMcalib_df_clustering, QMMcalib_TEL_clustering, QMMcalib_TUL_clustering, QMMcalib_summary_clustering = f_clustering(test_pred, train2_pred, target_var, num_ratings, QMM_calibration)
        QMMcalib_df_RT, QMMcalib_TEL_RT, QMMcalib_TUL_RT, QMMcalib_summary_RT = f_regression_tree(test_pred, train2_pred, target_var, num_ratings, QMM_calibration)
        #QMMcalib_df_SS, QMMcalib_TEL_SS, QMMcalib_TUL_SS, QMMcalib_summary_SS = f_search_scale(test_pred, train2_pred, target_var, num_ratings, QMM_calibration)
        QMMcalib_df_equalcount, QMMcalib_TEL_equalcount, QMMcalib_TUL_equalcount, QMMcalib_summary_equalcount = f_binning_methods(test_pred, train2_pred, target_var, num_ratings, bin_by_equal_count, QMM_calibration)
        QMMcalib_df_equalwidth, QMMcalib_TEL_equalwidth, QMMcalib_TUL_equalwidth, QMMcalib_summary_equalwidth = f_binning_methods(test_pred, train2_pred, target_var, num_ratings, bin_by_equal_width, QMM_calibration)
        QMMcalib_df_equalpd, QMMcalib_TEL_equalpd, QMMcalib_TUL_equalpd, QMMcalib_summary_equalpd = f_binning_methods(test_pred, train2_pred, target_var, num_ratings, bin_by_equal_pd, QMM_calibration)
        QMMcalib_df_objective, QMMcalib_TEL_objective, QMMcalib_TUL_objective, QMMcalib_summary_objective = f_objective_funct_2(test_pred, train2_pred, test, target_var, num_ratings, QMM_calibration)
        
        SLRcalib_df_ERR, SLRcalib_TEL_ERR, SLRcalib_TUL_ERR, SLRcalib_summary_ERR, thresholds = f_ERR(test_pred, train2_pred, target_var, num_ratings, x_min, x_max, SLR_calib)
        SLRcalib_df_clustering, SLRcalib_TEL_clustering, SLRcalib_TUL_clustering, SLRcalib_summary_clustering = f_clustering(test_pred, train2_pred, target_var, num_ratings, SLR_calib)
        SLRcalib_df_RT, SLRcalib_TEL_RT, SLRcalib_TUL_RT, SLRcalib_summary_RT = f_regression_tree(test_pred, train2_pred, target_var, num_ratings, SLR_calib)
        # SLRcalib_df_SS, SLRcalib_TEL_SS, SLRcalib_TUL_SS, SLRcalib_summary_SS = f_search_scale(test_pred, train2_pred, target_var, num_ratings, SLR_calib)
        SLRcalib_df_equalcount, SLRcalib_TEL_equalcount, SLRcalib_TUL_equalcount, SLRcalib_summary_equalcount = f_binning_methods(test_pred, train2_pred, target_var, num_ratings, bin_by_equal_count, SLR_calib)
        SLRcalib_df_equalwidth, SLRcalib_TEL_equalwidth, SLRcalib_TUL_equalwidth, SLRcalib_summary_equalwidth = f_binning_methods(test_pred, train2_pred, target_var, num_ratings, bin_by_equal_width, SLR_calib)
        SLRcalib_df_equalpd, SLRcalib_TEL_equalpd, SLRcalib_TUL_equalpd, SLRcalib_summary_equalpd = f_binning_methods(test_pred, train2_pred, target_var, num_ratings, bin_by_equal_pd, SLR_calib)
        SLRcalib_df_objective, SLRcalib_TEL_objective, SLRcalib_TUL_objective, SLRcalib_summary_objective = f_objective_funct_2(test_pred, train2_pred, test, target_var, num_ratings, SLR_calib)
        
        IRcalib_df_ERR, IRcalib_TEL_ERR, IRcalib_TUL_ERR, IRcalib_summary_ERR, thresholds = f_ERR(test_pred, train2_pred, target_var, num_ratings, x_min, x_max, IR_calib)
        IRcalib_df_clustering, IRcalib_TEL_clustering, IRcalib_TUL_clustering, IRcalib_summary_clustering = f_clustering(test_pred, train2_pred, target_var, num_ratings, IR_calib)
        IRcalib_df_RT, IRcalib_TEL_RT, IRcalib_TUL_RT, IRcalib_summary_RT = f_regression_tree(test_pred, train2_pred, target_var, num_ratings, IR_calib)
        #IRcalib_df_SS, IRcalib_TEL_SS, IRcalib_TUL_SS, IRcalib_summary_SS = f_search_scale(test_pred, train2_pred, target_var, num_ratings, IR_calib)
        IRcalib_df_equalcount, IRcalib_TEL_equalcount, IRcalib_TUL_equalcount, IRcalib_summary_equalcount = f_binning_methods(test_pred, train2_pred, target_var, num_ratings, bin_by_equal_count, IR_calib)
        IRcalib_df_equalwidth, IRcalib_TEL_equalwidth, IRcalib_TUL_equalwidth, IRcalib_summary_equalwidth = f_binning_methods(test_pred, train2_pred, target_var, num_ratings, bin_by_equal_width, IR_calib)
        IRcalib_df_equalpd, IRcalib_TEL_equalpd, IRcalib_TUL_equalpd, IRcalib_summary_equalpd = f_binning_methods(test_pred, train2_pred, target_var, num_ratings, bin_by_equal_pd, IR_calib)
        IRcalib_df_objective, IRcalib_TEL_objective, IRcalib_TUL_objective, IRcalib_summary_objective = f_objective_funct_2(test_pred, train2_pred, test, target_var, num_ratings, IR_calib)
    
    else:
        HDcalib_df_ERR, HDcalib_TEL_ERR, HDcalib_TUL_ERR, HDcalib_summary_ERR, HDcalib_thresholds_ERR = f_ERR(test_pred, train2_pred, target_var, num_ratings, x_min, x_max, HD_calib)
        HDcalib_df_clustering, HDcalib_TEL_clustering, HDcalib_TUL_clustering, HDcalib_summary_clustering = f_clustering(test_pred, train2_pred, target_var, num_ratings, HD_calib)
        HDcalib_df_RT, HDcalib_TEL_RT, HDcalib_TUL_RT, HDcalib_summary_RT = f_regression_tree(test_pred, train2_pred, target_var, num_ratings, HD_calib)
        # HDcalib_df_SS, HDcalib_TEL_SS, HDcalib_TUL_SS, HDcalib_summary_SS = None
        HDcalib_df_equalcount, HDcalib_TEL_equalcount, HDcalib_TUL_equalcount, HDcalib_summary_equalcount = f_binning_methods(test_pred, train2_pred, target_var, num_ratings, bin_by_equal_count, HD_calib)
        HDcalib_df_equalwidth, HDcalib_TEL_equalwidth, HDcalib_TUL_equalwidth, HDcalib_summary_equalwidth = f_binning_methods(test_pred, train2_pred, target_var, num_ratings, bin_by_equal_width, HD_calib)
        HDcalib_df_equalpd, HDcalib_TEL_equalpd, HDcalib_TUL_equalpd, HDcalib_summary_equalpd = f_binning_methods(test_pred, train2_pred, target_var, num_ratings, bin_by_equal_pd, HD_calib)
        HDcalib_df_objective, HDcalib_TEL_objective, HDcalib_TUL_objective, HDcalib_summary_objective = f_objective_funct_2(test_pred, train2_pred, test, target_var, num_ratings, HD_calib)
        
        TTCcalib_df_ERR, TTCcalib_TEL_ERR, TTCcalib_TUL_ERR, TTCcalib_summary_ERR, thresholds = f_ERR(test_pred, train2_pred, target_var, num_ratings, x_min, x_max, TTC_calib)
        TTCcalib_df_clustering, TTCcalib_TEL_clustering, TTCcalib_TUL_clustering, TTCcalib_summary_clustering = f_clustering(test_pred, train2_pred, target_var, num_ratings, TTC_calib)
        TTCcalib_df_RT, TTCcalib_TEL_RT, TTCcalib_TUL_RT, TTCcalib_summary_RT = f_regression_tree(test_pred, train2_pred, target_var, num_ratings, TTC_calib)
        # TTCcalib_df_SS = TTCcalib_TEL_SS = TTCcalib_TUL_SS = TTCcalib_summary_SS = None
        TTCcalib_df_equalcount, TTCcalib_TEL_equalcount, TTCcalib_TUL_equalcount, TTCcalib_summary_equalcount = f_binning_methods(test_pred, train2_pred, target_var, num_ratings, bin_by_equal_count, TTC_calib)
        TTCcalib_df_equalwidth, TTCcalib_TEL_equalwidth, TTCcalib_TUL_equalwidth, TTCcalib_summary_equalwidth = f_binning_methods(test_pred, train2_pred, target_var, num_ratings, bin_by_equal_width, TTC_calib)
        TTCcalib_df_equalpd, TTCcalib_TEL_equalpd, TTCcalib_TUL_equalpd, TTCcalib_summary_equalpd = f_binning_methods(test_pred, train2_pred, target_var, num_ratings, bin_by_equal_pd, TTC_calib)
        TTCcalib_df_objective, TTCcalib_TEL_objective, TTCcalib_TUL_objective, TTCcalib_summary_objective = f_objective_funct_2(test_pred, train2_pred, test, target_var, num_ratings, TTC_calib)
    
        QMMcalib_df_ERR, QMMcalib_TEL_ERR, QMMcalib_TUL_ERR, QMMcalib_summary_ERR, thresholds = f_ERR(test_pred, train2_pred, target_var, num_ratings, x_min, x_max, QMM_calibration)
        QMMcalib_df_clustering, QMMcalib_TEL_clustering, QMMcalib_TUL_clustering, QMMcalib_summary_clustering = f_clustering(test_pred, train2_pred, target_var, num_ratings, QMM_calibration)
        QMMcalib_df_RT, QMMcalib_TEL_RT, QMMcalib_TUL_RT, QMMcalib_summary_RT = f_regression_tree(test_pred, train2_pred, target_var, num_ratings, QMM_calibration)
        # QMMcalib_df_SS = QMMcalib_TEL_SS = QMMcalib_TUL_SS = QMMcalib_summary_SS = None
        QMMcalib_df_equalcount, QMMcalib_TEL_equalcount, QMMcalib_TUL_equalcount, QMMcalib_summary_equalcount = f_binning_methods(test_pred, train2_pred, target_var, num_ratings, bin_by_equal_count, QMM_calibration)
        QMMcalib_df_equalwidth, QMMcalib_TEL_equalwidth, QMMcalib_TUL_equalwidth, QMMcalib_summary_equalwidth = f_binning_methods(test_pred, train2_pred, target_var, num_ratings, bin_by_equal_width, QMM_calibration)
        QMMcalib_df_equalpd, QMMcalib_TEL_equalpd, QMMcalib_TUL_equalpd, QMMcalib_summary_equalpd = f_binning_methods(test_pred, train2_pred, target_var, num_ratings, bin_by_equal_pd, QMM_calibration)
        QMMcalib_df_objective, QMMcalib_TEL_objective, QMMcalib_TUL_objective, QMMcalib_summary_objective = f_objective_funct_2(test_pred, train2_pred, test, target_var, num_ratings, QMM_calibration)
        
        SLRcalib_df_ERR, SLRcalib_TEL_ERR, SLRcalib_TUL_ERR, SLRcalib_summary_ERR, thresholds = f_ERR(test_pred, train2_pred, target_var, num_ratings, x_min, x_max, SLR_calib)
        SLRcalib_df_clustering, SLRcalib_TEL_clustering, SLRcalib_TUL_clustering, SLRcalib_summary_clustering = f_clustering(test_pred, train2_pred, target_var, num_ratings, SLR_calib)
        SLRcalib_df_RT, SLRcalib_TEL_RT, SLRcalib_TUL_RT, SLRcalib_summary_RT = f_regression_tree(test_pred, train2_pred, target_var, num_ratings, SLR_calib)
        # SLRcalib_df_SS = SLRcalib_TEL_SS = SLRcalib_TUL_SS = SLRcalib_summary_SS = None
        SLRcalib_df_equalcount, SLRcalib_TEL_equalcount, SLRcalib_TUL_equalcount, SLRcalib_summary_equalcount = f_binning_methods(test_pred, train2_pred, target_var, num_ratings, bin_by_equal_count, SLR_calib)
        SLRcalib_df_equalwidth, SLRcalib_TEL_equalwidth, SLRcalib_TUL_equalwidth, SLRcalib_summary_equalwidth = f_binning_methods(test_pred, train2_pred, target_var, num_ratings, bin_by_equal_width, SLR_calib)
        SLRcalib_df_equalpd, SLRcalib_TEL_equalpd, SLRcalib_TUL_equalpd, SLRcalib_summary_equalpd = f_binning_methods(test_pred, train2_pred, target_var, num_ratings, bin_by_equal_pd, SLR_calib)
        SLRcalib_df_objective, SLRcalib_TEL_objective, SLRcalib_TUL_objective, SLRcalib_summary_objective = f_objective_funct_2(test_pred, train2_pred, test, target_var, num_ratings, SLR_calib)
        
        IRcalib_df_ERR, IRcalib_TEL_ERR, IRcalib_TUL_ERR, IRcalib_summary_ERR, thresholds = f_ERR(test_pred, train2_pred, target_var, num_ratings, x_min, x_max, IR_calib)
        IRcalib_df_clustering, IRcalib_TEL_clustering, IRcalib_TUL_clustering, IRcalib_summary_clustering = f_clustering(test_pred, train2_pred, target_var, num_ratings, IR_calib)
        IRcalib_df_RT, IRcalib_TEL_RT, IRcalib_TUL_RT, IRcalib_summary_RT = f_regression_tree(test_pred, train2_pred, target_var, num_ratings, IR_calib)
        # IRcalib_df_SS = IRcalib_TEL_SS = IRcalib_TUL_SS = IRcalib_summary_SS = None
        IRcalib_df_equalcount, IRcalib_TEL_equalcount, IRcalib_TUL_equalcount, IRcalib_summary_equalcount = f_binning_methods(test_pred, train2_pred, target_var, num_ratings, bin_by_equal_count, IR_calib)
        IRcalib_df_equalwidth, IRcalib_TEL_equalwidth, IRcalib_TUL_equalwidth, IRcalib_summary_equalwidth = f_binning_methods(test_pred, train2_pred, target_var, num_ratings, bin_by_equal_width, IR_calib)
        IRcalib_df_equalpd, IRcalib_TEL_equalpd, IRcalib_TUL_equalpd, IRcalib_summary_equalpd = f_binning_methods(test_pred, train2_pred, target_var, num_ratings, bin_by_equal_pd, IR_calib)
        IRcalib_df_objective, IRcalib_TEL_objective, IRcalib_TUL_objective, IRcalib_summary_objective = f_objective_funct_2(test_pred, train2_pred, test, target_var, num_ratings, IR_calib)

    import os
    import pandas as pd
    from joblib import dump

    output_dir = f"C:/Users/rikte/VS Code Python Projects/thesis_riktgs/output/{num_ratings}"
    os.makedirs(output_dir, exist_ok=True)

    def safe_save_csv(obj, name):
        try:
            path = os.path.join(output_dir, f"{name}.csv")
            
            if isinstance(obj, pd.DataFrame):
                obj.to_csv(path, index=False)
            elif isinstance(obj, np.ndarray):
                df = pd.DataFrame(obj)
                df.to_csv(path, index=False)
            else:
                raise TypeError(f"Unsupported type: {type(obj)}")
            
            print(f"✅ CSV saved: {path}")
        except Exception as e:
            print(f"⚠️ CSV error for {name}: {e}")

    def safe_save_txt(value, name):
        try:
            path = os.path.join(output_dir, f"{name}.txt")
            with open(path, "w") as f:
                f.write(str(value))
        except Exception as e:
            print(f"⚠️ TXT-fout bij {name}: {e}")

    # 1. DataFrames als CSV
    variables_copy = list(globals().items())  # veilige kopie van alle globale variabelen
    for var_name, obj in variables_copy:
        if isinstance(obj, pd.DataFrame) or isinstance(obj, np.ndarray):
            safe_save_csv(obj, f'{var_name}')

    # 2. TEL en TUL waarden als TXT
    for var_name, obj in variables_copy:
        if isinstance(obj, (float, int)):
            safe_save_txt(obj, f'{var_name}')

    # winsound.Beep(1000, 300)  # Beep sound to indicate completion

# winsound.Beep(500, 1500)  # Beep sound to indicate completion