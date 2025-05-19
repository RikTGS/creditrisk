import pandas as pd
import os
import winsound
from scales import f_ERR
from clustering import f_clustering
from regression_tree import f_regression_tree
from search_scale import f_search_scale
from logreg import logreg_model, logreg_model_cv  
from gmsc_preprocessing import train_preprocess_GMSC, test_preprocess_GMSC
from binning_techn import f_binning_methods, bin_by_equal_count, bin_by_equal_width, bin_by_equal_pd
from objective_funct import f_objective_funct_2
from pd_calib import HD_calib, TTC_calib, QMM_calibration, SLR_calib, IR_calib
from sklearn.model_selection import KFold
from collections import defaultdict
from sklearn.metrics import brier_score_loss


# Load dataset
df = pd.read_csv('C:/Users/rikte/VS Code Python Projects/thesis_riktgs/data/givemesomecredit dataset/gmsc_dataset.csv')
target_var = 'SeriousDlqin2yrs'

# Define cross-validation and result container
kf = KFold(n_splits=10, shuffle=True, random_state=42)
fold_results = defaultdict(list)

# Logging helper
def log_fold_results(name, fold, TEL, TUL, brier_score):
    fold_results[name].append({
        "Fold": fold,
        "TEL": TEL,
        "TUL": TUL,
        "TEL (norm)": TEL / len(test_pred),
        "TUL (norm)": TUL / len(test_pred),
        "Brier": brier_score
    })

# Start CV loop
fold_num = 0
for train_index, test_index in kf.split(df):
    train_df = df.iloc[train_index].copy()
    test_df = df.iloc[test_index].copy()
    test_pred, df_pred, train2_pred, test = logreg_model_cv(train_df, test_df, target_var, train_preprocess_GMSC, test_preprocess_GMSC)    
    x_min = test_pred['Prediction'].quantile(0.05)
    x_max = test_pred['Prediction'].quantile(0.95)

    number_ratings = [7]
    for num_ratings in number_ratings:
        # Only 7-ratings pipeline for now
        # # ERR + HD
        HDcalib_df_ERR, HDcalib_TEL_ERR, HDcalib_TUL_ERR, _, _ = f_ERR(test_pred, train2_pred, target_var, num_ratings, x_min, x_max, HD_calib)
        brier = brier_score_loss(test[target_var], HDcalib_df_ERR['method Cal_PD'])
        log_fold_results("ERR_HD", fold_num, HDcalib_TEL_ERR, HDcalib_TUL_ERR, brier)

        # Clustering + HD
        HDcalib_df_clustering, HDcalib_TEL_clustering, HDcalib_TUL_clustering, _ = f_clustering(test_pred, train2_pred, target_var, num_ratings, HD_calib)
        brier = brier_score_loss(test[target_var], HDcalib_df_clustering['method Cal_PD'])
        log_fold_results("clustering_HD", fold_num, HDcalib_TEL_clustering, HDcalib_TUL_clustering, brier)

        # RT + HD
        HDcalib_df_RT, HDcalib_TEL_RT, HDcalib_TUL_RT, _ = f_regression_tree(test_pred, train2_pred, target_var, num_ratings, HD_calib)
        brier = brier_score_loss(test[target_var], HDcalib_df_RT['method Cal_PD'])
        log_fold_results("RT_HD", fold_num, HDcalib_TEL_RT, HDcalib_TUL_RT, brier)

        # SS + HD
        HDcalib_df_SS, HDcalib_TEL_SS, HDcalib_TUL_SS, _ = f_search_scale(test_pred, train2_pred, target_var, num_ratings, HD_calib)
        brier = brier_score_loss(test[target_var], HDcalib_df_SS['method Cal_PD'])
        log_fold_results("SS_HD", fold_num, HDcalib_TEL_SS, HDcalib_TUL_SS, brier)

        # Equal Count + HD
        HDcalib_df_equalcount, HDcalib_TEL_equalcount, HDcalib_TUL_equalcount, _ = f_binning_methods(test_pred, train2_pred, target_var, num_ratings, bin_by_equal_count, HD_calib)
        brier = brier_score_loss(test[target_var], HDcalib_df_equalcount['method Cal_PD'])
        log_fold_results("equalcount_HD", fold_num, HDcalib_TEL_equalcount, HDcalib_TUL_equalcount, brier)

        # Equal Width + HD
        HDcalib_df_equalwidth, HDcalib_TEL_equalwidth, HDcalib_TUL_equalwidth, _ = f_binning_methods(test_pred, train2_pred, target_var, num_ratings, bin_by_equal_width, HD_calib)
        brier = brier_score_loss(test[target_var], HDcalib_df_equalwidth['method Cal_PD'])
        log_fold_results("equalwidth_HD", fold_num, HDcalib_TEL_equalwidth, HDcalib_TUL_equalwidth, brier)

        # Equal PD + HD
        HDcalib_df_equalpd, HDcalib_TEL_equalpd, HDcalib_TUL_equalpd, _ = f_binning_methods(test_pred, train2_pred, target_var, num_ratings, bin_by_equal_pd, HD_calib)
        brier = brier_score_loss(test[target_var], HDcalib_df_equalpd['method Cal_PD'])
        log_fold_results("equalpd_HD", fold_num, HDcalib_TEL_equalpd, HDcalib_TUL_equalpd, brier)

        # Objective functie + HD
        HDcalib_df_objective, HDcalib_TEL_objective, HDcalib_TUL_objective, _ = f_objective_funct_2(test_pred, train2_pred, test, target_var, num_ratings, HD_calib)
        brier = brier_score_loss(test[target_var], HDcalib_df_objective['method Cal_PD'])
        log_fold_results("objective_HD", fold_num, HDcalib_TEL_objective, HDcalib_TUL_objective, brier)

        # ERR + TTC
        TTCcalib_df_ERR, TTCcalib_TEL_ERR, TTCcalib_TUL_ERR, _, _ = f_ERR(test_pred, train2_pred, target_var, num_ratings, x_min, x_max, TTC_calib)
        brier = brier_score_loss(test[target_var], TTCcalib_df_ERR['method Cal_PD'])
        log_fold_results("ERR_TTC", fold_num, TTCcalib_TEL_ERR, TTCcalib_TUL_ERR, brier)

        # Clustering + TTC
        TTCcalib_df_clustering, TTCcalib_TEL_clustering, TTCcalib_TUL_clustering, _ = f_clustering(test_pred, train2_pred, target_var, num_ratings, TTC_calib)
        brier = brier_score_loss(test[target_var], TTCcalib_df_clustering['method Cal_PD'])
        log_fold_results("clustering_TTC", fold_num, TTCcalib_TEL_clustering, TTCcalib_TUL_clustering, brier)

        # RT + TTC
        TTCcalib_df_RT, TTCcalib_TEL_RT, TTCcalib_TUL_RT, _ = f_regression_tree(test_pred, train2_pred, target_var, num_ratings, TTC_calib)
        brier = brier_score_loss(test[target_var], TTCcalib_df_RT['method Cal_PD'])
        log_fold_results("RT_TTC", fold_num, TTCcalib_TEL_RT, TTCcalib_TUL_RT, brier)

        # SS + TTC
        # TTCcalib_df_SS, TTCcalib_TEL_SS, TTCcalib_TUL_SS, _ = f_search_scale(test_pred, train2_pred, target_var, num_ratings, TTC_calib)
        # brier = brier_score_loss(test[target_var], TTCcalib_df_SS['method Cal_PD'])
        # log_fold_results("SS_TTC", fold_num, TTCcalib_TEL_SS, TTCcalib_TUL_SS, brier)

        # Equal Count + TTC
        TTCcalib_df_equalcount, TTCcalib_TEL_equalcount, TTCcalib_TUL_equalcount, _ = f_binning_methods(test_pred, train2_pred, target_var, num_ratings, bin_by_equal_count, TTC_calib)
        brier = brier_score_loss(test[target_var], TTCcalib_df_equalcount['method Cal_PD'])
        log_fold_results("equalcount_TTC", fold_num, TTCcalib_TEL_equalcount, TTCcalib_TUL_equalcount, brier)

        # Equal Width + TTC
        TTCcalib_df_equalwidth, TTCcalib_TEL_equalwidth, TTCcalib_TUL_equalwidth, _ = f_binning_methods(test_pred, train2_pred, target_var, num_ratings, bin_by_equal_width, TTC_calib)
        brier = brier_score_loss(test[target_var], TTCcalib_df_equalwidth['method Cal_PD'])
        log_fold_results("equalwidth_TTC", fold_num, TTCcalib_TEL_equalwidth, TTCcalib_TUL_equalwidth, brier)

        # Equal PD + TTC
        TTCcalib_df_equalpd, TTCcalib_TEL_equalpd, TTCcalib_TUL_equalpd, _ = f_binning_methods(test_pred, train2_pred, target_var, num_ratings, bin_by_equal_pd, TTC_calib)
        brier = brier_score_loss(test[target_var], TTCcalib_df_equalpd['method Cal_PD'])
        log_fold_results("equalpd_TTC", fold_num, TTCcalib_TEL_equalpd, TTCcalib_TUL_equalpd, brier)

        # # Objective functie + TTC
        TTCcalib_df_objective, TTCcalib_TEL_objective, TTCcalib_TUL_objective, _ = f_objective_funct_2(test_pred, train2_pred, test, target_var, num_ratings, TTC_calib)
        brier = brier_score_loss(test[target_var], TTCcalib_df_objective['method Cal_PD'])
        log_fold_results("objective_TTC", fold_num, TTCcalib_TEL_objective, TTCcalib_TUL_objective, brier)

        # ERR + QMM
        QMMcalib_df_ERR, QMMcalib_TEL_ERR, QMMcalib_TUL_ERR, _, _ = f_ERR(test_pred, train2_pred, target_var, num_ratings, x_min, x_max, QMM_calibration)
        brier = brier_score_loss(test[target_var], QMMcalib_df_ERR['method Cal_PD'])
        log_fold_results("ERR_QMM", fold_num, QMMcalib_TEL_ERR, QMMcalib_TUL_ERR, brier)

        # Clustering + QMM
        QMMcalib_df_clustering, QMMcalib_TEL_clustering, QMMcalib_TUL_clustering, _ = f_clustering(test_pred, train2_pred, target_var, num_ratings, QMM_calibration)
        brier = brier_score_loss(test[target_var], QMMcalib_df_clustering['method Cal_PD'])
        log_fold_results("clustering_QMM", fold_num, QMMcalib_TEL_clustering, QMMcalib_TUL_clustering, brier)

        # RT + QMM
        QMMcalib_df_RT, QMMcalib_TEL_RT, QMMcalib_TUL_RT, _ = f_regression_tree(test_pred, train2_pred, target_var, num_ratings, QMM_calibration)
        brier = brier_score_loss(test[target_var], QMMcalib_df_RT['method Cal_PD'])
        log_fold_results("RT_QMM", fold_num, QMMcalib_TEL_RT, QMMcalib_TUL_RT, brier)

        # SS + QMM
        # QMMcalib_df_SS, QMMcalib_TEL_SS, QMMcalib_TUL_SS, _ = f_search_scale(test_pred, train2_pred, target_var, num_ratings, QMM_calibration)
        # brier = brier_score_loss(test[target_var], QMMcalib_df_SS['method Cal_PD'])
        # log_fold_results("SS_QMM", fold_num, QMMcalib_TEL_SS, QMMcalib_TUL_SS, brier)

        # Equal Count + QMM
        QMMcalib_df_equalcount, QMMcalib_TEL_equalcount, QMMcalib_TUL_equalcount, _ = f_binning_methods(test_pred, train2_pred, target_var, num_ratings, bin_by_equal_count, QMM_calibration)
        brier = brier_score_loss(test[target_var], QMMcalib_df_equalcount['method Cal_PD'])
        log_fold_results("equalcount_QMM", fold_num, QMMcalib_TEL_equalcount, QMMcalib_TUL_equalcount, brier)

        # Equal Width + QMM
        QMMcalib_df_equalwidth, QMMcalib_TEL_equalwidth, QMMcalib_TUL_equalwidth, _ = f_binning_methods(test_pred, train2_pred, target_var, num_ratings, bin_by_equal_width, QMM_calibration)
        brier = brier_score_loss(test[target_var], QMMcalib_df_equalwidth['method Cal_PD'])
        log_fold_results("equalwidth_QMM", fold_num, QMMcalib_TEL_equalwidth, QMMcalib_TUL_equalwidth, brier)

        # Equal PD + QMM
        QMMcalib_df_equalpd, QMMcalib_TEL_equalpd, QMMcalib_TUL_equalpd, _ = f_binning_methods(test_pred, train2_pred, target_var, num_ratings, bin_by_equal_pd, QMM_calibration)
        brier = brier_score_loss(test[target_var], QMMcalib_df_equalpd['method Cal_PD'])
        log_fold_results("equalpd_QMM", fold_num, QMMcalib_TEL_equalpd, QMMcalib_TUL_equalpd, brier)

        # Objective functie + QMM
        QMMcalib_df_objective, QMMcalib_TEL_objective, QMMcalib_TUL_objective, _ = f_objective_funct_2(test_pred, train2_pred, test, target_var, num_ratings, QMM_calibration)
        brier = brier_score_loss(test[target_var], QMMcalib_df_objective['method Cal_PD'])
        log_fold_results("objective_QMM", fold_num, QMMcalib_TEL_objective, QMMcalib_TUL_objective, brier)

        # ERR + SLR
        SLRcalib_df_ERR, SLRcalib_TEL_ERR, SLRcalib_TUL_ERR, _, _ = f_ERR(test_pred, train2_pred, target_var, num_ratings, x_min, x_max, SLR_calib)
        brier = brier_score_loss(test[target_var], SLRcalib_df_ERR['method Cal_PD'])
        log_fold_results("ERR_SLR", fold_num, SLRcalib_TEL_ERR, SLRcalib_TUL_ERR, brier)

        # Clustering + SLR
        SLRcalib_df_clustering, SLRcalib_TEL_clustering, SLRcalib_TUL_clustering, _ = f_clustering(test_pred, train2_pred, target_var, num_ratings, SLR_calib)
        brier = brier_score_loss(test[target_var], SLRcalib_df_clustering['method Cal_PD'])
        log_fold_results("clustering_SLR", fold_num, SLRcalib_TEL_clustering, SLRcalib_TUL_clustering, brier)

        # RT + SLR
        SLRcalib_df_RT, SLRcalib_TEL_RT, SLRcalib_TUL_RT, _ = f_regression_tree(test_pred, train2_pred, target_var, num_ratings, SLR_calib)
        brier = brier_score_loss(test[target_var], SLRcalib_df_RT['method Cal_PD'])
        log_fold_results("RT_SLR", fold_num, SLRcalib_TEL_RT, SLRcalib_TUL_RT, brier)

        # SS + SLR
        # SLRcalib_df_SS, SLRcalib_TEL_SS, SLRcalib_TUL_SS, _ = f_search_scale(test_pred, train2_pred, target_var, num_ratings, SLR_calib)
        # brier = brier_score_loss(test[target_var], SLRcalib_df_SS['method Cal_PD'])
        # log_fold_results("SS_SLR", fold_num, SLRcalib_TEL_SS, SLRcalib_TUL_SS, brier)

        # Equal Count + SLR
        SLRcalib_df_equalcount, SLRcalib_TEL_equalcount, SLRcalib_TUL_equalcount, _ = f_binning_methods(test_pred, train2_pred, target_var, num_ratings, bin_by_equal_count, SLR_calib)
        brier = brier_score_loss(test[target_var], SLRcalib_df_equalcount['method Cal_PD'])
        log_fold_results("equalcount_SLR", fold_num, SLRcalib_TEL_equalcount, SLRcalib_TUL_equalcount, brier)

        # Equal Width + SLR
        SLRcalib_df_equalwidth, SLRcalib_TEL_equalwidth, SLRcalib_TUL_equalwidth, _ = f_binning_methods(test_pred, train2_pred, target_var, num_ratings, bin_by_equal_width, SLR_calib)
        brier = brier_score_loss(test[target_var], SLRcalib_df_equalwidth['method Cal_PD'])
        log_fold_results("equalwidth_SLR", fold_num, SLRcalib_TEL_equalwidth, SLRcalib_TUL_equalwidth, brier)
        
        # Equal PD + SLR
        SLRcalib_df_equalpd, SLRcalib_TEL_equalpd, SLRcalib_TUL_equalpd, _ = f_binning_methods(test_pred, train2_pred, target_var, num_ratings, bin_by_equal_pd, SLR_calib)
        brier = brier_score_loss(test[target_var], SLRcalib_df_equalpd['method Cal_PD'])
        log_fold_results("equalpd_SLR", fold_num, SLRcalib_TEL_equalpd, SLRcalib_TUL_equalpd, brier)

        # Objective functie + SLR
        SLRcalib_df_objective, SLRcalib_TEL_objective, SLRcalib_TUL_objective, _ = f_objective_funct_2(test_pred, train2_pred, test, target_var, num_ratings, SLR_calib)
        brier = brier_score_loss(test[target_var], SLRcalib_df_objective['method Cal_PD'])
        log_fold_results("objective_SLR", fold_num, SLRcalib_TEL_objective, SLRcalib_TUL_objective, brier)

        # ERR + IR
        IRcalib_df_ERR, IRcalib_TEL_ERR, IRcalib_TUL_ERR, _, _ = f_ERR(test_pred, train2_pred, target_var, num_ratings, x_min, x_max, IR_calib)
        brier = brier_score_loss(test[target_var], IRcalib_df_ERR['method Cal_PD'])
        log_fold_results("ERR_IR", fold_num, IRcalib_TEL_ERR, IRcalib_TUL_ERR, brier)

        # Clustering + IR
        IRcalib_df_clustering, IRcalib_TEL_clustering, IRcalib_TUL_clustering, _ = f_clustering(test_pred, train2_pred, target_var, num_ratings, IR_calib)
        brier = brier_score_loss(test[target_var], IRcalib_df_clustering['method Cal_PD'])
        log_fold_results("clustering_IR", fold_num, IRcalib_TEL_clustering, IRcalib_TUL_clustering, brier)

        # RT + IR
        IRcalib_df_RT, IRcalib_TEL_RT, IRcalib_TUL_RT, _ = f_regression_tree(test_pred, train2_pred, target_var, num_ratings, IR_calib)
        brier = brier_score_loss(test[target_var], IRcalib_df_RT['method Cal_PD'])
        log_fold_results("RT_IR", fold_num, IRcalib_TEL_RT, IRcalib_TUL_RT, brier)

        # SS + IR
        # IRcalib_df_SS, IRcalib_TEL_SS, IRcalib_TUL_SS, _ = f_search_scale(test_pred, train2_pred, target_var, num_ratings, IR_calib)
        # brier = brier_score_loss(test[target_var], IRcalib_df_SS['method Cal_PD'])
        # log_fold_results("SS_IR", fold_num, IRcalib_TEL_SS, IRcalib_TUL_SS, brier)

        # Equal Count + IR
        IRcalib_df_equalcount, IRcalib_TEL_equalcount, IRcalib_TUL_equalcount, _ = f_binning_methods(test_pred, train2_pred, target_var, num_ratings, bin_by_equal_count, IR_calib)
        brier = brier_score_loss(test[target_var], IRcalib_df_equalcount['method Cal_PD'])
        log_fold_results("equalcount_IR", fold_num, IRcalib_TEL_equalcount, IRcalib_TUL_equalcount, brier)

        # Equal Width + IR
        IRcalib_df_equalwidth, IRcalib_TEL_equalwidth, IRcalib_TUL_equalwidth, _ = f_binning_methods(test_pred, train2_pred, target_var, num_ratings, bin_by_equal_width, IR_calib)
        brier = brier_score_loss(test[target_var], IRcalib_df_equalwidth['method Cal_PD'])
        log_fold_results("equalwidth_IR", fold_num, IRcalib_TEL_equalwidth, IRcalib_TUL_equalwidth, brier)

        # Equal PD + IR
        IRcalib_df_equalpd, IRcalib_TEL_equalpd, IRcalib_TUL_equalpd, _ = f_binning_methods(test_pred, train2_pred, target_var, num_ratings, bin_by_equal_pd, IR_calib)
        brier = brier_score_loss(test[target_var], IRcalib_df_equalpd['method Cal_PD'])
        log_fold_results("equalpd_IR", fold_num, IRcalib_TEL_equalpd, IRcalib_TUL_equalpd, brier)

        # Objective functie + IR
        IRcalib_df_objective, IRcalib_TEL_objective, IRcalib_TUL_objective, _ = f_objective_funct_2(test_pred, train2_pred, test, target_var, num_ratings, IR_calib)
        brier = brier_score_loss(test[target_var], IRcalib_df_objective['method Cal_PD'])
        log_fold_results("objective_IR", fold_num, IRcalib_TEL_objective, IRcalib_TUL_objective, brier)


    fold_num += 1
    # winsound.Beep(500, 300)  # Indicate fold completion

# Export results per pipeline
summary_output_dir = "C:/Users/rikte/VS Code Python Projects/thesis_riktgs/output/cross_val"
os.makedirs(summary_output_dir, exist_ok=True)

for name, results in fold_results.items():
    df_results = pd.DataFrame(results)
    df_results.to_csv(os.path.join(summary_output_dir, f"{name}.csv"), index=False)

winsound.Beep(1000, 1500)  # Final completion beep