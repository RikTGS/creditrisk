import os
import pandas as pd
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import statsmodels.api as sm
from sklearn.metrics import brier_score_loss
from sklearn.calibration import calibration_curve
from scipy.stats import shapiro, f_oneway, kruskal, probplot
import scikit_posthocs as sp
from plot_save import save_plot
from scales import f_ERR
from pd_calib import HD_calib

# variables 
test_pred = None
HDcalib_df_ERR = None
HDcalib_TEL_ERR = None
HDcalib_TUL_ERR = None
HDcalib_summary_ERR = None
HDcalib_thresholds_ERR = None
HDcalib_df_clustering = None
HDcalib_TEL_clustering = None
HDcalib_TUL_clustering = None
HDcalib_summary_clustering = None
HDcalib_df_RT = None
HDcalib_TEL_RT = None
HDcalib_TUL_RT = None
HDcalib_summary_RT = None
HDcalib_df_SS = None
HDcalib_TEL_SS = None
HDcalib_TUL_SS = None
HDcalib_summary_SS = None
HDcalib_df_equalcount = None
HDcalib_TEL_equalcount = None
HDcalib_TUL_equalcount = None
HDcalib_summary_equalcount = None
HDcalib_df_equalwidth = None
HDcalib_TEL_equalwidth = None
HDcalib_TUL_equalwidth = None
HDcalib_summary_equalwidth = None
HDcalib_df_equalpd = None
HDcalib_TEL_equalpd = None
HDcalib_TUL_equalpd = None
HDcalib_summary_equalpd = None
HDcalib_df_objective = None
HDcalib_TEL_objective = None
HDcalib_TUL_objective = None
HDcalib_summary_objective = None
full_df = None
TTCcalib_summary_ERR = None
QMMcalib_summary_ERR = None
SLRcalib_summary_ERR = None
IRcalib_summary_ERR = None
TTCcalib_df_ERR = None
QMMcalib_df_ERR = None
SLRcalib_df_ERR = None
IRcalib_df_ERR = None
test = None
IRcalib_summary_clustering = None
IRcalib_summary_RT = None
IRcalib_summary_SS = None
IRcalib_summary_equalcount = None
IRcalib_summary_equalwidth = None
IRcalib_summary_equalpd = None
IRcalib_summary_objective = None
TTCcalib_TEL_ERR = None
TTCcalib_TUL_ERR = None
QMMcalib_TEL_ERR = None
QMMcalib_TUL_ERR = None
SLRcalib_TEL_ERR = None
SLRcalib_TUL_ERR = None
IRcalib_TEL_ERR = None
IRcalib_TUL_ERR = None
df_pred = None
num_ratings = None

# numbers = [2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 20, 50, 100]
numbers = [7]
for num in numbers:
    output_dir = f'C:/Users/rikte/VS Code Python Projects/thesis_riktgs/output/{num}'
    dir = num
    print(num)


    # Loop over alle bestanden en maak variabelen aan
    for file in os.listdir(output_dir):
        file_path = os.path.join(output_dir, file)
        var_name, ext = os.path.splitext(file)  # 'HDcalib_df_ERR', '.csv' of '.txt'

        if ext == '.csv':
            try:
                df = pd.read_csv(file_path)
                globals()[var_name] = df
            except Exception as e:
                print(f"⚠️ Fout bij inladen van {var_name}: {e}")

        elif ext == '.txt':
            try:
                with open(file_path, 'r') as f:
                    val = f.read().strip()
                    # Probeer te casten naar float of int
                    try:
                        val = float(val) if '.' in val else int(val)
                    except:
                        pass
                    globals()[var_name] = val
            except Exception as e:
                print(f"⚠️ Fout bij inladen van {var_name}: {e}")

    target_var = 'SeriousDlqin2yrs'
    # GIVE ME SOME CREDIT DATASET

    x_min = test_pred['Prediction'].quantile(0.05)   # bv. 5e percentiel
    x_max = test_pred['Prediction'].quantile(0.95)   # bv. 95e percentiel

    markers = ['o', 's', 'x', 'D', '^', 'v', '*', '+', '1', '2', '3', '4']

        # Plot thresholds of HDcalib ERR on test_predictions with prediction on the x-as and index on y-as
    HDcalib_thresholds_ERR.columns = ['Threshold']
    sorted_test_pred = test_pred.sort_values(by='Prediction').reset_index(drop=True)
    plt.figure()
    plt.plot(sorted_test_pred['Prediction'], sorted_test_pred.index, label='Predictions', color='blue')
    for threshold in HDcalib_thresholds_ERR['Threshold']:
        plt.axvline(x=threshold, color='red', linestyle='-', linewidth=1)
    plt.title('Thresholds of Exponential Risk Relationship with Historical Default calibration on Test Predictions')
    plt.xlabel('Prediction / Threshold')
    plt.ylabel('Index')
    plt.legend(['Sorted Predictions', 'Exponential Risk Relationship with Historical Default calibration Thresholds'])
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout()
    save_plot(dir, 'Thresholds of ERR with HD calib on Test Predictions')
    plt.close()
    # plt.show()
    predictions = test_pred['Prediction']
    plt.figure()
    # Plot de PDF via KDE
    sns.kdeplot(predictions, fill=True, label='PDF of Predictions', color='blue')
    # Voeg de thresholds toe als verticale lijnen
    for threshold in HDcalib_thresholds_ERR['Threshold']:
        plt.axvline(x=threshold, color='red', linestyle='--', linewidth=1)
    # Titels en labels
    plt.title('Probability Density Function (PDF) of Test Predictions')
    plt.xlabel('Prediction')
    plt.ylabel('Density')
    plt.legend(['PDF of Predictions', 'Thresholds'])
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout()
    # Opslaan en/of tonen
    save_plot(dir, 'PDF of Test Predictions with Thresholds')
    plt.close()
    
        # Compare methods based on average PD and rating counts for each rating
    methods = ["Exponential Risk Relationship", "Clustering", "Regression Tree", "Search Scale", "Equal Count", "Equal Width", "Equal PD", "Objective Function"]
    # methods = ["ERR", "Clustering", "Regression Trees", "Equal Count", "Equal Width", "Equal PD"]
    summaries = [
        HDcalib_summary_ERR,
        HDcalib_summary_clustering,
        HDcalib_summary_RT,
        HDcalib_summary_SS,
        HDcalib_summary_equalcount,
        HDcalib_summary_equalwidth,
        HDcalib_summary_equalpd,
        HDcalib_summary_objective
    ]

    type = ["avg PD","test counts"]  # or "avg PD" based on your requirement
    for t in type:
            # Extract average PD for each rating from summaries
        avg_pd_data = {method: summary[t].values for method, summary in zip(methods, summaries)}
            # Create a DataFrame with methods as rows and ratings as columns
        avg_pd_df = pd.DataFrame(avg_pd_data)
        avg_pd_df.index = [f"{i}" for i in range(avg_pd_df.shape[0])]
        avg_pd_df.columns.name = t
        print(avg_pd_df)
            # graph 1 (avg PD) and 2 (test counts)
        plt.figure()
        for i, method in enumerate(avg_pd_df.columns):
            marker = markers[i % len(markers)]  # Use modulo to cycle through markers
            plt.plot(avg_pd_df.index, avg_pd_df[method], marker=marker, label=method)
        if t == "avg PD":
            plt.title("Pooled PD per rating for different rating methods")
            plt.ylabel("Pooled PD")
        elif   t == "test counts":
            plt.title("Count per rating for different rating methods")
            plt.ylabel("Count")
        plt.xlabel("Rating")
        plt.legend(title="Rating Methods", bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.grid(True, linestyle='--', alpha=0.6)
        plt.tight_layout()
        save_plot(dir, f"{t} per rating for different rating methods") 
        plt.close()
        # plt.show()

        # Analysis: exponential risk ERR x_min and x_max
    x_min_base = x_min
    x_max_base = x_max
    steps=5
    delta=0.2
    results = []

    x_min_range = np.linspace(x_min_base * (1 - delta), x_min_base * (1 + delta), steps)
    x_max_range = np.linspace(x_max_base * (1 - delta), x_max_base * (1 + delta), steps)
    calib_method = HD_calib
    for x_min in x_min_range:
        for x_max in x_max_range:
            _, _, _, rating_summary, thresholds = f_ERR(test_pred, df_pred, target_var, num, x_min, x_max, calib_method)
            rating_summary['x_min'] = x_min
            rating_summary['x_max'] = x_max
            results.append(rating_summary)

    full_df = pd.concat(results, ignore_index=True)

            # Eerst: converteer ratings naar int zodat je goed kunt sorteren
    full_df['Rating'] = full_df['Rating'].astype(int)

        # Plot: effect van x_min op avg PD of test counts, bij vaste x_max
    fixed_x_max = full_df['x_max'].median()
    subset_min = full_df[full_df['x_max'] == fixed_x_max]

    for t in type:
        plt.figure()
        for i, rating in enumerate(sorted(subset_min['Rating'].unique())):
            df_r = subset_min[subset_min['Rating'] == rating]
            marker = markers[i % len(markers)]  # Use modulo to cycle through markers
            plt.plot(df_r['x_min'], df_r[t], marker=marker, label=f'Rating {rating}')
        if t == "avg PD":
            plt.title(f'Effect of x_min on average PD (x_max = {fixed_x_max:.4f})')
        elif t == "test counts":
            plt.title(f'Effect of x_min on counts (x_max = {fixed_x_max:.4f})')
        plt.xlabel('x_min')
        plt.ylabel(t.capitalize())
        plt.grid(True, linestyle='--', alpha=0.6)
        plt.legend(title="Rating", bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.tight_layout()
        save_plot(dir, f'Effect of x_min on {t} (x_max = {fixed_x_max:.4f})')
        plt.close()
        # plt.show()

        # Plot: effect van x_max op avg PD of test counts, bij vaste x_min
    fixed_x_min = full_df['x_min'].median()
    subset_max = full_df[full_df['x_min'] == fixed_x_min]

    for t in type:
        plt.figure()
        for i, rating in enumerate(sorted(subset_max['Rating'].unique())):
            df_r = subset_max[subset_max['Rating'] == rating]
            marker = markers[i % len(markers)]  # Use modulo to cycle through markers
            plt.plot(df_r['x_max'], df_r[t], marker=marker, label=f'{rating}')
        if t == "avg PD":
            plt.title(f'Effect of x_max on pooled PD (x_min = {fixed_x_min:.4f})')
            plt.ylabel("Pooled PD")
        elif t == "test counts":
            plt.title(f'Effect of x_max on rating count (x_min = {fixed_x_min:.4f})')
            plt.ylabel("Count")
        plt.xlabel('x_max')
        plt.grid(True, linestyle='--', alpha=0.6)
        plt.legend(title="Rating", bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.tight_layout()
        save_plot(dir, f'Effect of x_max on {t} (x_min = {fixed_x_min:.4f})')
        plt.close()
        # plt.show()

        # Statistical analysis: linear regression for each rating
            # Lege lijst om resultaten op te slaan
    pval_tables = {}
    effect_tables = {}
    for t in type:
        pval_rows, effect_rows = [], []
        for rating in sorted(full_df['Rating'].unique()):
            df_rating = full_df[full_df['Rating'] == rating]
            X = df_rating[['x_min', 'x_max']]
            X = sm.add_constant(X)
            y = df_rating[t]
            model = sm.OLS(y, X).fit()
            coef_xmin = model.params['x_min']
            coef_xmax = model.params['x_max']
            pval_xmin = model.pvalues['x_min']
            pval_xmax = model.pvalues['x_max']
            pval_rows.append({
                'Rating': rating,
                'p x_min': f"{pval_xmin:.2e}",
                'p x_max': f"{pval_xmax:.2e}"
            })
            effect_rows.append({
                'Rating': rating,
                'Effect x_min': f"{coef_xmin:.4f}{' *' if pval_xmin < 0.05 else ''}",
                'Effect x_max': f"{coef_xmax:.4f}{' *' if pval_xmax < 0.05 else ''}"
            })
            # Store results per metric
        pval_df = pd.DataFrame(pval_rows).set_index('Rating')
        effect_df = pd.DataFrame(effect_rows).set_index('Rating')
        pval_tables[t] = pval_df
        effect_tables[t] = effect_df
            # Print results
        print(f"\nP-values (scientific notation) {t}:")
        print(pval_df.to_string())
        print(f"\nEffect sizes {t} (with * if p < 0.05):")
        print(effect_df.to_string())

        # Impact on EL and UL
            # Create a DataFrame with methods as rows and TEL/TUL as columns
    tel_tul_data = {
        "Rating Method": [
            "Exponential Risk Relationship", "Clustering", "Regression Tree", "Search Scale",
            "Equal Count", "Equal Width", "Equal PD", "Objective Function"
        ],
        "TEL": [
            HDcalib_TEL_ERR, HDcalib_TEL_clustering, HDcalib_TEL_RT, HDcalib_TEL_SS,
            HDcalib_TEL_equalcount, HDcalib_TEL_equalwidth, HDcalib_TEL_equalpd, HDcalib_TEL_objective
        ],
        "TUL": [
            HDcalib_TUL_ERR, HDcalib_TUL_clustering, HDcalib_TUL_RT, HDcalib_TUL_SS,
            HDcalib_TUL_equalcount, HDcalib_TUL_equalwidth, HDcalib_TUL_equalpd, HDcalib_TUL_objective
        ]
    }

    tel_tul_df = pd.DataFrame(tel_tul_data)
    norm_tel_tul_df = tel_tul_df.copy()
    norm_tel_tul_df[["TEL", "TUL"]] = norm_tel_tul_df[["TEL", "TUL"]] / len(test_pred)
    print(tel_tul_df)
    print(norm_tel_tul_df)
    # Plot TEL and TUL as a bar chart
    plt.figure()
    bar_width = 0.35
    x = np.arange(len(tel_tul_df['Rating Method']))
    plt.bar(x - bar_width / 2, tel_tul_df['TEL'], bar_width, label='TEL', color='black')
    plt.bar(x + bar_width / 2, tel_tul_df['TUL'], bar_width, label='TUL', color='gray')
    plt.xlabel('Rating Method')
    plt.ylabel('Total Losses')
    plt.title('TEL and TUL per rating method')
    plt.xticks(x, tel_tul_df['Rating Method'], rotation=45, ha='right')
    plt.legend(title="Credit Loss", bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    save_plot(dir, 'TEL_TUL_Barchart')
    plt.close()
    plt.figure()
    plt.bar(x - bar_width / 2, tel_tul_df['TEL'], bar_width, label='TEL', color='black')
    plt.xlabel('Rating Method')
    plt.ylabel('Total Expected Losses')
    plt.title('TEL per rating method')
    plt.xticks(x, tel_tul_df['Rating Method'], rotation=45, ha='right')
    plt.tight_layout()
    save_plot(dir, 'TEL_Barchart')
    plt.close()
    plt.figure()
    plt.bar(x - bar_width / 2, tel_tul_df['TUL'], bar_width, label='TUL', color='gray')
    plt.xlabel('Rating Method')
    plt.ylabel('Total Unexpected Losses')
    plt.title('TUL per rating method')
    plt.xticks(x, tel_tul_df['Rating Method'], rotation=45, ha='right')
    plt.tight_layout()
    save_plot(dir, 'TUL_Barchart')
    plt.close()
    # plt.show()

    # Calibration methods
        # Compare Rating Methods based on method Cal_PD and rating counts for each rating
    # Compare Rating Methods based on avg PD and method Cal_PD for each rating


        # Compare Calibration Methods based on method Cal_PD for each rating
    methods = ["Historical Default", "Through-the-Cycle", "Quasi-Moment Matching", "Scaled Likelihood Ratio", "Isotonic Regression"]
    summaries = [
        HDcalib_summary_ERR,
        TTCcalib_summary_ERR,   
        QMMcalib_summary_ERR,
        SLRcalib_summary_ERR,
        IRcalib_summary_ERR
    ]
    type = ["method Cal_PD","test counts"]  # or "avg PD" based on your requirement
    for t in type:
            # Extract average PD for each rating from summaries
        method_Cal_PD_data = {method: summary[t].values for method, summary in zip(methods, summaries)}
            # Create a DataFrame with methods as rows and ratings as columns
        method_cal_PD_df = pd.DataFrame(method_Cal_PD_data)
        method_cal_PD_df.index = [f"{i}" for i in range(method_cal_PD_df.shape[0])]
        method_cal_PD_df.columns.name = t
        print(method_cal_PD_df)
        # graph 
            # graph 1 (method Cal_PD) and 2 (test counts)
        plt.figure()
        for i, method in enumerate(method_cal_PD_df.columns):
            marker = markers[i % len(markers)]  # Use modulo to cycle through markers
            plt.plot(method_cal_PD_df.index, method_cal_PD_df[method], marker=marker, label=method)
        if t == "method Cal_PD":
            plt.title("Calibrated PD for Exponential Risk Relationship")
            plt.ylabel("Calibrated PD")
        elif   t == "test counts":
            plt.title("Count per rating for Exponential Risk Relationship")
            plt.ylabel("Count")
        plt.xlabel("Rating")
        plt.legend(title="Calibration Methods", bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.grid(True, linestyle='--', alpha=0.6)
        plt.tight_layout()
        save_plot(dir, f'{t} per rating for different calibration methods (ERR)')
        plt.close()
        # plt.show()

        # Brier score calculation: hoe lager hoe beter
            # Brier score is a measure of how close the predicted probabilities are to the actual outcomes.
    y_test = test[target_var]
    y_test = y_test.to_frame(name=target_var) if isinstance(y_test, pd.Series) else y_test
    y_test = y_test.reset_index(drop=True)
            # Alle calibratiemethoden
    method_dfs = {
        "Historical Default": HDcalib_df_ERR,
        "Through-the-Cycle": TTCcalib_df_ERR,
        "Quasi-Moment Matching": QMMcalib_df_ERR,
        "Scaled Likelihood Ratio": SLRcalib_df_ERR,
        "Isotonic Regression": IRcalib_df_ERR
    }
    brier_scores = {}
    for method, df in method_dfs.items():
        if df is not None and 'method Cal_PD' in df.columns:
            df = df.reset_index(drop=True)
            if len(df) == len(y_test):
                combined = pd.concat([y_test, df['method Cal_PD']], axis=1)
                brier_scores[method] = brier_score_loss(combined[target_var], combined['method Cal_PD'])
            else:
                brier_scores[method] = None
        elif df is not None and 'pooled pd' in df.columns and method == "No Calib":
            df = df.reset_index(drop=True)
            if len(df) == len(y_test):
                combined = pd.concat([y_test, df['pooled pd']], axis=1)
                brier_scores[method] = brier_score_loss(combined[target_var], combined['pooled pd'])
            else:
                brier_scores[method] = None
        else:
            brier_scores[method] = None
            # Resultaten tonen
    # Convert Brier scores to a DataFrame
    brier_scores_df = pd.DataFrame.from_dict(brier_scores, orient='index', columns=['Brier Score'])
    brier_scores_df.index.name = 'Method'
    brier_scores_df.reset_index(inplace=True)
    print("\nBrier Scores DataFrame:")
    print(brier_scores_df)
    # print("\nBrier scores per calibration method:")
    # for method, score in brier_scores.items():
    #     print(f"{method:20s}: {score}" if score is not None else f"{method:20s}: N/A")
        
                # Plotten van de Brier scores
    methods_ordered = sorted(brier_scores.items(), key=lambda x: x[1])
    labels = [m[0] for m in methods_ordered]
    scores = [m[1] for m in methods_ordered]

    plt.figure()
    plt.barh(labels, scores)
    plt.xlabel('Brier Score')
    plt.title('Brier Score per Calibration Method')
    plt.grid(axis='x', linestyle='--', alpha=0.6)
    plt.tight_layout()
    save_plot(dir, 'Brier Score per Calibration Method')
    plt.close()
    # plt.show()

            # Calibration curves per method
    plt.figure()
    for i, (method, df) in enumerate(method_dfs.items()):
        if df is not None and 'method Cal_PD' in df.columns:
            df = df.reset_index(drop=True)
            if len(df) == len(y_test):
                y_true = y_test[target_var].values
                y_prob = df['method Cal_PD'].values
                prob_true, prob_pred = calibration_curve(y_true, y_prob, n_bins=10, strategy='quantile')
                marker = markers[i % len(markers)]  # Use modulo to cycle through markers
                plt.plot(prob_pred, prob_true, marker=marker, label=method)
            # Perfect line
    plt.plot([0, 0.4], [0, 0.4], linestyle='--', color='gray', label='Perfect Calibration')
    plt.xlabel('Calibrated PD')
    plt.ylabel('Observed default rate (target var)')
    plt.title('Calibration Curves per Method')
    plt.legend(title="Calibration Methods", bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout()
    save_plot(dir, 'Calibration Curves per Method')
    plt.close()
    # plt.show()

        # TEL and TUL ERR for every calib method
    tel_tul_data = {
        "Calibration Method": [
            "Historical Default", "Through-the-Cycle", "Quasi-Moment Matching", "Scaled Likelihood Ratio", "Isotonic Regression"],
        "TEL": [
            HDcalib_TEL_ERR, TTCcalib_TEL_ERR, QMMcalib_TEL_ERR, SLRcalib_TEL_ERR, IRcalib_TEL_ERR],
        "TUL": [
            HDcalib_TUL_ERR, TTCcalib_TUL_ERR, QMMcalib_TUL_ERR, SLRcalib_TUL_ERR, IRcalib_TUL_ERR]
    }
    tel_tul_df = pd.DataFrame(tel_tul_data)
    norm_tel_tul_df = tel_tul_df.copy()
    norm_tel_tul_df[["TEL", "TUL"]] = norm_tel_tul_df[["TEL", "TUL"]] / len(test_pred)
    print(tel_tul_df)
    print(norm_tel_tul_df)

    # Dataframe met method Cal_PD per calibratie en rating methode
    cal_pd_data = []
    calib_methods = ['HDcalib', 'TTCcalib', 'QMMcalib', 'SLRcalib', 'IRcalib']
    # calib_methods = ['HDcalib', 'IRcalib']
    rating_methods = ['ERR', 'clustering', 'RT', 'equalcount', 'equalwidth', 'equalpd', 'objective']
    for calib in calib_methods:
        for rating_m in rating_methods:
            var_name = f"{calib}_summary_{rating_m}"
            try:
                summary_df = eval(var_name)
                if summary_df is not None and 'method Cal_PD' in summary_df.columns and 'Rating' in summary_df.columns:
                    for rating, cal_pd in zip(summary_df['Rating'], summary_df['method Cal_PD']):
                        cal_pd_data.append({
                            'Pipeline': f"{rating_m}_{calib}",
                            'Rating': rating,
                            'Calibrated PD': cal_pd
                        })
            except NameError:
                continue  # Sommige combinaties bestaan niet (zoals SS bij non-PD_calib)
    custom_rating_titles = {
    "ERR": "ERR",
    "RT": "RT",
    "clustering": "Clustering",
    "equalcount": "Equal Count",
    "equalwidth": "Equal Width",
    "equalpd": "Equal PD",
    "objective": "Objective Function"
    }
    custom_calib_titles = {
    "HDcalib": "Historical Default",
    "TTCcalib": "Through-the-Cycle",
    "QMMcalib": "Quasi-Moment Matching",
    "SLRcalib": "Scaled Likelihood Ratio",
    "IRcalib": "Isotonic Regression"
    }
    cal_pd_df = pd.DataFrame(cal_pd_data)
    # Split pipeline string
    cal_pd_df['Rating Method'] = cal_pd_df['Pipeline'].str.split('_').str[0]
    cal_pd_df['Calibration Method'] = cal_pd_df['Pipeline'].str.split('_').str[1].str.replace('calib', '')
    cal_pd_df.sort_values(['Rating Method', 'Calibration Method'], inplace=True)
    cal_pd_df['Combination'] = cal_pd_df['Rating Method'] + ' + ' + cal_pd_df['Calibration Method']
    print("📊 Calibrated PD per pipeline:\n", cal_pd_df)
    # Zet Seaborn-stijl
    plt.figure()
    sns.set(style="whitegrid")
    # Scatterplot
    ax = sns.lineplot(data=cal_pd_df, x="Rating", y="Calibrated PD", hue="Rating Method", style="Calibration Method", markers=True, dashes=False, markersize=8)
    # Optioneel: labels bij punten tonen
    plt.title("Calibrated PD per rating for all Rating Methods and Calibration Combinations")
    plt.xlabel("Rating")
    plt.ylabel("Calibrated PD")
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    save_plot(dir, 'Calibrated PD per rating for all Rating and Calibration Combinations')
    plt.close()
    # plt.show()


    # Expected and unexpected loss; provisions and capital requirements
            # Verzamel alle pipeline combinaties met hun EL en UL
    el_ul_data = []
            # Defineer calibratiemethoden en bijhorende summaries
    calib_methods = ['HDcalib', 'TTCcalib', 'QMMcalib', 'SLRcalib', 'IRcalib']
    rating_methods = ['ERR', 'clustering', 'RT', 'SS', 'equalcount', 'equalwidth', 'equalpd', 'objective']
    for calib in calib_methods:
        for rating in rating_methods:
            var_name = f"{calib}_df_{rating}"
            try:
                summary_df = eval(var_name)
                if summary_df is not None and 'EL' in summary_df.columns and 'UL' in summary_df.columns:
                    for el, ul in zip(summary_df['EL'], summary_df['UL']):
                        el_ul_data.append({
                            'Pipeline': f"{rating}_{calib}",
                            'EL': el,
                            'UL': ul
                        })
            except NameError:
                continue  # Sommige combinaties bestaan niet (zoals SS bij non-PD_calib)
            # Omzetten naar DataFrame
    el_ul_df = pd.DataFrame(el_ul_data)
    # statistische tests op el ul df?
    print("📊 EL en UL per pipeline:\n", el_ul_df)
    # Calculate total EL and UL for each pipeline
    total_el_ul_data = []
    for pipeline, group in el_ul_df.groupby('Pipeline'):
        total_el = group['EL'].sum()
        total_ul = group['UL'].sum()
        total_el_ul_data.append({
            'Pipeline': pipeline,
            'TEL': total_el,
            'TUL': total_ul
        })

    # Create a DataFrame for total EL and UL
    tel_tul_df = pd.DataFrame(total_el_ul_data)
    print("📊 Total EL and UL per pipeline:\n", tel_tul_df)
    # Split pipeline string
    tel_tul_df['Rating Method'] = tel_tul_df['Pipeline'].str.split('_').str[0]
    tel_tul_df = tel_tul_df[~tel_tul_df['Rating Method'].isin(['SS'])]
    tel_tul_df['Calibration Method'] = tel_tul_df['Pipeline'].str.split('_').str[1].str.replace('calib', '')
    tel_tul_df.sort_values(['Rating Method', 'Calibration Method'], inplace=True)
    tel_tul_df['Combination'] = tel_tul_df['Rating Method'] + ' + ' + tel_tul_df['Calibration Method']
    tel_tul_df['Rating Method'] = tel_tul_df['Rating Method'].replace({
    'ERR': 'Exponential Risk Relationship',
    'RT': 'Regression Tree',
    'clustering': 'Clustering',
    'equalcount': 'Equal Count',
    'equalwidth': 'Equal Width',
    'equalpd': 'Equal PD',
    'objective': 'Objective Function'
    })
    tel_tul_df['Calibration Method'] = tel_tul_df['Calibration Method'].replace({
    'HD': 'Historical Default',
    'TTC': 'Through-the-Cycle',
    'QMM': 'Quasi-Moment Matching',
    'SLR': 'Scaled Likelihood Ratio',
    'IR': 'Isotonic Regression'
    })
    print("📊 TEL and TUL per pipeline:\n", tel_tul_df)
    # Zet Seaborn-stijl
    plt.figure(figsize=(10, 6))
    sns.set(style="whitegrid")
    # Scatterplot
    ax = sns.scatterplot(
        data=tel_tul_df, x="TEL", y="TUL", hue="Calibration Method", style="Rating Method", s=100)
    # Optioneel: labels bij punten tonen
    plt.title("TEL vs TUL for all Rating and Calibration Combinations")
    plt.xlabel("TEL")
    plt.ylabel("TUL")
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    save_plot(dir, 'TEL vs TUL for all Rating and Calibration Combinations')
    plt.close()
    # plt.show()
# to improve: chart
    # Plot total EL and UL as a bar chart
    plt.figure()
    bar_width = 0.35
    x = np.arange(len(tel_tul_df['Pipeline']))
    plt.bar(x - bar_width / 2, tel_tul_df['TEL'], bar_width, label='TEL', color='black')
    plt.bar(x + bar_width / 2, tel_tul_df['TUL'], bar_width, label='TUL', color='gray')
    plt.xlabel('Pipeline')
    plt.ylabel('Total Losses')
    plt.title('TEL and TUL per Pipeline')
    plt.xticks(x, tel_tul_df['Pipeline'], rotation=45, ha='right')
    plt.legend(title="Loss Type", bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    save_plot(dir, 'Total_EL_UL_per_Pipeline')
    plt.close()
    # plt.show()

        # EL ANALYSE: significante verschillen in EL per pipeline

        
        # TEL per rating method per calibration method
    tel_plot_data = []
    rating_methods = ['ERR', 'clustering', 'RT', 'SS', 'equalcount', 'equalwidth', 'equalpd', 'objective']
    calib_methods = ['HDcalib', 'TTCcalib', 'QMMcalib', 'SLRcalib', 'IRcalib']
    for rating in rating_methods:
        for calib in calib_methods:
            tel_var_name = f"{calib}_TEL_{rating}"
            try:
                tel_val = eval(tel_var_name)
                if tel_val is not None:
                    tel_plot_data.append({
                        'Rating Method': rating.capitalize(),
                        'Calibration': calib.replace('calib', '').upper(),
                        'TEL': tel_val
                    })
            except NameError:
                continue  # combinatie bestaat niet (bijv. SS bij TTC-calib)

    tel_df = pd.DataFrame(tel_plot_data)
    tel_df['Rating Method'] = tel_df['Rating Method'].replace({
    'ERR': 'Exponential Risk Relationship',
    'RT': 'Regression Tree',
    'clustering': 'Clustering',
    'equalcount': 'Equal Count',
    'equalwidth': 'Equal Width',
    'equalpd': 'Equal PD',
    'objective': 'Objective Function'
    })
    tel_df['Calibration Method'] = tel_df['Calibration'].replace({
    'HD': 'Historical Default',
    'TTC': 'Through-the-Cycle',
    'QMM': 'Quasi-Moment Matching',
    'SLR': 'Scaled Likelihood Ratio',
    'IR': 'Isotonic Regression'
    })
    plt.figure()
    sns.set(style="whitegrid")
    sns.barplot(data=tel_df,x='Rating Method',y='TEL',hue='Calibration',palette='muted')
    plt.title("Total Expected Loss (TEL) per Rating Method and Calibration Method")
    plt.ylabel("TEL")
    plt.xlabel("Rating Method")
    plt.legend(title="Calibration Method", bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    save_plot(dir, "TEL_per_Rating_and_Calibration")
    plt.close()
    # plt.show()

    # UL ANALYSE: significante verschillen in UL per pipeline


        # TUL per rating method per calibration method
    tul_plot_data = []
    rating_methods = ['ERR', 'clustering', 'RT', 'SS', 'equalcount', 'equalwidth', 'equalpd', 'objective']
    calib_methods = ['HDcalib', 'TTCcalib', 'QMMcalib', 'SLRcalib', 'IRcalib']

    for rating in rating_methods:
        for calib in calib_methods:
            tul_var_name = f"{calib}_TUL_{rating}"
            try:
                tul_val = eval(tul_var_name)
                if tul_val is not None:
                    tul_plot_data.append({
                        'Rating Method': rating.capitalize(),
                        'Calibration': calib.replace('calib', '').upper(),
                        'TUL': tul_val
                    })
            except NameError:
                continue  # combinatie bestaat niet (bijv. SS bij TTC-calib)

    tul_df = pd.DataFrame(tul_plot_data)
    tul_df['Rating Method'] = tul_df['Rating Method'].replace({
    'ERR': 'Exponential Risk Relationship',
    'RT': 'Regression Tree',
    'clustering': 'Clustering',
    'equalcount': 'Equal Count',
    'equalwidth': 'Equal Width',
    'equalpd': 'Equal PD',
    'objective': 'Objective Function'
    })
    tul_df['Calibration Method'] = tul_df['Calibration'].replace({
    'HD': 'Historical Default',
    'TTC': 'Through-the-Cycle',
    'QMM': 'Quasi-Moment Matching',
    'SLR': 'Scaled Likelihood Ratio',
    'IR': 'Isotonic Regression'
    })
    plt.figure()
    sns.set(style="whitegrid")
    sns.barplot(data=tul_df,x='Rating Method',y='TUL',hue='Calibration',palette='muted')
    plt.title("Total Unexpected Loss (TUL) per Rating Method and Calibration Method")
    plt.ylabel("TUL")
    plt.xlabel("Rating Method")
    plt.legend(title="Calibration Method", bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    save_plot(dir, "TUL_per_Rating_and_Calibration")
    plt.close()
    # plt.show()

        # Brier Score per rating and calibration method
    # Brier score calculation: hoe lager hoe beter
            # Brier score is a measure of how close the predicted probabilities are to the actual outcomes.
    y_test = test[target_var]
    y_test = y_test.to_frame(name=target_var) if isinstance(y_test, pd.Series) else y_test
    y_test = y_test.reset_index(drop=True)
            # Alle calibratiemethoden
    calib_name_map = {
    'HDcalib': 'Historical Default',
    'TTCcalib': 'Through-the-Cycle',
    'QMMcalib': 'Quasi-Moment Matching',
    'SLRcalib': 'Scaled Likelihood Ratio',
    'IRcalib': 'Isotonic Regression'
    }
    rating_name_map = {
        'ERR': 'Exponential Risk Relationship',
        'clustering': 'Clustering',
        'RT': 'Regression Tree',
        'equalcount': 'Equal Count',
        'equalwidth': 'Equal Width',
        'equalpd': 'Equal PD',
        'objective': 'Objective Function'
    }
    calib_methods = list(calib_name_map.keys())
    rating_methods = list(rating_name_map.keys())
    method_dfs = {}
    for rating in rating_methods:
        for calib in calib_methods:
            var_name = f"{calib}_df_{rating}"
            try:
                val = eval(var_name)
                if val is not None:
                    method_dfs[f"{rating_name_map[rating]} + {calib_name_map[calib]}"] = val
            except NameError:
                continue  # combinatie bestaat niet (bijv. SS bij TTC-calib)
    
    brier_scores = {}
    for method, df in method_dfs.items():
        if df is not None and 'method Cal_PD' in df.columns:
            df = df.reset_index(drop=True)
            if len(df) == len(y_test):
                combined = pd.concat([y_test, df['method Cal_PD']], axis=1)
                brier_scores[method] = brier_score_loss(combined[target_var], combined['method Cal_PD'])
            else:
                brier_scores[method] = None
        else:
            brier_scores[method] = None
            # Resultaten tonen
    # Convert Brier scores to a DataFrame
    brier_scores_df = pd.DataFrame.from_dict(brier_scores, orient='index', columns=['Brier Score'])
    brier_scores_df.index.name = 'Pipeline'
    brier_scores_df.reset_index(inplace=True)
    brier_scores_df = brier_scores_df.sort_values('Brier Score', ascending=True)
    # Remove 'calib' from the index names in the Brier scores DataFrame
    # brier_scores_df['Pipeline'] = brier_scores_df['Pipeline'].str.replace('calib', '', regex=False)
    plt.figure(figsize=(8, max(6, len(brier_scores_df) * 0.35)))
    sns.barplot(data=brier_scores_df, y='Pipeline', x='Brier Score', palette='Blues_r')
    plt.title('Brier Score for all Rating and Calibration Methods')
    plt.grid(axis='x', linestyle='--', alpha=0.6)
    plt.tight_layout()
    save_plot(dir, 'Brier Score for all Rating and Calibration Methods')
    plt.close()


    # EXTRAS 
        # Influence/impact of IR Regression (see clustering and equal width)
            # Extract method Cal_PD for IR regression per rating method
    methods = ["Exponential Risk Relationship", "Clustering", "Regression Tree", "Equal Count", "Equal Width", "Equal PD", "Objective Function"]
    # methods = ["ERR", "Clustering", "Regression Trees", "Equal Count", "Equal Width", "Equal PD"]# , "Objective Function"]
    summaries = [
        IRcalib_summary_ERR,
        IRcalib_summary_clustering,
        IRcalib_summary_RT,
        IRcalib_summary_equalcount,
        IRcalib_summary_equalwidth,
        IRcalib_summary_equalpd,
        IRcalib_summary_objective
    ]
    type = "method Cal_PD" 
            # Extract average PD for each rating from summaries
    method_Cal_PD_data = {method: summary[type].values for method, summary in zip(methods, summaries)}
            # Create a DataFrame with methods as rows and ratings as columns
    method_Cal_PD_df = pd.DataFrame(method_Cal_PD_data)
    method_Cal_PD_df.index = [f"{i}" for i in range(method_Cal_PD_df.shape[0])]
    method_Cal_PD_df.columns.name = type
    print(method_Cal_PD_df)
        # graph 
    plt.figure()
    for i, method in enumerate(method_Cal_PD_df.columns):
        marker = markers[i % len(markers)]  # Use modulo to cycle through markers
        plt.plot(method_Cal_PD_df.index, method_Cal_PD_df[method], marker=marker, label=method)
    plt.title("Calibrated Probability of Default (PD) per rating for different rating methods (Isotonic Regression)")
    plt.xlabel("Rating")
    plt.ylabel("Calibrated PD")
    plt.legend(title="Rating Methods", bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout()
    save_plot(dir, 'method Cal_PD per rating for different rating methods (IR)')
    plt.close()
    # plt.show()

            # Extract method Cal_PD for HD per rating method
    methods = ["Exponential Risk Relationship", "Clustering", "Regression Tree", "Equal Count", "Equal Width", "Equal PD", "Objective Function"]
    # methods = ["ERR", "Clustering", "Regression Trees", "Equal Count", "Equal Width", "Equal PD"]# , "Objective Function"]
    summaries = [
        HDcalib_summary_ERR,
        HDcalib_summary_clustering,
        HDcalib_summary_RT,
        HDcalib_summary_equalcount,
        HDcalib_summary_equalwidth,
        HDcalib_summary_equalpd,
        HDcalib_summary_objective
    ]
    type = "method Cal_PD" 
            # Extract average PD for each rating from summaries
    method_cal_pd_data = {method: summary[type].values for method, summary in zip(methods, summaries)}
            # Create a DataFrame with methods as rows and ratings as columns
    method_Cal_PD_df = pd.DataFrame(method_cal_pd_data)
    method_Cal_PD_df.index = [f"{i}" for i in range(method_Cal_PD_df.shape[0])]
    method_Cal_PD_df.columns.name = type
    print(method_Cal_PD_df)
        # graph 
    plt.figure()
    for i, method in enumerate(method_Cal_PD_df.columns):
        marker = markers[i % len(markers)]  # Use modulo to cycle through markers
        plt.plot(method_Cal_PD_df.index, method_Cal_PD_df[method], marker=marker, label=method)
    plt.title("Calibrated Probability of Default (PD) per rating for different rating methods (Historical Default)")
    plt.xlabel("Rating")
    plt.ylabel("Calibrated PD")
    plt.legend(title="Rating Methods", bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout()
    save_plot(dir, 'method Cal_PD per rating for different rating methods (HD)')
    plt.close()
    # plt.show()

    # Enkel HD and IR
        # Split pipeline string
    total_el_ul_data = []
    for pipeline, group in el_ul_df.groupby('Pipeline'):
        total_el = group['EL'].sum()
        total_ul = group['UL'].sum()
        total_el_ul_data.append({
            'Pipeline': pipeline,
            'TEL': total_el,
            'TUL': total_ul
        })

    # Create a DataFrame for total EL and UL
    tel_tul_df = pd.DataFrame(total_el_ul_data)
    print("📊 Total EL and UL per pipeline:\n", tel_tul_df)
    # Split pipeline string
    tel_tul_df['Rating Method'] = tel_tul_df['Pipeline'].str.split('_').str[0]
    tel_tul_df = tel_tul_df[~tel_tul_df['Rating Method'].isin(['SS'])]
    tel_tul_df['Calibration Method'] = tel_tul_df['Pipeline'].str.split('_').str[1].str.replace('calib', '')
    tel_tul_df = tel_tul_df[~tel_tul_df['Calibration Method'].isin(['QMM', 'SLR', 'TTC'])]
    tel_tul_df.sort_values(['Rating Method', 'Calibration Method'], inplace=True)
    tel_tul_df['Combination'] = tel_tul_df['Rating Method'] + ' + ' + tel_tul_df['Calibration Method']
    tel_tul_df['Rating Method'] = tel_tul_df['Rating Method'].replace({
    'ERR': 'Exponential Risk Relationship',
    'RT': 'Regression Tree',
    'clustering': 'Clustering',
    'equalcount': 'Equal Count',
    'equalwidth': 'Equal Width',
    'equalpd': 'Equal PD',
    'objective': 'Objective Function'
    })
    tel_tul_df['Calibration Method'] = tel_tul_df['Calibration Method'].replace({
    'HD': 'Historical Default',
    'IR': 'Isotonic Regression'
    })
    print("📊 TEL and TUL per pipeline:\n", tel_tul_df)
    # Zet Seaborn-stijl
    plt.figure(figsize=(12,6))
    sns.set(style="whitegrid")
    # Scatterplot
    ax = sns.scatterplot(
        data=tel_tul_df, x="TEL", y="TUL", hue="Calibration Method", style="Rating Method", s=100)
    # Optioneel: labels bij punten tonen
    plt.title("TEL vs TUL for Historical Default and Isotonic Regression")
    plt.xlabel("TEL")
    plt.ylabel("TUL")
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    save_plot(dir, 'TEL vs TUL for all Rating and Calibration Combinations (HD and IR)')
    plt.close()
    # plt.show()

        # Dataframe met method Cal_PD per calibratie en rating methode
    cal_pd_data = []
    calib_methods = ['HDcalib', 'IRcalib']
    rating_methods = ['ERR', 'clustering', 'RT', 'equalcount', 'equalwidth', 'equalpd', 'objective']
    for calib in calib_methods:
        for rating_m in rating_methods:
            var_name = f"{calib}_summary_{rating_m}"
            try:
                summary_df = eval(var_name)
                if summary_df is not None and 'method Cal_PD' in summary_df.columns and 'Rating' in summary_df.columns:
                    for rating, cal_pd in zip(summary_df['Rating'], summary_df['method Cal_PD']):
                        cal_pd_data.append({
                            'Pipeline': f"{rating_m}_{calib}",
                            'Rating': rating,
                            'Calibrated PD': cal_pd
                        })
            except NameError:
                continue  # Sommige combinaties bestaan niet (zoals SS bij non-PD_calib)
    cal_pd_df = pd.DataFrame(cal_pd_data)
    # Split pipeline string
    cal_pd_df['Rating Method'] = cal_pd_df['Pipeline'].str.split('_').str[0]
    cal_pd_df['Calibration Method'] = cal_pd_df['Pipeline'].str.split('_').str[1].str.replace('calib', '')
    cal_pd_df.sort_values(['Rating Method', 'Calibration Method'], inplace=True)
    cal_pd_df['Combination'] = cal_pd_df['Rating Method'] + ' + ' + cal_pd_df['Calibration Method']
    cal_pd_df['Rating Method'] = cal_pd_df['Rating Method'].replace({
    'ERR': 'Exponential Risk Relationship',
    'RT': 'Regression Tree',
    'clustering': 'Clustering',
    'equalcount': 'Equal Count',
    'equalwidth': 'Equal Width',
    'equalpd': 'Equal PD',
    'objective': 'Objective Function'
    })
    cal_pd_df['Calibration Method'] = cal_pd_df['Calibration Method'].replace({
    'HD': 'Historical Default',
    'IR': 'Isotonic Regression'
    })
    print("📊 Calibrated PD per pipeline:\n", cal_pd_df)
    # Zet Seaborn-stijl
    plt.figure(figsize=(10,6))
    sns.set(style="whitegrid")
    # Scatterplot
    ax = sns.lineplot(data=cal_pd_df, x="Rating", y="Calibrated PD", hue="Rating Method", style="Calibration Method", markers=True, dashes=False, markersize=8)
    # Optioneel: labels bij punten tonen
    plt.title("Calibrated PD per rating for HD and IR Calibration")
    plt.xlabel("Rating")
    plt.ylabel("Calibrated PD")
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    save_plot(dir, 'Calibrated PD per rating for HD and IR Calibration')
    plt.close()
    # plt.show()