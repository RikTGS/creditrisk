import os
import pandas as pd
import pandas as pd
from logreg import logreg_model  
from gmsc_preprocessing import train_preprocess_GMSC, test_preprocess_GMSC
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
import time


numbers = [2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14]
# numbers = [3, 7, 14]
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
                globals()[f"{var_name}_{num}"] = df
            except Exception as e:
                print(f"⚠️ Fout bij inladen van {var_name}_{num}: {e}")

        elif ext == '.txt':
            try:
                with open(file_path, 'r') as f:
                    val = f.read().strip()
                    # Probeer te casten naar float of int
                    try:
                        val = float(val) if '.' in val else int(val)
                    except:
                        pass
                    globals()[f"{var_name}_{num}"]  = val
            except Exception as e:
                print(f"⚠️ Fout bij inladen van {var_name}_{num}: {e}")
dir = "num_rating"

target_var = 'SeriousDlqin2yrs'
# GIVE ME SOME CREDIT DATASET

x_min = test_pred_7['Prediction'].quantile(0.05)   # bv. 5e percentiel
x_max = test_pred_7['Prediction'].quantile(0.95)   # bv. 95e percentiel

markers = ['o', 's', 'x', 'D', '^', 'v', '*', '+', '1', '2', '3', '4']

# print(HDcalib_summary_clustering_9)
# print(HDcalib_df_clustering_9.head())
# print(HDcalib_TEL_clustering_9)
# print(HDcalib_TUL_clustering_9)

# Brier score berekenen voor elke pipeline per aantal ratings
y_test = test_7[target_var]
y_test = y_test.to_frame(name=target_var) if isinstance(y_test, pd.Series) else y_test
y_test = y_test.reset_index(drop=True)
calib_methods = ['HD', 'TTC', 'QMM', 'SLR', 'IR']
rating_methods = ['ERR', 'clustering', 'RT', 'equalcount', 'equalwidth', 'equalpd', 'objective']
brier_records = []

# Loop over combinaties
for num in numbers:
    for rating in rating_methods:
        for calib in calib_methods:
            var_name = f"{calib}calib_df_{rating}_{num}"
            try:
                df = eval(var_name)
                if df is not None and 'method Cal_PD' in df.columns:
                    df = df.reset_index(drop=True)
                    if len(df) == len(y_test):
                        combined = pd.concat([y_test, df['method Cal_PD']], axis=1)
                        brier = brier_score_loss(combined[target_var], combined['method Cal_PD'])
                        brier_records.append({
                            'rating_method': rating,
                            'calibration_method': calib,
                            'num_ratings': num,
                            'brier_score': brier
                        })
            except NameError:
                continue  # combinatie bestaat niet

# DataFrame aanmaken
brier_df = pd.DataFrame(brier_records)
  # df met aantal ratings in rijen, in kolom de rating method, in volgende kolom de calibration method en dan TEL, TUL, Brier score 

all_data = []
rating_methods = ['ERR', 'clustering', 'RT', 'equalcount', 'equalwidth', 'equalpd', 'objective']
calibration_methods = ['HD', 'TTC', 'QMM', 'SLR', 'IR']
for num in numbers:
    for rating in rating_methods:
        for calib in calibration_methods:
            try:
                # Append to the main DataFrame
                all_data.append({
                    'num_ratings': num,
                    'Rating Method': rating,
                    'Calibration Method': calib,
                    'TEL': globals().get(f"{calib}calib_TEL_{rating}_{num}", None),
                    'TEL (norm)': globals().get(f"{calib}calib_TEL_{rating}_{num}", None)/len(y_test) if globals().get(f"{calib}calib_TEL_{rating}_{num}", None) is not None else None,
                    'TUL': globals().get(f"{calib}calib_TUL_{rating}_{num}", None),
                    'TUL (norm)': globals().get(f"{calib}calib_TUL_{rating}_{num}", None)/len(y_test) if globals().get(f"{calib}calib_TUL_{rating}_{num}", None) is not None else None,
                    'Brier Score': brier_df.loc[(brier_df['num_ratings'] == num) & (brier_df['rating_method'] == rating) & (brier_df['calibration_method'] == calib), 'brier_score'].values[0] if not brier_df[(brier_df['num_ratings'] == num) & (brier_df['rating_method'] == rating) & (brier_df['calibration_method'] == calib)].empty else None
                })

            except NameError:
                continue  # Sommige combinaties bestaan niet (zoals SS bij non-PD_calib)
all_df = pd.DataFrame(all_data) # df met kolommen: num_ratings, rating method, calibration method, TEL, TUL
all_df['Rating Method'] = all_df['Rating Method'].replace({
'ERR': 'Exponential Risk Relationship',
'RT': 'Regression Tree',
'clustering': 'Clustering',
'equalcount': 'Equal Count',
'equalwidth': 'Equal Width',
'equalpd': 'Equal PD',
'objective': 'Objective Function'
})
all_df['Calibration Method'] = all_df['Calibration Method'].replace({
'HD': 'Historical Default',
'TTC': 'Through-the-Cycle',
'QMM': 'Quasi-Moment Matching',
'SLR': 'Scaled Likelihood Ratio',
'IR': 'Isotonic Regression'
})
print(all_df.head())
sorted_brier = all_df.sort_values(by='Brier Score', ascending=True)
print(sorted_brier.head(10))


# Find the lowest Brier Score for each Rating Method
lowest_brier_by_rating = all_df.loc[all_df.groupby('Rating Method')['Brier Score'].idxmin()]
lowest_brier_by_rating = lowest_brier_by_rating.sort_values(by='Brier Score', ascending=True)
print("Lowest Brier Score for each Rating Method:")
print(lowest_brier_by_rating)

# Find the lowest Brier Score for each Calibration Method
lowest_brier_by_calibration = all_df.loc[all_df.groupby('Calibration Method')['Brier Score'].idxmin()]
lowest_brier_by_calibration = lowest_brier_by_calibration.sort_values(by='Brier Score', ascending=True)
print("Lowest Brier Score for each Calibration Method:")
print(lowest_brier_by_calibration)

# sensitiviteit num ratings: x-as (aantal ratings), y-as (metriek), values = rating method, per calib methode een grafiek
    # metriek = calibrated PD: moeilijk aangezien aantal ratings verschillen
    # metriek = TEL: 
# Plot per Calibration Method
for calib in calibration_methods:
    calib_df = all_df[all_df['Calibration Method'] == calib]
    plt.figure()
    sns.lineplot(data=calib_df, x='num_ratings', y='TEL', hue='Rating Method', marker='o')
    plt.title(f"Sensitivity Analysis Number of Ratings - TEL - Calibration Method: {calib}")
    plt.xlabel("Number of ratings")
    plt.ylabel("TEL")
    plt.grid(True)
    plt.legend(title="Rating Method", bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.xticks(sorted(calib_df['num_ratings'].unique()))
    # save_plot(dir, f"TEL_Sensitivity_NumRatings_{calib}")
    plt.close()
    # plt.show()
# Plot 5 graphs te samen
g = sns.FacetGrid(all_df, col="Calibration Method", col_wrap=3, height=4, sharey=True)
g.map_dataframe(sns.lineplot, x="num_ratings", y="TEL", hue="Rating Method", marker='o')
g.set_axis_labels("Number of ratings", "TEL")
g.set_titles(col_template="{col_name}")
plt.subplots_adjust(top=0.9)
plt.legend(title="Rating Method", bbox_to_anchor=(1.05, 1), loc='upper left')
g.fig.suptitle("Total Expected Loss (TEL) Sensitivity to Rating Granularity")
plt.xticks(sorted(all_df['num_ratings'].unique()))
save_plot(dir, "TEL_per_rating_method")
plt.close()
# plt.show()


    # metriek = TUL
# Plot per Calibration Method
for calib in calibration_methods:
    calib_df = all_df[all_df['Calibration Method'] == calib]
    plt.figure()
    sns.lineplot(data=calib_df, x='num_ratings', y='TUL', hue='Rating Method', marker='o')
    plt.title(f"Sensitivity Analysis Number of Ratings - TUL - Calibration Method: {calib}")
    plt.xlabel("Number of ratings")
    plt.ylabel("TUL")
    plt.grid(True)
    plt.legend(title="Rating Method", bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.xticks(sorted(calib_df['num_ratings'].unique()))
    # save_plot(dir, f"TUL_Sensitivity_NumRatings_{calib}")
    plt.close()
    # plt.show()
# Plot 5 graphs te samen
g = sns.FacetGrid(all_df, col="Calibration Method", col_wrap=3, height=4, sharey=True)
g.map_dataframe(sns.lineplot, x="num_ratings", y="TUL", hue="Rating Method", marker='o')
g.set_axis_labels("number of ratings", "TUL")
g.set_titles(col_template="{col_name}")
plt.subplots_adjust(top=0.9)
plt.legend(title="Rating Method", bbox_to_anchor=(1.05, 1), loc='upper left')
g.fig.suptitle("Total Unexpected Loss (TUL) Sensitivity to Rating Granularity")
plt.xticks(sorted(all_df['num_ratings'].unique()))
save_plot(dir, "TUL_per_rating_method")
plt.close()
# plt.show()

    # metriek = Brier Score
# Plot per Calibration Method
for calib in calibration_methods:
    calib_df = all_df[all_df['Calibration Method'] == calib]
    plt.figure()
    sns.lineplot(data=calib_df, x='num_ratings', y='Brier Score', hue='Rating Method', marker='o')
    plt.title(f"Sensitivity Analysis Number of Ratings - Brier Score - Calibration Method: {calib}")
    plt.xlabel("Number of ratings")
    plt.ylabel("Brier Score")
    plt.grid(True)
    plt.legend(title="Rating Method", bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.xticks(sorted(calib_df['num_ratings'].unique()))
    # save_plot(dir, f"Brier Score_Sensitivity_NumRatings_{calib}")
    plt.close()
    # plt.show()
# Plot 5 graphs te samen
g = sns.FacetGrid(all_df, col="Calibration Method", col_wrap=3, height=4, sharey=True)
g.map_dataframe(sns.lineplot, x="num_ratings", y="Brier Score", hue="Rating Method", marker='o')
g.set_axis_labels("Number of ratings", "Brier Score")
g.set_titles(col_template="{col_name}")
plt.subplots_adjust(top=0.9)
plt.legend(title="Rating Method", bbox_to_anchor=(1.05, 1), loc='upper left')
g.fig.suptitle("Brier Score Sensitivity to Rating Granularity")
plt.xticks(sorted(all_df['num_ratings'].unique()))
save_plot(dir, "Brier Score_per_rating_method")
plt.close()
# plt.show()












# sensitiviteit num ratings: x-as (aantal ratings), y-as (metriek), values = rating method, per calib methode een grafiek
    # metriek = calibrated PD: moeilijk aangezien aantal ratings verschillen
    # metriek = TEL: 
custom_titles = {
    "ERR": "ERR",
    "RT": "RT",
    "clustering": "Clustering",
    "equalcount": "Equal Count",
    "equalwidth": "Equal Width",
    "equalpd": "Equal PD",
    "objective": "Objective Function"
}
# Plot per Rating Method
for rating in rating_methods:
    rating_df = all_df[all_df['Rating Method'] == rating]
    plt.figure()
    sns.lineplot(data=rating_df, x='num_ratings', y='TEL', hue='Calibration Method', marker='o')
    plt.title(f"Sensitivity Analysis Number of Ratings - TEL - Rating Method: {rating}")
    plt.xlabel("Number of ratings")
    plt.ylabel("TEL")
    plt.grid(True)
    plt.legend(title="Calibration Method", bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.xticks(sorted(rating_df['num_ratings'].unique()))
    # save_plot(dir, f"TEL_Calibration_Sensitivity_NumRatings_{rating}")
    plt.close()
    # plt.show()
# Plot 5 graphs te samen
g = sns.FacetGrid(all_df, col="Rating Method", col_wrap=4, height=4, sharey=True)
g.map_dataframe(sns.lineplot, x="num_ratings", y="TEL", hue="Calibration Method", marker='o')
g.set_axis_labels("Number of ratings", "TEL")
g.set_titles(col_template="{col_name}")
# for ax, col_val in zip(g.axes.flat, g.col_names):
    # ax.set_title(custom_titles.get(col_val, col_val))  # fallback naar originele naam
plt.subplots_adjust(top=0.9)
plt.legend(title="Calibration Method", bbox_to_anchor=(1.05, 1), loc='upper left')
g.fig.suptitle("Total Expected Loss (TEL) Sensitivity to Rating Granularity")
plt.xticks(sorted(all_df['num_ratings'].unique()))
save_plot(dir, "TEL_per_calibration_method")
plt.close()
# plt.show()


    # metriek = TUL
# Plot per Rating Method
for rating in rating_methods:
    rating_df = all_df[all_df['Rating Method'] == rating]
    plt.figure()
    sns.lineplot(data=rating_df, x='num_ratings', y='TUL', hue='Calibration Method', marker='o')
    plt.title(f"Sensitivity Analysis Number of Ratings - TUL - Rating Method: {rating}")
    plt.xlabel("Number of ratings")
    plt.ylabel("TUL")
    plt.grid(True)
    plt.legend(title="Calibration Method", bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.xticks(sorted(rating_df['num_ratings'].unique()))
    # save_plot(dir, f"TUL_Calibration_Sensitivity_NumRatings_{rating}")
    plt.close()
    # plt.show()
# Plot 5 graphs te samen
g = sns.FacetGrid(all_df, col="Rating Method", col_wrap=4, height=4, sharey=True)
g.map_dataframe(sns.lineplot, x="num_ratings", y="TUL", hue="Calibration Method", marker='o')
g.set_axis_labels("Number of ratings", "TUL")
g.set_titles(col_template="{col_name}")
# for ax, col_val in zip(g.axes.flat, g.col_names):
    # ax.set_title(custom_titles.get(col_val, col_val))  # fallback naar originele naam
plt.subplots_adjust(top=0.9)
plt.legend(title="Calibration Method", bbox_to_anchor=(1.05, 1), loc='upper left')
g.fig.suptitle("Total Unexpected Loss (TUL) Sensitivity to Rating Granularity")
plt.xticks(sorted(all_df['num_ratings'].unique()))
save_plot(dir, "TUL_per_calibration_method")
plt.close()
# plt.show()

    # metriek = Brier Score
# Plot per Rating Method
for rating in rating_methods:
    rating_df = all_df[all_df['Rating Method'] == rating]
    plt.figure()
    sns.lineplot(data=rating_df, x='num_ratings', y='Brier Score', hue='Calibration Method', marker='o')
    plt.title(f"Sensitivity Analysis Number of Ratings - Brier Score - Rating Method: {rating}")
    plt.xlabel("Number of ratings")
    plt.ylabel("Brier Score")
    plt.grid(True)
    plt.legend(title="Calibration Method", bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.xticks(sorted(rating_df['num_ratings'].unique()))
    # save_plot(dir, f"Brier Score_Calibration_Sensitivity_NumRatings_{rating}")
    plt.close()
    # plt.show()
# Plot 5 graphs te samen
g = sns.FacetGrid(all_df, col="Rating Method", col_wrap=4, height=4, sharey=True)
g.map_dataframe(sns.lineplot, x="num_ratings", y="Brier Score", hue="Calibration Method", marker='o')
g.set_axis_labels("Number of ratings", "Brier Score")
g.set_titles(col_template="{col_name}")
# g.fig.tight_layout()
# for ax, col_val in zip(g.axes.flat, g.col_names):
    # ax.set_title(custom_titles.get(col_val, col_val))  # fallback naar originele naam
plt.subplots_adjust(top=0.9)
plt.legend(title="Calibration Method", bbox_to_anchor=(1.05, 1), loc='upper left')
g.fig.suptitle("Brier Score Sensitivity to Rating Granularity")
plt.xticks(sorted(all_df['num_ratings'].unique()))
save_plot(dir, "Brier Score_per_calibration_method")
plt.close()
# plt.show()