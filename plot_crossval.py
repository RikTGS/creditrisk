import os
import pandas as pd
from plot_save import save_plot
import matplotlib.pyplot as plt
from scipy.stats import friedmanchisquare, wilcoxon
import itertools
import pandas as pd
import numpy as np
import itertools
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats as stats
import statsmodels.api as sm
from statsmodels.formula.api import ols
from statsmodels.stats.multicomp import pairwise_tukeyhsd

ERR_HD = None
ERR_TTC = None
ERR_QMM = None
ERR_SLR = None
ERR_IR = None
clustering_HD = None
clustering_TTC = None
clustering_QMM = None
clustering_SLR = None
clustering_IR = None
RT_HD = None
RT_HD = None
RT_QMM = None
RT_SLR = None
RT_IR = None
SS_HD = None
equalcount_HD = None
equalcount_TTC = None
equalcount_QMM = None
equalcount_SLR = None
equalcount_IR = None
equalwidth_HD = None
equalwidth_TTC = None
equalwidth_QMM = None
equalwidth_SLR = None
equalwidth_IR = None
equalpd_HD = None
equalpd_TTC = None
equalpd_QMM = None
equalpd_SLR = None
equalpd_IR = None
objective_HD = None


output_dir = f'C:/Users/rikte/VS Code Python Projects/thesis_riktgs/output/cross_val'
# Loop over alle bestanden en maak variabelen aan
for file in os.listdir(output_dir):
    file_path = os.path.join(output_dir, file)
    var_name, ext = os.path.splitext(file)  # 'ERR_HD', '.csv' of '.txt'

    if ext == '.csv':
        try:
            df = pd.read_csv(file_path)
            globals()[f"{var_name}"] = df
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
                globals()[f"{var_name}"]  = val
        except Exception as e:
            print(f"⚠️ Fout bij inladen van {var_name}: {e}")
dir = "cross_val"
# Selecteer de metric kolommen
metrics = ["TEL", "TUL", "Brier"]

# Verzamel alle geladen DataFrames met juiste kolommen
all_methods = {name: df for name, df in globals().items()
               if isinstance(df, pd.DataFrame) and all(metric in df.columns for metric in metrics)}
            # Verwijder 'df' uit all_methods als het aanwezig is
if "df" in all_methods:
    del all_methods["df"]
if "SS_HD" in all_methods:
    del all_methods["SS_HD"]

print(f"\n📊 Aantal methodes gevonden: {len(all_methods)}")
print("Methodes:", list(all_methods.keys()))

# Stap 1: Gemiddelde en standaardafwijking
print("\n📈 Gemiddelde en standaardafwijking per metric:")
summary_stats = pd.DataFrame()
for method_name, df in all_methods.items():
    for metric in metrics:
        mean = df[metric].mean()
        std = df[metric].std()
        summary_stats.loc[method_name, f"{metric} (mean)"] = mean
        summary_stats.loc[method_name, f"{metric} (std)"] = std

pd.set_option('display.float_format', lambda x: f'{x:0.6f}')
# Sorteer summary_stats op index
summary_stats = summary_stats.sort_index()

x = summary_stats["Brier (mean)"]
y = summary_stats["Brier (std)"]
plt.figure(figsize=(6.4, 4.8))
plt.scatter(x, y)
plt.legend()
plt.xlabel("Mean")
plt.ylabel("Std")
plt.title("Brier Score")
plt.grid(True)
save_plot(dir, "Brier_Score_Mean_vs_Std")
# plt.show()

summary_stats.reset_index(inplace=True)
summary_stats.rename(columns={'index': 'Pipeline'}, inplace=True) 
summary_stats['Calibration_Method'] = summary_stats['Pipeline'].apply(lambda x: x.split('_')[-1])
summary_stats['Rating_Method'] = summary_stats['Pipeline'].apply(lambda x: '_'.join(x.split('_')[:-1]))
summary_stats['Rating_Method'] = summary_stats['Rating_Method'].replace({
    'clustering': 'Clustering',
    'equalcount': 'Equal Count',
    'equalwidth': 'Equal Width',
    'equalpd': 'Equal PD',
    'objective': 'Objective Function'
})

print(summary_stats)

# ================= ANALYSE PER CALIBRATION METHOD =================

for metric in ['TEL (mean)', 'TUL (mean)']:
    print(f"\n== {metric} per Calibration Method ==")

    for group in summary_stats['Calibration_Method'].unique():
        data = summary_stats[summary_stats['Calibration_Method'] == group][metric]
        stat, p = stats.shapiro(data)
        print(f"Shapiro-Wilk for {group}: p = {p:.6f}")
    
    grouped = [summary_stats[summary_stats['Calibration_Method'] == group][metric] for group in summary_stats['Calibration_Method'].unique()]
    stat, p = stats.levene(*grouped)
    print(f"Levene’s test p = {p:.6f}")

    if all(stats.shapiro(group)[1] > 0.05 for group in grouped) and p > 0.05:
        print("=> Use ANOVA:")
        model = ols(f'Q("{metric}") ~ C(Calibration_Method)', data=summary_stats).fit()
        anova_table = sm.stats.anova_lm(model, typ=2)
        print(anova_table)

        if anova_table['PR(>F)'][0] < 0.05:
            tukey = pairwise_tukeyhsd(endog=summary_stats[metric], groups=summary_stats['Calibration_Method'], alpha=0.05)
            print(tukey)
    else:
        print("=> Use Kruskal-Wallis:")
        stat, p = stats.kruskal(*grouped)
        print(f"Kruskal-Wallis statistic = {stat:.6f}, p = {p:.6f}")
    
    # Plot
    plt.figure()
    sns.boxplot(data=summary_stats, x='Calibration_Method', y=metric)
    plt.title(f"{metric} per Calibration Method")
    plt.xlabel("Calibration Method")
    plt.ylabel(metric)
    plt.grid(True)
    plt.xticks(rotation=45)
    plt.tight_layout()
    save_plot(dir, f"{metric}_per_Calibration_Method")
    plt.close()
    # plt.show()

# ================= ANALYSE PER RATING METHOD =================

for metric in ['TEL (mean)', 'TUL (mean)']:
    print(f"\n== {metric} per Rating Method ==")

    for group in summary_stats['Rating_Method'].unique():
        data = summary_stats[summary_stats['Rating_Method'] == group][metric]
        stat, p = stats.shapiro(data)
        print(f"Shapiro-Wilk for {group}: p = {p:.6f}")

    grouped = [summary_stats[summary_stats['Rating_Method'] == group][metric] for group in summary_stats['Rating_Method'].unique()]
    stat, p = stats.levene(*grouped)
    print(f"Levene’s test p = {p:.6f}")

    if all(stats.shapiro(group)[1] > 0.05 for group in grouped) and p > 0.05:
        print("=> Use ANOVA:")
        model = ols(f'Q("{metric}") ~ C(Rating_Method)', data=summary_stats).fit()
        anova_table = sm.stats.anova_lm(model, typ=2)
        print(anova_table)

        if anova_table['PR(>F)'][0] < 0.05:
            tukey = pairwise_tukeyhsd(endog=summary_stats[metric], groups=summary_stats['Rating_Method'], alpha=0.05)
            print(tukey)
    else:
        print("=> Use Kruskal-Wallis:")
        stat, p = stats.kruskal(*grouped)
        print(f"Kruskal-Wallis statistic = {stat:.6f}, p = {p:.6f}")

    # Plot
    plt.figure()
    sns.boxplot(data=summary_stats, x='Rating_Method', y=metric)
    plt.title(f"{metric} per Rating Method")
    plt.xlabel("Rating Method")
    plt.ylabel(metric)
    plt.grid(True)
    plt.xticks(rotation=45)
    plt.tight_layout()
    save_plot(dir, f"{metric}_per_Rating_Method")
    plt.close()
    # plt.show()


# Analyse met alle folds ipv mean
from scipy.stats import friedmanchisquare
# Zorg dat 'metric' exact zo heet in je DataFrames
metrics = ["TEL", "TUL", "Brier"]
group = ["Rating", "Calibration"]  # of "Rating", "Calibration"
for group_type in group:
    print(f"\n== {group_type} ==")
    for metric in metrics:
        print(f"\n== {metric} per {group_type} ==")

        # Maak dict van: {groepnaam: [df1, df2, df3...]} op basis van naam
        groups = {}
        for name, df in all_methods.items():
            group = name.split('_')[-1] if group_type == "Calibration" else '_'.join(name.split('_')[:-1])
            if group not in groups:
                groups[group] = []
            groups[group].append(df[metric].values)  # voeg array van 10 folds toe

        # Filter alleen groepen met voldoende methodes (>=3)
        groups = {k: v for k, v in groups.items() if len(v) >= 3}

        print(f"\n== Friedman-test op '{metric}' per {group_type}-method ==")
        for group_name, values_list in groups.items():
            try:
                # Zorg dat alle sets even lang zijn en transponeer
                values_matrix = pd.DataFrame(values_list).T
                stat, p = friedmanchisquare(*[values_matrix[col] for col in values_matrix.columns])
                print(f"{group_name}: p = {p:.2e} (stat = {stat:.6f})")
            except Exception as e:
                print(f"{group_name}: fout bij testen: {e}")



# opnieuw: 
import pandas as pd
import scipy.stats as stats
import matplotlib.pyplot as plt
import seaborn as sns
import scikit_posthocs as sp
from statsmodels.stats.multitest import multipletests

for metric in metrics:
    print(f"\n== {metric} per Rating Method ==")
    # Verzamel alle TEL-scores per ratingmethode
    rating_scores = []

    for name, df in all_methods.items():
        rating_method = '_'.join(name.split('_')[:-1])
        scores = df[metric].values
        for score in scores:
            rating_scores.append({'Rating_Method': rating_method, metric: score})

    df_ratings = pd.DataFrame(rating_scores) # df van 6 verschillende ratingmethods met 5 soorten calib en 10 folds -> 300 rijen

    # Check groepen
    groups = [group[metric].values for name, group in df_ratings.groupby('Rating_Method')]
    labels = df_ratings['Rating_Method'].unique()

    # ✅ Kruskal-Wallis toets
    stat, p = stats.kruskal(*groups)
    print(f"\n📊 Kruskal-Wallis test op '{metric}' tussen ratingmethodes:")
    print(f"Statistiek = {stat:.4f}, p-waarde = {p:.2e}")

    # ➕ Post-hoc test als significant
    if p < 0.05:
        print("\n✅ Significante verschillen gevonden — voer Dunn post-hoc test uit:")
        
        # Dunn-test met Bonferroni correctie
        posthoc = sp.posthoc_dunn(df_ratings, val_col=metric, group_col='Rating_Method', p_adjust='bonferroni')
        print("\n🎯 Post-hoc Dunn-test (Bonferroni):")
        print(posthoc.round(6))

        # Plot met annotaties
        plt.figure(figsize=(10, 6))
        sns.boxplot(data=df_ratings, x='Rating_Method', y=metric)
        plt.title(f"{metric} per Rating Method\n(Kruskal-Wallis p = {p:.2e})")
        plt.grid(True)
        plt.xticks(rotation=45)
        plt.tight_layout()
        # plt.show()

        # Optioneel: toon welke paren significant zijn
        print("\n🔍 Significante verschillen tussen methodes:")
        for i in range(len(posthoc)):
            for j in range(i + 1, len(posthoc)):
                pval = posthoc.iloc[i, j]
                if pval < 0.05:
                    print(f"  {posthoc.index[i]} vs {posthoc.columns[j]}: p = {pval:.2e}")
    else:
        print("❌ Geen significant verschil tussen ratingmethodes — post-hoc test niet nodig.")

# calibration
# Kies de metric: "TEL", "TUL" of "Brier"

for metric in metrics:
    print(f"\n== {metric} per Calibration Method ==")
    # Verzamel alle scores per Calibration Method (HD, QMM, SLR, ...)
    calib_scores = []

    for name, df in all_methods.items():
        calib_method = name.split('_')[-1]
        scores = df[metric].values
        for score in scores:
            calib_scores.append({'Calibration_Method': calib_method, metric: score})

    df_calibs = pd.DataFrame(calib_scores)

    # Check groepen
    groups = [group[metric].values for name, group in df_calibs.groupby('Calibration_Method')]
    labels = df_calibs['Calibration_Method'].unique()

    # ✅ Kruskal-Wallis toets
    stat, p = stats.kruskal(*groups)
    print(f"\n📊 Kruskal-Wallis test op '{metric}' tussen calibratiemethodes:")
    print(f"Statistiek = {stat:.4f}, p-waarde = {p:.2e}")

    # ➕ Post-hoc test als significant
    if p < 0.05:
        print("\n✅ Significante verschillen gevonden — voer Dunn post-hoc test uit:")

        # Dunn-test met Bonferroni correctie
        posthoc = sp.posthoc_dunn(df_calibs, val_col=metric, group_col='Calibration_Method', p_adjust='bonferroni')
        print("\n🎯 Post-hoc Dunn-test (Bonferroni):")
        print(posthoc.round(6))

        # Plot met annotaties
        plt.figure(figsize=(8, 6))
        sns.boxplot(data=df_calibs, x='Calibration_Method', y=metric)
        plt.title(f"{metric} per Calibration Method\n(Kruskal-Wallis p = {p:.2e})")
        plt.grid(True)
        plt.tight_layout()
        # plt.show()

        # Optioneel: toon significante paren
        print("\n🔍 Significante verschillen tussen calibration methods:")
        for i in range(len(posthoc)):
            for j in range(i + 1, len(posthoc)):
                pval = posthoc.iloc[i, j]
                if pval < 0.05:
                    print(f"  {posthoc.index[i]} vs {posthoc.columns[j]}: p = {pval:.2e}")
    else:
        print("❌ Geen significant verschil tussen calibration methods — post-hoc test niet nodig.")