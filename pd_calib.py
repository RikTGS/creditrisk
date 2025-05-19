from logreg import evaluate_model
from scipy.stats import norm
import numpy as np
from scipy.optimize import minimize
from sklearn.metrics import roc_auc_score
import pandas as pd
from sklearn.isotonic import IsotonicRegression
from scipy.optimize import minimize_scalar
import matplotlib.pyplot as plt
from plot_save import save_plot
from sklearn.linear_model import LogisticRegression
import numpy as np



def pd_calibration(df, rating_col, prediction_col):
    # Calculate the average prediction per rating
    rating_avg = df.groupby(rating_col)[prediction_col].transform('mean')
    # Add the PD estimate column to the dataframe
    df['pooled pd'] = rating_avg
    # rating_avg per rating
    pd_series = (
        df.groupby(rating_col)[prediction_col]
        .mean()
        .sort_index())
    
    return pd_series

#AVG pd per group obv echte data

def PD_cal(df, preprocess, logreg_model, rating_df, train_num_mean, train_cat_mode, train_mean, train_stdev, target_var, method, thresholds):
    
    df = preprocess(df, train_num_mean, train_cat_mode, train_mean, train_stdev)
    # Evaluate the model with whole set
    df_pred  = evaluate_model(logreg_model, df, target_var)
    df['Prediction'] = df_pred
    # Print the number of rows that each rating contains
    rating_counts = df['Rating'].value_counts()
    rating_counts = rating_counts.sort_index()
    #print("df rating count: \n", rating_counts)
    # Calculate the sum of the target_var column per rating
    target_var_sum = df.groupby('Rating')[target_var].sum()
    #print("df target 1 counts: \n", target_var_sum)
    # Divide the sum of target_var by the number of rows per rating
    pd_per_rating = target_var_sum / rating_counts
    #print("df pd per rating: \n",pd_per_rating)
    # Add the "Cal_PD" column to rating_df
    rating_df['Cal_PD'] = rating_df['Rating'].map(pd_per_rating)
    
    return rating_df

    
def df_prediction(df, preprocess, logreg_model, train_num_mean, train_cat_mode, train_mean, train_stdev, target_var):
    df = preprocess(df, train_num_mean, train_cat_mode, train_mean, train_stdev)
    # Evaluate the model with whole set
    df_pred  = evaluate_model(logreg_model, df, target_var)
    df['Prediction'] = df_pred
    return df

def HD_calib(df_new, rating_df, target_var):
    # number of rows that each rating contains
    rating_counts = df_new['Rating'].value_counts()
    rating_counts = rating_counts.sort_index()
    #print("df rating count: \n", rating_counts)
    # Calculate the sum of the target_var column per rating
    target_var_sum = df_new.groupby('Rating')[target_var].sum()
    #print("df target 1 counts: \n", target_var_sum)
    # Divide the sum of target_var by the number of rows per rating
    pd_per_rating = target_var_sum / rating_counts
    #print("df pd per rating: \n",pd_per_rating)
    # Add the "Cal_PD" column to rating_df
    rating_df['HD_Cal_PD'] = rating_df['Rating'].map(pd_per_rating)
    # Cap Cal_PD at a minimum value of 0.0003: Basel floor
    rating_df['HD_Cal_PD'] = rating_df['HD_Cal_PD'].clip(lower=0.0003)
    rating_df['method Cal_PD'] = rating_df['HD_Cal_PD']
    return rating_df


def TTC_calib(df_new, rating_df, target_var):
    # Calculate the pooled PD for each rating
    pooled_pd_test = rating_df.groupby('Rating')['Prediction'].mean()
    pooled_pd_df = df_new.groupby('Rating')['Prediction'].mean()
    # Map the pooled PD to the rating_df
    rating_df['pooled_pd'] = rating_df['Rating'].map(pooled_pd_test)
    df_new['pooled_pd'] = df_new['Rating'].map(pooled_pd_df)
    # calculate DD PIT
    rating_df['DD_PIT'] = norm.ppf(1 - rating_df['pooled_pd'])
    df_new['DD_PIT'] = norm.ppf(1 - df_new['pooled_pd'])
    # Calculate the mean of DD_PIT
    mean_dd_pit = rating_df['DD_PIT'].mean()
    mean_dd_pit_df = df_new['DD_PIT'].mean()
    std_dd_pit = df_new['DD_PIT'].std()
    # Calculate Z
    z = (mean_dd_pit - mean_dd_pit_df) / std_dd_pit # Interpretation: 𝑍=0.787→Credit conditions in Sector A in March 2025 are better than normal. 
                                                    # So, when converting PIT to TTC PDs, you would adjust downward the DD to remove this favorable effect.
    # Remove the Cyclical Component to Obtain TTC DD
    rating_df['DD_TTC'] = rating_df['DD_PIT'] - z
    # Convert TTC DD Back to TTC PD
    rating_df['method Cal_PD'] = 1 - norm.cdf(rating_df['DD_TTC'])
    # Cap Cal_PD at a minimum value of 0.0003: Basel floor
    rating_df['method Cal_PD'] = rating_df['method Cal_PD'].clip(lower=0.0003)

    return rating_df


def QMM_calibration(df_new, rating_df, target_var):
    """
    QMM-calibratie met:
    - normalisatie van predictions
    - data-gedreven startwaarden voor a en b (via logistic regression)
    - L-BFGS-B optimalisatie met bounds
    """
    # 1. Normaliseer prediction
    df_new['Prediction_norm'] = (df_new['Prediction'] - df_new['Prediction'].mean()) / df_new['Prediction'].std()
    # 2. Targetwaarden bepalen
    target_mean_pd = df_new[target_var].mean()
    auc = roc_auc_score(df_new[target_var], df_new['Prediction'])
    target_gini = 2 * auc - 1
    # 3. Datagedreven init: logistische regressie op Prediction_norm
    X = df_new[['Prediction_norm']]
    y = df_new[target_var]
    base_logreg = LogisticRegression()
    base_logreg.fit(X, y)
    a_0 = base_logreg.intercept_[0]
    b_0 = base_logreg.coef_[0][0]
    # 4. Sigmoidfunctie
    def pd_curve(x, a, b):
        return 1 / (1 + np.exp(-(a + b * x)))
    # 5. Objective-functie
    def objective(params):
        a, b = params
        pd_est = pd_curve(df_new['Prediction_norm'], a, b)
        mean_pd = pd_est.mean()
        gini = 2 * roc_auc_score(df_new[target_var], pd_est) - 1
        return (mean_pd - target_mean_pd) ** 2 + (gini - target_gini) ** 2
    # 6. Optimalisatie
    res = minimize(objective, x0=[a_0, b_0], bounds=[(-10, 10), (0.1, 5)], method='L-BFGS-B')
    a_opt, b_opt = res.x
    # 7. Toepassen van calibratie
    df_new['Cal_PD'] = pd_curve(df_new['Prediction_norm'], a_opt, b_opt)
    # 8. Gemiddelde PD per rating
    avg_cal_pd_per_rating = df_new.groupby('Rating')['Cal_PD'].mean()
    rating_df['method Cal_PD'] = rating_df['Rating'].map(avg_cal_pd_per_rating)
    rating_df['method Cal_PD'] = rating_df['method Cal_PD'].clip(lower=0.0003)

    return rating_df


def SLR_calib(df_new, rating_df, target_var, plot_lambda=True):
    """
    Voert Scaled Likelihood Ratio calibratie uit op pooled PDs per rating.
    Optioneel: toont gevoeligheidsplot voor verschillende lambda-waarden.
    
    Parameters:
    - df_new: DataFrame met predicties en targets.
    - rating_df: DataFrame met ratinggroepen.
    - target_var: naam van targetkolom (bijv. 'SeriousDlqin2yrs').
    - plot_lambda: boolean, als True wordt gevoeligheidsplot gegenereerd.

    Returns:
    - rating_df met gekalibreerde 'method Cal_PD'.
    """

    # 1. Bepaal de pooled PD en observed default rate per rating
    pooled_pd = df_new.groupby('Rating')['Prediction'].mean()
    observed_pd = df_new.groupby('Rating')[target_var].mean()

    # 2. Verwijder eventuele lege rijen (kan gebeuren bij zeldzame ratings)
    df_slr = pd.DataFrame({
        'pooled_pd': pooled_pd,
        'observed_pd': observed_pd
    }).dropna()

    # 3. Bepaal likelihood ratio’s
    lr_raw = df_slr['pooled_pd'] / (1 - df_slr['pooled_pd'])
    lr_obs = df_slr['observed_pd'] / (1 - df_slr['observed_pd'])

    # 4. Fit schaalfactor lambda via optimalisatie
    def objective(lmbda):
        slr = lmbda * lr_raw
        pd_cal = slr / (1 + slr)
        return np.sum((pd_cal - df_slr['observed_pd']) ** 2)

    res = minimize_scalar(objective, bounds=(0.01, 100), method='bounded')
    lmbda_opt = res.x

    # 5. Toepassen van gekalibreerde PD's
    rating_df['pooled_pd'] = rating_df['Rating'].map(pooled_pd)
    lr = rating_df['pooled_pd'] / (1 - rating_df['pooled_pd'])
    slr_scaled = lmbda_opt * lr
    rating_df['method Cal_PD'] = slr_scaled / (1 + slr_scaled)
    rating_df['method Cal_PD'] = rating_df['method Cal_PD'].clip(lower=0.0003)
    rating_df.drop(columns=['pooled_pd'], inplace=True)

    # 6. Plot gevoeligheid voor lambda-waarden (optioneel)
    if plot_lambda:
        lambdas = np.linspace(0.1, 3.0, 5)
        ratings = df_slr.index.tolist()

        plt.figure(figsize=(10, 6))
        for lmbda in lambdas:
            scaled = lmbda * lr_raw
            pd_scaled = scaled / (1 + scaled)
            plt.plot(ratings, pd_scaled, marker='o', label=f'λ = {lmbda:.2f}')

        # Toon originele waarden ter referentie
        plt.plot(ratings, df_slr['pooled_pd'], linestyle='--', color='gray', label='Original pooled PD')
        plt.plot(ratings, df_slr['observed_pd'], linestyle='--', color='black', label='Observed PD')

        # Markeer geoptimaliseerde lambda
        scaled_opt = lmbda_opt * lr_raw
        pd_opt = scaled_opt / (1 + scaled_opt)
        plt.plot(ratings, pd_opt, marker='o', linestyle='-', linewidth=3, color='red', label=f'Optimized λ = {lmbda_opt:.2f}')

        plt.title('Scaled Likelihood Ratio: Effect of λ on calibrated PD per rating (Exponential Risk Relationship)')
        plt.xlabel('Rating')
        plt.ylabel('Calibrated PD')
        plt.grid(True, linestyle='--', alpha=0.6)
        plt.legend(title='Lambda', bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.tight_layout()
        dir = "lambda"
        save_plot(dir,'SLR_lambda_sensitivity')
        plt.show()

    return rating_df



def IR_calib(df_new, rating_df, target_var):
    """
    Performs Isotonic Regression PD calibration.
    zorgt voor mooi monotonistisch verloop van de PD curve, wanneer al monotoon, weinig toe te voegen

    Parameters:
    - df_new: DataFrame with predictions and true defaults (Prediction column: 'Prediction')
    - rating_df: DataFrame with Ratings assigned (based on test set)
    - target_var: name of the default indicator column (e.g., 'SeriousDlqin2yrs')

    Returns:
    - rating_df with calibrated 'method Cal_PD' column
    """
    # Step 1: Calculate average prediction and observed default rate per rating
    avg_pred_per_rating = df_new.groupby('Rating')['Prediction'].mean()
    obs_default_rate_per_rating = df_new.groupby('Rating')[target_var].mean()

    # Step 2: Fit isotonic regression
    iso_reg = IsotonicRegression(out_of_bounds='clip')
    iso_reg.fit(avg_pred_per_rating.values, obs_default_rate_per_rating.values)

    # Step 3: Apply calibrated PDs to ratings
    calibrated_pds = iso_reg.predict(avg_pred_per_rating.values)

    # Step 4: Map calibrated PDs back to the rating_df
    mapping_calibrated_pd = dict(zip(avg_pred_per_rating.index, calibrated_pds))
    rating_df['method Cal_PD'] = rating_df['Rating'].map(mapping_calibrated_pd)
    # Cap Cal_PD at a minimum value of 0.0003: Basel floor
    rating_df['method Cal_PD'] = rating_df['method Cal_PD'].clip(lower=0.0003)

    return rating_df