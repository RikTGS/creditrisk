import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GridSearchCV

def train_logreg(df, target_var):
    features = df.drop(columns=[target_var])
    target = df[target_var]
    # Splits de data in trainings- en validationdata
    X_train, X_val, y_train, y_val = train_test_split(features, target, test_size=0.2, random_state=42)
    # Hyperparameter tuning using GridSearchCV
    param_grid = {'C': [0.01, 0.05, 0.1, 0.5, 1]}
    # No penalization because otherwise EL goes up (and because Ridge is used when features are highly correlated)
    grid_search = GridSearchCV(LogisticRegression(max_iter=10000, solver='liblinear'), param_grid, cv=5, scoring='roc_auc')
    grid_search.fit(X_train, y_train)
    logreg_model = grid_search.best_estimator_
    # Print the best hyperparameters from GridSearchCV
    print(f'Best hyperparameters: {grid_search.best_params_}')

    # Fit het model op de trainingsdata
    logreg_model.fit(X_train, y_train)

    # Maak voorspellingen op de validation data
    y_pred = logreg_model.predict(X_val)
    y_pred_proba = logreg_model.predict_proba(X_val)[:, 1]

    # Evalueer de prestaties van het model
    mse = mean_squared_error(y_val, y_pred)
    r2 = r2_score(y_val, y_pred)
    auc = roc_auc_score(y_val, y_pred_proba)
   
    #print(f'Mean Squared Error: {mse}')
    #print(f'R^2 Score: {r2}')
    #print(f'AUC: {auc}')
    
    # Retrain the model on the entire dataset (train + validation)
    logreg_model.fit(features, target)

    return logreg_model

def evaluate_model(model, df, target_var):
    X_test = df.drop(columns=[target_var])
    # Maak voorspellingen op de test data
    y_pred_proba = model.predict_proba(X_test)[:, 1]
    y_pred_proba = pd.DataFrame(y_pred_proba, columns=['Prediction'])

    return y_pred_proba

def logreg_model(df, target_var, train_preprocess, test_preprocess):
    # Drop the first column of the DataFrame because it is the same as index
    df.drop(df.columns[0], axis=1, inplace=True)
    # Split the DataFrame into train and test sets
    train, test = train_test_split(df, test_size=0.2, random_state=10)

    train2 = train.copy()
    
    # Preprocess the data 
    train, train_num_mean, train_cat_mode, train_mean, train_stdev, train_outlier_mean, train_outlier_std = train_preprocess(train)
    test = test_preprocess(test, train_num_mean, train_cat_mode, train_mean, train_stdev, train_outlier_mean, train_outlier_std)
    df = test_preprocess(df, train_num_mean, train_cat_mode, train_mean, train_stdev, train_outlier_mean, train_outlier_std)
    train2 = test_preprocess(train2, train_num_mean, train_cat_mode, train_mean, train_stdev, train_outlier_mean, train_outlier_std)

    # Logistic Regression step (not Ridge): train the model
    logreg_model = train_logreg(train, target_var)
    # Evaluate the model with test set, df and train set
    test_pred = evaluate_model(logreg_model, test, target_var)
    df_pred = evaluate_model(logreg_model, df, target_var)
    train2_pred = evaluate_model(logreg_model, train2, target_var)
    
    # Assign the predictions to the respective DataFrames
    df = df.reset_index(drop=True)
    df['Prediction'] = df_pred
    df_pred = df.copy()

    train2 = train2.reset_index(drop=True)
    train2['Prediction'] = train2_pred
    train2_pred = train2.copy()

    return test_pred, df_pred, train2_pred, test


def logreg_model_cv(train_df, test_df, target_var, train_preprocess, test_preprocess):
    train2 = train_df.copy()

    # Preprocess
    train_df, train_num_mean, train_cat_mode, train_mean, train_stdev, train_outlier_mean, train_outlier_std = train_preprocess(train_df)
    test_df = test_preprocess(test_df, train_num_mean, train_cat_mode, train_mean, train_stdev, train_outlier_mean, train_outlier_std)
    train2 = test_preprocess(train2, train_num_mean, train_cat_mode, train_mean, train_stdev, train_outlier_mean, train_outlier_std)

    # Train model
    model = train_logreg(train_df, target_var)

    # Predictions
    test_pred = evaluate_model(model, test_df, target_var)
    train2_pred = evaluate_model(model, train2, target_var)

    test_df = test_df.reset_index(drop=True)
    test_df['Prediction'] = test_pred

    train2 = train2.reset_index(drop=True)
    train2['Prediction'] = train2_pred

    return test_pred, test_df, train2, test_df  # last test_df has target column intact for brier_score