import numpy as np

def assign_thresholds(df, prediction_col, thresholds):
    conditions = []

    for i in range(0, len(thresholds)+1):
        if i == 0:
            conditions.append(df[prediction_col] <= thresholds[i])
        elif i == len(thresholds):
            conditions.append(df[prediction_col] > thresholds[i - 1])
        else:
            conditions.append((df[prediction_col] > thresholds[i - 1]) & (df[prediction_col] <= thresholds[i]))

    num_ratings = len(conditions)
    ratings = list(range(0, num_ratings))
    df['Rating'] = np.select(conditions, ratings)
    return df