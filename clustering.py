from pd_calib import PD_cal, df_prediction, HD_calib, pd_calibration
from EL_UL_K_Calc import EL_Calc, UL_Calc
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from threshold_assignment import assign_thresholds
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

def f_clustering(test_pred, df_pred, target_var, number_of_clusters, calib_method):
    # Apply clustering to the predictions
    prediction_col = 'Prediction'
    rating_df, kmeans = perform_clustering(test_pred, number_of_clusters)
    thresholds = extract_kmeans_thresholds_from_centroids(kmeans)
    df_new = assign_thresholds(df_pred, prediction_col, thresholds)
    # Print the number of rows that each rating contains
    rating_counts = rating_df['Rating'].value_counts()
    rating_counts = rating_counts.sort_index()
    #print(rating_counts)
    # PD calibration
    calibrated_pd = HD_calib(df_new, rating_df, target_var)
    calibrated_pd_method = calib_method(df_new, rating_df, target_var)
    calibrated_pd['method Cal_PD'] = calibrated_pd_method['method Cal_PD']
    # Print the average Cal_PD per rating
    avg_cal_pd_per_rating = calibrated_pd.groupby('Rating')['HD_Cal_PD'].mean()
    avg_cal_pd_per_rating_method = calibrated_pd_method.groupby('Rating')['method Cal_PD'].mean()
    #print(avg_cal_pd_per_rating)
    # Map rating counts to avg_cal_pd_per_rating
    avg_pd_rating = pd_calibration(calibrated_pd, 'Rating', 'Prediction')
    avg_pd_rating_df = pd_calibration(df_new, 'Rating', 'Prediction')
    df_rating_counts = df_new['Rating'].value_counts()
    df_rating_counts = df_rating_counts.sort_index()
    df_default_counts = df_new.groupby('Rating')[target_var].sum()
    rating_summary = pd.DataFrame({
        'test counts': rating_counts,
        'avg PD': avg_pd_rating,
        'df counts': df_rating_counts,
        'df avg PD': avg_pd_rating_df,
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

def perform_clustering(df, n):
    df_clust = df[['Prediction']].copy()
    # Perform KMeans clustering
    kmeans = KMeans(n_clusters=n, random_state=42)
    df['Cluster'] = kmeans.fit_predict(df_clust)

    # Assign ratings based on cluster predictions
    # Extract cluster centers
    cluster_centers = kmeans.cluster_centers_.flatten()
    #print('cluster_centers: \n', cluster_centers)
    # Sorteer clusters o.b.v. hun gemiddelde prediction
    cluster_order = cluster_centers.argsort()  # van laag naar hoog
    #print('cluster_order: \n', cluster_order)
    # Map clusters to ratings starting from 0
    rating_map = {cluster: rating for rating, cluster in enumerate(cluster_order)}
    
    df['Rating'] = df['Cluster'].map(rating_map)
    
    return df, kmeans

def find_optimal_clusters(df, max_clusters):
    # Find the optimal number of clusters    
    silhouette_scores = []
    for n_clusters in range(2, max_clusters + 1):
        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        cluster_labels = kmeans.fit_predict(df)
        silhouette_avg = silhouette_score(df, cluster_labels)
        silhouette_scores.append(silhouette_avg)
    
    # Plot silhouette scores
    plt.plot(range(2, max_clusters + 1), silhouette_scores)
    plt.xlabel('Number of clusters')
    plt.ylabel('Silhouette Score')
    plt.title('Silhouette Scores for Different Number of Clusters')
    plt.show()
    
    optimal_clusters = silhouette_scores.index(max(silhouette_scores)) + 2
    return optimal_clusters

def extract_kmeans_thresholds_from_centroids(kmeans_model):
    """
    Bereken de prediction thresholds tussen clusters op basis van de centroids.
    De cutoff tussen cluster A en B ligt halverwege hun centroids.
    """
    centroids = kmeans_model.cluster_centers_.flatten()
    sorted_centroids = np.sort(centroids)

    # Thresholds liggen tussen de centroids
    thresholds = [(sorted_centroids[i] + sorted_centroids[i+1]) / 2 for i in range(len(sorted_centroids) - 1)]
    # Converteer np.float64 naar gewone float
    thresholds = [float(threshold) for threshold in thresholds]
    return thresholds

