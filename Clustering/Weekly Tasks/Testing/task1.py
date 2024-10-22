import unittest
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans, AgglomerativeClustering, DBSCAN
from sklearn.mixture import GaussianMixture
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score

# Import the functions from your Task 1 script
from Task1 import (
    prepare_data_for_clustering,
    apply_clustering_methods,
    evaluate_clustering_performance,
    visualize_clustering_results
)

class TestClusteringAnalysis(unittest.TestCase):
    def setUp(self):
        # Load a sample dataset for testing
        self.df = pd.read_csv(r"E:\Uni\2nd year\Intro to AI\Clustering\assignment-1-ryantigi254\Clustering\Weekly Tasks\data\further_cleaned_standardized_housing_dataset.csv")
        
    def test_prepare_data_for_clustering(self):
        X_scaled, df_prepared = prepare_data_for_clustering(self.df)
        self.assertIsInstance(X_scaled, np.ndarray)
        self.assertIsInstance(df_prepared, pd.DataFrame)
        self.assertEqual(X_scaled.shape[0], self.df.shape[0])
        
    def test_apply_clustering_methods(self):
        X_scaled, _ = prepare_data_for_clustering(self.df)
        clustering_methods = [
            ("K-Means", KMeans(n_clusters=4, random_state=42)),
            ("Agglomerative", AgglomerativeClustering(n_clusters=4)),
            ("DBSCAN", DBSCAN(eps=0.5, min_samples=5)),
            ("GaussianMixture", GaussianMixture(n_components=4, random_state=42))
        ]
        clusters = apply_clustering_methods(X_scaled, clustering_methods)
        self.assertEqual(len(clusters), len(clustering_methods))
        for name, labels in clusters.items():
            self.assertEqual(len(labels), X_scaled.shape[0])
            
    def test_evaluate_clustering_performance(self):
        X_scaled, _ = prepare_data_for_clustering(self.df)
        clusters = {
            "K-Means": KMeans(n_clusters=4, random_state=42).fit_predict(X_scaled)
        }
        results_df = evaluate_clustering_performance(X_scaled, clusters)
        self.assertIsInstance(results_df, pd.DataFrame)
        self.assertEqual(results_df.shape[0], len(clusters))
        self.assertTrue("Silhouette Score" in results_df.columns)
        self.assertTrue("Davies-Bouldin Index" in results_df.columns)
        self.assertTrue("Calinski-Harabasz Score" in results_df.columns)
        
    def test_visualize_clustering_results(self):
        X_scaled, _ = prepare_data_for_clustering(self.df)
        clusters = {
            "K-Means": KMeans(n_clusters=4, random_state=42).fit_predict(X_scaled)
        }
        # This test just checks if the function runs without errors
        try:
            visualize_clustering_results(X_scaled, clusters)
            test_passed = True
        except Exception as e:
            test_passed = False
        self.assertTrue(test_passed)

if __name__ == '__main__':
    unittest.main()