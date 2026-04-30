#!/usr/bin/env python
# coding: utf-8

# # Imports
# 
# 

# In[ ]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.metrics import silhouette_score
from scipy.cluster.hierarchy import linkage, dendrogram
from scipy.spatial.distance import cdist

# # Data Import

# In[6]:


#import to dataframe
file_path = "Longotor1delta.xls"
df = pd.read_excel(file_path)

#preview of dataset
print("First 5 rows:")
print(df.head())
print("\nColumns:")
print(df.columns.tolist())
print("\nShape:")
print(df.shape)

# # Data Cleaning
# We identified the columns used for identification and the three columns containing numeric data about the genes. We removed all other columns, 'Unnamed: 6', 'Unnamed: 7', and 'Unnamed: 8', then dropped all N/A values. We normalized the numerical data with StandardScalar to ensure they are comparible for accurate clustering.
# 

# In[7]:


#keep needed columns
id_cols = ["Public ID", "Gene", "Gene description"]
feature_cols = ["sch9/wt", "ras2/wt", "tor1/wt"]
df = df[id_cols + feature_cols].copy()
#drop missing values
df = df.dropna(subset=feature_cols)
print("\nCleaned dataset shape:")
print(df.shape)

# In[8]:


#isolate columns with numerical data
X = df[feature_cols].values
#scale
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

print("\nFirst 5 rows of normalized data:")
print(X_scaled[:5])


# In[9]:


def k_means(X, n_clusters=3, random_state=13, n_init=10):
    """runs and reports results of k means clustering on input dataset"""

    model = KMeans(n_clusters=n_clusters, random_state=random_state, n_init=n_init)
    labels = model.fit_predict(X)

    result = {
        "model": model,
        "labels": labels,
        "silhouette_score": silhouette_score(X, labels) if n_clusters > 1 else None,
        "cluster_centers": model.cluster_centers_
    }
    return result

# In[10]:


def hierarchical(X, n_clusters=3, linkage_method="ward", distance_metric="euclidean"):
    """runs and reports results of agglomerative hierarchical clustering on input dataset
        Note: ward linkage only works with euclidean distance"""
    
    if linkage_method == "ward" and distance_metric != "euclidean":
        raise ValueError("Ward linkage only supports euclidean distance.")

    model = AgglomerativeClustering(
        n_clusters=n_clusters,
        metric=distance_metric,
        linkage=linkage_method
    )
    labels = model.fit_predict(X)

    result = {
        "model": model,
        "labels": labels,
        "silhouette_score": silhouette_score(X, labels) if n_clusters > 1 else None
    }
    return result

# In[11]:


def attach_cluster_labels(df, labels, cluster_col_name="Cluster"):
    """
    Attach cluster labels back to the original dataframe.
    """
    df_copy = df.copy()
    df_copy[cluster_col_name] = labels
    return df_copy


def print_cluster_members(df, id_cols, cluster_col_name="Cluster", max_rows=10):
    """
    Print sample members from each cluster.
    """
    for cluster_id in sorted(df[cluster_col_name].unique()):
        print(f"\nCluster {cluster_id}")
        print(df[df[cluster_col_name] == cluster_id][id_cols].head(max_rows))

# ### Visualization Helper Functions
# 
# 

# In[12]:


def plot_clusters_2d(X, labels, feature_cols, title, filename=None):
    """
    Plot first two normalized features for cluster visualization.
    """
    plt.figure(figsize=(8, 6))
    plt.scatter(X[:, 0], X[:, 1], c=labels, s=20)
    plt.xlabel(feature_cols[0])
    plt.ylabel(feature_cols[1])
    plt.title(title)
    plt.grid(True)

    if filename:
        plt.savefig(filename, dpi=300, bbox_inches="tight")

    plt.show()


def plot_dendrogram(X, method="ward", filename=None):
    """
    Plot dendrogram for hierarchical clustering.
    """
    linked = linkage(X, method=method)
    plt.figure(figsize=(12, 6))
    dendrogram(linked, no_labels=True)
    plt.title(f"Dendrogram ({method} linkage)")
    plt.xlabel("Data Points")
    plt.ylabel("Distance")

    if filename:
        plt.savefig(filename, dpi=300, bbox_inches="tight")

    plt.show()


# K-Means Experiments: Testing Different k Values Beyond k=3
kmeans_experiment_results = []

for k in range(2, 8):
    result = k_means(X_scaled, n_clusters=k)

    cluster_sizes = pd.Series(result["labels"]).value_counts().sort_index().to_dict()

    kmeans_experiment_results.append({
        "Algorithm": "K-Means",
        "k": k,
        "Linkage": "N/A",
        "Metric": "euclidean",
        "Silhouette Score": result["silhouette_score"],
        "Cluster Sizes": cluster_sizes
    })

kmeans_results_df = pd.DataFrame(kmeans_experiment_results)

print("\nK-Means Experiment Results:")
print(kmeans_results_df)


# Hierarchical Clustering Experiments: Testing Linkage Methods
# Tested with Single, Complete, Average, and Ward

hierarchical_experiment_results = []

linkage_methods = ["single", "complete", "average", "ward"]

for linkage_method in linkage_methods:
    result = hierarchical(
        X_scaled,
        n_clusters=3,
        linkage_method=linkage_method,
        distance_metric="euclidean"
    )

    cluster_sizes = pd.Series(result["labels"]).value_counts().sort_index().to_dict()

    hierarchical_experiment_results.append({
        "Algorithm": "Hierarchical",
        "k": 3,
        "Linkage": linkage_method,
        "Metric": "euclidean",
        "Silhouette Score": result["silhouette_score"],
        "Cluster Sizes": cluster_sizes
    })

hierarchical_results_df = pd.DataFrame(hierarchical_experiment_results)

print("\nHierarchical Linkage Experiment Results:")
print(hierarchical_results_df)


# Hierarchical Clustering Experiments: Testing Distance Metrics
# Ward linkage only supports Euclidean distance, so it is excluded here.

distance_metric_results = []

distance_metrics = ["euclidean", "manhattan", "cosine"]
linkage_methods_for_metrics = ["single", "complete", "average"]

for linkage_method in linkage_methods_for_metrics:
    for metric in distance_metrics:
        result = hierarchical(
            X_scaled,
            n_clusters=3,
            linkage_method=linkage_method,
            distance_metric=metric
        )

        cluster_sizes = pd.Series(result["labels"]).value_counts().sort_index().to_dict()

        distance_metric_results.append({
            "Algorithm": "Hierarchical",
            "k": 3,
            "Linkage": linkage_method,
            "Metric": metric,
            "Silhouette Score": result["silhouette_score"],
            "Cluster Sizes": cluster_sizes
        })

distance_metric_results_df = pd.DataFrame(distance_metric_results)

print("\nHierarchical Distance Metric Experiment Results:")
print(distance_metric_results_df)


# Combined Experimental Results Table
all_results_df = pd.concat(
    [
        kmeans_results_df,
        hierarchical_results_df,
        distance_metric_results_df
    ],
    ignore_index=True
)

print("\nAll Clustering Experiment Results:")
print(all_results_df)

all_results_df.to_csv("clustering_experiment_results.csv", index=False)

# In[16]:

# Final Selected Models and Visualizations

# K-Means with k=3
kmeans_result = k_means(X_scaled, n_clusters=3)

plot_clusters_2d(
    X_scaled,
    kmeans_result["labels"],
    feature_cols,
    "K-Means Clustering (k=3)",
    filename="kmeans_k3_plot.png"
)

df_kmeans = attach_cluster_labels(df, kmeans_result["labels"], "KMeans_Cluster")

print("\nK-Means Cluster Members:")
print_cluster_members(df_kmeans, id_cols, "KMeans_Cluster")


# Hierarchical clustering with Ward linkage and k=3
hier_result = hierarchical(
    X_scaled,
    n_clusters=3,
    linkage_method="ward",
    distance_metric="euclidean"
)

plot_clusters_2d(
    X_scaled,
    hier_result["labels"],
    feature_cols,
    "Hierarchical Clustering (Ward Linkage, k=3)",
    filename="hierarchical_ward_k3_plot.png"
)

df_hierarchical = attach_cluster_labels(df, hier_result["labels"], "Hierarchical_Cluster")

print("\nHierarchical Cluster Members:")
print_cluster_members(df_hierarchical, id_cols, "Hierarchical_Cluster")


# Dendrogram for hierarchical clustering
plot_dendrogram(
    X_scaled,
    method="ward",
    filename="ward_dendrogram.png"
)


# Best Result Summary
def is_valid_clustering(cluster_sizes, min_ratio=0.05):
    sizes = list(cluster_sizes.values())
    total = sum(sizes)
    return min(sizes) / total >= min_ratio


valid_results_df = all_results_df[
    all_results_df["Cluster Sizes"].apply(is_valid_clustering)
]

best_result = valid_results_df.loc[
    valid_results_df["Silhouette Score"].idxmax()
]

print("\nBest Valid Clustering Result Based on Silhouette Score:")
print(best_result)