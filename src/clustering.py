"""
IncidentLens — incident clustering.

K-Means clustering over TF-IDF vectors to surface recurring incident
patterns. A cluster is a discovered grouping of similar language, NOT
a claim about a true underlying root cause — see docs/clustering.md.
"""
import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score


class IncidentClusterer:
    """K-Means clustering wrapper with silhouette/inertia evaluation."""

    def __init__(self, n_clusters: int, random_state: int = 42):
        self.n_clusters = n_clusters
        self.random_state = random_state
        self.model = KMeans(
            n_clusters=n_clusters, random_state=random_state, n_init=10
        )
        self._fitted = False

    def fit(self, matrix):
        self.model.fit(matrix)
        self._fitted = True
        return self

    def predict(self, matrix):
        if not self._fitted:
            raise RuntimeError("Clusterer must be fit before predict().")
        return self.model.predict(matrix)

    def fit_predict(self, matrix):
        labels = self.model.fit_predict(matrix)
        self._fitted = True
        return labels

    def inertia(self) -> float:
        if not self._fitted:
            raise RuntimeError("Clusterer must be fit first.")
        return self.model.inertia_

    def top_terms_per_cluster(self, matrix, feature_names, top_n: int = 10):
        """Return {cluster_id: [top terms by centroid weight]}."""
        if not self._fitted:
            raise RuntimeError("Clusterer must be fit first.")
        centroids = self.model.cluster_centers_
        result = {}
        for cluster_id in range(self.n_clusters):
            top_indices = centroids[cluster_id].argsort()[::-1][:top_n]
            result[cluster_id] = [feature_names[i] for i in top_indices]
        return result


def evaluate_k_range(matrix, k_values, sample_size_for_silhouette=5000, random_state=42):
    """
    Compute real inertia (all k) and silhouette score (on a sample, since
    silhouette is O(n^2) and infeasible on 100K rows) for a range of k.
    Returns a list of dicts: k, inertia, silhouette.
    """
    rng = np.random.RandomState(random_state)
    n_samples = matrix.shape[0]
    sample_idx = rng.choice(
        n_samples, size=min(sample_size_for_silhouette, n_samples), replace=False
    )
    sample_matrix = matrix[sample_idx]

    results = []
    for k in k_values:
        model = KMeans(n_clusters=k, random_state=random_state, n_init=10)
        labels = model.fit_predict(matrix)
        sample_labels = labels[sample_idx]

        # silhouette_score requires at least 2 distinct labels in the sample
        if len(set(sample_labels)) < 2:
            sil = None
        else:
            sil = silhouette_score(sample_matrix, sample_labels)

        results.append({"k": k, "inertia": model.inertia_, "silhouette": sil})
        print(f"  k={k}: inertia={model.inertia_:.2f}, silhouette={sil}")

    return results
