# coding: utf-8
import copy
from collections import defaultdict
from itertools import product
from math import gamma

import numpy as np
import pandas as pd
from scipy.spatial.distance import squareform, pdist
from sklearn.neighbors import NearestNeighbors, BallTree
from tqdm import tqdm


# coding: utf-8
def partition(arr, l, r):
    x = arr[r]
    i = l
    for j in range(l, r):

        if arr[j] <= x:
            arr[i], arr[j] = arr[j], arr[i]
            i += 1

    arr[i], arr[r] = arr[r], arr[i]
    return i


def QuickSelectWithLR(array, left, right, k):
    if 0 < k <= right - left + 1:
        index = partition(array, left, right)
        if index - left == k - 1:
            return array[index]
        if index - left > k - 1:
            return QuickSelectWithLR(array, left, index - 1, k)
        return QuickSelectWithLR(array, index + 1, right,
                                 k - index + left - 1)


def QuickSelect(array, k):
    return QuickSelectWithLR(array, 0, len(array) - 1, k)


# import cupy as cp
# from cupyx.scipy.spatial.distance import cdist
# from itertools import product
from collections import defaultdict
import math

def volume(radius, dim):
    return np.pi ** (dim / 2) * radius ** dim / gamma(dim / 2 + 1)


class UnionFind:
    def __init__(self):
        self.parent = {}

    def find(self, x):
        if x not in self.parent:
            self.parent[x] = x
        if self.parent[x] != x:
            self.parent[x] = self.find(self.parent[x])
        return self.parent[x]

    def union(self, x, y):
        fx = self.find(x)
        fy = self.find(y)
        if fx != fy:
            self.parent[fy] = fx


class Wishart:
    def __init__(self, k, mu):
        self.k, self.mu = k, mu
        self.labels_ = None
        self.clusters_centers_ = None
        self.center = None

    def significant(self, cluster_points_significances):
        max_diff = 0
        # Перебор всех пар точек в кластере
        for i, j in product(cluster_points_significances, repeat=2):
            diff = abs(i - j)
            if diff > max_diff:
                max_diff = diff
        return max_diff >= self.mu

    def fit(self, z_vectors,tqdms = False):
        z_vectors = np.asarray(z_vectors)
        n, dim = z_vectors.shape
        labels = np.zeros(n, dtype=int)
        completed = {0: False}
        cluster_counter = 1
        uf = UnionFind()
        knn = NearestNeighbors(n_neighbors=self.k + 1)
        knn.fit(z_vectors)
        k_distances = knn.kneighbors(z_vectors, return_distance=True)[0][:, self.k]
        r = k_distances
        volumes = (np.pi ** (dim / 2) * r ** dim) / math.gamma(dim / 2 + 1)
        significance_values = self.k / (volumes * n)

        processed_order = np.argsort(k_distances)
        tree = BallTree(z_vectors)
        processed_order = tqdm(processed_order) if tqdms else processed_order
        for i in processed_order:
            xi = z_vectors[i:i + 1]
            neighbors = tree.query_radius(xi, r=k_distances[i])[0]
            neighbors = np.setdiff1d(neighbors, [i])

            neighbor_roots = set()
            cluster_members = {}
            for n in neighbors:
                lbl = labels[n]
                if lbl == 0:
                    continue
                root = uf.find(lbl)
                neighbor_roots.add(root)
                if root not in cluster_members:
                    cluster_members[root] = []
                cluster_members[root].append(n)

            neighbor_roots = [r for r in neighbor_roots if not completed.get(r, False)]

            if len(neighbor_roots) == 0:
                new_label = cluster_counter
                labels[i] = new_label
                uf.union(new_label, new_label)
                completed[new_label] = False
                cluster_counter += 1
                continue

            if len(neighbor_roots) == 1:
                target = neighbor_roots[0]
                labels[i] = target
                continue

            cluster_significances = {}
            for root in neighbor_roots:
                members = cluster_members[root]
                if not members:
                    continue
                cluster_significances[root] = significance_values[members]

            significant_clusters = [
                r for r in cluster_significances
                if self.significant(cluster_significances[r])
            ]

            if len(significant_clusters) > 1:
                labels[i] = 0
                for r in significant_clusters:
                    completed[r] = True
                continue

            if significant_clusters:
                target = significant_clusters[0]
            else:
                target = neighbor_roots[0]
            labels[i] = target
            for r in neighbor_roots:
                if r != target:
                    uf.union(target, r)
        for i in range(n):
            if labels[i] != 0:
                labels[i] = uf.find(labels[i])

        unique_labels = np.unique(labels)
        self.clusters_centers_ = {}
        for l in unique_labels:
            if l == 0:
                continue
            mask = (labels == l)
            self.clusters_centers_[l] = z_vectors[mask].mean(axis=0)

        self.labels_ = labels
        self.center = z_vectors.mean(axis=0)
        return self