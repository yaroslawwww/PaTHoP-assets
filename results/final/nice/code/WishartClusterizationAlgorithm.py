import math
import numpy as np
from scipy.special import gamma
from sklearn.neighbors import NearestNeighbors, BallTree
from tqdm import tqdm


class UnionFind:
    def __init__(self):
        self.parent = {}
        # 0 - специальный корень для шума/границ/удаленных кластеров
        self.parent[0] = 0

    def find(self, x):
        if x not in self.parent:
            self.parent[x] = x
        if self.parent[x] != x:
            # Сжатие путей для гарантии амортизированного O(1)
            self.parent[x] = self.find(self.parent[x])
        return self.parent[x]


class Wishart:
    def __init__(self, k, mu):
        self.k = k
        self.mu = mu
        self.labels_ = None
        self.clusters_centers_ = None
        self.center = None

    def fit(self, z_vectors, tqdms=False):
        z_vectors = np.asarray(z_vectors)
        n, dim = z_vectors.shape

        # Инициализация -1 для явного обозначения необработанных точек
        labels = np.full(n, -1, dtype=int)

        uf = UnionFind()
        completed = {}
        cluster_min_p = {}
        cluster_max_p = {}
        cluster_counter = 1

        # 1-2. Находим k-расстояния и сортируем
        # k-расстояние требует k+1 соседей (точка + k соседей)
        knn = NearestNeighbors(n_neighbors=self.k + 1)
        knn.fit(z_vectors)
        k_distances = knn.kneighbors(z_vectors, return_distance=True)[0][:, self.k]

        # Избегаем деления на ноль для совпадающих точек
        r_dist = np.maximum(k_distances, 1e-10)

        # Объем d-мерной сферы: V = π^(d/2) * r^d / Γ(d/2 + 1)
        volumes = (np.pi ** (dim / 2) * r_dist ** dim) / gamma(dim / 2 + 1)
        p_values = self.k / (volumes * n)

        # Сортировка по возрастанию радиуса = по убыванию плотности p(x)
        processed_order = np.argsort(r_dist)
        tree = BallTree(z_vectors)

        if tqdms:
            processed_order = tqdm(processed_order)

        for i in processed_order:
            xi = z_vectors[i:i + 1]

            # Строим подграф: ищем соседей в пределах d_k(x_q)
            neighbors = tree.query_radius(xi, r=r_dist[i])[0]

            active_clusters = set()
            completed_clusters = set()

            for n_idx in neighbors:
                if n_idx == i:
                    continue

                # Обрабатываем только уже пройденные вершины (т.е. с p(x_j) >= p(x_i))
                if labels[n_idx] == -1:
                    continue

                # ВСЕГДА используем uf.find для получения актуального кластера за O(1)
                # Это решает проблему "отложенного обновления"
                root = uf.find(labels[n_idx])

                # Если сосед — шум или граница, просто игнорируем его при сборе кластеров.
                # Шум не должен блокировать создание нового кластера!
                if root == 0:
                    continue

                if completed.get(root, False):
                    completed_clusters.add(root)
                else:
                    active_clusters.add(root)

            active_clusters = list(active_clusters)

            # Строки 7-8 из статьи: если нет связей с активными или завершенными кластерами
            # (даже если есть связи с чистым шумом), создаем новый кластер
            if len(active_clusters) == 0 and len(completed_clusters) == 0:
                new_label = cluster_counter
                cluster_counter += 1

                labels[i] = new_label
                uf.parent[new_label] = new_label
                completed[new_label] = False

                # Инициализация статистики нового кластера текущей плотностью
                cluster_max_p[new_label] = p_values[i]
                cluster_min_p[new_label] = p_values[i]
                continue

            # Строка 11: если все кластеры, с которыми есть связь, уже завершены
            if len(active_clusters) == 0 and len(completed_clusters) > 0:
                labels[i] = 0
                continue

            # Строка 14: Определение значимости (до присоединения текущей точки)
            significant_clusters = []
            insignificant_clusters = []
            for root in active_clusters:
                # Внутрикластерная вариация плотности
                if (cluster_max_p[root] - cluster_min_p[root]) >= self.mu:
                    significant_clusters.append(root)
                else:
                    insignificant_clusters.append(root)

            # Строки 15-18: Точка является границей между несколькими значимыми кластерами
            if len(significant_clusters) > 1:
                labels[i] = 0  # Точка становится границей (шумом)

                # Значимые кластеры помечаем как завершенные
                for r in significant_clusters:
                    completed[r] = True

                # Гениальный O(1) трюк: перенаправляем их корень в 0 (шум)
                for r in insignificant_clusters:
                    uf.parent[r] = 0

            # Строки 19-22: Слияние кластеров (<= 1 значимый кластер)
            else:
                if len(significant_clusters) == 1:
                    primary_root = significant_clusters[0]
                else:
                    # Если значимых нет, сливаем в самый плотный (старый) активный кластер
                    primary_root = max(active_clusters, key=lambda r: cluster_max_p[r])

                labels[i] = primary_root

                for r in active_clusters:
                    if r != primary_root:
                        uf.parent[r] = primary_root  # O(1) обновление UnionFind

                        # Объединяем плотностную статистику объединяемых кластеров
                        cluster_max_p[primary_root] = max(cluster_max_p[primary_root], cluster_max_p[r])
                        cluster_min_p[primary_root] = min(cluster_min_p[primary_root], cluster_min_p[r])

                # Обновляем пределы объединенного кластера плотностью новой присоединенной точки
                cluster_max_p[primary_root] = max(cluster_max_p[primary_root], p_values[i])
                cluster_min_p[primary_root] = min(cluster_min_p[primary_root], p_values[i])

        # Финальное разрешение меток (Flattening)
        for i in range(n):
            if labels[i] > 0:
                labels[i] = uf.find(labels[i])
            elif labels[i] == -1:
                labels[i] = 0  # На всякий случай подчищаем, если вдруг какие-то зависли

        # Извлечение центров (мотивов)
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