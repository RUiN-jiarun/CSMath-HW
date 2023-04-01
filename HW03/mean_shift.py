import numpy as np

class MeanShift(object):
    def __init__(self, data, kernel_bandwidth, stop_threshold, cluster_threshold):
        self.points = data
        self.kernel_bandwidth = kernel_bandwidth
        self.stop_threshold = stop_threshold
        self.cluster_threshold = cluster_threshold

    @staticmethod
    def dist(a, b):
        return np.linalg.norm(np.array(a) - np.array(b))

    @staticmethod
    def gaussian_kernel(distance, bandwidth):
        return (1 / (bandwidth * np.sqrt(2 * np.pi))) * np.exp(-0.5 * ((distance / bandwidth)) ** 2)

    def fit(self):

        shift_points = np.array(self.points)
        shifting = [True] * self.points.shape[0]

        while True:
            max_dist = 0
            for i in range(0, len(shift_points)):
                print(i + 1, '/', len(shift_points), 'max_dist:', max_dist)
                if not shifting[i]:
                    continue
                p_shift_init = shift_points[i].copy()
                shift_points[i] = self._shift_point(shift_points[i], self.points)
                dist = self.dist(shift_points[i], p_shift_init)
                max_dist = max(max_dist, dist)
                shifting[i] = dist > self.stop_threshold

            if max_dist < self.stop_threshold:
                break

        cluster_ids = self._cluster_points(shift_points.tolist())
        return cluster_ids

    def _shift_point(self, point):
        shift = np.array([0., 0.])
        scale = 0.
        for p in self.points:
            dist = self.dist(point, p)
            weight = self.gaussian_kernel(dist, self.kernel_bandwidth)
            shift += weight * p
            scale += weight
        shift /= scale
        return shift

    def _cluster_points(self):
        cluster_ids = []
        cluster_idx = 0
        cluster_centers = []

        for i, point in enumerate(self.points):
            if len(cluster_ids) == 0:
                cluster_ids.append(cluster_idx)
                cluster_centers.append(point)
                cluster_idx += 1
            else:
                for center in cluster_centers:
                    dist = self.dist(point, center)
                    if(dist < self.cluster_threshold):
                        cluster_ids.append(cluster_centers.index(center))
                if len(cluster_ids) < i + 1:
                    cluster_ids.append(cluster_idx)
                    cluster_centers.append(point)
                    cluster_idx += 1
        return cluster_ids