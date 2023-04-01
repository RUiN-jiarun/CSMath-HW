import numpy as np
import matplotlib.pyplot as plt
from mean_shift import MeanShift

def gen_MoG(mu, sigma, size=150):
    assert type(mu) == list
    assert type(sigma) == list
    assert len(mu) == len(sigma)
    
    n = len(mu)
    g = [np.random.multivariate_normal(mu[i], np.diag(sigma[i]), size) for i in range(n)]

    return g

def gen_data():
    G_data = gen_MoG(mu=[[0,0]], sigma=[[0.25, 0.25]])

    MoG_data = gen_MoG(mu=[[-1, -0.5], [-1, 0.5], [1, 1], [1, -1]], sigma=[[0.16, 0.16]] * 4)

    color = ['red', 'green', 'blue', 'cyan', 'yellow', 'gray', ]
    for i, p in enumerate(G_data):
        plt.scatter(p[:, 0], p[:, 1], s=1, color=color[i])
        plt.savefig("img/G2d.png")
    plt.show()
    for i, p in enumerate(MoG_data):
        plt.scatter(p[:, 0], p[:, 1], s=1, color=color[i])
        plt.savefig("img/MoG2d.png")
    plt.show()

    return MoG_data



if __name__ == '__main__':
    data = gen_data()
    points = np.concatenate(data, axis=0)
    ms = MeanShift(0.5, 1e-2, 1e-1)      
    pts, cluster = ms.fit(points)
    color = ['red', 'green', 'blue', 'cyan', 'yellow', 'gray', ]
    for i, p in enumerate(points):
        plt.scatter(p[0], p[1], s=1, color=color[cluster[i] - 1])
    plt.savefig("img/mean_shift.png")
    plt.show()