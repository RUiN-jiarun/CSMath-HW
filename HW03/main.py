import numpy as np
import matplotlib.pyplot as plt
from mean_shift import MeanShift
from em import EM

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

    # Mean Shift
    ms = MeanShift(points, 0.5, 1e-2, 1e-1)      
    cluster = ms.fit()
    color = ['red', 'green', 'blue', 'cyan', 'yellow', 'gray', ]
    for i, p in enumerate(points):
        plt.scatter(p[0], p[1], s=1, color=color[cluster[i] - 1])
    plt.savefig("img/mean_shift.png")
    plt.show()

    # EM
    # init by yourself
    em = EM(data=points, ndim=4, p_sample = [0.25, 0.25, 0.25, 0.25],
                        init_mu = [[-1, -1.5], [-1.3, 0.7], [0.75, 1.1], [1.2, -0.5]],
                        init_sigma = [
                            [[0.2,0],[0,0.23]],
                            [[0.13,0],[0,0.17]],
                            [[0.15,0],[0,0.14]],
                            [[0.16,0],[0,0.1]],
                        ])
    em.EM()