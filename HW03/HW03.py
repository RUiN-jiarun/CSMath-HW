import numpy as np
import matplotlib.pyplot as plt

def gen_MoG(mu, sigma, size=1000):
    assert type(mu) == list
    assert type(sigma) == list
    assert len(mu) == len(sigma)
    
    n = len(mu)
    g = [np.random.multivariate_normal(mu[i], np.diag(sigma[i]), size) for i in range(n)]

    return g

def gen_data():
    G_data = gen_MoG(mu=[[0,0]], sigma=[[0.25, 0.25]])

    MoG_data = gen_MoG(mu=[[-1, -1], [-0.5, 0.5], [2, -0.5], [1, 1]], sigma=[[0.25, 0.25]] * 4)

    color = ['red', 'green', 'blue', 'cyan', 'yellow', 'gray', ]
    for i, p in enumerate(G_data):
        plt.scatter(p[:, 0], p[:, 1], s=1, color=color[i])
        plt.savefig("img/G2d.png")
    plt.show()
    for i, p in enumerate(MoG_data):
        plt.scatter(p[:, 0], p[:, 1], s=1, color=color[i])
        plt.savefig("img/MoG2d.png")
    plt.show()

if __name__ == '__main__':
    gen_data()