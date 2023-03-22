import numpy as np
import matplotlib.pyplot as plt
import torch

from utils import read_ori_data, get_data_with_label, get_focus, vis_2_comp

def Hbeta(D, beta=1.0):
    P = torch.exp(-D.clone() * beta)

    sumP = torch.sum(P)

    H = torch.log(sumP) + beta * torch.sum(D * P) / sumP
    P = P / sumP

    return H, P

def pji(X, tol=1e-15, perp=30.0):
    n, d = X.shape

    D = torch.add(torch.add(-2 * torch.mm(X, X.t()), torch.sum(X*X, 1)).t(), torch.sum(X*X, 1))
    P = torch.zeros(n, n)

    beta = torch.ones(n, 1)
    logU = torch.log(torch.tensor([perp]))

    n_list = [i for i in range(n)]

    for i in range(n):

        # Compute the Gaussian kernel and entropy for the current precision
        # there may be something wrong with this setting None
        betamin = None
        betamax = None
        Di = D[i, n_list[0:i]+n_list[i+1:n]]

        H, thisP = Hbeta(Di, beta[i])

        # Evaluate whether the perplexity is within tolerance
        Hdiff = H - logU
        tries = 0
        while torch.abs(Hdiff) > tol and tries < 50:

            # If not, increase or decrease precision
            if Hdiff > 0:
                betamin = beta[i].clone()
                if betamax is None:
                    beta[i] = beta[i] * 2.
                else:
                    beta[i] = (beta[i] + betamax) / 2.
            else:
                betamax = beta[i].clone()
                if betamin is None:
                    beta[i] = beta[i] / 2.
                else:
                    beta[i] = (beta[i] + betamin) / 2.

            # Recompute the values
            H, thisP = Hbeta(Di, beta[i])

            Hdiff = H - logU
            tries += 1

        # Set the final row of P
        P[i, n_list[0:i]+n_list[i+1:n]] = thisP

    print("Mean value of sigma: %f" % torch.mean(torch.sqrt(1 / beta)))
    # Return final P-matrix
    return P

def pca(X, dims=50):
    print("Preprocessing the data using PCA...")
    n, d = X.shape
    X = X - torch.mean(X, 0)

    l, M = torch.linalg.eig(torch.mm(X.t(), X))
    l, M = l.clone().detach().float(), M.clone().detach().float()

    Y = torch.mm(X, M[:, 0:dims])
    return Y


def tsne(X, dims=2, initial_dims=50, perp=30.0, max_iter=1000):

    # Initialize variables
    X = pca(X, initial_dims)
    n, d = X.shape

    initial_momentum = 0.5
    final_momentum = 0.8
    eta = 500
    min_gain = 0.01
    Y = torch.randn(n, dims)
    dY = torch.zeros(n, dims)
    iY = torch.zeros(n, dims)
    gains = torch.ones(n, dims)

    # Compute P-values
    P = pji(X, 1e-5, perp)
    P = P + P.t()
    P = P / torch.sum(P)
    P = P * 4.    # early exaggeration
    print("get P shape", P.shape)
    P = torch.max(P, torch.tensor([1e-21]))

    loss_list = []
    # Run iterations
    for iter in range(max_iter):

        # Compute pairwise affinities
        sum_Y = torch.sum(Y*Y, 1)
        num = -2. * torch.mm(Y, Y.t())
        num = 1. / (1. + torch.add(torch.add(num, sum_Y).t(), sum_Y))
        # num[range(n), range(n)] = 0.
        num[torch.eye(n, n, dtype=torch.long) == 1] = 0.
        Q = num / torch.sum(num)
        Q = torch.max(Q, torch.tensor([1e-12]))

        # Compute gradient
        PQ = P - Q
        for i in range(n):
            dY[i, :] = torch.sum((PQ[:, i] * num[:, i]).repeat(dims, 1).t() * (Y[i, :] - Y), 0)

        # Perform the update
        if iter < 20:
            momentum = initial_momentum
        else:
            momentum = final_momentum

        gains = (gains + 0.2) * ((dY > 0.) != (iY > 0.)).double() + (gains * 0.8) * ((dY > 0.) == (iY > 0.)).double()
        gains[gains < min_gain] = min_gain
        iY = momentum * iY - eta * (gains * dY)
        Y = Y + iY
        Y = Y - torch.mean(Y, 0)

        loss = torch.sum(P * torch.log(P / Q))
        loss_list.append(loss)


        if (iter + 1) % 100 == 0:
            print("Iteration %d: loss = %f" % (iter + 1, loss))

        # Stop exaggeration
        if iter == 100:
            P = P / 4.

    plt.plot(loss_list)
    plt.show()
    plt.savefig("img/tsne_loss.png")
    # Return solution
    return Y

if __name__ == '__main__':
    train_dataset = read_ori_data('data/optdigits-orig.tra')
    data = get_data_with_label(train_dataset, [3])

    with torch.no_grad():
        Y = tsne(data, 2, 10, 30.0, 1000)
        # Y = tsne(data, 2, 50, 20.0, 1000)

    axv = np.linspace(-10, 10, 5, endpoint=True)
    axh = np.linspace(-10, 10, 5, endpoint=True)
    focus = get_focus(Y[:, 0], Y[:, 1], axv, axh)
    vis_2_comp(data, Y[:, 0], Y[:, 1], axv, axh, focus, method='tsne', plot=True)