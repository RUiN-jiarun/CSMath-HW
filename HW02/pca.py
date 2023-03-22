import numpy as np
import matplotlib.pyplot as plt
import torch

from utils import read_ori_data, get_data_with_label, get_focus, vis_2_comp

def PCA2(data):
    """PCA with 2 components
    return: Xm, V2, m
    """
    m = data.mean(axis=0, keepdims=True)
    X = (data - m).T 

    H = X.T @ X 
    w, U = torch.linalg.eig(H)
    w, U = w.clone().detach().float(), U.clone().detach().float()   # ComplexFloat to Float
    D = torch.diag(torch.sqrt(w))
    V = (torch.linalg.inv(U @ D) @ X.T).T

    V2 = V[:, :2]       # Select 2 principle components
    Xm = V2.T @ X

    return Xm, V2, m


def main():

    # Convert from original dataset
    train_dataset = read_ori_data('data/optdigits-orig.tra')
    data = get_data_with_label(train_dataset, 3)

    # Display a sample of number 3
    plt.imshow(data[0].reshape(32, -1), cmap='gray')
    plt.savefig("img/sample_3.png")
    plt.show()

    # PCA 
    Xm, V2, m = PCA2(data)

    # Display eigen (mean) image
    plt.imshow((V2 @ Xm + m.T)[:, 0].reshape(32, 32), cmap='gray')
    plt.savefig("img/mean_3.png")
    plt.show()

    # Plot PCA result
    axv = np.linspace(-4, 4, 5, endpoint=True)
    axh = np.linspace(-4, 4, 5, endpoint=True)
    focus = get_focus(Xm[0], Xm[1], axv, axh)
    vis_2_comp(data, Xm[0], Xm[1], axv, axh, focus, method='pca')


if __name__ == '__main__':
    main()