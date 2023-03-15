import numpy as np
import matplotlib.pyplot as plt
import torch

from utils import read_ori_data, get_data_with_label

def PCA2(data):
    """PCA with 2 components
    return: Xm, V2, m
    """
    m = data.mean(axis=0, keepdims=True)
    X = (data - m).T 

    H = X.T @ X 
    w, U = torch.linalg.eig(H)
    w, U = torch.tensor(w, dtype=torch.float32), torch.tensor(U, dtype=torch.float32)   # ComplexFloat to Float
    D = torch.diag(torch.sqrt(w))
    V = (torch.linalg.inv(U @ D) @ X.T).T

    V2 = V[:, :2]       # Select 2 principle components
    Xm = V2.T @ X

    return Xm, V2, m


def get_focus(x, y, axv, axh):
    focus = []
    for v in axv:
        for h in axh:
            best = 0
            for i in range(int(list(x.size())[0])):
                if pow((x[i] - v), 2) + pow((y[i] - h), 2) < pow((x[best] - v), 2) + pow((y[best] - h), 2):
                    best = i
            focus.append(best)
    return focus

def vis_pca2(data, x, y, axv, axh, focus):
    plt.xlabel('First Principle Component')
    plt.ylabel('Second Principle Component')   
    for v in axv:
        plt.axvline(x=v,color="grey",ls="--",lw=1)
    for h in axh:
        plt.axhline(y=h,color="grey",ls="--",lw=1)
    # print(x, y)
    plt.scatter(x, y, color='lime', s=5)
    
    fx, fy = [x[i] for i in focus], [y[i] for i in focus]
    plt.scatter(fx, fy, color='none', marker='o', edgecolor='red')
    plt.savefig("img/pca2_scatter.png")
    plt.show()

    padding = 2
    L = 32 * 5 + padding * 6
    canvas = np.zeros((L, L, 3))
    canvas[:, :, 0] = np.ones((L, L))

    for i in range(5):
        x = padding * (i + 1) + i * 32
        for j in range(5):
            y = padding * (j + 1) + j * 32
            img = data[focus[i * 5 + j]].reshape(32, 32)
            img = np.stack((img, img, img), axis=-1)
            canvas[x: x + 32, y: y + 32] = img
            
    plt.axis('off')
    plt.imshow(canvas)
    plt.savefig("img/pca2_eigen.png")
    plt.show()


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
    vis_pca2(data, Xm[0], Xm[1], axv, axh, focus)


if __name__ == '__main__':
    main()