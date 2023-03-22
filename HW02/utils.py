import numpy as np
import matplotlib.pyplot as plt
import torch

def read_ori_data(file):
    dataset = []
    f = open(file, encoding='utf-8')
    for i in range(21):
        f.readline()
    while True:
        s = f.readline().strip()
        if not s:
            break
        for i in range(31):
            s += f.readline().strip()
        label = int(f.readline().strip())
        dataset.append((torch.tensor(list(map(int, s))), label))
    return dataset

def get_data_with_label(dataset, label):
    d = []

    for i in dataset:
        if i[1] == label:
            d.append(torch.unsqueeze(i[0], dim=0))

    for i in range(len(d)):
        if i == 0:
            data = d[i]
        else:
            data = torch.cat((data, d[i]), dim=0)
    
    return data.clone().detach().float()

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

def vis_2_comp(data, x, y, axv, axh, focus, method='pca'):
    plt.xlabel('First Principal Component')
    plt.ylabel('Second Principal Component')   
    for v in axv:
        plt.axvline(x=v,color="grey",ls="--",lw=1)
    for h in axh:
        plt.axhline(y=h,color="grey",ls="--",lw=1)
    # print(x, y)
    plt.scatter(x, y, color='lime', s=5)
    
    fx, fy = [x[i] for i in focus], [y[i] for i in focus]
    plt.scatter(fx, fy, color='none', marker='o', edgecolor='red')
    plt.savefig("img/"+method+"_scatter.png")
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
    plt.savefig("img/"+method+"_eigen.png")
    plt.show()