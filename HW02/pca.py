import numpy as np
import matplotlib.pyplot as plt
import torch

def read_data(file):
    data = []
    f = open(file,encoding='utf-8')
    for i in range(21):
        f.readline()
    while True:
        s = f.readline().strip()
        if not s:
            break
        for i in range(31):
            s += f.readline().strip()
        label = int(f.readline().strip())
        data.append((torch.tensor(list(map(int, s))), label))
    return data

train_dataset = read_data('data/optdigits-orig.tra')
# test_dataset = read_data('data/optdigits.tes')

three = []

for i in train_dataset:
    if i[1] == 3:
        three.append(torch.unsqueeze(i[0], dim=0))
# print(data)

for i in range(len(three)):
    if i == 0:
        data = three[i]
    else:
        data = torch.cat((data, three[i]), dim=0)

# plt.imshow(data[0].reshape(32, -1), cmap='gray')
# plt.show()

# data = np.array(data).astype(np.float32)
data = data.float()
m = data.mean(axis=0, keepdims=True)
X = (data - m).T 

H = X.T @ X 
w, U = torch.linalg.eig(H)
w, U = torch.tensor(w, dtype=torch.float32), torch.tensor(U, dtype=torch.float32)
S = torch.diag(torch.sqrt(w))
V = (torch.linalg.inv(U @ S) @ X.T).T

V_ = V[:, :2]
X_ = V_.T @ X
print(X_)

print("error =", torch.linalg.norm(V_ @ X_ - X))
# plt.imshow((V_ @ X_ + m.T)[:, 0].reshape(32, 32), cmap='gray')
# plt.show()


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

def plot(x, y, axv, axh, focus):
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
    plt.show()

axv = np.linspace(-4, 4, 5, endpoint=True)
axh = np.linspace(-4, 4, 5, endpoint=True)
focus = get_focus(X_[0], X_[1], axv, axh)
print(focus)
plot(X_[0], X_[1], axv, axh, focus)

padding = 2
L = 32 * 5 + padding * 6
canvas = np.zeros((L, L, 3))
canvas[:, :, 0] = np.ones((L, L))

idx = 0
for i in range(5):
    x = padding * (i + 1) + i * 32
    for j in range(5):
        y = padding * (j + 1) + j * 32
        img = data[focus[i * 5 + j]].reshape(32, 32)
        img = np.stack((img, img, img), axis=-1)
        canvas[x: x + 32, y: y + 32] = img
        
plt.axis('off')
plt.imshow(canvas)
plt.show()