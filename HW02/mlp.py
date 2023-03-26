import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn

from utils import read_ori_data, get_focus, vis_2_comp
from tsne import pca

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

class MLP(nn.Module):
    def __init__(self, input, classes=10):
        super(MLP, self).__init__()

        self.layer1 = nn.Linear(input, 512)
        self.relu1 = nn.ReLU()
        self.layer2 = nn.Linear(512, 128)
        self.relu2 = nn.ReLU()
        self.layer3 = nn.Linear(128, classes)
        self.softmax = nn.Softmax()

    def forward(self, input):
        out = self.layer1(input)
        out = self.relu1(out)
        out = self.layer2(out)
        out = self.relu2(out)
        out = self.layer3(out)
        out = self.softmax(out)
        return out


def train(train_loader, test_loader, model, criterion, optimizer, num_epochs):
    # test the initial acc
    acc_list = []
    acc_list.append(get_acc(test_loader, model))

    # train the model
    total_step = len(train_loader)
    loss_list = []
    for epoch in range(num_epochs):
        model.train()
        for i, (images, labels) in enumerate(train_loader):
            images = images.float().to(device)
            labels = labels.to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)

            optimizer.zero_grad()

            loss.backward()
            optimizer.step()

            loss_list.append(loss.item())
            if (i + 1) % 10 == 0:
                print ('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}'
                    .format(epoch+1, num_epochs, i+1, total_step, loss.item()))
        
        acc_list.append(get_acc(test_loader, model))

    plt.plot(loss_list)
    plt.savefig("img/mlp_loss.png")
    plt.show()
    
    plt.plot(acc_list)
    plt.savefig("img/mlp_acc.png")
    plt.show()
    


def get_acc(test_loader, model):
    # test the model
    model.eval()
    with torch.no_grad():
        correct = 0
        total = 0
        for images, labels in test_loader:

            images = images.float().to(device)
            labels = labels.to(device)

            outputs = model(images)

            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            acc = correct / total
        print('Test Accuracy of the model on the test images: {} '.format(acc))

    return acc



def main():
    input_size = 32 * 32
    num_epochs = 10
    batch_size = 100
    learning_rate = 0.001

    train_dataset = read_ori_data('data/optdigits-orig.tra') + read_ori_data('data/optdigits-orig.wdep') + read_ori_data('data/optdigits-orig.windep')
    test_dataset = read_ori_data('data/optdigits-orig.cv')

    # data loader (input pipeline)
    train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                            batch_size=batch_size,
                                            shuffle=True)

    test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                            batch_size=batch_size,
                                            shuffle=False)
    
    ori_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                            batch_size=1,
                                            shuffle=True)

    # init model, loss and optimizer
    model = MLP(input_size).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    # train
    train(train_loader, test_loader, model, criterion, optimizer, num_epochs=num_epochs)


    # save the model checkpoint
    torch.save(model.state_dict(), 'model.ckpt')

    pred_three = []
    model.eval()
    for images, labels in ori_loader:
        images = images.float().to(device)
        labels = labels.to(device)

        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)
        if predicted == 3:
            pred_three.append(images)
    
    for i in range(len(pred_three)):
        if i == 0:
            data = pred_three[i]
        else:
            data = torch.cat((data, pred_three[i]), dim=0)
    
    data = data.clone().detach().float().cpu()
    
    
    Y = pca(data, dims=2)
    axv = np.linspace(-10, 10, 5, endpoint=True)
    axh = np.linspace(-10, 10, 5, endpoint=True)
    focus = get_focus(Y[:, 0], Y[:, 1], axv, axh)
    vis_2_comp(data, Y[:, 0], Y[:, 1], axv, axh, focus, method='mlp', plot=True)



if __name__ == '__main__':
    main()
