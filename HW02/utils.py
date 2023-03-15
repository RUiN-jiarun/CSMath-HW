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
    three = []

    for i in dataset:
        if i[1] == label:
            three.append(torch.unsqueeze(i[0], dim=0))

    for i in range(len(three)):
        if i == 0:
            data = three[i]
        else:
            data = torch.cat((data, three[i]), dim=0)
    
    return data.clone().detach().float()