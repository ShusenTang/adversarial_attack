import time
import torch
from torch import nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
import numpy as np
import matplotlib.pyplot as plt



class FlattenLayer(torch.nn.Module):
    def __init__(self):
        super(FlattenLayer, self).__init__()
    def forward(self, x): # x shape: (batch, *, *, ...)
        return x.view(x.shape[0], -1)

class white_model(nn.Module):
    def __init__(self):
        super(white_model, self).__init__()
        self.infer = nn.Sequential(
            nn.Conv2d(1, 32, 5, padding = 2), 
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(32, 64, 5, padding = 2),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            FlattenLayer(),
            nn.Linear(64*7*7, 1024),
            nn.ReLU(),
            nn.Dropout(p=0.4),
            nn.Linear(1024, 10)
            
        )

    def forward(self, img):
        return self.infer(img)
    
    

def load_data_fashion_mnist(batch_size, resize=None, root='~/Datasets/FashionMNIST'):
    trans = []
    if resize:
        trans.append(torchvision.transforms.Resize(size=resize))
    trans.append(torchvision.transforms.ToTensor())
    
    transform = torchvision.transforms.Compose(trans)
    mnist_train = torchvision.datasets.FashionMNIST(root=root, train=True, download=True, transform=transform)
    mnist_test = torchvision.datasets.FashionMNIST(root=root, train=False, download=True, transform=transform)

    num_workers = 4
    train_iter = torch.utils.data.DataLoader(mnist_train, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    test_iter = torch.utils.data.DataLoader(mnist_test, batch_size=batch_size, shuffle=True, num_workers=num_workers)

    return train_iter, test_iter


def evaluate_accuracy(data_iter, net):
    device = list(net.parameters())[0].device
    net.eval() # 评估模式, 这会关闭dropout
    acc_sum, n = 0.0, 0
    with torch.no_grad():
        for X, y in data_iter:
            acc_sum += (net(X.to(device)).argmax(dim=1) == y.to(device)).float().sum().cpu().item()
            n += y.shape[0]
    net.train() # 改回训练模式
    return acc_sum / n


def train(net, train_iter, test_iter, batch_size, optimizer, device, num_epochs):
    net = net.to(device)
    print("training on ", device)
    loss = torch.nn.CrossEntropyLoss()
    for epoch in range(num_epochs):
        train_l_sum, train_acc_sum, n, start = 0.0, 0.0, 0, time.time()
        for X, y in train_iter:
            X = X.to(device)
            y = y.to(device)
            y_hat = net(X)
            l = loss(y_hat, y)
            optimizer.zero_grad()
            l.backward()
            optimizer.step()
            train_l_sum += l.cpu().item()
            train_acc_sum += (y_hat.argmax(dim=1) == y).sum().cpu().item()
            n += y.shape[0]
        test_acc = evaluate_accuracy(test_iter, net)
        print('epoch %d, loss %.4f, train acc %.3f, test acc %.3f, time %.1f sec'
              % (epoch + 1, train_l_sum / n, train_acc_sum / n, test_acc, time.time() - start))



def get_fashion_mnist_labels(labels):
    text_labels = ['0:t-shirt', '1:trouser', '2:pullover', '3:dress', '4:coat',
                   '5:sandal', '6:shirt', '7:sneaker', '8:bag', '9:ankle boot']
    return [text_labels[int(i)] for i in labels]


def show_fashion_mnist(images, labels):
    _, figs = plt.subplots(1, len(images), figsize=(15, 15))
    for f, img, lbl in zip(figs, images, labels):
        f.imshow(img.view((28, 28)).numpy())
        f.set_title(lbl)
        f.axes.get_xaxis().set_visible(False)
        f.axes.get_yaxis().set_visible(False)
    plt.show()
    
    
    
def select_right_sample(eval_net, data_iter, num = 1000):
    assert not eval_net.training # 确保关闭了dropout
    count = 0
    x_list = []
    y_list = []
    device = list(eval_net.parameters())[0].device
    for X, y in data_iter:
        correct_idx = (eval_net(X.to(device)).argmax(dim=1) == y.to(device)).float()
        for i in range(X.shape[0]):
            if correct_idx[i] == 1.0:  
                x_list.append(X[i])
                y_list.append(y[i])
                count += 1
                if count == num:
                    return torch.stack(x_list), torch.stack(y_list)
    return torch.stack(x_list), torch.stack(y_list) 


def white_box_attack(eval_net, X, y, lr = 0.01, max_step = 500): # X shape: (1, 1, 28, 28)
    """白盒攻击"""
    device = list(eval_net.parameters())[0].device
    X = X.to(device)
    y = y.to(device)
    X_ = X.clone().detach().to(device)
    X_.requires_grad_(True)
    loss = torch.nn.CrossEntropyLoss()
    y_target = (y + 1) % 10
    
    for step in range(max_step):
        #print(X_.device)
        y_hat = eval_net(X_)
        if (y_hat.argmax(dim=1) == y_target).float().sum().cpu().item():
            # print("Successful attack at setp %d " % step)
            return X_.cpu().data, y_hat.argmax(dim=1).cpu().data, True
        
        l = loss(y_hat, y_target) 
        if X_.grad is not None:
            X_.grad.data.zero_() # 梯度清0
        l.backward()
        X_.data -= lr * X_.grad
        
    # print("Attack failure!")
    return X_.cpu().data, y_hat.argmax(dim=1).cpu().data, False


def black_box_attack(eval_net, X, y, sigma=0.01, max_step=100): # X shape: (1, 1, 28, 28)
    """MCMC黑盒攻击"""
    device = list(eval_net.parameters())[0].device
    X = X.to(device)
    y = y.to(device)
    X_ = X.clone().detach().to(device)
    y_target = (y + 1) % 10
    
    for step in range(max_step):
        tmp_X = X_.data + torch.tensor(np.random.normal(0, sigma, size=X_.size()), dtype=torch.float32, device=device)
        y_hat = eval_net(tmp_X)
        y_hat = nn.functional.softmax(y_hat, dim=-1)
        
        if torch.randn(1)[0] > y_hat[0, y_target[0]].item(): # 接受
            X_ = tmp_X
            
        if (y_hat.argmax(dim=1) == y_target).float().sum().cpu().item():
            return X_.cpu().data, y_hat.argmax(dim=1).cpu().data, True                    
        
    # print("Attack failure!")
    return X_.cpu().data, y_hat.argmax(dim=1).cpu().data, False