import copy
import datetime
import numpy as np
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import accuracy_score
from tqdm import tqdm
import torch
import pandas as pd
import torch.nn as nn
from torch.nn.utils import weight_norm
from FDRandomWalkSSA import RandomWalkSSA
from utils import load_data, plot_history_torch, plot_heat_map
from myOptimizer_adamZZB import MyAdam,SGD,Adagrad
# project root path
project_path = "../"
# define log directory
log_dir = project_path + "logs/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
model_path = project_path + "ecg_model.pt"

# the device to use
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print("Using {} device".format(device))


# define the dataset class
class ECGDataset(Dataset):
    def __init__(self, x, y):
        self.x = x
        self.y = y

    def __getitem__(self, index):
        x = torch.tensor(self.x[index], dtype=torch.float32)
        y = torch.tensor(self.y[index], dtype=torch.long)
        return x, y

    def __len__(self):
        return len(self.x)


# build the tcn model
class Chomp1d(nn.Module):
    def __init__(self, chomp_size):
        super(Chomp1d, self).__init__()
        self.chomp_size = chomp_size

    def forward(self, x):
        """
        其实这就是一个裁剪的模块，裁剪多出来的padding
        """
        return x[:, :, :-self.chomp_size].contiguous()


class TemporalBlock(nn.Module):
    def __init__(self, n_inputs, n_outputs, kernel_size, stride, dilation, padding, dropout=0.2):
        """
        相当于一个Residual block

        :param n_inputs: int, 输入通道数    5 32 16 4
        :param n_outputs: int, 输出通道数   32 16 4  1
        :param kernel_size: int, 卷积核尺寸 3
        :param stride: int, 步长，一般为1
        :param dilation: int, 膨胀系数
        :param padding: int, 填充系数
        :param dropout: float, dropout比率
        """
        super(TemporalBlock, self).__init__()
        self.conv1 = weight_norm(nn.Conv1d(n_inputs, n_outputs, kernel_size,
                                           stride=stride, padding=padding, dilation=dilation))
        # 经过conv1，输出的size其实是(Batch, input_channel, seq_len + padding)
        self.chomp1 = Chomp1d(padding)  # 裁剪掉多出来的padding部分，维持输出时间步为seq_len
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout)

        self.conv2 = weight_norm(nn.Conv1d(n_outputs, n_outputs, kernel_size,
                                           stride=stride, padding=padding, dilation=dilation))
        self.chomp2 = Chomp1d(padding)  # 裁剪掉多出来的padding部分，维持输出时间步为seq_len
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(dropout)

        self.net = nn.Sequential(self.conv1, self.chomp1, self.relu1, self.dropout1,
                                 self.conv2, self.chomp2, self.relu2, self.dropout2)
        self.downsample = nn.Conv1d(n_inputs, n_outputs, 1) if n_inputs != n_outputs else None
        self.relu = nn.ReLU()
        self.init_weights()

    def init_weights(self):
        """
        参数初始化

        :return:
        """
        self.conv1.weight.data.normal_(0, 0.01)
        self.conv2.weight.data.normal_(0, 0.01)
        if self.downsample is not None:
            self.downsample.weight.data.normal_(0, 0.01)

    def forward(self, x):
        """
        :param x: size of (Batch, input_channel, seq_len)
        :return:
        """
        out = self.net(x)
        res = x if self.downsample is None else self.downsample(x)

        return (out + res)


class TCN(nn.Module):
    def __init__(self, num_inputs, num_channels, kernel_size=2, dropout=0.2):#tcn(num_inputs=5,num_channels=[32,16,4,1],kernel_size=3, dropout=0.3)
        """
        TCN，目前paper给出的TCN结构很好的支持每个时刻为一个数的情况，即sequence结构，
        对于每个时刻为一个向量这种一维结构，勉强可以把向量拆成若干该时刻的输入通道，
        对于每个时刻为一个矩阵或更高维图像的情况，就不太好办。

        :param num_inputs: int， 输入通道数
        :param num_channels: list，每层的hidden_channel数，例如[25,25,25,25]表示有4个隐层，每层hidden_channel数为25
        :param kernel_size: int, 卷积核尺寸
        :param dropout: float, drop_out比率
        """
        super(TCN, self).__init__()
        self.linear = nn.Linear(num_channels[-1], 5)
        layers = []
        num_levels = len(num_channels)
        for i in range(num_levels):
            dilation_size = 2 ** i  # 膨胀系数：1，2，4，8……
            in_channels = num_inputs if i == 0 else num_channels[i - 1]  # 确定每一层的输入通道数# 5 32 16 4
            out_channels = num_channels[i]  # 确定每一层的输出通道数                               32 16 4  1
            layers += [TemporalBlock(in_channels, out_channels, kernel_size, stride=1, dilation=dilation_size,
                                     padding=(kernel_size - 1) * dilation_size, dropout=dropout)]

        self.network = nn.Sequential(*layers)

    def forward(self, x):

        #print("x=",x.shape)#x= torch.Size([64, 5, 6])
        """
        输入x的结构不同于RNN，一般RNN的size为(Batch, seq_len, channels)或者(seq_len, Batch, channels)，
        这里把seq_len放在channels后面，把所有时间步的数据拼起来，当做Conv1d的输入尺寸，实现卷积跨时间步的操作，
        很巧妙的设计。

        :param x: size of (Batch, input_channel, seq_len)
        :return: size of (Batch, output_channel, seq_len)
        """
        x=x.unsqueeze(dim=2)
        #print("x=",x.shape)#x= torch.Size([64, 5, 6])
        out=self.network(x)
        #print("out1=",out.shape)
        out=out[:,:,-1]
        #print("out=", out.shape)

        out = self.linear(out)

        return out



# define the training function and validation function
def train_steps(loop, model, criterion, optimizer):
    train_loss = []
    train_acc = []
    model.train()
    for step_index, (X, y) in loop:
        X, y = X.to(device), y.to(device)
        pred = model(X)
        loss = criterion(pred, y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        loss = loss.item()
        train_loss.append(loss)
        pred_result = torch.argmax(pred, dim=1).detach().cpu().numpy()
        y = y.detach().cpu().numpy()
        acc = accuracy_score(y, pred_result)
        train_acc.append(acc)
        loop.set_postfix(loss=loss, acc=acc)
    return {"loss": np.mean(train_loss),
            "acc": np.mean(train_acc)}


def eval_steps(loop, model, criterion):
    val_loss = []
    val_acc = []
    model.eval()
    with torch.no_grad():
        for step_index, (X, y) in loop:
            X, y = X.to(device), y.to(device)
            pred = model(X)
            loss = criterion(pred, y).item()

            val_loss.append(loss)
            pred_result = torch.argmax(pred, dim=1).detach().cpu().numpy()
            y = y.detach().cpu().numpy()
            acc = accuracy_score(y, pred_result)
            val_acc.append(acc)
            loop.set_postfix(loss=loss, acc=acc)
    return {"loss": np.mean(val_loss),
            "acc": np.mean(val_acc)}


def train_epochs(train_dataloader, val_dataloader,  model, criterion, optimizer, config):
    num_epochs = config['num_epochs']
    train_loss_ls = []
    train_loss_acc = []
    val_loss_ls = []
    val_loss_acc = []


    for epoch in range(num_epochs):
        train_loop = tqdm(enumerate(train_dataloader), total=len(train_dataloader), disable=True)
        test_loop = tqdm(enumerate(val_dataloader), total=len(val_dataloader), disable=True)
        train_loop.set_description(f'Epoch [{epoch + 1}/{num_epochs}]')
        test_loop.set_description(f'Epoch [{epoch + 1}/{num_epochs}]')

        train_metrix = train_steps(train_loop, model, criterion, optimizer)
        test_metrix = eval_steps(test_loop, model, criterion)

        train_loss_ls.append(train_metrix['loss'])
        train_loss_acc.append(train_metrix['acc'])
        val_loss_ls.append(test_metrix['loss'])
        val_loss_acc.append(test_metrix['acc'])



    return {
        'train_loss': train_loss_ls,
        'train_acc': train_loss_acc,
        'val_loss': val_loss_ls,
        'val_acc': val_loss_acc
    }


def evaluate_model(lr, batch_size, dropout_rate, kernel_size):
    batch_size = int(batch_size)
    kernel_size = int(kernel_size)
    # Update the configuration
    config = {
        'seed': 1,
        'test_ratio': 0.1,  # the ratio of the test set
        'val_ratio' : 0.1,
        'num_epochs': 1,
        'batch_size': batch_size,
        'lr': lr,
        'kernel_size': kernel_size,  # 添加kernel_size到配置中
    }
    # Load the data
    X_train, X_val, X_test, y_train, y_val, y_test = load_data(config['test_ratio'], config['val_ratio'],config['seed'])
    train_dataset, test_dataset = ECGDataset(X_train, y_train), ECGDataset(X_test, y_test)
    val_dataset = ECGDataset(X_val, y_val)
    train_dataloader = DataLoader(train_dataset, batch_size=config['batch_size'], shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=config['batch_size'], shuffle=False)
    test_dataloader = DataLoader(test_dataset, batch_size=config['batch_size'], shuffle=False)

    # Define the model
    model = TCN(num_inputs=300, num_channels=[128, 64, 32, 5], kernel_size=config['kernel_size'], dropout=dropout_rate)
    model = model.to(device)

    # Define the loss function and the optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=config['lr'])

    # Train and evaluate model
    history = train_epochs(train_dataloader, val_dataloader, model, criterion, optimizer, config)

    # Here, the validation loss is considered as the objective to minimize
    objective = np.mean(history['val_loss'])
    return objective


def main():
    pop = 30
    dim = 4
    lb = [0.0001, 32, 0.001, 3]
    ub = [0.1, 512, 0.5, 7]
    Max_iter = 50

    # Run FDSSA optimization
    best_score, best_params = RandomWalkSSA(pop, dim, lb, ub, Max_iter, evaluate_model)
    # print('Best parameters found: lr = {}, batch_size = {}, dropout = {}'.format(best_params[0], int(best_params[1]),
    #                                                         best_params[2]))
    print('FDRWSSA end:',best_score, best_params)

main()
