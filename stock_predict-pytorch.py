import torch
from torch.autograd import Variable
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from torch import nn
import torch.utils.data as Data

f = open('dataset_3.csv')
df = pd.read_csv(f)  # 读入股票数据
data = np.array(df['high'])  # 获取最高价序列
data = data[::-1]  # 反转，使数据按照日期先后顺序排列
# 以折线图展示data

plt.figure()
plt.plot(data)
# plt.show()

time_step = 60  # 时间步
num_layers = 1  # 层数
hidden_size = 60  # hidden layer units
batch_size = 20  # 每一批次训练多少个样例
input_size = 1  # 输入层维度
output_size = 1  # 输出层维度
lr = 0.0006  # 学习率
train_x, train_y = [], []  # 训练集

normalize_data = (data - np.mean(data)) / np.std(data)  # 标准化
mean_data = np.mean(data)
std_data = np.std(data)
# print('mean_data=',mean_data,'std_data=',std_data)

normalize_data = normalize_data[:, np.newaxis]  # 增加维度x

for i in range(len(normalize_data) - time_step - 1):
    x = normalize_data[i:i + time_step]
    y = normalize_data[i + 1:i + time_step + 1]
    train_x.append(x.tolist())
    train_y.append(y.tolist())

train_x = torch.Tensor(train_x)
train_y = torch.Tensor(train_y)
train_x = Variable(train_x)
var_x = Variable(train_x)
var_y = Variable(train_y)
"""torch_dataset=Data.TensorDataset(train_x,train_y)
loader=Data.DataLoader(
    dataset=torch_dataset,
    batch_size=batch_size,
    shuffle=False,
)"""


# ————————搭建模型——————————
class lstm_reg(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layers):
        super(lstm_reg, self).__init__()
        self.rnn = nn.LSTM(input_size, hidden_size, num_layers)  # 隐藏层
        self.reg = nn.Linear(hidden_size, output_size)  # 输出层

    def forward(self, x):
        x, (h_n, h_c) = self.rnn(x, None)  # None 表示 hidden state 会用全0的 state
        x = self.reg(x)
        return x


net1 = lstm_reg(input_size, hidden_size, output_size, num_layers)


# ———————————训练模型——————————
def train_lstm(distance_test):
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(net1.parameters(), lr=lr)
    for i in range(2):
        start = 0
        end = start + batch_size
        while (end < len(train_x) - distance_test):
            out = net1(var_x[start:end]).view(-1)
            p = var_y[start:end].view(-1)
            loss = criterion(out, p)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            start += batch_size
            end = start + batch_size  # 每次用一批，即batch_size进行训练
        real_distance_test = end - batch_size  # 循环跳出来之前，end已经加了一个batch_size，这里要减回去
        print(i)
    return real_distance_test


# ————————————预测模型————————————
def prediction(real_distance_test, offset, predict_number):
    # net1 = lstm_reg(input_size, hidden_size, output_size, num_layers)
    end2 = real_distance_test - offset  # 扣除offset的起始点位置
    prev_seq = train_x[end2]  # 预测起始点的输入
    label = []  # 测试标签
    pre_predict = []  # 记录用训练数据预测的结果，数据无意义，仅用于隐含层记忆历史数据
    predict = []  # 有效的预测结果
    # 得到之后100个预测结果
    for i in range(offset + predict_number):
        prev_seq = torch.Tensor(prev_seq)
        prev_seq = torch.unsqueeze(prev_seq, 1)
        next_seq = net1(prev_seq)
        label.append(train_y[end2 + i][-1])
        if i < offset:  # 用训练集输入用于预测，预测结果无意义
            pre_predict.append(next_seq[-1])
            prev_seq = train_x[end2 + i + 1]
            # print('old=',prev_seq,'\n')
        else:  # 用上步预测结果作为当前步的输入，进行连续有效预测
            predict.append(next_seq[-1].detach().numpy())
            # 每次得到最后一个时间步的预测结果，与之前的数据加在一起，形成新的测试样本
            prev_seq = torch.squeeze(prev_seq, dim=2)
            prev_seq = torch.cat((prev_seq[1:], next_seq[-1]))
            prev_seq = prev_seq.tolist()
            # print('new=',prev_seq,'\n')
    label = np.array(label).reshape(-1, 1)
    # label=label[0]
    predict = np.array(predict)
    predict = predict.reshape(-1, 1)
    label = label * std_data + mean_data
    predict = predict * std_data + mean_data
    print('label=\n', label, '\n predict=\n', predict)

    np.savez('./index.npz', label, pre_predict, predict)  # 保存数据，用于画图。可运行draw.py作图


if __name__ == "__main__":
    distance_test = 450  # 训练数据的截止点离最新数据的距离
    predict_number = 10  # 连续预测天数
    # 已经训练过的输入数据作为预测时的输入。由于LSTM隐含层和历史输入数据相关，
    # 当用于预测时，需要用一段训练数据作为预测输入，但该段数据
    # 的预测结果没有意义，仅仅是让模型隐含层记忆历史数据
    offset = 0
    # 训练数据的截止点离最近数据的真实距离，因为训练是以batch_size为单位进行训练的。
    # 因此real_distance_test大于等于distance_test.
    real_distance_test = train_lstm(distance_test)
    prediction(real_distance_test, offset, predict_number)

    D = np.load('./index.npz')
    label = D['arr_0']
    pre_predict = D['arr_1']
    predict = D['arr_2']

    plt.figure()
    plt.plot(list(range(len(label))), label, color='b')
    plt.plot(list(range(len(pre_predict))), pre_predict, color='r')
    plt.plot(list(range(len(pre_predict), len(pre_predict) + len(predict))), predict, color='y')
    plt.show()
