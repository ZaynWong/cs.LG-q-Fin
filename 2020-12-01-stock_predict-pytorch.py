import torch
import torch.nn as nn
#import torchvision
from torch.autograd import Variable
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
#from torch import nn
from torch.utils.data import DataLoader
#import tushare as ts   #系统需要先安装tushar模块



#ts.set_token('xxx')#需要自己到网站www.tushare.pro注册获得token
#df = ts.pro_bar(ts_code='300059.SZ', adj='qfq', start_date='20190101', end_date='20201230')


#---参数配置----------------------------

#-----------------------------
#0--代表每次运算都随机初始化模型的权重参数，运行完以权重文件形式存储权重参数
#1（非零即可）--代表每次运算都使用上次运算的模型权重参数（上次运算权重已经存储在文件中）
#注意，若没有权重文件存在，而运行1，则程序会由于找不到权重文件而出错。
switch_model=1
#------------------------

epoch=10      
time_step=30   #输入的时间步，即一次输入多少天数据
hidden_size=10       #隐含层神经元数目
batch_size=2     #每一批次训练多少个样例
input_size=6      #输入维度
output_size=6     #输出维度
num_layers=1      #层数
lr=0.0001         #学习率
#-------end----------------



# -----获取数据--------------------------------
#f=open('dataset_3.csv')
#f=open('dataset_4.csv')
#f=open('300059-1.csv') #100天数据
f=open('300059.csv') #全部数据
df=pd.read_csv(f)     #读入股票数据
data=np.array(df.loc[:,['open','high','low','amount','vol','close']])
f.close()
#----------end-------------



#----处理数据--------------------- 
data=data[::-1]      #反转，使数据按照日期先后顺序排列

#以折线图展示data
#plt.figure()
#plt.plot(data[:,-1])
#plt.show()


length=data.shape[1]
for i in range (length):
    mean_data=np.mean(data[:,i])
    std_data=np.std(data[:,i])
    data[:,i]=(data[:,i]-mean_data)/std_data  #标准化
#print('normalize_data= \n',normalize_data)
#print('mean_data= \n',mean_data)
#print('std_data= \n',std_data)

#print('mean_data=',mean_data,'std_data=',std_data)
#print('normalize_data= \n',normalize_data)
#normalize_data=normalize_data[:,np.newaxis]       #增加维度x
#print('normalize_data_2= \n',normalize_data)
train_x,train_y=[],[]   #训练集
for i in range(len(data)-time_step):
    x=data[i:i+time_step]
    y=data[i+1:i+time_step+1]
    train_x.append(x)
    train_y.append(y)


train_x=torch.Tensor(train_x)
train_y=torch.Tensor(train_y)


#---------CNN算法增加的代码---------------

train_x=torch.unsqueeze(train_y,1) #在原有3维数组的第1维后面增加1维
train_y=torch.unsqueeze(train_y,1)#在原有3维数组的第1维后面增加1维
#print(train_x.shape)
#---------------------------------------


#train_x=Variable(train_x)
var_x = Variable(train_x)
var_y = Variable(train_y)

#-------end-------------

# 定义网络结构
class CNNnet(torch.nn.Module):
    def __init__(self):
        super(CNNnet,self).__init__()
        self.conv1 = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels=1,#in_channels,即图片的通道数量,黑白为1,RGB彩色为3,filter的层数默认与此数字一致
                            out_channels=5,#out_channels,即filter的数量
                            kernel_size=2,#kernel_size,4代表(4,4)即正方形的filter，若为长方形，则(height,width)
                            stride=1,#filter移动的步长,2代表(2,2)表示右移和下移都是一个像素，否则用(n,m)表示步长
                            padding=1),#图片外围每一条边补充0的层数,output_size=1+(input_size+2*padding-filter_size)/stride
            #torch.nn.BatchNorm2d(16),
            torch.nn.ReLU(True),
            #nn.MaxPool2d(kernel_size=1)
        )
        self.conv2 = torch.nn.Sequential(
            torch.nn.Conv2d(5,10,4,1,1),
            #torch.nn.BatchNorm2d(1),
            torch.nn.ReLU(True),
            #nn.MaxPool2d(kernel_size=1)
        )
        self.conv3 = torch.nn.Sequential(
            torch.nn.Conv2d(10,20,2,1,1),
            #torch.nn.BatchNorm2d(64),
            torch.nn.ReLU()
        )
        self.conv4 = torch.nn.Sequential(
            torch.nn.Conv2d(20,1,4,1,1),
            #torch.nn.BatchNorm2d(64),
            torch.nn.ReLU()
        )
        self.mlp1 = torch.nn.Linear(2*2*64,100)
        self.mlp2 = torch.nn.Linear(100,10)
    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        #x = self.conv3(x)
        #x = self.conv4(x)
        #x = self.mlp1(x.view(x.size(0),-1))
        #x = self.mlp2(x)
        return x

#————————算法模型——————————
class lstm_reg(nn.Module):
    def __init__(self,input_size,hidden_size,output_size,num_layers,batch_first):
        super(lstm_reg,self).__init__()
        self.rnn=nn.LSTM(input_size,hidden_size,num_layers,batch_first)   #隐藏层
        self.reg=nn.Linear(hidden_size,output_size)    #输出层

    def forward(self,x):
        x, (h_n, h_c) = self.rnn(x, None)  # None 表示 hidden state 会用全0的 state
        x = self.reg(x)
        return x



#-------end----------------------

class train_prediction():
    def __init__(self,distance_test):

        #使用GPU时，启用下一行
        self.device = torch.device(type='cuda')

        self.distance_test=distance_test
        #存储模型
        #if switch_model==0:
        #    self.net1 = lstm_reg(input_size,hidden_size,output_size,num_layers,batch_first=True)
        #else:
        #    self.net1=torch.load('./model_10.pkl') 

        #存储权重
        #self.net1 = lstm_reg(input_size,hidden_size,output_size,num_layers,batch_first=True)
        self.net1=CNNnet()
        if switch_model!=0:
            self.net1.load_state_dict(torch.load('./model_w.pkl'))

        #使用GPU时，启用下一行
        self.net1=self.net1.to(self.device)


    #———————————训练模型——————————
    def train_lstm(self):
        #criterion = nn.CrossEntropyLoss()#nn.MSELoss()
        criterion = torch.nn.MSELoss()
        #criterion = nn.BCELoss()
        optimizer = torch.optim.Adam(self.net1.parameters(),lr=lr)
        pair=[]
        for j in range(len(train_x) - self.distance_test):
            pair.append((var_x[j],var_y[j]))
        
        for i in range(epoch):
                                
            train_batches = DataLoader(dataset=pair, batch_size=batch_size, shuffle=True, pin_memory=True)            
            for (x,y) in train_batches:

                #使用GPU时，同时启用以下两行
                x = x.to(self.device)
                y=y.to(self.device)
                
                out=self.net1(x)
                loss=criterion(out,y)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

        #------------------------------------
        #若是GPU运算启用下一行（以下2行代码二选一）
        print('i=',i,'   loss=',loss.cpu().detach().numpy())
        #若是CPU运算，则启用下一行
        #print('i=',i,'   loss=',loss.detach().numpy())
        #--------------------------------------


        #torch.save(self.net1,'./model_10.pkl')
        torch.save(self.net1.state_dict(),'./model_w.pkl')

        #----------end-------------------

    #————————————用模型预测未来数据————————————
    def prediction(self,offset, predict_number):
        #net1 = lstm_reg(input_size, hidden_size, output_size, num_layers)
        end2 =len(train_x) - self.distance_test - offset  # 扣除offset的起始点位置
        prev_seq = train_x[end2]  # 预测起始点的输入
        label = []  # 测试标签
        pre_predict = []  # 记录用训练数据预测的结果，数据无意义，仅用于隐含层记忆历史数据
        predict = []  # 有效的预测结果

        error_all=0
        error_count=0
        prev_seq=torch.Tensor(prev_seq)
        prev_seq=torch.unsqueeze(prev_seq,0)
        prev_seq=Variable(prev_seq)
        for i in range(offset + predict_number):
            #print('prev_seq_0=',prev_seq)

            #GPU运算需要启用下一行
            #print('prev_seq=',prev_seq.shape)
            prev_seq=prev_seq.to(self.device)

            next_seq = self.net1(prev_seq)
            if i<self.distance_test+offset:
                label.append(train_y[end2 + i][-1][-1][-1])
            #if i==0:
            #    print('label_0=',train_y[end2 + i][-1][-1])
            if i < offset:  # 用训练集输入用于预测，预测结果无意义

                #------------------------------------
                #若是GPU运算启用下一行（以下2行代码二选一）
                pre_predict.append(next_seq[-1][-1][-1][-1].cpu().detach().numpy())
                #若是CPU运算，则启用下一行
                #pre_predict.append(next_seq[-1][-1][-1][-1].detach().numpy())
                #--------------------------------------

                prev_seq = train_y[end2 + i]
                prev_seq=torch.Tensor(prev_seq)
                prev_seq=torch.unsqueeze(prev_seq,0)
                prev_seq=Variable(prev_seq)
                # print('old=',prev_seq,'\n')
            else:  # 用上步预测结果作为当前步的输入，进行连续有效预测

                #------------------------------------
                #若是GPU运算启用下一行（以下2行代码二选一）
                predict.append(next_seq[-1][-1][-1][-1].cpu().detach().numpy())
                #若是CPU运算，则启用下一行
                #predict.append(next_seq[-1][-1][-1][-1].detach().numpy())
                #--------------------------------------
                if i<=len(var_x)-end2-1:
                    error_label=train_y[end2 + i][-1][-1][-1].numpy()
                    #------------------------------------
                    #若是GPU运算启用下一行（以下2行代码二选一）
                    error_predict=next_seq[-1][-1][-1][-1].cpu().detach().numpy()
                    #若是CPU运算，则启用下一行
                    #error_predict=next_seq[-1][-1][-1][-1].detach().numpy()
                    #--------------------------------------
                    error=abs((error_predict-error_label)/error_label)

                    error_all=error_all+error
                    error_count=error_count+1
                # 每次得到最后一个时间步的预测结果，与之前的数据加在一起，形成新的测试样本
                #prev_seq = torch.squeeze(prev_seq,dim=0)
                #if i==offset:
                #    print('prev_seq[-1,-1,1:]=',prev_seq[-1,-1,1:])
                #    print('next_seq[-1,-1,-1]=',next_seq[-1,-1,-1])
                #    print('next_seq=',next_seq)
                #    print('prev_seq.shape',prev_seq)
                #    print('next_seq.shape',next_seq)
                #prev_seq = torch.squeeze(prev_seq,dim=0)
                prev_seq=torch.cat((prev_seq[-1,-1,1:], next_seq[-1,:,-1]))
                prev_seq = torch.unsqueeze(prev_seq,dim=0)
                prev_seq = torch.unsqueeze(prev_seq,dim=0)                
                #if i==offset:
                #    print('prev_seq_2=',prev_seq)
                    

        label=np.array(label)
        error_all=error_all/(1+error_count)
        predict=np.array(predict)
        pre_predict=np.array(pre_predict)
        #---------end---------------
        
        #-------数据复原-------------------------
        label=label*std_data+mean_data
        predict=predict*std_data+mean_data
        pre_predict=pre_predict*std_data+mean_data
        
        #----------end-----------
        
        #-------数据输出--------------        
        #print('label=\n',label,'\n predict=\n',predict,'\n pre_predict=\n',pre_predict)

        np.savez('./index.npz', label, pre_predict, predict)  # 保存数据，用于画图。可运行draw.py作图
        return error_all
    #--------end---------------------------

if __name__ == "__main__":

    #----参数配置--------------------
    distance_test = 90  # 训练数据的截止点离最新数据的距离

    predict_number = 30  # 连续预测天数
    # 已经训练过的输入数据作为预测时的输入。由于LSTM隐含层和历史输入数据相关，
    # 当用于预测时，需要用一段训练数据作为预测输入，但该段数据
    # 的预测结果没有意义，仅仅是让模型隐含层记忆历史数据

    offset = 30
    # 训练数据的截止点离最近数据的真实距离，因为训练是以batch_size为单位进行训练的。
    # 因此real_distance_test大于等于distance_test.
    #------end------------------

    #-------训练和预测过程--------------------
    instance_train_prediction=train_prediction(distance_test)
    instance_train_prediction.train_lstm()
    #instance_train_prediction.prediction(offset,predict_number)
    error_all=instance_train_prediction.prediction(offset,predict_number)
    print('error_all=',error_all)
    #----------------end-----------------------
    
    #---------数据可视化表示----------------    
    D = np.load('./index.npz')
    label = D['arr_0']
    pre_predict = D['arr_1']
    predict = D['arr_2']
    D.close()

    plt.figure()
    plt.plot(list(range(len(label))), label, color='b')
    plt.plot(list(range(len(pre_predict))), pre_predict, color='r')
    plt.plot(list(range(len(pre_predict), len(pre_predict) + len(predict))), predict, color='y')
    plt.show()
#---------------end--------------------------    