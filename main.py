import os
import time
import torch
import csv
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader

from stmgcn.net.dataset import LoadData
from stmgcn.net.models import GCN

def main():

    # Loading Dataset 数据下载 -----------------------------
    train_data = LoadData(data_path=["data/gps_20161101", "data/order_20161101"], num_nodes=None, divide_days=[23, 7],
                          time_interval=5, history_length=6,
                          train_mode="train")
    train_loader = DataLoader(train_data, batch_size=64, shuffle=True, num_workers=32)

    test_data = LoadData(data_path=["data/gps_20161101", "data/order_20161101"], num_nodes=None, divide_days=[23, 7],
                         time_interval=5, history_length=6,
                         train_mode="test")

    test_loader = DataLoader(test_data, batch_size=64, shuffle=True, num_workers=32)


    # Loading Model  导入模型 -------------------------
    my_net = STMGCN(nfeat1=None,nfeat2 =None,nfeat3=None,
                    nhid1 =None,nhid2=None,nhid3 =None,
                    nclass1=None,nclass2=None,nclass3=None,dropout=None)
  #设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    #将网络送入到GPU
    my_net = my_net.to(device)
     #损失函数
    criterion = nn.MSELoss()
    #adam优化器
    optimizer = optim.Adam(params=my_net.parameters())


    # Train model  训练模型
    Epoch = 200

    my_net.train()
    for epoch in range(Epoch):
        epoch_loss = 0.0  #损失值初始化
        start_time = time.time()
        for data in train_loader:  # ["graph": [B, N, N] , "flow_x": [B, N, H, D], "flow_y": [B, N, 1, D]]
            #模型梯度清零
            my_net.zero_grad()
            #获取预测值
            predict_value = my_net(data, device).to(torch.device("cpu"))  # [0, 1] -> recover
             #计算损失函数（预测值，真实值）-----------------------------
            loss = criterion(predict_value, data["flow_y"])

            epoch_loss += loss.item()
             #损失反向传播
            loss.backward()
            #修正参数
            optimizer.step()
        end_time = time.time()
             #查看每批次损失函数，每批次数据运行时间
        print("Epoch: {:04d}, Loss: {:02.4f}, Time: {:02.2f} mins".format(epoch, 1000 * epoch_loss / len(train_data),
                                                                          (end_time - start_time) / 60))

    # Test Model  导入测试模型
    # TODO: Visualize the Prediction Result
    # TODO: Measure the results with metrics MAE, MAPE, and RMSE
    my_net.eval()  #打开测试模式
    with torch.no_grad(): #模型梯度关闭

        total_loss = 0.0   #整体损失值初始化
        for data in test_loader:
            predict_value = my_net(data, device).to(torch.device("cpu"))  # [B, N, 1, D]
                #---------------------------------------------
            loss = criterion(predict_value, data["flow_y"])
            total_loss += loss.item()

        print("Test Loss: {:02.4f}".format(1000 * total_loss / len(test_data)))

        print("helloworld")




if __name__ == '__main__':

  main()
