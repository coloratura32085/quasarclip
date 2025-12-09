import torch
from torch import nn


class MappingNetwork(nn.Module):
    def __init__(self, input_dim=120, hidden_dims=[256, 512], output_dim=128):
        super(MappingNetwork, self).__init__()

        # 构建隐藏层
        layers = []
        current_dim = input_dim
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(current_dim, hidden_dim))
            layers.append(nn.ReLU())  # 激活函数
            current_dim = hidden_dim

        # 输出层
        layers.append(nn.Linear(current_dim, output_dim))

        # 将所有层组合成一个序列模型
        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)


# 实例化模型
photo_net = MappingNetwork(input_dim=120, hidden_dims=[256, 512,1024,512,256], output_dim=128)
#
# # 打印模型结构
# print(model)
#
# # 测试输入
# input_data = torch.randn(1, 120)  # 随机生成一个样本，维度为 120
# output = model(input_data)  # 前向传播
# print("Output shape:", output.shape)  # 输出形状应为 [1, 128]