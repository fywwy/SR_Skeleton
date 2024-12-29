# import torch
#
# import torch.nn as nn
# import torch.nn.functional as F
#
# # 这个代码中省略了输入矩阵x转变为qkv的过程！！！
# class MultiHeadAttention(nn.Module):
#     def __init__(self, heads, d_model):
#         super(MultiHeadAttention, self).__init__()
#         self.d_model = d_model
#         self.heads = heads
#         # 定义K, Q, V的权重矩阵
#         # 多头注意力中K、Q、V的线性层具有相同输入和输出尺寸是一种常见且实用的设计选择！！！
#         self.k_linear = nn.Linear(d_model, d_model)
#         self.q_linear = nn.Linear(d_model, d_model)
#         self.v_linear = nn.Linear(d_model, d_model)
#         # 分头后的维度
#         self.d_token = d_model // heads
#         # 定义输出权重矩阵
#         self.out = nn.Linear(d_model, d_model)
#
#     def forward(self, q, k, v):
#         # 计算batch大小
#         batch = q.size(0)
#
#         # 线性变换后的Q, K, V，然后分割成多个头
#         k = self.k_linear(k).view(batch, -1, self.heads, self.d_token)
#         q = self.q_linear(q).view(batch, -1, self.heads, self.d_token)
#         v = self.v_linear(v).view(batch, -1, self.heads, self.d_token)
#
#         # 转置调整维度，以计算注意力分数
#         k = k.transpose(1, 2)  # 形状变为 [batch, heads, seq_len, d_token]
#         q = q.transpose(1, 2)
#         v = v.transpose(1, 2)
#
#         # 计算自注意力分数
#         scores = self.attention(q, k, v, self.d_token)
#
#         # 调整形状以进行拼接
#         scores = scores.transpose(1, 2).contiguous().view(batch, -1, self.d_model)
#
#         # 通过输出权重矩阵进行线性变换
#         output = self.out(scores)
#         return output
#
#     @staticmethod
#     def attention(q, k, v, d_token):
#         # 计算注意力分数 (q @ k^T) / sqrt(d_token)
#         scores = torch.matmul(q, k.transpose(-2, -1)) / torch.sqrt(d_token)
#         # 应用softmax归一化（沿着最后一个维度（dim=-1））
#         attn = F.softmax(scores, dim=-1)
#         # 计算加权的V
#         output = torch.matmul(attn, v)
#         return output
# # 设置模型参数
# batch_size = 2   # 批大小
# seq_len = 5      # 序列长度
# d_model = 16     # 输入特征维度（需要是 heads 的整数倍）
# heads = 4        # 注意力头数
#
# # 创建 MultiHeadAttention 模型
# multi_head_attention = MultiHeadAttention(heads=heads, d_model=d_model)
#
# # 生成随机输入数据
# # q, k, v 的形状应为 [batch_size, seq_len, d_model]
# q = torch.randn(batch_size, seq_len, d_model)
# k = torch.randn(batch_size, seq_len, d_model)
# v = torch.randn(batch_size, seq_len, d_model)
#
# # 调用 MultiHeadAttention 模型
# output = multi_head_attention(q, k, v)
#
# # 打印输入和输出形状
# print("输入 Q 形状:", q.shape)  # [batch_size, seq_len, d_model]
# print("输入 K 形状:", k.shape)  # [batch_size, seq_len, d_model]
# print("输入 V 形状:", v.shape)  # [batch_size, seq_len, d_model]
# print("输出形状:", output.shape)  # [batch_size, seq_len, d_model]
import torch
import torch.nn as nn

class Upsample(nn.Module):
    def __init__(self):
        super(Upsample, self).__init__()
        # 使用转置卷积实现上采样，目标是将高度从 1 上采样到 300
        self.deconv = nn.ConvTranspose2d(3, 3, kernel_size=(3, 1), stride=(300, 1), padding=(1, 0))

    def forward(self, x):
        return self.deconv(x)

# 示例输入
x = torch.randn(8, 3, 1, 64)  # 输入形状: (batch_size=8, channels=3, height=1, width=64)

# 创建上采样模块
upsample = Upsample()

# 前向传播
output = upsample(x)
print(output.shape)  # 输出形状
