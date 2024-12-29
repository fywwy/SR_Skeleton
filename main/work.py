'''
odel(
  (data_bn): SyncBatchNorm(150, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
  (layers): ModuleList(
    (0): MST_GCN_block(
      (msgcn): Basic_GCN_layer(
        (conv): SpatialGraphCov(
          (gcn): Conv2d(3, 336, kernel_size=(1, 1), stride=(1, 1))
        )
        (bn): SyncBatchNorm(112, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (residual): Zero_Layer()
        (act): Activations(
          (act): ReLU(inplace=True)
        )
      )
      (mstcn): Basic_TCN_layer(
        (conv): Conv2d(112, 112, kernel_size=(9, 1), stride=(1, 1), padding=(4, 0))
        (bn): SyncBatchNorm(112, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (residual): Zero_Layer()
        (act): Activations(
          (act): ReLU(inplace=True)
        )
      )
    )
    (1-3): 3 x MST_GCN_block(
      (msgcn): MS_GCN_layer(
        (act): Activations(
          (act): ReLU(inplace=True)
        )
        (bn_in): SyncBatchNorm(112, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (conv_bn): ModuleList(
          (0-3): 4 x Sequential(
            (0): SpatialGraphCov(
              (gcn): Conv2d(28, 84, kernel_size=(1, 1), stride=(1, 1))
            )
            (1): SyncBatchNorm(28, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          )
        )
        (residual): Identity()
      )
      (mstcn): MS_TCN_layer(
        (act): Activations(
          (act): ReLU(inplace=True)
        )
        (bn_in): SyncBatchNorm(112, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (conv_bn): ModuleList(
          (0-3): 4 x Sequential(
            (0): Conv2d(28, 28, kernel_size=(9, 1), stride=(1, 1), padding=(4, 0))
            (1): SyncBatchNorm(28, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          )
        )
        (residual): Identity()
      )
    )
    (4): MST_GCN_block(
      (msgcn): MS_GCN_layer(
        (act): Activations(
          (act): ReLU(inplace=True)
        )
        (bn_in): SyncBatchNorm(112, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (conv_bn): ModuleList(
          (0-3): 4 x Sequential(
            (0): SpatialGraphCov(
              (gcn): Conv2d(28, 168, kernel_size=(1, 1), stride=(1, 1))
            )
            (1): SyncBatchNorm(56, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          )
        )
        (residual): Sequential(
          (0): Conv2d(112, 224, kernel_size=(1, 1), stride=(1, 1))
          (1): SyncBatchNorm(224, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        
      )
      (mstcn): MS_TCN_layer(
        (act): Activations(
          (act): ReLU(inplace=True)
        )
        (bn_in): SyncBatchNorm(224, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (conv_bn): ModuleList(
          (0-3): 4 x Sequential(
            (0): Conv2d(56, 56, kernel_size=(9, 1), stride=(2, 1), padding=(4, 0))
            (1): SyncBatchNorm(56, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          )
        )
        (residual): Sequential(
          (0): Conv2d(224, 224, kernel_size=(1, 1), stride=(2, 1))
          (1): SyncBatchNorm(224, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        )
      )
    )
    (5-6): 2 x MST_GCN_block(
      (msgcn): MS_GCN_layer(
        (act): Activations(
          (act): ReLU(inplace=True)
        )
        (bn_in): SyncBatchNorm(224, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (conv_bn): ModuleList(
          (0-3): 4 x Sequential(
            (0): SpatialGraphCov(
              (gcn): Conv2d(56, 168, kernel_size=(1, 1), stride=(1, 1))
            )
            (1): SyncBatchNorm(56, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          )
        )
        (residual): Identity()
      )
      (mstcn): MS_TCN_layer(
        (act): Activations(
          (act): ReLU(inplace=True)
        )
        (bn_in): SyncBatchNorm(224, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (conv_bn): ModuleList(
          (0-3): 4 x Sequential(
            (0): Conv2d(56, 56, kernel_size=(9, 1), stride=(1, 1), padding=(4, 0))
            (1): SyncBatchNorm(56, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          )
        )
        (residual): Identity()
      )
    )
    (7): MST_GCN_block(
      (msgcn): MS_GCN_layer(
        (act): Activations(
          (act): ReLU(inplace=True)
        )
        (bn_in): SyncBatchNorm(224, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (conv_bn): ModuleList(
          (0-3): 4 x Sequential(
            (0): SpatialGraphCov(
              (gcn): Conv2d(56, 336, kernel_size=(1, 1), stride=(1, 1))
            )
            (1): SyncBatchNorm(112, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          )
        )
        (residual): Sequential(
          (0): Conv2d(224, 448, kernel_size=(1, 1), stride=(1, 1))
          (1): SyncBatchNorm(448, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        )
      )
      (mstcn): MS_TCN_layer(
        (act): Activations(
          (act): ReLU(inplace=True)
        )
        (bn_in): SyncBatchNorm(448, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (conv_bn): ModuleList(
          (0-3): 4 x Sequential(
            (0): Conv2d(112, 112, kernel_size=(9, 1), stride=(2, 1), padding=(4, 0))
            (1): SyncBatchNorm(112, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          )
        )
        (residual): Sequential(
          (0): Conv2d(448, 448, kernel_size=(1, 1), stride=(2, 1))
          (1): SyncBatchNorm(448, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        )
      )
    )
    (8-9): 2 x MST_GCN_block(
      (msgcn): MS_GCN_layer(
        (act): Activations(
          (act): ReLU(inplace=True)
        )
        (bn_in): SyncBatchNorm(448, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (conv_bn): ModuleList(
          (0-3): 4 x Sequential(
            (0): SpatialGraphCov(
              (gcn): Conv2d(112, 336, kernel_size=(1, 1), stride=(1, 1))
            )
            (1): SyncBatchNorm(112, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          )
        )
        (residual): Identity()
      )
      (mstcn): MS_TCN_layer(
        (act): Activations(
          (act): ReLU(inplace=True)
        )
        (bn_in): SyncBatchNorm(448, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (conv_bn): ModuleList(
          (0-3): 4 x Sequential(
            (0): Conv2d(112, 112, kernel_size=(9, 1), stride=(1, 1), padding=(4, 0))
            (1): SyncBatchNorm(112, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          )
        )
        (residual): Identity()
      )
    )
  )
  (gap): AdaptiveAvgPool2d(output_size=1)
  (fc): Linear(in_features=448, out_features=60, bias=True)
)
'''


# class Mammal:    



#
#     def __init__(self, name, sex):
#
#         self.name = name
#         self.sex = sex
#         self.num_eyes = 2
#
#     def breathe(self):
#         print(self.name + "在呼吸...")
#
#     def poop(self):
#         print(self.name + "在玩手机...")
#
# class Human(Mammal):
#
#     def __init__(self, name, sex):
#         super().__init__(name, sex)
#         self.have_tail = False
#
#     def read(self):
#         print(self.name + "在阅读...")
#
# class Cat(Mammal):
#
#     def __init__(self, name, sex):
#         super().__init__(name, sex)
#         self.have_tail = True
#
#     def scrath_sofa(self):
#         print(self.name + "在抓沙发,..")
#         return 1
#
# cat1 = Cat('Join', '男')
# print(cat1.scrath_sofa())

# cat1.scrath_sofa()

# class Employee:
#
#     def __init__(self, name, id):
#         self.name = name
#         self.id = id
#
#     def print_info(self):
#         print(self.name + "的工号是" + self.id)
#
#
# class FullTimeEmployee(Employee):
#
#     def __init__(self, name, id, monthly_salary):
#         super().__init__(name, id)
#         self.monthly_salary = monthly_salary
#
#     def calculate_monthly_pay(self):
#         return self.monthly_salary
#
# class PartTimeEmployeee(Employee):
#
#     def __init__(self, name, id, daily_salary, work_days):
#         super().__init__(name, id)
#         self.daily_salary = daily_salary
#         self.work_days = work_days
#
#     def calculate_monthly_pay(self):
#         return self.daily_salary * self.work_days
#
# zhangsan = FullTimeEmployee("张三", '01', 8000)
# zhangsan.print_info()
# print(zhangsan.name)
# print(zhangsan.calculate_monthly_pay())
'''
Model(
  (data_bn): SyncBatchNorm(150, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
  (layers): ModuleList(
    (0): MST_GCN_block(
      (msgcn): Basic_GCN_layer(
        (conv): SpatialGraphCov(
          (gcn): Conv2d(3, 336, kernel_size=(1, 1), stride=(1, 1))
        )
        (bn): SyncBatchNorm(112, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (residual): Zero_Layer()
        (act): Activations(
          (act): ReLU(inplace=True)
        )
      )
      (mstcn): Basic_TCN_layer(
        (conv): Conv2d(112, 112, kernel_size=(9, 1), stride=(1, 1), padding=(4, 0))
        (bn): SyncBatchNorm(112, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (residual): Zero_Layer()
        (act): Activations(
          (act): ReLU(inplace=True)
        )
      )
    )
    (1-3): 3 x MST_GCN_block(
      (msgcn): MS_GCN_layer(
        (act): Activations(
          (act): ReLU(inplace=True)
        )
        (bn_in): SyncBatchNorm(112, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (conv_bn): ModuleList(
          (0-3): 4 x Sequential(
            (0): SpatialGraphCov(
              (gcn): Conv2d(28, 84, kernel_size=(1, 1), stride=(1, 1))
            )
            (1): SyncBatchNorm(28, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          )
        )
        (residual): Identity()
      )
      (mstcn): MS_TCN_layer(
        (act): Activations(
          (act): ReLU(inplace=True)
        )
        (bn_in): SyncBatchNorm(112, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (conv_bn): ModuleList(
          (0-3): 4 x Sequential(
            (0): Conv2d(28, 28, kernel_size=(9, 1), stride=(1, 1), padding=(4, 0))
            (1): SyncBatchNorm(28, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          )
        )
        (residual): Identity()
      )
      (att): Attention_Layer(
        (att): SelfAttention_Att(
          (query_conv): Conv2d(112, 56, kernel_size=(1, 1), stride=(1, 1))
          (key_conv): Conv2d(112, 56, kernel_size=(1, 1), stride=(1, 1))
          (value_conv): Conv2d(112, 112, kernel_size=(1, 1), stride=(1, 1))
          (bn): SyncBatchNorm(112, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          (act): ReLU(inplace=True)
        )
        (bn): SyncBatchNorm(112, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (act): Activations(
          (act): ReLU(inplace=True)
        )
      )
    )
    (4): MST_GCN_block(
      (msgcn): MS_GCN_layer(
        (act): Activations(
          (act): ReLU(inplace=True)
        )
        (bn_in): SyncBatchNorm(112, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (conv_bn): ModuleList(
          (0-3): 4 x Sequential(
            (0): SpatialGraphCov(
              (gcn): Conv2d(28, 168, kernel_size=(1, 1), stride=(1, 1))
            )
            (1): SyncBatchNorm(56, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          )
        )
        (residual): Sequential(
          (0): Conv2d(112, 224, kernel_size=(1, 1), stride=(1, 1))
          (1): SyncBatchNorm(224, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        )
      )
      (mstcn): MS_TCN_layer(
        (act): Activations(
          (act): ReLU(inplace=True)
        )
        (bn_in): SyncBatchNorm(224, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (conv_bn): ModuleList(
          (0-3): 4 x Sequential(
            (0): Conv2d(56, 56, kernel_size=(9, 1), stride=(2, 1), padding=(4, 0))
            (1): SyncBatchNorm(56, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          )
        )
        (residual): Sequential(
          (0): Conv2d(224, 224, kernel_size=(1, 1), stride=(2, 1))
          (1): SyncBatchNorm(224, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        )
      )
      (att): Attention_Layer(
        (att): SelfAttention_Att(
          (query_conv): Conv2d(224, 112, kernel_size=(1, 1), stride=(1, 1))
          (key_conv): Conv2d(224, 112, kernel_size=(1, 1), stride=(1, 1))
          (value_conv): Conv2d(224, 224, kernel_size=(1, 1), stride=(1, 1))
          (bn): SyncBatchNorm(224, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          (act): ReLU(inplace=True)
        )
        (bn): SyncBatchNorm(224, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (act): Activations(
          (act): ReLU(inplace=True)
        )
      )
    )
    (5-6): 2 x MST_GCN_block(
      (msgcn): MS_GCN_layer(
        (act): Activations(
          (act): ReLU(inplace=True)
        )
        (bn_in): SyncBatchNorm(224, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (conv_bn): ModuleList(
          (0-3): 4 x Sequential(
            (0): SpatialGraphCov(
              (gcn): Conv2d(56, 168, kernel_size=(1, 1), stride=(1, 1))
            )
            (1): SyncBatchNorm(56, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          )
        )
        (residual): Identity()
      )
      (mstcn): MS_TCN_layer(
        (act): Activations(
          (act): ReLU(inplace=True)
        )
        (bn_in): SyncBatchNorm(224, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (conv_bn): ModuleList(
          (0-3): 4 x Sequential(
            (0): Conv2d(56, 56, kernel_size=(9, 1), stride=(1, 1), padding=(4, 0))
            (1): SyncBatchNorm(56, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          )
        )
        (residual): Identity()
      )
    )
    (7): MST_GCN_block(
      (msgcn): MS_GCN_layer(
        (act): Activations(
          (act): ReLU(inplace=True)
        )
        (bn_in): SyncBatchNorm(224, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (conv_bn): ModuleList(
          (0-3): 4 x Sequential(
            (0): SpatialGraphCov(
              (gcn): Conv2d(56, 336, kernel_size=(1, 1), stride=(1, 1))
            )
            (1): SyncBatchNorm(112, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          )
        )
        (residual): Sequential(
          (0): Conv2d(224, 448, kernel_size=(1, 1), stride=(1, 1))
          (1): SyncBatchNorm(448, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        )
      )
      (mstcn): MS_TCN_layer(
        (act): Activations(
          (act): ReLU(inplace=True)
        )
        (bn_in): SyncBatchNorm(448, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (conv_bn): ModuleList(
          (0-3): 4 x Sequential(
            (0): Conv2d(112, 112, kernel_size=(9, 1), stride=(2, 1), padding=(4, 0))
            (1): SyncBatchNorm(112, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          )
        )
        (residual): Sequential(
          (0): Conv2d(448, 448, kernel_size=(1, 1), stride=(2, 1))
          (1): SyncBatchNorm(448, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        )
      )
    )
    (8-9): 2 x MST_GCN_block(
      (msgcn): MS_GCN_layer(
        (act): Activations(
          (act): ReLU(inplace=True)
        )
        (bn_in): SyncBatchNorm(448, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (conv_bn): ModuleList(
          (0-3): 4 x Sequential(
            (0): SpatialGraphCov(
              (gcn): Conv2d(112, 336, kernel_size=(1, 1), stride=(1, 1))
            )
            (1): SyncBatchNorm(112, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          )
        )
        (residual): Identity()
      )
      (mstcn): MS_TCN_layer(
        (act): Activations(
          (act): ReLU(inplace=True)
        )
        (bn_in): SyncBatchNorm(448, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (conv_bn): ModuleList(
          (0-3): 4 x Sequential(
            (0): Conv2d(112, 112, kernel_size=(9, 1), stride=(1, 1), padding=(4, 0))
            (1): SyncBatchNorm(112, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          )
        )
        (residual): Identity()
      )
    )
  )
  (gap): AdaptiveAvgPool2d(output_size=1)
  (fc): Linear(in_features=448, out_features=60, bias=True)
)
'''

import torch
import torch.nn as nn

class Downsample(nn.Module):
    def __init__(self):
        super(Downsample, self).__init__()
        # 使用卷积层进行时间维度下采样，kernel_size=(3, 1)表示只在时间维度上进行卷积
        self.conv = nn.Conv2d(3, 3, kernel_size=(3, 1), stride=(2, 1), padding=(1, 0))  # 这样就能将时间维度从300下采样到150

    def forward(self, x):
        return self.conv(x)

# 假设输入的张量形状为 (batch_size, channels, time_steps, keypoints) = (8, 3, 300, 25)
x = torch.randn(8, 3, 300, 25)

# 创建并应用 Downsample 层
Downsample_T = Downsample()
output = Downsample_T(x)

# 输出形状，时间维度将被下采样为150
print(output.shape)  # 输出形状应该是 (8, 3, 150, 25)
""