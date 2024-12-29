import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np
import math

import sys
sys.path.append('../')
from model.layers import Basic_Layer, Basic_TCN_layer, MS_TCN_layer, Temporal_Bottleneck_Layer, \
    MS_Temporal_Bottleneck_Layer, Temporal_Sep_Layer, Basic_GCN_layer, MS_GCN_layer, Spatial_Bottleneck_Layer, \
    MS_Spatial_Bottleneck_Layer, SpatialGraphCov, Spatial_Sep_Layer
from model.activations import Activations
from model.utils import import_class, conv_branch_init, conv_init, bn_init
from model.attentions import Attention_Layer
from sr.infer import sr_inference
from sr.infer import main

__block_type__ = {
    'basic': (Basic_GCN_layer, Basic_TCN_layer),
    'bottle': (Spatial_Bottleneck_Layer, Temporal_Bottleneck_Layer),
    'sep': (Spatial_Sep_Layer, Temporal_Sep_Layer),
    'ms': (MS_GCN_layer, MS_TCN_layer),
    'ms_bottle': (MS_Spatial_Bottleneck_Layer, MS_Temporal_Bottleneck_Layer),
}

class PositionalEncoding():

    def __init__(self, channel, joint_num, max_frame, domain):
        super(PositionalEncoding, self).__init__()
        self.joint_num = joint_num
        self.max_frame = max_frame
        self.domain = domain

        if domain == 'temporal':
            pos_list = []
            for t in range(self.max_frame):
                for j in range(self.joint_num):
                    pos_list.append(t)

        elif domain == 'spatial':
            pos_list = []
            for t in range(self.max_frame):
                for j_id in range(self.joint_num):
                    pos_list.append(j_id)

        position = torch.from_numpy(np.array(pos_list)).unsqueeze(1).float()
        pe = torch.zeros(self.max_frame * self.joint_num, channel)
        div_term = torch.exp(torch.arange(0, channel, 2).float() *
                             -(math.log(10000.0) / channel))  # channel//2

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.view(max_frame, joint_num, channel).permute(2, 0, 1).unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):  # nctv
        x = x + self.pe[:, :, :x.size(2)]
        return x


# class claculate_threshold_from_attention_weights(sum_attention_weights, keypoints=25, top_k=10):
#
#     sum_attention_weights = sum_attention_weights
#     top_k = top_k
#
#     left_wrist_idx = 7
#     right_wrist_idx = 11
#
#     sorted_weights, sorted_indices = torch.sort(sum_attention_weights, descending=True)
#     # 选择排名前 top_k 个重要的关键点
#     top_k_indices = sorted_indices[:top_k]  # 选择权重最高的前 top_k 个关键点的索引
#     # if left_wrist_index in top_k_indices or right_wrist_index in top_k_indices:
#     left_wrist_in_top_k = (top_k_indices == left_wrist_idx)
#     right_wrist_in_top_k = (top_k_indices == right_wrist_idx)
#     if left_wrist_in_top_k.any() or right_wrist_in_top_k.any():
#         # print("左手腕或右手腕排名在前 {}，调用超分辨率模型".format(K))



class Model(nn.Module):

    # def __init__(self, num_class, num_point, num_person, block_args, graph, graph_args, kernel_size, block_type, atten,
    #              **kwargs):
    def __init__(self, num_class, num_point, num_person, block_args, graphs, kernel_size, block_type, atten,
                 **kwargs):
        super(Model, self).__init__()
        kwargs['act'] = Activations(kwargs['act'])  # 调用激活函数
        atten = None if atten == 'None' else atten

        A = []
        for graph in graphs:
            if graph['graph'] is None:
                raise ValueError()
            else:
                Graph = import_class(graph['graph'])
                self.graph = Graph(**graph['graph_args'])
            A.append(self.graph.A)        # 单位矩阵

        self.data_bn = nn.BatchNorm1d(num_person * block_args[0][0] * num_point)

        self.layers = nn.ModuleList()
        self.sr_layers = nn.ModuleList()  # 超分之后的时空特征提取层

        # self.M = nn.Parameter(torch.rand(num_point, num_point), requires_grad=True)     # 可学习的参数矩阵

        for i, block in enumerate(block_args):

            if i == 0:  # 第一层没有残差
                self.layers.append(MST_GCN_block(in_channels=block[0], out_channels=block[1], residual=block[2],
                                                 kernel_size=kernel_size, stride=block[3], A=A[0], block_type='basic',
                                                 atten=None, **kwargs))
            else:
                if i <= 3:
                    self.layers.append(MST_GCN_block(in_channels=block[0], out_channels=block[1], residual=block[2],
                                                     kernel_size=kernel_size, stride=block[3], A=A[0], block_type=block_type,
                                                     atten='sa', **kwargs))

                else:
                    self.layers.append(MST_GCN_block(in_channels=block[0], out_channels=block[1], residual=block[2],
                                                     kernel_size=kernel_size, stride=block[3], A=A[0], block_type=block_type,
                                                     atten=atten, **kwargs))

        for i, block in enumerate(block_args):
            self.sr_layers.append(MST_GCN_block(in_channels=block[0], out_channels=block[1], residual=block[2],
                                                kernel_size=kernel_size, stride=block[3], A=A[1], block_type='basic',
                                                atten=None, **kwargs))      # A不应该是之前的邻接矩阵，而是更新后的邻接矩阵

        self.gap = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(block_args[-1][1], num_class)

        for m in self.modules():
            if isinstance(m, SpatialGraphCov) or isinstance(m, Spatial_Sep_Layer):
                for mm in m.modules():
                    if isinstance(mm, nn.Conv2d):
                        conv_branch_init(mm, self.graph.A.shape[0])
                    if isinstance(mm, nn.BatchNorm2d):
                        bn_init(mm, 1)
            elif isinstance(m, nn.Conv2d):
                conv_init(m)
            elif isinstance(m, nn.BatchNorm2d) or isinstance(m, nn.BatchNorm1d):
                bn_init(m, 1)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, math.sqrt(2. / num_class))



    def forward(self, x):

        N, C, T, V, M = x.size()

        x = x.permute(0, 4, 3, 1, 2).contiguous().view(N, M * V * C, T)  # N C T V M --> N M V C T
        x = self.data_bn(x)
        x = x.view(N, M, V, C, T).permute(0, 1, 3, 4, 2).contiguous().view(N * M, C, T, V)

        for i, layer in enumerate(self.layers):

            if 0 < i < 5:
                x = layer(x)
                if x.shape[3] == 64:  # 对是否是超分过后的数据进行判断
                    break
            else:
                x = layer(x)
        if x.shape[3] == 64:
            for i, layer in enumerate(self.sr_layers):
                x = layer(x)

        features = x

        x = self.gap(x).view(N, M, -1).mean(dim=1)
        x = self.fc(x)

        return features, x



class MST_GCN_block(nn.Module):

    def __init__(self, in_channels, out_channels, residual, kernel_size, stride, A, block_type, atten, **kwargs):
        super(MST_GCN_block, self).__init__()
        self.atten = atten
        self.msgcn = __block_type__[block_type][0](in_channels=in_channels, out_channels=out_channels, A=A,
                                                   residual=residual, **kwargs)
        self.A = A
        if atten is not None:
            self.att = Attention_Layer(out_channels, att_type=atten, **kwargs)

        self.mstcn = __block_type__[block_type][1](channels=out_channels, kernel_size=kernel_size, stride=stride,
                                                   residual=residual, **kwargs)

        self.left_wrist_index = 7
        self.right_wrist_index = 11

        self.ConvTranspose2d = nn.ConvTranspose2d(
            in_channels=112,
            out_channels=3,
            kernel_size=(3, 3),
            stride=1,
            padding=1
        )

    def forward(self, x):
        # return self.att(self.mstcn(self.msgcn(x))) if self.atten is not None else self.mstcn(self.msgcn(x))
        msgcn_output = self.msgcn(x)  # x 输入到 msgcn 层

        if self.atten is not None:
            output, sum_attention_weights = self.att(msgcn_output)
            sorted_weights, sorted_indices = torch.sort(sum_attention_weights, descending=True)
            # 在这里进行筛选，调用超分模块
            top_k_indices = sorted_indices[:12]

            is_left_wrist_in_top_k_idc = (self.left_wrist_index in top_k_indices)
            is_right_wrist_in_top_k_idc = (self.right_wrist_index in top_k_indices)

            if is_left_wrist_in_top_k_idc or is_right_wrist_in_top_k_idc:
                # 反卷积
                x1 = self.ConvTranspose2d(output)
                # 调用超分模块
                # model = main()
                output = sr_inference(x1)
                return output


        else:
            output = msgcn_output
        # 然后通过 MSTCN 层进行处理
        mstcn_output = self.mstcn(output)  # msgcn 层的输出输入到 mstcn 层

        return mstcn_output

if __name__ == '__main__':
    import sys
    import time

    parts = [
        np.array([5, 6, 7, 8, 22, 23]) - 1,  # left_arm
        np.array([9, 10, 11, 12, 24, 25]) - 1,  # right_arm
        np.array([13, 14, 15, 16]) - 1,  # left_leg
        np.array([17, 18, 19, 20]) - 1,  # right_leg
        np.array([1, 2, 3, 4, 21]) - 1  # torso
    ]

    warmup_iter = 3
    test_iter = 10
    sys.path.append('/home/chenzhan/mywork/MST-GCN/')
    from thop import profile
    basic_channels = 112
    cfgs = {
        'num_class': 2,
        'num_point': 25,
        'num_person': 1,
        'block_args': [[2, basic_channels, False, 1],
                       [basic_channels, basic_channels, True, 1], [basic_channels, basic_channels, True, 1], [basic_channels, basic_channels, True, 1],
                       [basic_channels, basic_channels*2, True, 1], [basic_channels*2, basic_channels*2, True, 1], [basic_channels*2, basic_channels*2, True, 1],
                       [basic_channels*2, basic_channels*4, True, 1], [basic_channels*4, basic_channels*4, True, 1], [basic_channels*4, basic_channels*4, True, 1]],
        'graph': 'graph.ntu_rgb_d.Graph',
        'graph_args': {'labeling_mode': 'spatial'},
        'kernel_size': 9,
        'block_type': 'ms',
        'reduct_ratio': 2,
        'expand_ratio': 0,
        't_scale': 4,
        'layer_type': 'sep',
        'act': 'relu',
        's_scale': 4,
        'atten': 'stcja',
        'bias': True,
        'parts': parts
    }

    model = Model(**cfgs)

    N, C, T, V, M = 4, 2, 16, 25, 1
    inputs = torch.rand(N, C, T, V, M)

    for i in range(warmup_iter + test_iter):
        if i == warmup_iter:
            start_time = time.time()
        outputs = model(inputs)
    end_time = time.time()

    total_time = end_time - start_time
    print('iter_with_CPU: {:.2f} s/{} iters, persample: {:.2f} s/iter '.format(
        total_time, test_iter, total_time/test_iter/N))

    print(outputs.size())

    hereflops, params = profile(model, inputs=(inputs,), verbose=False)
    print('# GFlops is {} G'.format(hereflops / 10 ** 9 / N))
    print('# Params is {} M'.format(sum(param.numel() for param in model.parameters()) / 10 ** 6))




