import argparse
import logging
import torch
import sr.model as Model
import data as Data
import sr.core.logger as Logger
import torch.nn.functional as F
import numpy as np
import torch.nn as nn
from tqdm import tqdm
from tensorboardX import SummaryWriter


def calculate_conf(keypoints, wrist_index=[9, 10], sigma=1.0, beta=5):      # index=9对应[25:45]

    # 提取x,y坐标
    x_coords = keypoints[0].reshape(1, 64)
    y_coords = keypoints[1].reshape(1, 64)

    # 提取左右手掌超分关节点
    left_palm_x = x_coords[0, 25:45]
    left_palm_y = y_coords[0, 25:45]

    right_palm_x = x_coords[0, 45:]
    right_palm_y = y_coords[0, 45:]

    # 左右腕关节索引
    left_wrist_index = wrist_index[0]
    right_wrist_index = wrist_index[1]

    # 左右腕关节位置
    left_wrist_x = x_coords[0, left_wrist_index-1]
    left_wrist_y = y_coords[0, left_wrist_index-1]

    right_wrist_x = x_coords[0, right_wrist_index-1]
    right_wrist_y = y_coords[0, right_wrist_index-1]

    # 计算每个左右关节到腕关节的距离
    left_distance = np.sqrt((left_palm_x - left_wrist_x)**2 + (left_palm_y - left_wrist_y)**2)
    right_distance = np.sqrt((right_palm_x - right_wrist_x) ** 2 + (right_palm_y - right_wrist_y) ** 2)

    # # 计算左右手平均距离和标准差
    # left_avg_distance = torch.mean(left_distance)
    # left_std_distance = torch.std(left_distance)
    #
    # right_avg_distance = torch.mean(right_distance)
    # right_std_distance = torch.std(right_distance)
    # 平滑处理
    # epsilon = 0.1  # 平滑因子，适当调整
    # left_distance_norm_e = (left_distance - left_distance.min()) / (left_distance.max() - left_distance.min() + epsilon)
    # right_distance_norm_e = (right_distance - right_distance.min()) / (
    #             right_distance.max() - right_distance.min() + epsilon)
    #
    # left_confidences_e = 1 - left_distance_norm_e
    # right_confidences_e = 1 - right_distance_norm_e

    # 非线性归一化
    left_distance_norm = (left_distance - left_distance.min()) / (left_distance.max() - left_distance.min())
    right_distance_norm = (right_distance - right_distance.min()) / (right_distance.max() - right_distance.min())

    left_confidences = 1 / (1 + torch.exp(-beta * (1 - left_distance_norm)))
    right_confidences = 1 / (1 + torch.exp(-beta * (1 - right_distance_norm)))

    # 用置信度替换对应通道的值（后39个）
    confindence = torch.cat([left_confidences, right_confidences], dim=0)
    flattened = keypoints[2].view(-1)
    flattened[-39:] = confindence
    keypoints[2] = flattened.view(8, 8)

    return keypoints


class Downsample(nn.Module):
    def __init__(self):
        super(Downsample, self).__init__()
        self.conv = nn.Conv2d(3, 3, kernel_size=(3, 1), stride=(300, 1), padding=(1, 0))  # 沿着时间维度下采样

    def forward(self, x):
        return self.conv(x)

class Upsample(nn.Module):
    def __init__(self):
        super(Upsample, self).__init__()
        # 通过转置卷积实现上采样
        # 输入通道为3，输出通道为3，kernel_size设为(3, 1)保持与下采样时一致
        # stride设大于1以实现上采样，padding设置为适当的值以保持图像大小
        self.deconv = nn.ConvTranspose2d(3, 3, kernel_size=(3, 1), stride=(300, 1), padding=(1, 0))

    def forward(self, x):
        return self.deconv(x)



def sr_inference(x):

    # Feed the input into the model
    # N, C, T, V = x.size()

    device = x.device
    Downsample_T = Downsample().to(device)
    x = Downsample_T(x)
    N, C, T, V = x.size()
    model = main()
    x_upsampled = F.interpolate(x, size=(T, 64), mode='bilinear', align_corners=False)

    x_sr = []
    # for i in tqdm(range(N)):
    for i in range(N):
        for j in range(T):
            data = x_upsampled[i, :, j, :].reshape(1, 3, 8, 8)
            # data = F.interpolate(data, size=(8, 8), mode='bilinear', align_corners=False)
            # data = downsample_data(data)
            val_data = {
                'SR': data,
                'Index': torch.tensor(j)
            }
            model.feed_data(val_data)
            model.test(continous=False)  # Test without continuous mode
            visuals = model.get_current_visuals_val(need_LR=False)

            # Get the super-resolved data (SR) from the model
            sr_data = visuals['SR']  # The super-resolved data (SR)

            # Apply the confidence estimation function
            sr_data = calculate_conf(sr_data, sigma=1.0).view(3, -1)

            # 将 SR 数据加入结果列表
            # x_sr.append(sr_data)
            x_upsampled[i:, :, j, :] = sr_data

    x = F.interpolate(x, size=(300, 64), mode='bilinear', align_corners=False)

    return x


def main():
    """
    Main function to parse arguments, load data, initialize model, and run inference.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config', type=str, default='sr/config/ntu_sr3_25_65.json',
                        help='JSON file for configuration')
    parser.add_argument('-p', '--phase', type=str, choices=['val'], help='val(generation)', default='val')
    parser.add_argument('-gpu', '--gpu_ids', type=str, default=None)
    parser.add_argument('-debug', '-d', action='store_true')
    parser.add_argument('-enable_wandb', action='store_true')
    parser.add_argument('-log_infer', action='store_true')

    # Parse configurations
    args = parser.parse_args()
    opt = Logger.parse(args)
    opt = Logger.dict_to_nonedict(opt)

    # Initialize model
    model = Model.create_model(opt)

    model.set_new_noise_schedule(
        opt['model']['beta_schedule']['val'], schedule_phase='val')

    return model


if __name__ == "__main__":
    main()






