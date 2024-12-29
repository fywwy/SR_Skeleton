from io import BytesIO
import lmdb
import pickle

import torch
from PIL import Image
from torch.utils.data import Dataset
import random
import data.util as Util
import numpy as np
import os
from tqdm import tqdm

class LRHRDataset(Dataset):

    def __init__(self, dataroot, datatype, l_resolution=16, r_resolution=128, split='train', data_len=-1, need_LR=False):

        self.datatype = datatype
        self.l_res = l_resolution
        self.r_res = r_resolution
        self.data_len = data_len
        self.need_LR = need_LR
        self.split = split

        self.seq_length = 10

        if datatype == 'npy' and self.split == 'train':
            # 建立lr、hr、sr的路径
            self.lr_paths = os.path.join(dataroot, 'training_set_lr.npy')
            self.hr_paths = os.path.join(dataroot, 'training_set_hr.npy')
            self.sr_paths = os.path.join(dataroot, 'training_set_sr.npy')

            self.data_HR = np.load(self.hr_paths, allow_pickle=True)
            self.data_SR = np.load(self.sr_paths, allow_pickle=True)
            if need_LR:
                self.data_LR = np.load(self.lr_paths, allow_pickle=True)

            # 此处可以不加载数据集
            self.data = np.load(self.hr_paths, allow_pickle=True)
            self.dataset_len = len(self.data)

            if self.data_len <= 0:      # self.den_len=-1,代表利用全部数据
                self.data_len = self.dataset_len
            else:
                self.data_len = min(self.data_len, self.dataset_len)
        elif datatype == 'npy' and self.split == 'val':
            # 建立lr、hr、sr的路径
            self.lr_paths = os.path.join(dataroot, 'val_set_lr.npy')
            self.hr_paths = os.path.join(dataroot, 'val_set_hr.npy')
            self.sr_paths = os.path.join(dataroot, 'val_set_sr.npy')

            self.data_HR = np.load(self.hr_paths, allow_pickle=True)
            self.data_SR = np.load(self.sr_paths, allow_pickle=True)
            if need_LR:
                self.data_LR = np.load(self.lr_paths, allow_pickle=True)

            # 此处可以不加载数据集
            self.data = np.load(self.hr_paths, allow_pickle=True)
            self.dataset_len = len(self.data)

            if self.data_len <= 0:  # self.den_len=-1,代表利用全部数据
                self.data_len = self.dataset_len
            else:
                self.data_len = min(self.data_len, self.dataset_len)

        else:
            raise NotImplementedError(
                'data_type [{:s}] is not recognized.'.format(datatype))




    def __len__(self):
        return self.data_len

    def __getitem__(self, index):
        img_HR = None
        img_LR = None

        if self.datatype == 'lmdb':
            with self.env.begin(write=False) as txn:
                hr_img_bytes = txn.get(
                    'hr_{}_{}'.format(
                        self.r_res, str(index).zfill(5)).encode('utf-8')
                )
                sr_img_bytes = txn.get(
                    'sr_{}_{}_{}'.format(
                        self.l_res, self.r_res, str(index).zfill(5)).encode('utf-8')
                )
                if self.need_LR:
                    lr_img_bytes = txn.get(
                        'lr_{}_{}'.format(
                            self.l_res, str(index).zfill(5)).encode('utf-8')
                    )
                # skip the invalid index
                while (hr_img_bytes is None) or (sr_img_bytes is None):
                    new_index = random.randint(0, self.data_len-1)
                    hr_img_bytes = txn.get(
                        'hr_{}_{}'.format(
                            self.r_res, str(new_index).zfill(5)).encode('utf-8')
                    )
                    sr_img_bytes = txn.get(
                        'sr_{}_{}_{}'.format(
                            self.l_res, self.r_res, str(new_index).zfill(5)).encode('utf-8')
                    )
                    if self.need_LR:
                        lr_img_bytes = txn.get(
                            'lr_{}_{}'.format(
                                self.l_res, str(new_index).zfill(5)).encode('utf-8')
                        )
                img_HR = Image.open(BytesIO(hr_img_bytes)).convert("RGB")
                img_SR = Image.open(BytesIO(sr_img_bytes)).convert("RGB")
                if self.need_LR:
                    img_LR = Image.open(BytesIO(lr_img_bytes)).convert("RGB")
        elif self.datatype == 'img':
                img_HR = Image.open(self.hr_path[index]).convert("RGB")
                img_SR = Image.open(self.sr_path[index]).convert("RGB")
                if self.need_LR:
                    img_LR = Image.open(self.lr_path[index]).convert("RGB")

        else:
            data_HR = self.data_HR
            data_SR = self.data_SR
            if self.need_LR:
                data_LR = self.data_LR
            assert len(data_HR) == len(data_SR)

            if self.split == 'train':
                frame_dir_HR = data_HR[index]['frame_dir']
                x = data_HR[index]['keypoints'][:, :, :, 0]/1920
                y = data_HR[index]['keypoints'][:, :, :, 1]/1080
                sum_xy = x + y
                # 计算 x + y 的平均值
                z = sum_xy / 2
                img_HR_64 = np.stack([x, y, z], axis=-1)[:, :, :64, :]
                img_HR = img_HR_64.reshape(img_HR_64.shape[0], img_HR_64.shape[1], 8, 8, img_HR_64.shape[3]).transpose(0, 1, 4, 2, 3)


                frame_dir_SR = data_SR[index]['frame_dir']
                x = data_SR[index]['keypoints'][:, :, :, 0]/1920
                y = data_SR[index]['keypoints'][:, :, :, 1]/1080
                sum_xy = x + y
                # 计算 x + y 的平均值
                z = sum_xy / 2
                img_SR_64 = np.stack([x, y, z], axis=-1)[:, :, :64, :]
                img_SR = img_SR_64.reshape(img_SR_64.shape[0], img_SR_64.shape[1], 8, 8, img_SR_64.shape[3]).transpose(0, 1, 4, 2, 3)

                assert frame_dir_HR == frame_dir_SR

                if self.need_LR:
                    frame_dir_LR = data_LR[index]['frame_dir']
                    x = data_LR[index]['keypoints'][:, :, :, 0]/1920
                    y = data_LR[index]['keypoints'][:, :, :, 1]/1080
                    sum_xy = x + y
                    # 计算 x + y 的平均值
                    z = sum_xy / 2
                    img_LR_25 = np.stack([x, y, z], axis=-1)
                    img_LR = img_LR_25.reshape(img_LR_25.shape[0], img_LR_25.shape[1], 5, 5,
                                               img_LR_25.shape[3]).transpose(0, 1, 4, 2, 3)


                    # [img_LR, img_SR, img_HR] = Util.transform_augment([img_LR, img_SR, img_HR], split=self.split, min_max=(-1, 1))

                    return{'LR': img_LR, 'HR': img_HR, 'SR': img_SR, "Index": index}

                return{'HR': img_HR, 'SR': img_SR, "Index": index}
            else:

                frame_dir_HR = data_HR[index]['frame_dir']
                x = data_HR[index]['keypoints'][:, :, :, 0]/1920
                y = data_HR[index]['keypoints'][:, :, :, 1]/1080
                sum_xy = x + y

                # 计算 x + y 的平均值
                z = sum_xy / 2
                img_HR_64 = np.stack([x, y, z], axis=-1)[:, :, :64, :]
                img_HR = img_HR_64.reshape(img_HR_64.shape[0], img_HR_64.shape[1], 8, 8, img_HR_64.shape[3]).transpose(
                    0, 1, 4, 2, 3)

                frame_dir_SR = data_SR[index]['frame_dir']
                x = data_SR[index]['keypoints'][:, :, :, 0]/1920
                y = data_SR[index]['keypoints'][:, :, :, 1]/1080
                sum_xy = x + y

                # 计算 x + y 的平均值
                z = sum_xy / 2
                img_SR_64 = np.stack([x, y, z], axis=-1)[:, :, :64, :]
                img_SR = img_SR_64.reshape(img_SR_64.shape[0], img_SR_64.shape[1], 8, 8, img_SR_64.shape[3]).transpose(
                    0, 1, 4, 2, 3)

                assert frame_dir_HR == frame_dir_SR

                if self.need_LR:
                    frame_dir_LR = data_LR[index]['frame_dir']
                    x = data_LR[index]['keypoints'][:, :, :, 0]/1920
                    y = data_LR[index]['keypoints'][:, :, :, 1]/1080
                    sum_xy = x + y
                    # 计算 x + y 的平均值
                    z = sum_xy / 2
                    img_LR_25 = np.stack([x, y, z], axis=-1)
                    img_LR = img_LR_25.reshape(img_LR_25.shape[0], img_LR_25.shape[1], 5, 5,    # 低分辨率只有25个点
                                               img_LR_25.shape[3]).transpose(0, 1, 4, 2, 3)

                    # [img_LR, img_SR, img_HR] = Util.transform_augment([img_LR, img_SR, img_HR], split=self.split,

                    return {'LR': img_LR, 'HR': img_HR, 'SR': img_SR, "Index": index}

                return {'HR': img_HR, 'SR': img_SR, "Index": index}







