import argparse
import os
import numpy as np
from scipy.ndimage import zoom
from tqdm import tqdm


def load_npy(file_path):

    original_data = np.load(file_path, allow_pickle=True)
    return original_data


def save_npy(data, folder, size):
    sizes = parse_size(size)
    os.makedirs('{}/ntu60_sr_sr'.format(folder), exist_ok=True)
    output_path = os.path.join(folder, 'ntu60_expressive', '2024-11-20sr.npy')

    np.save(output_path, data)
    # print(f"Data saved successfully to {output_path}")


def processed_data(data, size):

    sizes = parse_size(size)
    ntu60_expressive_25_65 = []

    for item in tqdm(data, desc='Processing data', unit='item'):
        frame_dir = item['frame_dir']
        keypoints = item['keypoints']

        # keypoints = keypoints.astype(np.float32)
        processed_keypoints = zoom(keypoints.astype(np.float32), (1, 1, 65/25, 1))    # zoom不支持float16
        ntu60_expressive_25_65.append({'frame_dir': frame_dir, 'keypoints': processed_keypoints})
        # if frame_dir == 'S001C001P001R001A001':
        #     break


    return ntu60_expressive_25_65

def parse_size(size):

    return [int(s.strip()) for s in size.split(',')]

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Interpolation processing npy files")
    parser.add_argument('--input', '-i', type=str, default='../output/train/training_set_lr.npy')
    parser.add_argument('--output', '-o', type=str, default='../output')
    parser.add_argument('--size', type=str, default='25, 65')
    parser.add_argument('--num_worker', type=int, default=0)

    # 解析命令行参数


    args = parser.parse_args()
    data = load_npy(args.input)

    # 插值
    data_processed = processed_data(data, args.size)
    save_npy(data_processed, args.output, args.size)
    print(f'Processing is complete and the interpolated data has been saved to {args.output}')

