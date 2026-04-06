import os
import numpy as np
from os.path import join as pjoin

input_dir = '/dataset/HumanML3D/new_joints'
output_dir = '/dataset/HumanML3D/new_joints_66'
os.makedirs(output_dir, exist_ok=True)

# 遍历输入目录中的所有 .npy 文件
for file_name in os.listdir(input_dir):
    if file_name.endswith('.npy'):
        input_path = pjoin(input_dir, file_name)
        output_path = pjoin(output_dir, file_name)

        data = np.load(input_path)  # (T, 22, 3)

        reshaped_data = data.reshape(data.shape[0], -1)  # (T, 66)

        np.save(output_path, reshaped_data)

print("转换完成:", output_dir)
