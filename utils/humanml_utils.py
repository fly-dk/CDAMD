import numpy as np
import torch

HML_JOINT_NAMES = [
    'pelvis',
    'left_hip',
    'right_hip',
    'spine1',
    'left_knee',
    'right_knee',
    'spine2',
    'left_ankle',
    'right_ankle',
    'spine3',
    'left_foot',
    'right_foot',
    'neck',
    'left_collar',
    'right_collar',
    'head',
    'left_shoulder',
    'right_shoulder',
    'left_elbow',
    'right_elbow',
    'left_wrist',
    'right_wrist',
]

NUM_HML_JOINTS = len(HML_JOINT_NAMES)  # 22 SMPLH body joints

HML_LOWER_BODY_JOINTS = [HML_JOINT_NAMES.index(name) for name in ['pelvis', 'left_hip', 'right_hip', 'left_knee', 'right_knee', 'left_ankle', 'right_ankle', 'left_foot', 'right_foot',]]
SMPL_UPPER_BODY_JOINTS = [i for i in range(len(HML_JOINT_NAMES)) if i not in HML_LOWER_BODY_JOINTS]
# 上/下半身关节索引（基于上面的顺序）
UPPER_BODY_JOINTS = [3, 6, 9, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21]
LOWER_BODY_JOINTS = [0, 1, 2, 4, 5, 7, 8, 10, 11]
def build_upper_body_keep_mask(m_lens_frames, T, device):
    """
    返回 bool 掩码 (B, T, 22)：在有效帧范围内，上半身关节为 True（保留），其余为 False（可编辑）。
    m_lens_frames: 长度单位为帧（与 T 一致）
    """
    B = m_lens_frames.shape[0]
    m = torch.arange(T, device=device).unsqueeze(0) < m_lens_frames.unsqueeze(1)  # (B,T)
    keep = torch.zeros(B, T, NUM_HML_JOINTS, dtype=torch.bool, device=device)
    keep[:, :, UPPER_BODY_JOINTS] = True
    keep = keep & m.unsqueeze(-1)
    return keep
# Recover global angle and positions for rotation data
# root_rot_velocity (B, seq_len, 1)
# root_linear_velocity (B, seq_len, 2)
# root_y (B, seq_len, 1)
# ric_data (B, seq_len, (joint_num - 1)*3)
# rot_data (B, seq_len, (joint_num - 1)*6)
# local_velocity (B, seq_len, joint_num*3)
# foot contact (B, seq_len, 4)
HML_ROOT_BINARY = np.array([True] + [False] * (NUM_HML_JOINTS-1))
HML_ROOT_MASK = np.concatenate(([True]*(1+2+1),
                                HML_ROOT_BINARY[1:].repeat(3),
                                HML_ROOT_BINARY[1:].repeat(6),
                                HML_ROOT_BINARY.repeat(3),
                                [False] * 4))
HML_LOWER_BODY_JOINTS_BINARY = np.array([i in HML_LOWER_BODY_JOINTS for i in range(NUM_HML_JOINTS)])
HML_LOWER_BODY_MASK = np.concatenate(([True]*(1+2+1),
                                     HML_LOWER_BODY_JOINTS_BINARY[1:].repeat(3),
                                     HML_LOWER_BODY_JOINTS_BINARY[1:].repeat(6),
                                     HML_LOWER_BODY_JOINTS_BINARY.repeat(3),
                                     [True]*4))
HML_UPPER_BODY_MASK = ~HML_LOWER_BODY_MASK


ALL_JOINT_FALSE = np.full(*HML_ROOT_BINARY.shape, False)
HML_UPPER_BODY_JOINTS_BINARY = np.array([i in SMPL_UPPER_BODY_JOINTS for i in range(NUM_HML_JOINTS)])

UPPER_JOINT_Y_TRUE = np.array([ALL_JOINT_FALSE[1:], HML_UPPER_BODY_JOINTS_BINARY[1:], ALL_JOINT_FALSE[1:]])
UPPER_JOINT_Y_TRUE = UPPER_JOINT_Y_TRUE.T
UPPER_JOINT_Y_TRUE = UPPER_JOINT_Y_TRUE.reshape(ALL_JOINT_FALSE[1:].shape[0]*3)

UPPER_JOINT_Y_MASK = np.concatenate(([False]*(1+2+1),
                                UPPER_JOINT_Y_TRUE,
                                ALL_JOINT_FALSE[1:].repeat(6),
                                ALL_JOINT_FALSE.repeat(3),
                                [False] * 4))