import os
import torch
import numpy as np

from torch.utils.data import DataLoader
from os.path import join as pjoin

from models.vq.model import RVQVAE

from options.train_option import TrainT2MOptions

from utils.plot_script import plot_3d_motion
from utils.motion_process import recover_from_ric
from utils.get_opt import get_opt
from utils.fixseed import fixseed
from utils.paramUtil import t2m_kinematic_chain, kit_kinematic_chain

from data.t2m_dataset import Text2MotionDataset
from motion_loaders.dataset_motion_loader import get_dataset_motion_loader

from models.mardm_evaluators import Evaluators
from models.no_vq.AE import AE_models
from models.mask_transformer.abs_transformer import MaskTransformer
from models.mask_transformer.abs_transformer_trainer import MaskTransformerTrainer
def plot_t2m(data, save_dir, captions, m_lengths):
    data = train_dataset.inv_transform(data)

    # print(ep_curves.shape)
    for i, (caption, joint_data) in enumerate(zip(captions, data)):
        joint_data = joint_data[:m_lengths[i]]
        joint = recover_from_ric(torch.from_numpy(joint_data).float(), opt.joints_num).numpy()
        save_path = pjoin(save_dir, '%02d.mp4'%i)
        # print(joint.shape)
        # plot_3d_motion(save_path, kinematic_chain, joint, title=caption, fps=20)

def load_vq_model():
    opt_path = pjoin('/path/to/vq/t2m/opt.txt') #for t2m
    vq_opt = get_opt(opt_path, opt.device)
    vq_model = RVQVAE(vq_opt,
                63 if vq_opt.dataset_name == 'kit' else 66,
                vq_opt.nb_code,
                vq_opt.code_dim,
                vq_opt.output_emb_width,
                vq_opt.down_t,
                vq_opt.stride_t,
                vq_opt.width,
                vq_opt.depth,
                vq_opt.dilation_growth_rate,
                vq_opt.vq_act,
                vq_opt.vq_norm)

    # 绝对坐标
    ckpt = torch.load(pjoin('/path/to/vq/t2m/model/latest.tar'),
                            map_location='cpu') #for t2m
    model_key = 'vq_model' if 'vq_model' in ckpt else 'net'
    vq_model.load_state_dict(ckpt[model_key])
    print(f'Loading VQ Model {opt.vq_name}')
    return vq_model, vq_opt

if __name__ == '__main__':
    parser = TrainT2MOptions()
    opt = parser.parse()
    fixseed(opt.seed)

    opt.device = torch.device("cpu" if opt.gpu_id == -1 else "cuda:" + str(opt.gpu_id))
    torch.autograd.set_detect_anomaly(True)

    # setting this to true significantly increase training and sampling speed
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    opt.save_root = pjoin(opt.checkpoints_dir, opt.dataset_name, opt.name)
    # opt.checkpoints_dir: './log/t2m'; opt.dataset_name: 't2m'; opt.name: 't2m_nlayer8_nhead6_ld384_ff1024_cdp0.1_rvq6ns',help='Name of this trial'
    from exit.utils import init_save_folder
    init_save_folder(opt.save_root)
    opt.model_dir = pjoin(opt.save_root, 'model')
    # opt.meta_dir = pjoin(opt.save_root, 'meta')
    opt.eval_dir = pjoin(opt.save_root, 'animation')
    opt.log_dir = pjoin('./log/t2m/', opt.dataset_name, opt.name)

    os.makedirs(opt.model_dir, exist_ok=True)
    # os.makedirs(opt.meta_dir, exist_ok=True)
    os.makedirs(opt.eval_dir, exist_ok=True)
    os.makedirs(opt.log_dir, exist_ok=True)

    if opt.dataset_name == 't2m':
        opt.data_root = './dataset/HumanML3D'
        opt.motion_dir = pjoin(opt.data_root, 'new_joint_66')
        opt.joints_num = 22
        opt.max_motion_len = 55
        dim_pose = 66
        radius = 4
        fps = 20
        kinematic_chain = t2m_kinematic_chain
        dataset_opt_path = './checkpoints/t2m/Comp_v6_KLD005/opt.txt'

    elif opt.dataset_name == 'kit': 
        opt.data_root = './dataset/KIT-ML'
        opt.motion_dir = pjoin(opt.data_root, 'new_joint_63')
        opt.joints_num = 21
        radius = 240 * 8
        fps = 12.5
        dim_pose = 63
        opt.max_motion_len = 55
        kinematic_chain = kit_kinematic_chain
        dataset_opt_path = './checkpoints/kit/Comp_v6_KLD005/opt.txt'

    else:
        raise KeyError('Dataset Does Not Exist')

    opt.text_dir = pjoin(opt.data_root, 'texts')

    #####################################################################
    # load AE model                                                     #
    #####################################################################
    ae = AE_models['AE_Model'](input_width=dim_pose)

    # 绝对坐标AE权重
    model_dir = pjoin('./checkpoints/t2m/AE/model/latest.tar')
    checkpoint = torch.load(model_dir, map_location='cpu')
    print(checkpoint.keys())
    ae.load_state_dict(checkpoint['ae'])
    #####################################################################

    clip_version = 'ViT-B/32'
    #####################################################################
    # load VQVAE model                                                  #
    #####################################################################
    vq_model, vq_opt = load_vq_model()
    #####################################################################

    t2m_transformer = MaskTransformer(ae_dim=ae.output_emb_width, # ae.dim= output_emb_width(4) * joints_num(22)
                                      cond_mode='text',
                                      latent_dim=opt.latent_dim,
                                      ff_size=opt.ff_size,
                                      num_layers=opt.n_layers,
                                      num_heads=opt.n_heads,
                                      dropout=opt.dropout,
                                      clip_dim=512,
                                      cond_drop_prob=opt.cond_drop_prob,
                                      clip_version=clip_version,
                                      opt=opt)



    all_params = 0
    pc_transformer = sum(param.numel() for param in t2m_transformer.parameters_wo_clip())

    all_params += pc_transformer

    print('Total parameters of all models: {:.2f}M'.format(all_params / 1000_000))

    绝对坐标的均值和方差
    mean = np.load(pjoin('/dataset/HumanML3D/Mean_abs.npy')) # make sure this is computed
    std = np.load(pjoin('/dataset/HumanML3D/Std_abs.npy')) # make sure this is computed


    train_split_file = pjoin(opt.data_root, 'train.txt')
    val_split_file = pjoin(opt.data_root, 'val.txt')

    train_dataset = Text2MotionDataset(opt, mean, std, train_split_file)
    val_dataset = Text2MotionDataset(opt, mean, std, val_split_file)

    train_loader = DataLoader(train_dataset, batch_size=opt.batch_size, num_workers=4, shuffle=True, drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=opt.batch_size, num_workers=4, shuffle=True, drop_last=True)

    eval_val_loader, _ = get_dataset_motion_loader(dataset_opt_path, 32, 'val', device=opt.device)

    eval_wrapper = Evaluators(dataset_name=opt.dataset_name, device=opt.device)

    trainer = MaskTransformerTrainer(opt, t2m_transformer, ae, vq_model) #, res_model

    trainer.train(train_loader, val_loader, eval_val_loader, eval_wrapper=eval_wrapper, plot_eval=plot_t2m)

    #CUDA_LAUNCH_BLOCKING=1 python train_abs_transformer.py --dataset_name t2m --name dual_sparse_token_trans_4_all_quants --batch_size 64 --max_epoch 500 --milestones 50000 --trans cross_attn --latent_dim 512 --ff_size 1024 --n_heads 8 --n_layers 4 --gpu_id 2