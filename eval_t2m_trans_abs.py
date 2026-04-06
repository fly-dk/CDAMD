import os
from os.path import join as pjoin
import torch
from options.eval_option import EvalT2MOptions
from utils.get_opt import get_opt
from motion_loaders.dataset_motion_loader import get_dataset_motion_loader
from models.t2m_eval_wrapper import EvaluatorModelWrapper

import utils.eval_t2m as eval_t2m
from utils.fixseed import fixseed

import numpy as np
from models.mask_transformer.abs_transformer import MaskTransformer
from models.mardm_evaluators import Evaluators
from models.no_vq.AE import AE_models


def load_trans_model(model_opt, which_model):
    t2m_transformer = MaskTransformer(ae_dim=ae.output_emb_width,
                                      cond_mode='text',
                                      latent_dim=model_opt.latent_dim,
                                      ff_size=model_opt.ff_size,
                                      num_layers=model_opt.n_layers,
                                      num_heads=model_opt.n_heads,
                                      dropout=model_opt.dropout,
                                      clip_dim=512,
                                      cond_drop_prob=model_opt.cond_drop_prob,
                                      clip_version=clip_version,
                                      opt=model_opt)
    ckpt = torch.load(pjoin(model_opt.checkpoints_dir, model_opt.dataset_name, model_opt.name, 'model', which_model),
                      map_location=opt.device)
    model_key = 't2m_transformer' if 't2m_transformer' in ckpt else 'trans'
    missing_keys, unexpected_keys = t2m_transformer.load_state_dict(ckpt[model_key], strict=False)
    non_clip_missing = [k for k in missing_keys if not k.startswith('clip_model.')]
    if unexpected_keys:
        print(f"[Warn] Unexpected keys in checkpoint (ignored): {unexpected_keys}")
    if non_clip_missing:
        print(f"[Warn] Missing non-CLIP keys (use current init): {non_clip_missing}")
    else:
        print(f"[Info] Only CLIP weights are missing (expected).")
    print(f'Loading Mask Transformer {opt.name} from epoch {ckpt["ep"]}!')
    return t2m_transformer

if __name__ == '__main__':
    parser = EvalT2MOptions()
    opt = parser.parse(is_eval=True)
    fixseed(opt.seed)

    opt.device = torch.device("cpu" if opt.gpu_id == -1 else "cuda:" + str(opt.gpu_id))
    torch.autograd.set_detect_anomaly(True)

    dim_pose = 63 if opt.dataset_name == 'kit' else 66

    root_dir = pjoin(opt.checkpoints_dir, opt.dataset_name, opt.name)
    model_dir = pjoin(root_dir, 'model')
    out_dir = pjoin(root_dir, 'eval')
    os.makedirs(out_dir, exist_ok=True)

    out_path = pjoin(out_dir, "%s.log"%opt.ext)

    f = open(pjoin(out_path), 'w')

    model_opt_path = pjoin(root_dir, 'opt.txt')
    model_opt = get_opt(model_opt_path, device=opt.device)
    clip_version = 'ViT-B/32'

    #####################################################################
    # load AE model                                                     #
    #####################################################################
    ae = AE_models['AE_Model'](input_width=dim_pose)
    ae_model_dir = pjoin('/checkpoints/t2m/AE/model', 'latest.tar')
    checkpoint = torch.load(ae_model_dir, map_location='cpu')
    ae.load_state_dict(checkpoint['ae'])
    #####################################################################


    dataset_opt_path = 'checkpoints/kit/Comp_v6_KLD005/opt.txt' if opt.dataset_name == 'kit' \
        else 'checkpoints/t2m/Comp_v6_KLD005/opt.txt'

    eval_wrapper = Evaluators(dataset_name=opt.dataset_name, device=opt.device)

    ##### ---- Dataloader ---- #####
    opt.nb_joints = 21 if opt.dataset_name == 'kit' else 22

    eval_val_loader, _ = get_dataset_motion_loader(dataset_opt_path, 32, 'test', device=opt.device)

    # model_dir = pjoin(opt.)
    for file in os.listdir(model_dir):
        if opt.which_epoch != "all" and opt.which_epoch not in file:
            continue
        print('loading checkpoint {}'.format(file))
        t2m_transformer = load_trans_model(model_opt, file)
        t2m_transformer.eval()
        # vq_model.eval()
        ae.eval()

        t2m_transformer.to(opt.device)
        ae.to(opt.device)

        fid = []
        div = []
        top1 = []
        top2 = []
        top3 = []
        matching = []
        mm = []
        clip_scores = []

        repeat_time = 20# for editing 1, for t2m 20
        for i in range(repeat_time):
            with torch.no_grad():
                best_fid, best_div, Rprecision, best_matching, clip_score, best_mm, msg = \
                    eval_t2m.evaluation_abs_transformer_test(eval_val_loader, ae, t2m_transformer,
                                                                       i, eval_wrapper=eval_wrapper,
                                                         time_steps=opt.time_steps, cond_scale=opt.cond_scale,
                                                         temperature=opt.temperature, device=opt.device,
                                                                       force_mask=opt.force_mask, cal_mm=False, 
                                                                       save_motions=False,
                                                                       save_dir="custom_generated_motions_output",
                                                                       )
                print(msg, file=f, flush=True)
            fid.append(best_fid)
            div.append(best_div)
            top1.append(Rprecision[0])
            top2.append(Rprecision[1])
            top3.append(Rprecision[2])
            matching.append(best_matching)
            mm.append(best_mm)
            clip_scores.append(clip_score)

        fid = np.array(fid)
        div = np.array(div)
        top1 = np.array(top1)
        top2 = np.array(top2)
        top3 = np.array(top3)
        matching = np.array(matching)
        mm = np.array(mm)
        clip_scores = np.array(clip_scores)

        print(f'{file} final result:')
        print(f'{file} final result:', file=f, flush=True)

        msg_final = f"\tFID: {np.mean(fid):.3f}, conf. {np.std(fid) * 1.96 / np.sqrt(repeat_time):.3f}\n" \
                    f"\tDiversity: {np.mean(div):.3f}, conf. {np.std(div) * 1.96 / np.sqrt(repeat_time):.3f}\n" \
                    f"\tTOP1: {np.mean(top1):.3f}, conf. {np.std(top1) * 1.96 / np.sqrt(repeat_time):.3f}, TOP2. {np.mean(top2):.3f}, conf. {np.std(top2) * 1.96 / np.sqrt(repeat_time):.3f}, TOP3. {np.mean(top3):.3f}, conf. {np.std(top3) * 1.96 / np.sqrt(repeat_time):.3f}\n" \
                    f"\tMatching: {np.mean(matching):.3f}, conf. {np.std(matching) * 1.96 / np.sqrt(repeat_time):.3f}\n" \
                    f"\tMultimodality:{np.mean(mm):.3f}, conf.{np.std(mm) * 1.96 / np.sqrt(repeat_time):.3f}\n\n"\
                    f"\tCLIP-Score:{np.mean(clip_scores):.3f}, conf.{np.std(clip_scores) * 1.96 / np.sqrt(repeat_time):.3f}\n\n"
        # logger.info(msg_final)
        print(msg_final)
        print(msg_final, file=f, flush=True)

    f.close()


# python eval_t2m_trans_abs.py --name dual_sparse_token_trans_4 --dataset_name t2m --gpu_id 0 --cond_scale 4 --time_steps 10 --ext your_eval --checkpoints_dir ./log/t2m