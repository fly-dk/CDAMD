import os

import clip
import numpy as np
import torch
from scipy import linalg
from utils.metrics import *
import torch.nn.functional as F
# import visualization.plot_3d_global as plot_3d
from utils.motion_process_acmdm import recover_from_ric
from models.mask_transformer.tools import lengths_to_mask
from utils.back_process import back_process
from tqdm import tqdm
from einops import repeat
from data_loaders.humanml.utils.metrics import calculate_skating_ratio, compute_kps_error, calculate_trajectory_error, calculate_trajectory_diversity
from itertools import combinations
from utils.humanml_utils import build_upper_body_keep_mask

@torch.no_grad()
def evaluation_no_vqvae(out_dir, val_loader, net, writer, ep, best_fid, best_div, best_top1,
                     best_top2, best_top3, best_matching, eval_wrapper, save=True, draw=True):
    net.eval()

    motion_annotation_list = []
    motion_pred_list = []

    R_precision_real = 0
    R_precision = 0

    nb_sample = 0
    matching_score_real = 0
    matching_score_pred = 0
    for batch in val_loader:
        # print(len(batch))
        word_embeddings, pos_one_hots, caption, sent_len, motion, m_length, token = batch

        motion = motion.cuda()
        # print('motion shape:', motion.shape)  #motion shape: torch.Size([32, 196, 66])
        
        et, em = eval_wrapper.get_co_embeddings(word_embeddings, pos_one_hots, sent_len, caption, motion, m_length)
        bs, seq = motion.shape[0], motion.shape[1]

        pred_pose_eval, loss_ckl = net(motion) # no_vq_vae

        et_pred, em_pred = eval_wrapper.get_co_embeddings(word_embeddings, pos_one_hots, sent_len, caption, pred_pose_eval,
                                                          m_length)

        motion_pred_list.append(em_pred)
        motion_annotation_list.append(em)

        temp_R = calculate_R_precision(et.cpu().numpy(), em.cpu().numpy(), top_k=3, sum_all=True)
        temp_match = euclidean_distance_matrix(et.cpu().numpy(), em.cpu().numpy()).trace()
        R_precision_real += temp_R
        matching_score_real += temp_match
        temp_R = calculate_R_precision(et_pred.cpu().numpy(), em_pred.cpu().numpy(), top_k=3, sum_all=True)
        temp_match = euclidean_distance_matrix(et_pred.cpu().numpy(), em_pred.cpu().numpy()).trace()
        R_precision += temp_R
        matching_score_pred += temp_match

        nb_sample += bs

    motion_annotation_np = torch.cat(motion_annotation_list, dim=0).cpu().numpy()
    motion_pred_np = torch.cat(motion_pred_list, dim=0).cpu().numpy()
    gt_mu, gt_cov = calculate_activation_statistics(motion_annotation_np)
    mu, cov = calculate_activation_statistics(motion_pred_np)

    diversity_real = calculate_diversity(motion_annotation_np, 300 if nb_sample > 300 else 100)
    diversity = calculate_diversity(motion_pred_np, 300 if nb_sample > 300 else 100)

    R_precision_real = R_precision_real / nb_sample
    R_precision = R_precision / nb_sample

    matching_score_real = matching_score_real / nb_sample
    matching_score_pred = matching_score_pred / nb_sample

    fid = calculate_frechet_distance(gt_mu, gt_cov, mu, cov)

    msg = "--> \t Eva. Ep %d:, FID. %.4f, Diversity Real. %.4f, Diversity. %.4f, R_precision_real. (%.4f, %.4f, %.4f), R_precision. (%.4f, %.4f, %.4f), matching_score_real. %.4f, matching_score_pred. %.4f"%\
          (ep, fid, diversity_real, diversity, R_precision_real[0],R_precision_real[1], R_precision_real[2],
           R_precision[0],R_precision[1], R_precision[2], matching_score_real, matching_score_pred )
    # logger.info(msg)
    print(msg)

    if draw:
        writer.add_scalar('./Test/FID', fid, ep)
        writer.add_scalar('./Test/Diversity', diversity, ep)
        writer.add_scalar('./Test/top1', R_precision[0], ep)
        writer.add_scalar('./Test/top2', R_precision[1], ep)
        writer.add_scalar('./Test/top3', R_precision[2], ep)
        writer.add_scalar('./Test/matching_score', matching_score_pred, ep)

    if fid < best_fid:
        msg = "--> --> \t FID Improved from %.5f to %.5f !!!" % (best_fid, fid)
        if draw: print(msg)
        best_fid = fid
        if save:
            torch.save({'vq_model': net.state_dict(), 'ep': ep}, os.path.join(out_dir, 'net_best_fid.tar'))

    if abs(diversity_real - diversity) < abs(diversity_real - best_div):
        msg = "--> --> \t Diversity Improved from %.5f to %.5f !!!"%(best_div, diversity)
        if draw: print(msg)
        best_div = diversity
        # if save:
        #     torch.save({'net': net.state_dict()}, os.path.join(out_dir, 'net_best_div.pth'))

    if R_precision[0] > best_top1:
        msg = "--> --> \t Top1 Improved from %.5f to %.5f !!!" % (best_top1, R_precision[0])
        if draw: print(msg)
        best_top1 = R_precision[0]
        if save:
            torch.save({'vq_model': net.state_dict(), 'ep':ep}, os.path.join(out_dir, 'net_best_top1.tar'))

    if R_precision[1] > best_top2:
        msg = "--> --> \t Top2 Improved from %.5f to %.5f!!!" % (best_top2, R_precision[1])
        if draw: print(msg)
        best_top2 = R_precision[1]

    if R_precision[2] > best_top3:
        msg = "--> --> \t Top3 Improved from %.5f to %.5f !!!" % (best_top3, R_precision[2])
        if draw: print(msg)
        best_top3 = R_precision[2]

    if matching_score_pred < best_matching:
        msg = f"--> --> \t matching_score Improved from %.5f to %.5f !!!" % (best_matching, matching_score_pred)
        if draw: print(msg)
        best_matching = matching_score_pred
        if save:
            torch.save({'vq_model': net.state_dict(), 'ep': ep}, os.path.join(out_dir, 'net_best_mm.tar'))

    # if save:
    #     torch.save({'net': net.state_dict()}, os.path.join(out_dir, 'net_last.pth'))

    net.train()
    return best_fid, best_div, best_top1, best_top2, best_top3, best_matching, writer

@torch.no_grad()
def evaluation_vqvae(out_dir, val_loader, net, writer, ep, best_fid, best_div, best_top1,
                     best_top2, best_top3, best_matching, eval_wrapper, save=True, draw=True):
    net.eval()

    motion_annotation_list = []
    motion_pred_list = []

    R_precision_real = 0
    R_precision = 0

    nb_sample = 0
    matching_score_real = 0
    matching_score_pred = 0
    device = next(net.parameters()).device
    for batch in val_loader:
        # print(len(batch))
        word_embeddings, pos_one_hots, caption, sent_len, motion, m_length, token = batch

        motion = motion.cuda()
        bs, seq = motion.shape[0], motion.shape[1]

        # print('motion shape:', motion.shape)  #motion shape: torch.Size([32, 196, 66])

        (et, em), (_, _) = eval_wrapper.get_co_embeddings(word_embeddings, pos_one_hots, sent_len, caption, motion, m_length)

        # 原来的vq model
        pred_pose_eval, loss_commit, perplexity = net(motion) # vq_vae


        (et_pred, em_pred), (_, _) = eval_wrapper.get_co_embeddings(word_embeddings, pos_one_hots, sent_len, caption, pred_pose_eval,
                                                          m_length)

        motion_pred_list.append(em_pred)
        motion_annotation_list.append(em)

        temp_R = calculate_R_precision(et.cpu().numpy(), em.cpu().numpy(), top_k=3, sum_all=True)
        temp_match = euclidean_distance_matrix(et.cpu().numpy(), em.cpu().numpy()).trace()
        R_precision_real += temp_R
        matching_score_real += temp_match
        temp_R = calculate_R_precision(et_pred.cpu().numpy(), em_pred.cpu().numpy(), top_k=3, sum_all=True)
        temp_match = euclidean_distance_matrix(et_pred.cpu().numpy(), em_pred.cpu().numpy()).trace()
        R_precision += temp_R
        matching_score_pred += temp_match

        nb_sample += bs

    motion_annotation_np = torch.cat(motion_annotation_list, dim=0).cpu().numpy()
    motion_pred_np = torch.cat(motion_pred_list, dim=0).cpu().numpy()
    gt_mu, gt_cov = calculate_activation_statistics(motion_annotation_np)
    mu, cov = calculate_activation_statistics(motion_pred_np)

    diversity_real = calculate_diversity(motion_annotation_np, 300 if nb_sample > 300 else 100)
    diversity = calculate_diversity(motion_pred_np, 300 if nb_sample > 300 else 100)

    R_precision_real = R_precision_real / nb_sample
    R_precision = R_precision / nb_sample

    matching_score_real = matching_score_real / nb_sample
    matching_score_pred = matching_score_pred / nb_sample

    fid = calculate_frechet_distance(gt_mu, gt_cov, mu, cov)

    msg = "--> \t Eva. Ep %d:, FID. %.4f, Diversity Real. %.4f, Diversity. %.4f, R_precision_real. (%.4f, %.4f, %.4f), R_precision. (%.4f, %.4f, %.4f), matching_score_real. %.4f, matching_score_pred. %.4f"%\
          (ep, fid, diversity_real, diversity, R_precision_real[0],R_precision_real[1], R_precision_real[2],
           R_precision[0],R_precision[1], R_precision[2], matching_score_real, matching_score_pred )
    # logger.info(msg)
    print(msg)

    if draw:
        writer.add_scalar('./Test/FID', fid, ep)
        writer.add_scalar('./Test/Diversity', diversity, ep)
        writer.add_scalar('./Test/top1', R_precision[0], ep)
        writer.add_scalar('./Test/top2', R_precision[1], ep)
        writer.add_scalar('./Test/top3', R_precision[2], ep)
        writer.add_scalar('./Test/matching_score', matching_score_pred, ep)

    if fid < best_fid:
        msg = "--> --> \t FID Improved from %.5f to %.5f !!!" % (best_fid, fid)
        if draw: print(msg)
        best_fid = fid
        if save:
            torch.save({'vq_model': net.state_dict(), 'ep': ep}, os.path.join(out_dir, 'net_best_fid.tar'))

    if abs(diversity_real - diversity) < abs(diversity_real - best_div):
        msg = "--> --> \t Diversity Improved from %.5f to %.5f !!!"%(best_div, diversity)
        if draw: print(msg)
        best_div = diversity
        # if save:
        #     torch.save({'net': net.state_dict()}, os.path.join(out_dir, 'net_best_div.pth'))

    if R_precision[0] > best_top1:
        msg = "--> --> \t Top1 Improved from %.5f to %.5f !!!" % (best_top1, R_precision[0])
        if draw: print(msg)
        best_top1 = R_precision[0]
        if save:
            torch.save({'vq_model': net.state_dict(), 'ep':ep}, os.path.join(out_dir, 'net_best_top1.tar'))

    if R_precision[1] > best_top2:
        msg = "--> --> \t Top2 Improved from %.5f to %.5f!!!" % (best_top2, R_precision[1])
        if draw: print(msg)
        best_top2 = R_precision[1]

    if R_precision[2] > best_top3:
        msg = "--> --> \t Top3 Improved from %.5f to %.5f !!!" % (best_top3, R_precision[2])
        if draw: print(msg)
        best_top3 = R_precision[2]

    if matching_score_pred < best_matching:
        msg = f"--> --> \t matching_score Improved from %.5f to %.5f !!!" % (best_matching, matching_score_pred)
        if draw: print(msg)
        best_matching = matching_score_pred
        if save:
            torch.save({'vq_model': net.state_dict(), 'ep': ep}, os.path.join(out_dir, 'net_best_mm.tar'))

    # if save:
    #     torch.save({'net': net.state_dict()}, os.path.join(out_dir, 'net_last.pth'))

    net.train()
    return best_fid, best_div, best_top1, best_top2, best_top3, best_matching, writer

@torch.no_grad()
def evaluation_vqvae_plus_mpjpe(val_loader, net, repeat_id, eval_wrapper, num_joint):
    net.eval()

    motion_annotation_list = []
    motion_pred_list = []

    R_precision_real = 0
    R_precision = 0

    nb_sample = 0
    matching_score_real = 0
    matching_score_pred = 0
    mpjpe = 0
    num_poses = 0
    for batch in val_loader:
        # print(len(batch))
        word_embeddings, pos_one_hots, caption, sent_len, motion, m_length, token = batch

        motion = motion.cuda()
        et, em = eval_wrapper.get_co_embeddings(word_embeddings, pos_one_hots, sent_len, caption, motion, m_length)
        bs, seq = motion.shape[0], motion.shape[1]

        # num_joints = 21 if motion.shape[-1] == 251 else 22

        # pred_pose_eval = torch.zeros((bs, seq, motion.shape[-1])).cuda()

        pred_pose_eval, loss_commit, perplexity = net(motion)
        # all_indices,_  = net.encode(motion)
        # pred_pose_eval = net.forward_decoder(all_indices[..., :1])

        et_pred, em_pred = eval_wrapper.get_co_embeddings(word_embeddings, pos_one_hots, sent_len, pred_pose_eval,
                                                          m_length)

        bgt = val_loader.dataset.inv_transform(motion.detach().cpu().numpy())
        bpred = val_loader.dataset.inv_transform(pred_pose_eval.detach().cpu().numpy())
        for i in range(bs):
            gt = recover_from_ric(torch.from_numpy(bgt[i, :m_length[i]]).float(), num_joint)
            pred = recover_from_ric(torch.from_numpy(bpred[i, :m_length[i]]).float(), num_joint)

            mpjpe += torch.sum(calculate_mpjpe(gt, pred))
            # print(calculate_mpjpe(gt, pred).shape, gt.shape, pred.shape)
            num_poses += gt.shape[0]

        # print(mpjpe, num_poses)
        # exit()

        motion_pred_list.append(em_pred)
        motion_annotation_list.append(em)

        temp_R = calculate_R_precision(et.cpu().numpy(), em.cpu().numpy(), top_k=3, sum_all=True)
        temp_match = euclidean_distance_matrix(et.cpu().numpy(), em.cpu().numpy()).trace()
        R_precision_real += temp_R
        matching_score_real += temp_match
        temp_R = calculate_R_precision(et_pred.cpu().numpy(), em_pred.cpu().numpy(), top_k=3, sum_all=True)
        temp_match = euclidean_distance_matrix(et_pred.cpu().numpy(), em_pred.cpu().numpy()).trace()
        R_precision += temp_R
        matching_score_pred += temp_match

        nb_sample += bs

    motion_annotation_np = torch.cat(motion_annotation_list, dim=0).cpu().numpy()
    motion_pred_np = torch.cat(motion_pred_list, dim=0).cpu().numpy()
    gt_mu, gt_cov = calculate_activation_statistics(motion_annotation_np)
    mu, cov = calculate_activation_statistics(motion_pred_np)

    diversity_real = calculate_diversity(motion_annotation_np, 300 if nb_sample > 300 else 100)
    diversity = calculate_diversity(motion_pred_np, 300 if nb_sample > 300 else 100)

    R_precision_real = R_precision_real / nb_sample
    R_precision = R_precision / nb_sample

    matching_score_real = matching_score_real / nb_sample
    matching_score_pred = matching_score_pred / nb_sample
    mpjpe = mpjpe / num_poses

    fid = calculate_frechet_distance(gt_mu, gt_cov, mu, cov)

    msg = "--> \t Eva. Re %d:, FID. %.4f, Diversity Real. %.4f, Diversity. %.4f, R_precision_real. (%.4f, %.4f, %.4f), R_precision. (%.4f, %.4f, %.4f), matching_real. %.4f, matching_pred. %.4f, MPJPE. %.4f" % \
          (repeat_id, fid, diversity_real, diversity, R_precision_real[0], R_precision_real[1], R_precision_real[2],
           R_precision[0], R_precision[1], R_precision[2], matching_score_real, matching_score_pred, mpjpe)
    # logger.info(msg)
    print(msg)
    return fid, diversity, R_precision, matching_score_pred, mpjpe

@torch.no_grad()
def evaluation_vqvae_plus_l1(val_loader, net, repeat_id, eval_wrapper, num_joint):
    net.eval()

    motion_annotation_list = []
    motion_pred_list = []

    R_precision_real = 0
    R_precision = 0

    nb_sample = 0
    matching_score_real = 0
    matching_score_pred = 0
    l1_dist = 0
    num_poses = 1
    for batch in val_loader:
        # print(len(batch))
        word_embeddings, pos_one_hots, caption, sent_len, motion, m_length, token = batch

        motion = motion.cuda()
        et, em = eval_wrapper.get_co_embeddings(word_embeddings, pos_one_hots, sent_len, motion, m_length)
        bs, seq = motion.shape[0], motion.shape[1]

        # num_joints = 21 if motion.shape[-1] == 251 else 22

        # pred_pose_eval = torch.zeros((bs, seq, motion.shape[-1])).cuda()

        pred_pose_eval, loss_commit, perplexity = net(motion)
        # all_indices,_  = net.encode(motion)
        # pred_pose_eval = net.forward_decoder(all_indices[..., :1])

        et_pred, em_pred = eval_wrapper.get_co_embeddings(word_embeddings, pos_one_hots, sent_len, pred_pose_eval,
                                                          m_length)

        bgt = val_loader.dataset.inv_transform(motion.detach().cpu().numpy())
        bpred = val_loader.dataset.inv_transform(pred_pose_eval.detach().cpu().numpy())
        for i in range(bs):
            gt = recover_from_ric(torch.from_numpy(bgt[i, :m_length[i]]).float(), num_joint)
            pred = recover_from_ric(torch.from_numpy(bpred[i, :m_length[i]]).float(), num_joint)
            # gt = motion[i, :m_length[i]]
            # pred = pred_pose_eval[i, :m_length[i]]
            num_pose = gt.shape[0]
            l1_dist += F.l1_loss(gt, pred) * num_pose
            num_poses += num_pose

        motion_pred_list.append(em_pred)
        motion_annotation_list.append(em)

        temp_R = calculate_R_precision(et.cpu().numpy(), em.cpu().numpy(), top_k=3, sum_all=True)
        temp_match = euclidean_distance_matrix(et.cpu().numpy(), em.cpu().numpy()).trace()
        R_precision_real += temp_R
        matching_score_real += temp_match
        temp_R = calculate_R_precision(et_pred.cpu().numpy(), em_pred.cpu().numpy(), top_k=3, sum_all=True)
        temp_match = euclidean_distance_matrix(et_pred.cpu().numpy(), em_pred.cpu().numpy()).trace()
        R_precision += temp_R
        matching_score_pred += temp_match

        nb_sample += bs

    motion_annotation_np = torch.cat(motion_annotation_list, dim=0).cpu().numpy()
    motion_pred_np = torch.cat(motion_pred_list, dim=0).cpu().numpy()
    gt_mu, gt_cov = calculate_activation_statistics(motion_annotation_np)
    mu, cov = calculate_activation_statistics(motion_pred_np)

    diversity_real = calculate_diversity(motion_annotation_np, 300 if nb_sample > 300 else 100)
    diversity = calculate_diversity(motion_pred_np, 300 if nb_sample > 300 else 100)

    R_precision_real = R_precision_real / nb_sample
    R_precision = R_precision / nb_sample

    matching_score_real = matching_score_real / nb_sample
    matching_score_pred = matching_score_pred / nb_sample
    l1_dist = l1_dist / num_poses

    fid = calculate_frechet_distance(gt_mu, gt_cov, mu, cov)

    msg = "--> \t Eva. Re %d:, FID. %.4f, Diversity Real. %.4f, Diversity. %.4f, R_precision_real. (%.4f, %.4f, %.4f), R_precision. (%.4f, %.4f, %.4f), matching_real. %.4f, matching_pred. %.4f, mae. %.4f"%\
          (repeat_id, fid, diversity_real, diversity, R_precision_real[0],R_precision_real[1], R_precision_real[2],
           R_precision[0],R_precision[1], R_precision[2], matching_score_real, matching_score_pred, l1_dist)
    # logger.info(msg)
    print(msg)
    return fid, diversity, R_precision, matching_score_pred, l1_dist


@torch.no_grad()
def evaluation_res_plus_l1(val_loader, vq_model, res_model, repeat_id, eval_wrapper, num_joint, do_vq_res=True):
    vq_model.eval()
    res_model.eval()

    motion_annotation_list = []
    motion_pred_list = []

    R_precision_real = 0
    R_precision = 0

    nb_sample = 0
    matching_score_real = 0
    matching_score_pred = 0
    l1_dist = 0
    num_poses = 1
    for batch in val_loader:
        # print(len(batch))
        word_embeddings, pos_one_hots, caption, sent_len, motion, m_length, token = batch

        motion = motion.cuda()
        et, em = eval_wrapper.get_co_embeddings(word_embeddings, pos_one_hots, sent_len, motion, m_length)
        bs, seq = motion.shape[0], motion.shape[1]

        # num_joints = 21 if motion.shape[-1] == 251 else 22

        # pred_pose_eval = torch.zeros((bs, seq, motion.shape[-1])).cuda()

        if do_vq_res:
            code_ids, all_codes = vq_model.encode(motion)
            if len(code_ids.shape) == 3:
                pred_vq_codes = res_model(code_ids[..., 0])
            else:
                pred_vq_codes = res_model(code_ids)
            # pred_vq_codes = pred_vq_codes - pred_vq_res + all_codes[1:].sum(0)
            pred_pose_eval = vq_model.decoder(pred_vq_codes)
        else:
            rec_motions, _, _ = vq_model(motion)
            pred_pose_eval = res_model(rec_motions)        # all_indices,_  = net.encode(motion)
        # pred_pose_eval = net.forward_decoder(all_indices[..., :1])

        et_pred, em_pred = eval_wrapper.get_co_embeddings(word_embeddings, pos_one_hots, sent_len, pred_pose_eval,
                                                          m_length)

        bgt = val_loader.dataset.inv_transform(motion.detach().cpu().numpy())
        bpred = val_loader.dataset.inv_transform(pred_pose_eval.detach().cpu().numpy())
        for i in range(bs):
            gt = recover_from_ric(torch.from_numpy(bgt[i, :m_length[i]]).float(), num_joint)
            pred = recover_from_ric(torch.from_numpy(bpred[i, :m_length[i]]).float(), num_joint)
            # gt = motion[i, :m_length[i]]
            # pred = pred_pose_eval[i, :m_length[i]]
            num_pose = gt.shape[0]
            l1_dist += F.l1_loss(gt, pred) * num_pose
            num_poses += num_pose

        motion_pred_list.append(em_pred)
        motion_annotation_list.append(em)

        temp_R = calculate_R_precision(et.cpu().numpy(), em.cpu().numpy(), top_k=3, sum_all=True)
        temp_match = euclidean_distance_matrix(et.cpu().numpy(), em.cpu().numpy()).trace()
        R_precision_real += temp_R
        matching_score_real += temp_match
        temp_R = calculate_R_precision(et_pred.cpu().numpy(), em_pred.cpu().numpy(), top_k=3, sum_all=True)
        temp_match = euclidean_distance_matrix(et_pred.cpu().numpy(), em_pred.cpu().numpy()).trace()
        R_precision += temp_R
        matching_score_pred += temp_match

        nb_sample += bs

    motion_annotation_np = torch.cat(motion_annotation_list, dim=0).cpu().numpy()
    motion_pred_np = torch.cat(motion_pred_list, dim=0).cpu().numpy()
    gt_mu, gt_cov = calculate_activation_statistics(motion_annotation_np)
    mu, cov = calculate_activation_statistics(motion_pred_np)

    diversity_real = calculate_diversity(motion_annotation_np, 300 if nb_sample > 300 else 100)
    diversity = calculate_diversity(motion_pred_np, 300 if nb_sample > 300 else 100)

    R_precision_real = R_precision_real / nb_sample
    R_precision = R_precision / nb_sample

    matching_score_real = matching_score_real / nb_sample
    matching_score_pred = matching_score_pred / nb_sample
    l1_dist = l1_dist / num_poses

    fid = calculate_frechet_distance(gt_mu, gt_cov, mu, cov)

    msg = "--> \t Eva. Re %d:, FID. %.4f, Diversity Real. %.4f, Diversity. %.4f, R_precision_real. (%.4f, %.4f, %.4f), R_precision. (%.4f, %.4f, %.4f), matching_real. %.4f, matching_pred. %.4f, mae. %.4f"%\
          (repeat_id, fid, diversity_real, diversity, R_precision_real[0],R_precision_real[1], R_precision_real[2],
           R_precision[0],R_precision[1], R_precision[2], matching_score_real, matching_score_pred, l1_dist)
    # logger.info(msg)
    print(msg)
    return fid, diversity, R_precision, matching_score_pred, l1_dist

@torch.no_grad()
def evaluation_mask_transformer(out_dir, val_loader, trans, vq_model, writer, ep, best_fid, best_div,
                           best_top1, best_top2, best_top3, best_matching, eval_wrapper, plot_func,
                           save_ckpt=False, save_anim=False, res_model=None):

    def save(file_name, ep):
        t2m_trans_state_dict = trans.state_dict()
        clip_weights = [e for e in t2m_trans_state_dict.keys() if e.startswith('clip_model.')]
        for e in clip_weights:
            del t2m_trans_state_dict[e]
        state = {
            't2m_transformer': t2m_trans_state_dict,
            # 'opt_t2m_transformer': self.opt_t2m_transformer.state_dict(),
            # 'scheduler':self.scheduler.state_dict(),
            'ep': ep,
        }
        torch.save(state, file_name)

    trans.eval()
    vq_model.eval()

    motion_annotation_list = []
    motion_pred_list = []
    R_precision_real = 0
    R_precision = 0
    matching_score_real = 0
    matching_score_pred = 0
    time_steps = 3
    if "kit" in out_dir:
        cond_scale = 2
    else:
        cond_scale = 4

    # print(num_quantizer)

    # assert num_quantizer >= len(time_steps) and num_quantizer >= len(cond_scales)

    nb_sample = 0
    # for i in range(1):
    from tqdm import tqdm
    for batch in tqdm(val_loader):
        word_embeddings, pos_one_hots, clip_text, sent_len, pose, m_length, token = batch
        m_length = m_length.cuda()

        bs, seq = pose.shape[:2]
        # num_joints = 21 if pose.shape[-1] == 251 else 22

        # (b, seqlen)
        mids,_ = trans.generate(clip_text, m_length//4, time_steps, cond_scale, temperature=1)
        # for quick fix bug of old run that didn't predict end token
        # q = (mids == trans.end_id)
        # mids[q] = 0
        # pred_len = m_length
        #############################
        # print("mids", mids)
    #     mids (tensor([[468,  20, 247,  ..., 113, 336, 513],
    #     [265, 468, 347,  ..., 210, 472, 513],
    #     [287, 265,  70,  ..., 177, 320, 513],
    #     ...,
    #     [265, 265, 414,  ..., 223, 175, 513],
    #     [468, 460,  12,  ...,  80, 319, 513],
    #     [155, 468, 482,  ..., 114,  90, 513]], device='cuda:5'), tensor([25, 15, 24, 33, 23, 17, 46, 17, 12, 36, 34, 32, 35, 39, 15, 49, 14, 15,
    #     29, 41, 12, 19, 49, 10, 13, 16, 31, 18, 21, 18, 16, 45],
    #    device='cuda:5'))
        ###########################

        mids, pred_len = trans.pad_when_end(mids)
        pred_len = pred_len.clamp(1, 49) * 4

        # motion_codes = motion_codes.permute(0, 2, 1)
        if res_model is not None:
            mids = res_model.generate(mids, clip_text, pred_len // 4, temperature=1, cond_scale=5)
        else:
            mids =  mids.unsqueeze_(-1)
        
        #pred_motions_mask = lengths_to_mask(pred_len, 196).unsqueeze(-1) #(b, n, 1)
        pred_motions = vq_model.forward_decoder(mids) #* pred_motions_mask

        et_pred, em_pred = eval_wrapper.get_co_embeddings(word_embeddings, pos_one_hots, sent_len, clip_text, pred_motions.clone(),
                                                          pred_len)

        pose = pose.cuda().float()

        et, em = eval_wrapper.get_co_embeddings(word_embeddings, pos_one_hots, sent_len, clip_text, pose, m_length)
        motion_annotation_list.append(em)
        motion_pred_list.append(em_pred)

        temp_R = calculate_R_precision(et.cpu().numpy(), em.cpu().numpy(), top_k=3, sum_all=True)
        temp_match = euclidean_distance_matrix(et.cpu().numpy(), em.cpu().numpy()).trace()
        R_precision_real += temp_R
        matching_score_real += temp_match
        temp_R = calculate_R_precision(et_pred.cpu().numpy(), em_pred.cpu().numpy(), top_k=3, sum_all=True)
        temp_match = euclidean_distance_matrix(et_pred.cpu().numpy(), em_pred.cpu().numpy()).trace()
        R_precision += temp_R
        matching_score_pred += temp_match

        nb_sample += bs

    motion_annotation_np = torch.cat(motion_annotation_list, dim=0).cpu().numpy()
    motion_pred_np = torch.cat(motion_pred_list, dim=0).cpu().numpy()
    gt_mu, gt_cov = calculate_activation_statistics(motion_annotation_np)
    mu, cov = calculate_activation_statistics(motion_pred_np)

    diversity_real = calculate_diversity(motion_annotation_np, 300 if nb_sample > 300 else 100)
    diversity = calculate_diversity(motion_pred_np, 300 if nb_sample > 300 else 100)

    R_precision_real = R_precision_real / nb_sample
    R_precision = R_precision / nb_sample

    matching_score_real = matching_score_real / nb_sample
    matching_score_pred = matching_score_pred / nb_sample

    fid = calculate_frechet_distance(gt_mu, gt_cov, mu, cov)

    msg = f"--> \t Eva. Ep {ep} :, FID. {fid:.4f}, Diversity Real. {diversity_real:.4f}, Diversity. {diversity:.4f}, R_precision_real. {R_precision_real}, R_precision. {R_precision}, matching_score_real. {matching_score_real}, matching_score_pred. {matching_score_pred}"
    print(msg)

    # if draw:
    writer.add_scalar('./Test/FID', fid, ep)
    writer.add_scalar('./Test/Diversity', diversity, ep)
    writer.add_scalar('./Test/top1', R_precision[0], ep)
    writer.add_scalar('./Test/top2', R_precision[1], ep)
    writer.add_scalar('./Test/top3', R_precision[2], ep)
    writer.add_scalar('./Test/matching_score', matching_score_pred, ep)


    if fid < best_fid:
        msg = f"--> --> \t FID Improved from {best_fid:.5f} to {fid:.5f} !!!"
        print(msg)
        best_fid, best_ep = fid, ep
        if save_ckpt:
            save(os.path.join(out_dir, 'model', 'net_best_fid.tar'), ep)

    if matching_score_pred < best_matching:
        msg = f"--> --> \t matching_score Improved from {best_matching:.5f} to {matching_score_pred:.5f} !!!"
        print(msg)
        best_matching = matching_score_pred
        if save_ckpt:
            save(os.path.join(out_dir, 'model', 'net_best_mm.tar'), ep)

    if abs(diversity_real - diversity) < abs(diversity_real - best_div):
        msg = f"--> --> \t Diversity Improved from {best_div:.5f} to {diversity:.5f} !!!"
        print(msg)
        best_div = diversity

    if R_precision[0] > best_top1:
        msg = f"--> --> \t Top1 Improved from {best_top1:.4f} to {R_precision[0]:.4f} !!!"
        print(msg)
        best_top1 = R_precision[0]

    if R_precision[1] > best_top2:
        msg = f"--> --> \t Top2 Improved from {best_top2:.4f} to {R_precision[1]:.4f} !!!"
        print(msg)
        best_top2 = R_precision[1]

    if R_precision[2] > best_top3:
        msg = f"--> --> \t Top3 Improved from {best_top3:.4f} to {R_precision[2]:.4f} !!!"
        print(msg)
        best_top3 = R_precision[2]

    if save_anim:
        rand_idx = torch.randint(bs, (3,))
        data = pred_motions[rand_idx].detach().cpu().numpy()
        captions = [clip_text[k] for k in rand_idx]
        lengths = m_length[rand_idx].cpu().numpy()
        save_dir = os.path.join(out_dir, 'animation', 'E%04d' % ep)
        os.makedirs(save_dir, exist_ok=True)
        # print(lengths)
        # plot_func(data, save_dir, captions, lengths)


    return best_fid, best_div, best_top1, best_top2, best_top3, best_matching, writer

@torch.no_grad()
def evaluation_abs_transformer(out_dir, val_loader, trans, ae, writer, ep, best_fid, best_div,
                        best_top1, best_top2, best_top3, best_matching, eval_wrapper, device, clip_score_old, time_steps=None,
                        cond_scale=None, temperature=1, cal_mm=False, plot_func=None, draw=True, hard_pseudo_reorder=False, save_ckpt=False):

    def save(file_name, ep):
        t2m_trans_state_dict = trans.state_dict()
        clip_weights = [e for e in t2m_trans_state_dict.keys() if e.startswith('clip_model.')]
        for e in clip_weights:
            del t2m_trans_state_dict[e]
        state = {
            't2m_transformer': t2m_trans_state_dict,
            'ep': ep,
        }
        torch.save(state, file_name)

    trans.eval()
    ae.eval()

    motion_annotation_list = []
    motion_pred_list = []
    motion_multimodality = []
    R_precision_real = 0
    R_precision = 0
    matching_score_real = 0
    matching_score_pred = 0
    multimodality = 0
    if time_steps is None: time_steps = 10
    if cond_scale is None:
        if "kit" in out_dir:
            cond_scale = 2.0
        else:
            cond_scale = 4.0

    clip_score_real = 0
    clip_score_gt = 0

    nb_sample = 0
    if cal_mm:
        num_mm_batch = 3
    else:
        num_mm_batch = 0

    for i, batch in enumerate(tqdm(val_loader)):
        word_embeddings, pos_one_hots, clip_text, sent_len, pose, m_length, token = batch
        m_length = m_length.to(device)

        bs, seq = pose.shape[:2]
        if i < num_mm_batch:
            motion_multimodality_batch = []
            batch_clip_score_pred = 0
            for _ in tqdm(range(30)):
                pred_latents = trans.generate(clip_text, m_length//4 , time_steps, cond_scale,
                                                  temperature=temperature, hard_pseudo_reorder=hard_pseudo_reorder)
                
                pred_motions = ae.decode(pred_latents)
                (et_pred, em_pred), (et_pred_clip, em_pred_clip) = eval_wrapper.get_co_embeddings(word_embeddings, pos_one_hots, sent_len,
                                                                            clip_text,
                                                                            pred_motions.to(device),
                                                                            m_length)
                motion_multimodality_batch.append(em_pred.unsqueeze(1))
            motion_multimodality_batch = torch.cat(motion_multimodality_batch, dim=1) #(bs, 30, d)
            motion_multimodality.append(motion_multimodality_batch)
            for j in range(32):
                single_em = em_pred_clip[j]
                single_et = et_pred_clip[j]
                clip_score = (single_em @ single_et.T).item()
                batch_clip_score_pred += clip_score
            clip_score_real += batch_clip_score_pred

        else:
            pred_latents = trans.generate(clip_text, m_length//4 , time_steps, cond_scale,
                                              temperature=temperature, hard_pseudo_reorder=hard_pseudo_reorder)

            pred_motions = ae.decode(pred_latents)
            
            (et_pred, em_pred), (et_pred_clip, em_pred_clip) = eval_wrapper.get_co_embeddings(word_embeddings,
                                                                              pos_one_hots, sent_len,
                                                                              clip_text,
                                                                              pred_motions.to(device),
                                                                              m_length)
            batch_clip_score_pred = 0
            for j in range(32):
                single_em = em_pred_clip[j]
                single_et = et_pred_clip[j]
                clip_score = (single_em @ single_et.T).item()
                batch_clip_score_pred += clip_score
            clip_score_real += batch_clip_score_pred

        pose = pose.cuda().float()
        (et, em), (et_clip, em_clip) = eval_wrapper.get_co_embeddings(word_embeddings, pos_one_hots, sent_len, clip_text,
                                                          pose.clone(), m_length)
        batch_clip_score = 0
        for j in range(32):
            single_em = em_clip[j]
            single_et = et_clip[j]
            clip_score = (single_em @ single_et.T).item()
            batch_clip_score += clip_score
        clip_score_gt += batch_clip_score
        motion_annotation_list.append(em)
        motion_pred_list.append(em_pred)

        temp_R = calculate_R_precision(et.cpu().numpy(), em.cpu().numpy(), top_k=3, sum_all=True)
        temp_match = euclidean_distance_matrix(et.cpu().numpy(), em.cpu().numpy()).trace()
        R_precision_real += temp_R
        matching_score_real += temp_match
        temp_R = calculate_R_precision(et_pred.cpu().numpy(), em_pred.cpu().numpy(), top_k=3, sum_all=True)
        temp_match = euclidean_distance_matrix(et_pred.cpu().numpy(), em_pred.cpu().numpy()).trace()
        R_precision += temp_R
        matching_score_pred += temp_match

        nb_sample += bs

    motion_annotation_np = torch.cat(motion_annotation_list, dim=0).cpu().numpy()
    motion_pred_np = torch.cat(motion_pred_list, dim=0).cpu().numpy()
    gt_mu, gt_cov = calculate_activation_statistics(motion_annotation_np)
    mu, cov = calculate_activation_statistics(motion_pred_np)

    diversity_real = calculate_diversity(motion_annotation_np, 300 if nb_sample > 300 else 100)
    diversity = calculate_diversity(motion_pred_np, 300 if nb_sample > 300 else 100)

    R_precision_real = R_precision_real / nb_sample
    R_precision = R_precision / nb_sample

    clip_score_real = clip_score_real / nb_sample
    clip_score_gt = clip_score_gt / nb_sample

    matching_score_real = matching_score_real / nb_sample
    matching_score_pred = matching_score_pred / nb_sample

    if cal_mm:
        motion_multimodality = torch.cat(motion_multimodality, dim=0).cpu().numpy()
        multimodality = calculate_multimodality(motion_multimodality, 10)

    fid = calculate_frechet_distance(gt_mu, gt_cov, mu, cov)

    msg = f"--> \t Eva. Ep/Re {ep} :, FID. {fid:.4f}, Diversity Real. {diversity_real:.4f}, Diversity. {diversity:.4f}, R_precision_real. {R_precision_real}, R_precision. {R_precision}, matching_score_real. {matching_score_real}, matching_score_pred. {matching_score_pred} multimodality. {multimodality:.4f} clip score real. {clip_score_gt} clip score. {clip_score_real}"
    print(msg)

    if draw:
        writer.add_scalar('./Test/FID', fid, ep)
        writer.add_scalar('./Test/Diversity', diversity, ep)
        writer.add_scalar('./Test/top1', R_precision[0], ep)
        writer.add_scalar('./Test/top2', R_precision[1], ep)
        writer.add_scalar('./Test/top3', R_precision[2], ep)
        writer.add_scalar('./Test/matching_score', matching_score_pred, ep)
        writer.add_scalar('./Test/clip_score', clip_score_real, ep)


    if fid < best_fid:
        msg = f"--> --> \t FID Improved from {best_fid:.5f} to {fid:.5f} !!!"
        print(msg)
        best_fid, best_ep = fid, ep
        if save_ckpt:
            save(os.path.join(out_dir, 'model', 'net_best_fid.tar'), ep)

    if matching_score_pred < best_matching:
        msg = f"--> --> \t matching_score Improved from {best_matching:.5f} to {matching_score_pred:.5f} !!!"
        print(msg)
        best_matching = matching_score_pred

    if abs(diversity_real - diversity) < abs(diversity_real - best_div):
        msg = f"--> --> \t Diversity Improved from {best_div:.5f} to {diversity:.5f} !!!"
        print(msg)
        best_div = diversity

    if R_precision[0] > best_top1:
        msg = f"--> --> \t Top1 Improved from {best_top1:.4f} to {R_precision[0]:.4f} !!!"
        print(msg)
        best_top1 = R_precision[0]

    if R_precision[1] > best_top2:
        msg = f"--> --> \t Top2 Improved from {best_top2:.4f} to {R_precision[1]:.4f} !!!"
        print(msg)
        best_top2 = R_precision[1]

    if R_precision[2] > best_top3:
        msg = f"--> --> \t Top3 Improved from {best_top3:.4f} to {R_precision[2]:.4f} !!!"
        print(msg)
        best_top3 = R_precision[2]

    if clip_score_real > clip_score_old:
        msg = f"--> --> \t CLIP-score Improved from {clip_score_old:.4f} to {clip_score_real:.4f} !!!"
        print(msg)
        clip_score_old = clip_score_real

    if cal_mm:
        return best_fid, best_div, best_top1, best_top2, best_top3, best_matching, multimodality, clip_score_old, writer, save
    else:
        return best_fid, best_div, best_top1, best_top2, best_top3, best_matching, 0, clip_score_old, writer, save

@torch.no_grad()
def evaluation_res_transformer(out_dir, val_loader, trans, vq_model, writer, ep, best_fid, best_div,
                           best_top1, best_top2, best_top3, best_matching, eval_wrapper, plot_func,
                           save_ckpt=False, save_anim=False, cond_scale=2, temperature=1, t2m_transformer=None):

    def save(file_name, ep):
        res_trans_state_dict = trans.state_dict()
        clip_weights = [e for e in res_trans_state_dict.keys() if e.startswith('clip_model.')]
        for e in clip_weights:
            del res_trans_state_dict[e]
        state = {
            'res_transformer': res_trans_state_dict,
            # 'opt_t2m_transformer': self.opt_t2m_transformer.state_dict(),
            # 'scheduler':self.scheduler.state_dict(),
            'ep': ep,
        }
        torch.save(state, file_name)

    trans.eval()
    vq_model.eval()

    motion_annotation_list = []
    motion_pred_list = []
    R_precision_real = 0
    R_precision = 0
    matching_score_real = 0
    matching_score_pred = 0

    # print(num_quantizer)

    # assert num_quantizer >= len(time_steps) and num_quantizer >= len(cond_scales)

    nb_sample = 0
    # for i in range(1):
    for batch in val_loader:
        word_embeddings, pos_one_hots, clip_text, sent_len, pose, m_length, token = batch
        m_length = m_length.cuda().long()
        pose = pose.cuda().float()

        bs, seq = pose.shape[:2]
        # num_joints = 21 if pose.shape[-1] == 251 else 22

        code_indices, all_codes = vq_model.encode(pose)
        # (b, seqlen)
        if ep == 0:
            pred_ids = code_indices[..., 0:1]
        else:
            mids, _ = t2m_transformer.generate(clip_text, m_length // 4, timesteps=3, cond_scale=4,
                                      temperature=temperature)
            mids, pred_len = t2m_transformer.pad_when_end(mids)
            pred_ids = trans.generate(mids, clip_text, m_length//4,
                                      temperature=temperature, cond_scale=cond_scale)
            # pred_codes = trans(code_indices[..., 0], clip_text, m_length//4, force_mask=force_mask)

        pred_motions = vq_model.forward_decoder(pred_ids)

        et_pred, em_pred = eval_wrapper.get_co_embeddings(word_embeddings, pos_one_hots, sent_len, clip_text, pred_motions.clone(),
                                                          m_length)

        pose = pose.cuda().float()

        et, em = eval_wrapper.get_co_embeddings(word_embeddings, pos_one_hots, sent_len, clip_text, pose, m_length)
        motion_annotation_list.append(em)
        motion_pred_list.append(em_pred)

        temp_R = calculate_R_precision(et.cpu().numpy(), em.cpu().numpy(), top_k=3, sum_all=True)
        temp_match = euclidean_distance_matrix(et.cpu().numpy(), em.cpu().numpy()).trace()
        R_precision_real += temp_R
        matching_score_real += temp_match
        temp_R = calculate_R_precision(et_pred.cpu().numpy(), em_pred.cpu().numpy(), top_k=3, sum_all=True)
        temp_match = euclidean_distance_matrix(et_pred.cpu().numpy(), em_pred.cpu().numpy()).trace()
        R_precision += temp_R
        matching_score_pred += temp_match

        nb_sample += bs

    motion_annotation_np = torch.cat(motion_annotation_list, dim=0).cpu().numpy()
    motion_pred_np = torch.cat(motion_pred_list, dim=0).cpu().numpy()
    gt_mu, gt_cov = calculate_activation_statistics(motion_annotation_np)
    mu, cov = calculate_activation_statistics(motion_pred_np)

    diversity_real = calculate_diversity(motion_annotation_np, 300 if nb_sample > 300 else 100)
    diversity = calculate_diversity(motion_pred_np, 300 if nb_sample > 300 else 100)

    R_precision_real = R_precision_real / nb_sample
    R_precision = R_precision / nb_sample

    matching_score_real = matching_score_real / nb_sample
    matching_score_pred = matching_score_pred / nb_sample

    fid = calculate_frechet_distance(gt_mu, gt_cov, mu, cov)

    msg = f"--> \t Eva. Ep {ep} :, FID. {fid:.4f}, Diversity Real. {diversity_real:.4f}, Diversity. {diversity:.4f}, R_precision_real. {R_precision_real}, R_precision. {R_precision}, matching_score_real. {matching_score_real}, matching_score_pred. {matching_score_pred}"
    print(msg)

    # if draw:
    writer.add_scalar('./Test/FID', fid, ep)
    writer.add_scalar('./Test/Diversity', diversity, ep)
    writer.add_scalar('./Test/top1', R_precision[0], ep)
    writer.add_scalar('./Test/top2', R_precision[1], ep)
    writer.add_scalar('./Test/top3', R_precision[2], ep)
    writer.add_scalar('./Test/matching_score', matching_score_pred, ep)


    if fid < best_fid:
        msg = f"--> --> \t FID Improved from {best_fid:.5f} to {fid:.5f} !!!"
        print(msg)
        best_fid, best_ep = fid, ep
        if save_ckpt:
            save(os.path.join(out_dir, 'model', 'net_best_fid.tar'), ep)

    if matching_score_pred < best_matching:
        msg = f"--> --> \t matching_score Improved from {best_matching:.5f} to {matching_score_pred:.5f} !!!"
        print(msg)
        best_matching = matching_score_pred

    if abs(diversity_real - diversity) < abs(diversity_real - best_div):
        msg = f"--> --> \t Diversity Improved from {best_div:.5f} to {diversity:.5f} !!!"
        print(msg)
        best_div = diversity

    if R_precision[0] > best_top1:
        msg = f"--> --> \t Top1 Improved from {best_top1:.4f} to {R_precision[0]:.4f} !!!"
        print(msg)
        best_top1 = R_precision[0]

    if R_precision[1] > best_top2:
        msg = f"--> --> \t Top2 Improved from {best_top2:.4f} to {R_precision[1]:.4f} !!!"
        print(msg)
        best_top2 = R_precision[1]

    if R_precision[2] > best_top3:
        msg = f"--> --> \t Top3 Improved from {best_top3:.4f} to {R_precision[2]:.4f} !!!"
        print(msg)
        best_top3 = R_precision[2]

    if save_anim:
        rand_idx = torch.randint(bs, (3,))
        data = pred_motions[rand_idx].detach().cpu().numpy()
        captions = [clip_text[k] for k in rand_idx]
        lengths = m_length[rand_idx].cpu().numpy()
        save_dir = os.path.join(out_dir, 'animation', 'E%04d' % ep)
        os.makedirs(save_dir, exist_ok=True)
        # print(lengths)
        # plot_func(data, save_dir, captions, lengths)


    return best_fid, best_div, best_top1, best_top2, best_top3, best_matching, writer


@torch.no_grad()
def evaluation_res_transformer_plus_l1(val_loader, vq_model, trans, repeat_id, eval_wrapper, num_joint,
                                       cond_scale=2, temperature=1, topkr=0.9, cal_l1=True):


    trans.eval()
    vq_model.eval()

    motion_annotation_list = []
    motion_pred_list = []
    R_precision_real = 0
    R_precision = 0
    matching_score_real = 0
    matching_score_pred = 0

    # print(num_quantizer)

    # assert num_quantizer >= len(time_steps) and num_quantizer >= len(cond_scales)

    nb_sample = 0
    l1_dist = 0
    num_poses = 1
    # for i in range(1):
    for batch in val_loader:
        word_embeddings, pos_one_hots, clip_text, sent_len, pose, m_length, token = batch
        m_length = m_length.cuda().long()
        pose = pose.cuda().float()

        bs, seq = pose.shape[:2]
        # num_joints = 21 if pose.shape[-1] == 251 else 22

        code_indices, all_codes = vq_model.encode(pose)
        # print(code_indices[0:2, :, 1])

        pred_ids = trans.generate(code_indices[..., 0], clip_text, m_length//4, topk_filter_thres=topkr,
                                  temperature=temperature, cond_scale=cond_scale)
            # pred_codes = trans(code_indices[..., 0], clip_text, m_length//4, force_mask=force_mask)

        pred_motions = vq_model.forward_decoder(pred_ids)

        if cal_l1:
            bgt = val_loader.dataset.inv_transform(pose.detach().cpu().numpy())
            bpred = val_loader.dataset.inv_transform(pred_motions.detach().cpu().numpy())
            for i in range(bs):
                gt = recover_from_ric(torch.from_numpy(bgt[i, :m_length[i]]).float(), num_joint)
                pred = recover_from_ric(torch.from_numpy(bpred[i, :m_length[i]]).float(), num_joint)
                # gt = motion[i, :m_length[i]]
                # pred = pred_pose_eval[i, :m_length[i]]
                num_pose = gt.shape[0]
                l1_dist += F.l1_loss(gt, pred) * num_pose
                num_poses += num_pose

        et_pred, em_pred = eval_wrapper.get_co_embeddings(word_embeddings, pos_one_hots, sent_len, pred_motions.clone(),
                                                          m_length)

        pose = pose.cuda().float()

        et, em = eval_wrapper.get_co_embeddings(word_embeddings, pos_one_hots, sent_len, pose, m_length)
        motion_annotation_list.append(em)
        motion_pred_list.append(em_pred)

        temp_R = calculate_R_precision(et.cpu().numpy(), em.cpu().numpy(), top_k=3, sum_all=True)
        temp_match = euclidean_distance_matrix(et.cpu().numpy(), em.cpu().numpy()).trace()
        R_precision_real += temp_R
        matching_score_real += temp_match
        temp_R = calculate_R_precision(et_pred.cpu().numpy(), em_pred.cpu().numpy(), top_k=3, sum_all=True)
        temp_match = euclidean_distance_matrix(et_pred.cpu().numpy(), em_pred.cpu().numpy()).trace()
        R_precision += temp_R
        matching_score_pred += temp_match

        nb_sample += bs

    motion_annotation_np = torch.cat(motion_annotation_list, dim=0).cpu().numpy()
    motion_pred_np = torch.cat(motion_pred_list, dim=0).cpu().numpy()
    gt_mu, gt_cov = calculate_activation_statistics(motion_annotation_np)
    mu, cov = calculate_activation_statistics(motion_pred_np)

    diversity_real = calculate_diversity(motion_annotation_np, 300 if nb_sample > 300 else 100)
    diversity = calculate_diversity(motion_pred_np, 300 if nb_sample > 300 else 100)

    R_precision_real = R_precision_real / nb_sample
    R_precision = R_precision / nb_sample

    matching_score_real = matching_score_real / nb_sample
    matching_score_pred = matching_score_pred / nb_sample
    l1_dist = l1_dist / num_poses

    fid = calculate_frechet_distance(gt_mu, gt_cov, mu, cov)

    msg = "--> \t Eva. Re %d:, FID. %.4f, Diversity Real. %.4f, Diversity. %.4f, R_precision_real. (%.4f, %.4f, %.4f), R_precision. (%.4f, %.4f, %.4f), matching_real. %.4f, matching_pred. %.4f, mae. %.4f" % \
          (repeat_id, fid, diversity_real, diversity, R_precision_real[0], R_precision_real[1], R_precision_real[2],
           R_precision[0], R_precision[1], R_precision[2], matching_score_real, matching_score_pred, l1_dist)
    # logger.info(msg)
    print(msg)
    return fid, diversity, R_precision, matching_score_pred, l1_dist


@torch.no_grad()
def evaluation_mask_transformer_test(val_loader, vq_model, trans, repeat_id, eval_wrapper,
                                time_steps, cond_scale, temperature, topkr, gsample=True, force_mask=False, cal_mm=True):
    trans.eval()
    vq_model.eval()

    motion_annotation_list = []
    motion_pred_list = []
    motion_multimodality = []
    R_precision_real = 0
    R_precision = 0
    matching_score_real = 0
    matching_score_pred = 0
    multimodality = 0

    nb_sample = 0
    if cal_mm:
        num_mm_batch = 3
    else:
        num_mm_batch = 0

    for i, batch in enumerate(val_loader):
        # print(i)
        word_embeddings, pos_one_hots, clip_text, sent_len, pose, m_length, token = batch
        m_length = m_length.cuda()

        bs, seq = pose.shape[:2]
        # num_joints = 21 if pose.shape[-1] == 251 else 22

        # for i in range(mm_batch)
        if i < num_mm_batch:
        # (b, seqlen, c)
            motion_multimodality_batch = []
            for _ in range(30):
                mids = trans.generate(clip_text, m_length // 4, time_steps, cond_scale,
                                      temperature=temperature, topk_filter_thres=topkr,
                                      gsample=gsample, force_mask=force_mask)

                # motion_codes = motion_codes.permute(0, 2, 1)
                mids.unsqueeze_(-1)
                pred_motions = vq_model.forward_decoder(mids)

                et_pred, em_pred = eval_wrapper.get_co_embeddings(word_embeddings, pos_one_hots, sent_len, pred_motions.clone(),
                                                                  m_length)
                # em_pred = em_pred.unsqueeze(1)  #(bs, 1, d)
                motion_multimodality_batch.append(em_pred.unsqueeze(1))
            motion_multimodality_batch = torch.cat(motion_multimodality_batch, dim=1) #(bs, 30, d)
            motion_multimodality.append(motion_multimodality_batch)
        else:
            mids = trans.generate(clip_text, m_length // 4, time_steps, cond_scale,
                                  temperature=temperature, topk_filter_thres=topkr,
                                  force_mask=force_mask)

            # motion_codes = motion_codes.permute(0, 2, 1)
            mids.unsqueeze_(-1)
            pred_motions = vq_model.forward_decoder(mids)

            et_pred, em_pred = eval_wrapper.get_co_embeddings(word_embeddings, pos_one_hots, sent_len,
                                                              pred_motions.clone(),
                                                              m_length)

        pose = pose.cuda().float()

        et, em = eval_wrapper.get_co_embeddings(word_embeddings, pos_one_hots, sent_len, pose, m_length)
        motion_annotation_list.append(em)
        motion_pred_list.append(em_pred)

        temp_R = calculate_R_precision(et.cpu().numpy(), em.cpu().numpy(), top_k=3, sum_all=True)
        temp_match = euclidean_distance_matrix(et.cpu().numpy(), em.cpu().numpy()).trace()
        R_precision_real += temp_R
        matching_score_real += temp_match
        # print(et_pred.shape, em_pred.shape)
        temp_R = calculate_R_precision(et_pred.cpu().numpy(), em_pred.cpu().numpy(), top_k=3, sum_all=True)
        temp_match = euclidean_distance_matrix(et_pred.cpu().numpy(), em_pred.cpu().numpy()).trace()
        R_precision += temp_R
        matching_score_pred += temp_match

        nb_sample += bs

    motion_annotation_np = torch.cat(motion_annotation_list, dim=0).cpu().numpy()
    motion_pred_np = torch.cat(motion_pred_list, dim=0).cpu().numpy()
    if not force_mask and cal_mm:
        motion_multimodality = torch.cat(motion_multimodality, dim=0).cpu().numpy()
        multimodality = calculate_multimodality(motion_multimodality, 10)
    gt_mu, gt_cov = calculate_activation_statistics(motion_annotation_np)
    mu, cov = calculate_activation_statistics(motion_pred_np)

    diversity_real = calculate_diversity(motion_annotation_np, 300 if nb_sample > 300 else 100)
    diversity = calculate_diversity(motion_pred_np, 300 if nb_sample > 300 else 100)

    R_precision_real = R_precision_real / nb_sample
    R_precision = R_precision / nb_sample

    matching_score_real = matching_score_real / nb_sample
    matching_score_pred = matching_score_pred / nb_sample

    fid = calculate_frechet_distance(gt_mu, gt_cov, mu, cov)

    msg = f"--> \t Eva. Repeat {repeat_id} :, FID. {fid:.4f}, " \
          f"Diversity Real. {diversity_real:.4f}, Diversity. {diversity:.4f}, " \
          f"R_precision_real. {R_precision_real}, R_precision. {R_precision}, " \
          f"matching_score_real. {matching_score_real:.4f}, matching_score_pred. {matching_score_pred:.4f}," \
          f"multimodality. {multimodality:.4f}"
    print(msg)
    return fid, diversity, R_precision, matching_score_pred, multimodality


@torch.no_grad()
def evaluation_mask_transformer_test_plus_res(val_loader, vq_model, res_model, trans, repeat_id, eval_wrapper,
                                time_steps, cond_scale, temperature, topkr, gsample=True, force_mask=False,
                                              cal_mm=True, res_cond_scale=5, device=None, save_motions=False, save_dir=None):
    trans.eval()
    vq_model.eval()
    res_model.eval()

    motion_annotation_list = []
    motion_pred_list = []
    motion_multimodality = []
    R_precision_real = 0
    R_precision = 0
    matching_score_real = 0
    matching_score_pred = 0
    multimodality = 0

    clip_score_real = 0
    clip_score_gt = 0

    abs_mean = np.load('/home/dingkang/BAMM/dataset/HumanML3D/Mean_abs_kjd.npy')
    abs_std = np.load('/home/dingkang/BAMM/dataset/HumanML3D/Std_abs_kjd.npy')

    nb_sample = 0

    edit_task = None # inpainting, outpainting, prefix, suffix
    force_mask = False
    # If true need to add this to trans.generate&res_model.generate "force_mask=force_mask" (dont forget there are 2 places in if else)
    res_cond_scale = 6

    all_saved_results = [] # 存储所有生成的动作
    if force_mask or (not cal_mm):
        num_mm_batch = 0
    else:
        num_mm_batch = 3
    is_res = False
    gt_len = True
    len_estimator = None #'hml3d_estimator'

    if len_estimator == 'hml3d_estimator':
        from models.len_predictor_modules import MotionLenEstimatorBiGRU
        from utils.word_vectorizer import WordVectorizer, POS_enumerator
        unit_length = 4
        dim_word = 300
        dim_pos_ohot = len(POS_enumerator)
        num_classes = 200 // unit_length
        estimator = MotionLenEstimatorBiGRU(dim_word, dim_pos_ohot, 512, num_classes)

        cp = '/home/epinyoan/git/momask-codes/checkpoints/t2m/length_est_bigru/model/latest.tar'
        checkpoints = torch.load(cp, map_location='cpu')
        estimator.load_state_dict(checkpoints['estimator'], strict=True)
        estimator.cuda()
        estimator.eval()
        softmax = torch.nn.Softmax(-1)

    from tqdm import tqdm
    for i, batch in enumerate(tqdm(val_loader)):
        word_embeddings, pos_one_hots, clip_text, sent_len, pose, m_length, token = batch
        m_length = m_length.cuda()
        input_len = m_length


        if len_estimator == 'hml3d_estimator':
            pred_probs = estimator(word_embeddings.cuda().float(), pos_one_hots.cuda().float(), sent_len).detach()
            pred_probs = softmax(pred_probs)
            pred_tok_len = pred_probs.argsort(dim=-1, descending=True)[:, :5][..., 0]
            input_len = pred_tok_len*4

        bs, seq = pose.shape[:2]
        # num_joints = 21 if pose.shape[-1] == 251 else 22
        if edit_task in ['inpainting', 'outpainting']:
            tokens = trans.pad_id * torch.ones((bs, 50), dtype=torch.long).cuda()
            m_token_length = torch.ceil((m_length)/4).int().cpu().numpy()
            m_token_length_init = (m_token_length * .25).astype(int)
            m_length_init = (m_length * .25).int()

            for k in range(bs):
                l = m_length_init[k]
                l_token = m_token_length_init[k]

                if edit_task == 'inpainting':
                    # # start tokens
                    index_motion, _ = vq_model.encode(pose[k:k+1, :l].cuda())
                    index_motion = index_motion[..., 0]
                    tokens[k,:index_motion.shape[1]] = index_motion[0]

                    # # end tokens
                    index_motion, _ = vq_model.encode(pose[k:k+1, m_length[k]-l :m_length[k]].cuda())
                    index_motion = index_motion[..., 0]
                    tokens[k, m_token_length[k]-l_token :m_token_length[k]] = index_motion[0]
                elif edit_task == 'outpainting':
                    # inside tokens
                    index_motion, _ = vq_model.encode(pose[k:k+1, l:m_length[k]-l].cuda())
                    index_motion = index_motion[..., 0]
                    tokens[k, l_token: l_token+index_motion.shape[1]] = index_motion[0]
            tokens = tokens.scatter(-1, torch.from_numpy(m_token_length[..., None]).cuda().long(), trans.end_id)

        if edit_task in ['prefix', 'suffix']:
            tokens = trans.pad_id * torch.ones((bs, 50), dtype=torch.long).cuda()
            m_token_length = torch.ceil((m_length)/4).int().cpu().numpy()
            m_token_length_half = (m_token_length * .5).astype(int)
            m_length_half = (m_length * .5).int()
            for k in range(bs):
                if edit_task == 'prefix':
                    index_motion, _ = vq_model.encode(pose[k:k+1, :m_length_half[k]].cuda())
                    index_motion = index_motion[..., 0]
                    tokens[k, :m_token_length_half[k]] = index_motion[0]
                elif edit_task == 'suffix':
                    index_motion, _ = vq_model.encode(pose[k:k+1, m_length_half[k]:m_length[k]].cuda())
                    index_motion = index_motion[..., 0]
                    tokens[k, m_token_length[k]-m_token_length_half[k] :m_token_length[k]] = index_motion[0]
            tokens = tokens.scatter(-1, torch.from_numpy(m_token_length[..., None]).cuda().long(), trans.end_id)


        # for i in range(mm_batch)
        if i < num_mm_batch:
        # (b, seqlen, c)
            motion_multimodality_batch = []
            batch_clip_score_pred = 0
            for _ in range(30):
                if edit_task is not None:
                    # mids = trans.edit2(clip_text, tokens, force_mask=force_mask) #edit2 for my trans, edit2 without force_mask for original
                    mids = trans.edit2(clip_text, tokens)
                else:
                    mids, _ = trans.generate(clip_text, input_len // 4, time_steps, cond_scale,
                                        temperature=temperature, topk_filter_thres=topkr,
                                        gsample=gsample)
                # q = mids == trans.end_id
                # mids[q] = 0
                mids, pred_len = trans.pad_when_end(mids)
                if gt_len:
                    pred_len = input_len
                else:
                    pred_len = pred_len.clamp(1, 49) * 4


                # motion_codes = motion_codes.permute(0, 2, 1)
                if is_res:
                    pred_ids = res_model.generate(mids, clip_text, pred_len // 4, temperature=1, cond_scale=res_cond_scale)
                else:
                    pred_ids =  mids.unsqueeze_(-1)

                pred_motions = vq_model.forward_decoder(pred_ids) #* pred_motions_mask

                #==============for 66 test================
                pred_motions = val_loader.dataset.inv_transform(pred_motions.detach().cpu().numpy())
                pred_motions = torch.from_numpy(pred_motions).float().to(device)
                pred_motions = recover_from_ric(pred_motions, joints_num=22)#(B, T, 22,3)
                pred_motions = pred_motions.reshape(pred_motions.shape[0], pred_motions.shape[1], -1)#(B, T, 66)
                pred_motions = val_loader.dataset.transform(pred_motions.detach().cpu().numpy(), abs_mean, abs_std)
                #=========================================


                (et_pred, em_pred), (et_pred_clip, em_pred_clip) = eval_wrapper.get_co_embeddings(word_embeddings, pos_one_hots, sent_len, clip_text, torch.from_numpy(pred_motions).float().to(device),
                                                                  pred_len)
                # em_pred = em_pred.unsqueeze(1)  #(bs, 1, d)
                motion_multimodality_batch.append(em_pred.unsqueeze(1))
            motion_multimodality_batch = torch.cat(motion_multimodality_batch, dim=1) #(bs, 30, d)
            motion_multimodality.append(motion_multimodality_batch)
            for j in range(32):
                single_em = em_pred_clip[j]
                single_et = et_pred_clip[j]
                clip_score = (single_em @ single_et.T).item()
                batch_clip_score_pred += clip_score
            clip_score_real += batch_clip_score_pred
        else:
            if edit_task is not None:
                    # mids = trans.edit2(clip_text, tokens, force_mask=force_mask)
                    mids = trans.edit2(clip_text, tokens)
            else:
                custom_text = ['a person jogs forward for several seconds.'] * bs
                clip_text = tuple(custom_text)  # 测试自定义文本
                mids, _ = trans.generate(clip_text, input_len // 4, time_steps, cond_scale,
                                    temperature=temperature, topk_filter_thres=topkr)
            # q = mids == trans.end_id
            # mids[q] = 0
            mids, pred_len = trans.pad_when_end(mids)
            if gt_len:
                pred_len = input_len
            else:
                pred_len = pred_len.clamp(1, 49) * 4

                
            # motion_codes = motion_codes.permute(0, 2, 1)
            if is_res:
                pred_ids = res_model.generate(mids, clip_text, pred_len // 4, temperature=1, cond_scale=res_cond_scale)
            else:
                pred_ids =  mids.unsqueeze_(-1)

            #pred_motions_mask = lengths_to_mask(pred_len, 196).unsqueeze(-1) #(b, n, 1)
            pred_motions = vq_model.forward_decoder(pred_ids) #* pred_motions_mask

            #==============for 66 test================
            pred_motions = val_loader.dataset.inv_transform(pred_motions.detach().cpu().numpy())
            pred_motions = torch.from_numpy(pred_motions).float().to(device)
            pred_motions = recover_from_ric(pred_motions, joints_num=22)#(B, T, 22,3)
            pred_motions = pred_motions.reshape(pred_motions.shape[0], pred_motions.shape[1], -1)#(B, T, 66)
            pred_motions = val_loader.dataset.transform(pred_motions.detach().cpu().numpy(), abs_mean, abs_std)
            #=========================================

            # 如果需要保存动作数据
            if save_motions:
                pred_motions = val_loader.dataset.inv_transform(pred_motions, abs_mean, abs_std)
                pred_motions = torch.from_numpy(pred_motions).float().to(device)
                # m_length = [196] * bs  # 假设所有生成的动作长度都是196
                bs = len(clip_text)
                for j in range(1):#bs
                    current_motion = pred_motions[j]
                    current_length = 144 #m_length[j].item()
                    current_text = clip_text[j]
                    
                    # 截取到实际长度并重塑
                    actual_motion = current_motion[:current_length]
                    joints_data = actual_motion.reshape(current_length, 22, 3)
                    
                    res = {
                        'joints': joints_data.detach().cpu().numpy(),
                        'text': current_text,
                        'length': current_length,
                        'hint': None
                    }
                    
                    all_saved_results.append(res)
            # 保存所有生成的动作
            if save_motions and save_dir:
                import pickle
                os.makedirs(save_dir, exist_ok=True)
                for idx, res in enumerate(all_saved_results):
                    save_path = os.path.join(save_dir, f"app_vs_34_BAMM.pkl")
                    with open(save_path, 'wb') as f:
                        pickle.dump(res, f)
                print(f"Saved {len(all_saved_results)} motions to {save_dir}")
            import pdb; pdb.set_trace()
            #=========================================
            (et_pred, em_pred), (et_pred_clip, em_pred_clip) = eval_wrapper.get_co_embeddings(word_embeddings, pos_one_hots, sent_len, clip_text,
                                                              torch.from_numpy(pred_motions).float().to(device),
                                                              pred_len)
            batch_clip_score_pred = 0
            for j in range(32):
                single_em = em_pred_clip[j]
                single_et = et_pred_clip[j]
                clip_score = (single_em @ single_et.T).item()
                batch_clip_score_pred += clip_score
            clip_score_real += batch_clip_score_pred
        
        pose = pose.cuda().float()
        #==============for 66 test================
        pose = val_loader.dataset.inv_transform(pose.detach().cpu().numpy())
        pose = torch.from_numpy(pose).float().to(device)
        pose = recover_from_ric(pose, joints_num=22)#(B, T, 22,3)
        pose = pose.reshape(pose.shape[0], pose.shape[1], -1)#(B, T, 66)
        pose = val_loader.dataset.transform(pose.detach().cpu().numpy(), abs_mean, abs_std)
        pose = torch.from_numpy(pose).float().to(device)
        #=========================================


        (et, em), (et_clip, em_clip) = eval_wrapper.get_co_embeddings(word_embeddings, pos_one_hots, sent_len, clip_text, pose, input_len)
        batch_clip_score = 0
        for j in range(32):
            single_em = em_clip[j]
            single_et = et_clip[j]
            clip_score = (single_em @ single_et.T).item()
            batch_clip_score += clip_score
        clip_score_gt += batch_clip_score
        
        motion_annotation_list.append(em)
        motion_pred_list.append(em_pred)

        temp_R = calculate_R_precision(et.cpu().numpy(), em.cpu().numpy(), top_k=3, sum_all=True)
        temp_match = euclidean_distance_matrix(et.cpu().numpy(), em.cpu().numpy()).trace()
        R_precision_real += temp_R
        matching_score_real += temp_match
        # print(et_pred.shape, em_pred.shape)
        temp_R = calculate_R_precision(et_pred.cpu().numpy(), em_pred.cpu().numpy(), top_k=3, sum_all=True)
        temp_match = euclidean_distance_matrix(et_pred.cpu().numpy(), em_pred.cpu().numpy()).trace()
        R_precision += temp_R
        matching_score_pred += temp_match

        nb_sample += bs

    motion_annotation_np = torch.cat(motion_annotation_list, dim=0).cpu().numpy()
    motion_pred_np = torch.cat(motion_pred_list, dim=0).cpu().numpy()
    if not force_mask and cal_mm:
        motion_multimodality = torch.cat(motion_multimodality, dim=0).cpu().numpy()
        multimodality = calculate_multimodality(motion_multimodality, 10)
    gt_mu, gt_cov = calculate_activation_statistics(motion_annotation_np)
    mu, cov = calculate_activation_statistics(motion_pred_np)

    diversity_real = calculate_diversity(motion_annotation_np, 300 if nb_sample > 300 else 100)
    diversity = calculate_diversity(motion_pred_np, 300 if nb_sample > 300 else 100)

    R_precision_real = R_precision_real / nb_sample
    R_precision = R_precision / nb_sample

    clip_score_real = clip_score_real / nb_sample
    clip_score_gt = clip_score_gt / nb_sample

    matching_score_real = matching_score_real / nb_sample
    matching_score_pred = matching_score_pred / nb_sample

    fid = calculate_frechet_distance(gt_mu, gt_cov, mu, cov)

    msg = f"--> \t Eva. Repeat {repeat_id} :, FID. {fid:.4f}, " \
          f"Diversity Real. {diversity_real:.4f}, Diversity. {diversity:.4f}, " \
          f"R_precision_real. {R_precision_real}, R_precision. {R_precision}, " \
          f"matching_score_real. {matching_score_real:.4f}, matching_score_pred. {matching_score_pred:.4f}," \
          f"clip_score_real. {clip_score_real:.4f}, clip_score_gt. {clip_score_gt:.4f}," \
          f"multimodality. {multimodality:.4f}"
    print(msg)
    return fid, diversity, R_precision, matching_score_pred, clip_score_real, multimodality, msg

@torch.no_grad()
def evaluation_abs_transformer_test(val_loader, ae_model, trans, repeat_id, eval_wrapper,
                                time_steps, cond_scale, temperature, device, gsample=True, force_mask=False,
                                              cal_mm=True, save_motions=False, save_dir=None):
    trans.eval()
    ae_model.eval()

    motion_annotation_list = []
    motion_pred_list = []
    motion_multimodality = []
    R_precision_real = 0
    R_precision = 0
    matching_score_real = 0
    matching_score_pred = 0
    multimodality = 0

    clip_score_real = 0
    clip_score_gt = 0

    nb_sample = 0
    edit_task = None  # inpainting, outpainting, prefix, suffix, upper_body
    force_mask = False

    all_saved_results = []  # 存储所有生成的动作

    if force_mask or (not cal_mm):
        num_mm_batch = 0
    else:
        num_mm_batch = 3

    from tqdm import tqdm
    for i, batch in enumerate(tqdm(val_loader)):
        word_embeddings, pos_one_hots, clip_text, sent_len, pose, m_length, token = batch
        m_length = m_length.cuda()
        pose = pose.cuda().float()
        # print("clip_text:",clip_text) #tuple[str, ...]

        with torch.no_grad():
            latents = ae_model.encode(pose)

        bs, seq = pose.shape[:2]

        latent_seq_len = latents.shape[-1]  # 这通常是原序列长度的1/4
        edit_mask = None
        
        if edit_task in ['inpainting', 'outpainting', 'prefix', 'suffix']:
            # 创建编辑掩码，形状为 (bs, latent_seq_len)，注意这里使用latent空间的序列长度
            edit_mask = torch.zeros(bs, latent_seq_len, dtype=torch.bool, device=device)
            
            if edit_task in ['inpainting', 'outpainting']:
                # inpainting和outpainting使用25%的长度
                preserve_length = (m_length.float() * 0.25).long()
                
                for k in range(bs):
                    l = preserve_length[k].item()
                    actual_length = min(l, m_length[k].item())
                    
                    # 将原始长度转换为latent空间长度
                    latent_actual_length = actual_length // 4
                    latent_m_length = m_length[k].item() // 4
                    
                    # 确保索引在有效范围内
                    latent_actual_length = min(latent_actual_length, latent_seq_len)
                    latent_m_length = min(latent_m_length, latent_seq_len)
                    
                    if edit_task == 'inpainting' and latent_actual_length > 0:
                        # inpainting: 编辑中间部分，保留开头和结尾
                        start_preserve = latent_actual_length
                        end_preserve = max(0, latent_m_length - latent_actual_length)
                        if start_preserve < end_preserve:
                            edit_mask[k, start_preserve:end_preserve] = True
                        
                    elif edit_task == 'outpainting':
                        # outpainting: 编辑开头和结尾，保留中间
                        start_idx = latent_actual_length
                        end_idx = max(start_idx, latent_m_length - latent_actual_length)
                        
                        # 编辑开头部分
                        if start_idx > 0:
                            edit_mask[k, :start_idx] = True
                        
                        # 编辑结尾部分
                        if end_idx < latent_m_length:
                            edit_mask[k, end_idx:latent_m_length] = True
                        
            elif edit_task in ['prefix', 'suffix']:
                # prefix和suffix使用50%的长度
                half_length = (m_length.float() * 0.5).long()
                
                for k in range(bs):
                    l_half = half_length[k].item()
                    actual_half = min(l_half, m_length[k].item())
                    
                    # 将原始长度转换为latent空间长度
                    latent_actual_half = actual_half // 4
                    latent_m_length = m_length[k].item() // 4
                    
                    # 确保索引在有效范围内
                    latent_actual_half = min(latent_actual_half, latent_seq_len)
                    latent_m_length = min(latent_m_length, latent_seq_len)
                    
                    if edit_task == 'suffix' and latent_actual_half < latent_m_length:
                        # suffix: 编辑前50%，保留后50%
                        edit_mask[k, :latent_actual_half] = True

                    elif edit_task == 'prefix' and latent_actual_half > 0:
                        # prefix: 编辑后50%，保留前50%
                        edit_mask[k, latent_actual_half:latent_m_length] = True
        elif edit_task == 'upper_body':
            # 构造上半身保留掩码 (B,T,22)
            joint_keep_mask = build_upper_body_keep_mask(m_length, T=pose.shape[1], device=pose.device)

        if i < num_mm_batch:
            motion_multimodality_batch = []
            batch_clip_score_pred = 0
            for _ in tqdm(range(30)):
                if edit_task == 'upper_body':
                    pred_latents = trans.edit_with_joint_mask(
                        clip_text,
                        gt_pose=pose,                       # (B,T,66)
                        m_lens_tok=(m_length // 4),
                        timesteps=time_steps,
                        cond_scale=cond_scale,
                        ae_model=ae_model,
                        joint_keep_mask=joint_keep_mask,   # (B,T,22)
                        temperature=temperature,
                        force_mask=False
                    )
                elif edit_task is not None:
                    pred_latents = trans.edit(
                        clip_text, latents.clone(), m_length // 4,
                        timesteps=time_steps, cond_scale=cond_scale,
                        temperature=temperature, force_mask=False, edit_mask=edit_mask.clone()
                    )
                else:
                    pred_latents = trans.generate(clip_text, m_length//4 , time_steps, cond_scale,
                                                  temperature=temperature, hard_pseudo_reorder=False)

                pred_motions = ae_model.decode(pred_latents)

                (et_pred, em_pred), (et_pred_clip, em_pred_clip) = eval_wrapper.get_co_embeddings(word_embeddings, pos_one_hots, sent_len,
                                                                            clip_text,
                                                                            pred_motions.to(device),
                                                                            m_length)
                motion_multimodality_batch.append(em_pred.unsqueeze(1))
            motion_multimodality_batch = torch.cat(motion_multimodality_batch, dim=1) #(bs, 30, d)
            motion_multimodality.append(motion_multimodality_batch)
            # 绝对坐标
            for j in range(32):
                single_em = em_pred_clip[j]
                single_et = et_pred_clip[j]
                clip_score = (single_em @ single_et.T).item()
                batch_clip_score_pred += clip_score
            clip_score_real += batch_clip_score_pred
        else:
            if edit_task == 'upper_body':
                pred_latents = trans.edit_with_joint_mask(
                    clip_text,
                    gt_pose=pose,
                    m_lens_tok=(m_length // 4),
                    timesteps=time_steps,
                    cond_scale=cond_scale,
                    ae_model=ae_model,
                    joint_keep_mask=joint_keep_mask,
                    temperature=temperature,
                    force_mask=False
                )
            elif edit_task is not None:
                pred_latents = trans.edit(
                    clip_text, latents.clone(), m_length // 4,
                    timesteps=time_steps, cond_scale=cond_scale,
                    temperature=temperature, force_mask=False, edit_mask=edit_mask.clone()
                )
            else:
                pred_latents = trans.generate(clip_text, m_length // 4, time_steps, cond_scale,
                                              temperature=temperature, hard_pseudo_reorder=False)

            pred_motions = ae_model.decode(pred_latents)

            # 如果需要保存动作数据
            if save_motions:
                pred_motions = val_loader.dataset.inv_transform(pred_motions.detach().cpu().numpy())
                pred_motions = torch.from_numpy(pred_motions).float().to(device)
                bs = len(clip_text)
                for j in range(1):#bs
                    current_motion = pred_motions[j]
                    current_length = 121 #实际长度
                    current_text = clip_text[j]

                    actual_motion = current_motion[:current_length]
                    joints_data = actual_motion.reshape(current_length, 22, 3)
                    
                    res = {
                        'joints': joints_data.detach().cpu().numpy(),
                        'text': current_text,
                        'length': current_length,
                        'hint': None
                    }
                    
                    all_saved_results.append(res)
            # 保存所有生成的动作
            if save_motions and save_dir:
                import pickle
                os.makedirs(save_dir, exist_ok=True)
                for idx, res in enumerate(all_saved_results):
                    save_path = os.path.join(save_dir, f"your_{idx}.pkl") 
                    with open(save_path, 'wb') as f:
                        pickle.dump(res, f)
                print(f"Saved {len(all_saved_results)} motions to {save_dir}")

            (et_pred, em_pred), (et_pred_clip, em_pred_clip) = eval_wrapper.get_co_embeddings(word_embeddings,
                                                                              pos_one_hots, sent_len,
                                                                              clip_text,
                                                                              pred_motions.to(device),
                                                                              m_length)
            batch_clip_score_pred = 0
            for j in range(32):
                single_em = em_pred_clip[j]
                single_et = et_pred_clip[j]
                clip_score = (single_em @ single_et.T).item()
                batch_clip_score_pred += clip_score
            clip_score_real += batch_clip_score_pred

        pose = pose.cuda().float()
        (et, em), (et_clip, em_clip) = eval_wrapper.get_co_embeddings(word_embeddings, pos_one_hots, sent_len, clip_text,
                                                          pose.clone(), m_length)
        batch_clip_score = 0
        for j in range(32):
            single_em = em_clip[j]
            single_et = et_clip[j]
            clip_score = (single_em @ single_et.T).item()
            batch_clip_score += clip_score
        clip_score_gt += batch_clip_score
        motion_annotation_list.append(em)
        motion_pred_list.append(em_pred)

        temp_R = calculate_R_precision(et.cpu().numpy(), em.cpu().numpy(), top_k=3, sum_all=True)
        temp_match = euclidean_distance_matrix(et.cpu().numpy(), em.cpu().numpy()).trace()
        R_precision_real += temp_R
        matching_score_real += temp_match
        # print(et_pred.shape, em_pred.shape)
        temp_R = calculate_R_precision(et_pred.cpu().numpy(), em_pred.cpu().numpy(), top_k=3, sum_all=True)
        temp_match = euclidean_distance_matrix(et_pred.cpu().numpy(), em_pred.cpu().numpy()).trace()
        R_precision += temp_R
        matching_score_pred += temp_match

        nb_sample += bs

    motion_annotation_np = torch.cat(motion_annotation_list, dim=0).cpu().numpy()
    motion_pred_np = torch.cat(motion_pred_list, dim=0).cpu().numpy()
    if not force_mask and cal_mm:
        motion_multimodality = torch.cat(motion_multimodality, dim=0).cpu().numpy()
        multimodality = calculate_multimodality(motion_multimodality, 10)
    gt_mu, gt_cov = calculate_activation_statistics(motion_annotation_np)
    mu, cov = calculate_activation_statistics(motion_pred_np)

    diversity_real = calculate_diversity(motion_annotation_np, 300 if nb_sample > 300 else 100)
    diversity = calculate_diversity(motion_pred_np, 300 if nb_sample > 300 else 100)

    R_precision_real = R_precision_real / nb_sample
    R_precision = R_precision / nb_sample

    clip_score_real = clip_score_real / nb_sample
    clip_score_gt = clip_score_gt / nb_sample

    matching_score_real = matching_score_real / nb_sample
    matching_score_pred = matching_score_pred / nb_sample

    fid = calculate_frechet_distance(gt_mu, gt_cov, mu, cov)

    msg = f"--> \t Eva. Repeat {repeat_id} :, FID. {fid:.4f}, " \
          f"Diversity Real. {diversity_real:.4f}, Diversity. {diversity:.4f}, " \
          f"R_precision_real. {R_precision_real}, R_precision. {R_precision}, " \
          f"matching_score_real. {matching_score_real:.4f}, matching_score_pred. {matching_score_pred:.4f}," \
          f"multimodality. {multimodality:.4f}," \
          f"clip_score_real. {clip_score_real:.4f}, clip_score_gt. {clip_score_gt:.4f}"
    print(msg)
    return fid, diversity, R_precision, matching_score_pred, clip_score_real, multimodality, msg


