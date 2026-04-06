import torch
from collections import defaultdict
import torch.optim as optim
# import tensorflow as tf
from torch.utils.tensorboard import SummaryWriter
from collections import OrderedDict
from utils.utils import *
from os.path import join as pjoin
from utils.eval_t2m import evaluation_my_transformer, evaluation_res_transformer
from models.mask_transformer.tools import *

from einops import rearrange, repeat

def def_value():
    return 0.0

class MaskTransformerTrainer:
    def __init__(self, args, t2m_transformer, AE_model, vq_model):
        self.opt = args
        self.t2m_transformer = t2m_transformer
        self.AE_model = AE_model
        self.vq_model = vq_model
        self.device = args.device
        self.AE_model.eval()
        self.vq_model.eval()
        if args.is_train:
            self.logger = SummaryWriter(args.log_dir)


    def update_lr_warm_up(self, nb_iter, warm_up_iter, lr):

        current_lr = lr * (nb_iter + 1) / (warm_up_iter + 1)
        for param_group in self.opt_t2m_transformer.param_groups:
            param_group["lr"] = current_lr

        return current_lr


    def forward(self, batch_data):

        conds, motion, m_lens = batch_data # (B,T,22,3)
        motion = motion.detach().float().to(self.device)
        m_lens = m_lens.detach().long().to(self.device)

        # (b, t, d) - 连续的潜在表示
        latent = self.AE_model.encode(motion) # (B, 4, T//4, 22)
        latent = latent.permute(0, 2, 1)  # (b, t, 512)
        vq_latent, _ = self.vq_model.encode(motion) # (B, , T//4, 量化器数量：4)=(B, 49, 4) # 每个时间步对应4个量化器的索引
        # print('vq_latent:', vq_latent)
        
        m_lens = m_lens // 4

        conds = conds.to(self.device).float() if torch.is_tensor(conds) else conds

        _loss = self.t2m_transformer(latent, vq_latent[..., 0], conds, m_lens)  # 只使用第一个码本

        return _loss

    def update(self, batch_data, it):
        loss = self.forward(batch_data)
        self.opt_t2m_transformer.zero_grad()
        loss.backward()
        self.opt_t2m_transformer.step()
        if it >= self.opt.warm_up_iter:
            self.scheduler.step() # 每次batch迭代后调用scheduler
        return loss.item()

    def save(self, file_name, ep, total_it):
        t2m_trans_state_dict = self.t2m_transformer.state_dict()
        clip_weights = [e for e in t2m_trans_state_dict.keys() if e.startswith('clip_model.')]
        for e in clip_weights:
            del t2m_trans_state_dict[e]
        state = {
            't2m_transformer': t2m_trans_state_dict,
            'opt_t2m_transformer': self.opt_t2m_transformer.state_dict(),
            'scheduler':self.scheduler.state_dict(),
            'ep': ep,
            'total_it': total_it,
        }
        torch.save(state, file_name)

    def resume(self, model_dir):
        checkpoint = torch.load(model_dir, map_location=self.device)
        missing_keys, unexpected_keys = self.t2m_transformer.load_state_dict(checkpoint['t2m_transformer'], strict=False)
        assert len(unexpected_keys) == 0
        assert all([k.startswith('clip_model.') for k in missing_keys])

        try:
            self.opt_t2m_transformer.load_state_dict(checkpoint['opt_t2m_transformer']) # Optimizer

            self.scheduler.load_state_dict(checkpoint['scheduler']) # Scheduler
        except:
            print('Resume wo optimizer')
        return checkpoint['ep'], checkpoint['total_it']

    def train(self, train_loader, val_loader, eval_val_loader, eval_wrapper, plot_eval):
        self.t2m_transformer.to(self.device)

        self.AE_model.to(self.device)
        self.vq_model.to(self.device)

        self.opt_t2m_transformer = optim.AdamW(self.t2m_transformer.parameters(), betas=(0.9, 0.99), lr=self.opt.lr, weight_decay=1e-5)

        total_iters = self.opt.max_epoch * len(train_loader)
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
                                                                self.opt_t2m_transformer,
                                                                T_max=total_iters - self.opt.warm_up_iter,  # 余弦退火的最大周期，通常设为总epoch数
                                                                eta_min=1e-6  # 最小学习率
                                                            )

        epoch = 0
        it = 0

        if self.opt.is_continue:
            model_dir = pjoin('/path/to/model', 'latest.tar')  # TODO
            epoch, it = self.resume(model_dir)
            print("Load model epoch:%d iterations:%d"%(epoch, it))

        start_time = time.time()
        # total_iters = self.opt.max_epoch * len(train_loader)
        print(f'Total Epochs: {self.opt.max_epoch}, Total Iters: {total_iters}')
        print('Iters Per Epoch, Training: %04d, Validation: %03d' % (len(train_loader), len(val_loader)))
        logs = defaultdict(def_value, OrderedDict())

        best_fid=1000 
        best_iter=0 
        best_div=100 
        best_top1=0 
        best_top2=0 
        best_top3=0 
        best_matching=100
        clip_score = -1
        best_acc = 0.

        while epoch < self.opt.max_epoch:
            self.t2m_transformer.train()
            self.AE_model.eval()
            self.vq_model.eval()

            for i, batch in enumerate(train_loader):
                it += 1
                if it < self.opt.warm_up_iter:
                    self.update_lr_warm_up(it, self.opt.warm_up_iter, self.opt.lr)

                loss = self.update(batch_data=batch, it=it)
                logs['loss'] += loss
                logs['lr'] += self.opt_t2m_transformer.param_groups[0]['lr']

                if it % self.opt.log_every == 0:
                    mean_loss = OrderedDict()
                    for tag, value in logs.items():
                        self.logger.add_scalar('Train/%s'%tag, value / self.opt.log_every, it)
                        mean_loss[tag] = value / self.opt.log_every
                    logs = defaultdict(def_value, OrderedDict())
                    print_current_loss(start_time, it, total_iters, mean_loss, epoch=epoch, inner_iter=i)

                if it % self.opt.save_latest == 0:
                    self.save(pjoin(self.opt.model_dir, 'latest.tar'), epoch, it)
            self.save(pjoin(self.opt.model_dir, 'latest.tar'), epoch, it)
            epoch += 1

            print('Validation time:')
            self.AE_model.eval()
            self.vq_model.eval()
            self.t2m_transformer.eval()

            val_loss = []
            with torch.no_grad():
                for i, batch_data in enumerate(val_loader):
                    loss = self.forward(batch_data)
                    val_loss.append(loss.item())

            print(f"Validation loss:{np.mean(val_loss):.4f}")

            self.logger.add_scalar('Val/loss', np.mean(val_loss), epoch)

            if epoch % 500 == 0 or epoch == self.opt.max_epoch:
                best_fid, best_div, best_top1, best_top2, best_top3, best_matching, _, clip_score, writer, _ = evaluation_abs_transformer(
                    self.opt.save_root, eval_val_loader, self.t2m_transformer, self.AE_model, self.logger, epoch, best_fid=best_fid,
                    clip_score_old=clip_score, best_div=best_div, best_top1=best_top1, best_top2=best_top2, best_top3=best_top3,
                    best_matching=best_matching, eval_wrapper=eval_wrapper, device=self.device, save_ckpt=True
                )
