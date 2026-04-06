import torch
import torch.nn as nn
import numpy as np
# from networks.layers import *
import torch.nn.functional as F
import clip
from einops import rearrange, repeat
import math
from random import random
from tqdm.auto import tqdm
from typing import Callable, Optional, List, Dict
from copy import deepcopy
from functools import partial
from models.mask_transformer.tools import *
from torch.distributions.categorical import Categorical
from models.mask_transformer.transformer_block import TransformerEncoderLayer, TransformerDecoderLayer, TransformerEncoder
from models.mask_transformer.tools import lengths_to_mask
from models.DiffMLPs import DiffMLPs_models

from models.causal_crossattn import CausalCrossAttnTransformer, CausalCrossAttnTransformerBlock, StrictDualCausalTransformer
class InputProcess(nn.Module):
    def __init__(self, input_feats, latent_dim):
        super().__init__()
        self.input_feats = input_feats
        self.latent_dim = latent_dim
        self.poseEmbedding = nn.Linear(self.input_feats, self.latent_dim)

    def forward(self, x):
        # [bs, ntokens, input_feats]
        x = x.permute((1, 0, 2)) # [seqen, bs, input_feats]
        # print(x.shape)
        x = self.poseEmbedding(x)  # [seqlen, bs, d]
        return x

class PositionalEncoding(nn.Module):
    #Borrow from MDM, the same as above, but add dropout, exponential may improve precision
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1) #[max_len, 1, d_model]

        self.register_buffer('pe', pe)

    def forward(self, x):
        # not used in the final model
        x = x + self.pe[:x.shape[0], :]
        return self.dropout(x)

class OutputProcess_Bert(nn.Module):
    def __init__(self, out_feats, latent_dim):
        super().__init__()
        self.dense = nn.Linear(latent_dim, latent_dim)
        self.transform_act_fn = F.gelu
        self.LayerNorm = nn.LayerNorm(latent_dim, eps=1e-12)
        self.poseFinal = nn.Linear(latent_dim, out_feats) #Bias!
        self.out_feats = out_feats

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        hidden_states = self.dense(hidden_states)
        hidden_states = self.transform_act_fn(hidden_states)
        hidden_states = self.LayerNorm(hidden_states)
        output = self.poseFinal(hidden_states)  # [seqlen, bs, out_feats]
        output = output.permute(1, 2, 0)  # [bs, c, seqlen]
        return output

class OutputProcess(nn.Module):
    def __init__(self, out_feats, latent_dim):
        super().__init__()
        self.dense = nn.Linear(latent_dim, latent_dim)
        self.transform_act_fn = F.gelu
        self.LayerNorm = nn.LayerNorm(latent_dim, eps=1e-12)
        self.poseFinal = nn.Linear(latent_dim, out_feats) #Bias!

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        hidden_states = self.dense(hidden_states)
        hidden_states = self.transform_act_fn(hidden_states)
        hidden_states = self.LayerNorm(hidden_states)
        output = self.poseFinal(hidden_states)  # [seqlen, bs, out_feats]
        output = output.permute(1, 2, 0)  # [bs, e, seqlen]
        return output

class OutputProcessContinuous(nn.Module):
    def __init__(self, model_latent_dim, target_latent_dim):
        super().__init__()
        self.dense = nn.Linear(model_latent_dim, model_latent_dim)
        self.transform_act_fn = F.gelu
        self.LayerNorm = nn.LayerNorm(model_latent_dim, eps=1e-12)
        self.poseFinal = nn.Linear(model_latent_dim, target_latent_dim)  # 输出目标维度

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        # hidden_states: (seqlen, bs, model_latent_dim)
        hidden_states = self.dense(hidden_states)
        hidden_states = self.transform_act_fn(hidden_states)
        hidden_states = self.LayerNorm(hidden_states)
        output = self.poseFinal(hidden_states)  # (seqlen, bs, target_latent_dim)
        
        # 调整维度顺序：(seqlen, bs, target_latent_dim) -> (bs, target_latent_dim, seqlen)
        output = output.permute(1, 2, 0)
        return output

class MaskTransformer(nn.Module):
    def __init__(self, ae_dim, cond_mode, latent_dim=256, ff_size=1024, num_layers=8,
                 num_heads=4, dropout=0.1, clip_dim=512, cond_drop_prob=0.1,
                 clip_version=None, opt=None, **kargs):
        super().__init__()
        print(f'latent_dim: {latent_dim}, ff_size: {ff_size}, nlayers: {num_layers}, nheads: {num_heads}, dropout: {dropout}')

        self.ae_dim = ae_dim # 88
        self.latent_dim = latent_dim #1024
        self.clip_dim = clip_dim #4096
        self.dropout = dropout
        self.opt = opt

        self.cond_mode = cond_mode
        self.cond_drop_prob = cond_drop_prob
        
        # Motion Token 控制参数
        self.motion_token_dropout_prob = 0.7  # 70%概率完全丢弃motion token, 1.01 for ablate study
        self.motion_token_vocab_size = 1024     # VQ词汇表大小

        if self.cond_mode == 'action':
            assert 'num_actions' in kargs
        self.num_actions = kargs.get('num_actions', 1)

        '''
        Preparing Networks
        '''
        self.input_process = InputProcess(self.ae_dim, self.latent_dim)
        self.position_enc = PositionalEncoding(self.latent_dim, self.dropout)

        _num_tokens = 1024 + 3  # two dummy tokens, one for masking, one for padding
        self.token_emb = nn.Embedding(_num_tokens, self.latent_dim)

        # 每个码本一个可学习偏置
        self.max_codebooks = getattr(self.opt, 'max_codebooks', 4)
        self.codebook_type_emb = nn.Embedding(self.max_codebooks, self.latent_dim)

        self.use_dual_causal = True  # 是否启用双重因果掩码

        if self.opt.trans == 'cross_attn':
            if self.use_dual_causal:
                self.seqTransEncoder = StrictDualCausalTransformer(
                    num_layers=num_layers,
                    latent_dim=self.latent_dim,
                    cond_dim=self.latent_dim,
                    n_head=num_heads,
                    ff_size=ff_size,
                    dropout=dropout
                )
            else:
                self.seqTransEncoder = CausalCrossAttnTransformer(
                    num_layers=num_layers,
                    latent_dim=self.latent_dim,
                    cond_dim=self.latent_dim,
                    n_head=num_heads,
                    ff_size=ff_size,
                    dropout=dropout
                )
        elif self.opt.trans == 'official':
            seqTransEncoderLayer = nn.TransformerEncoderLayer(d_model=self.latent_dim,
                                                            nhead=num_heads,
                                                            dim_feedforward=ff_size,
                                                            dropout=dropout,
                                                            activation='gelu')

            self.seqTransEncoder = nn.TransformerEncoder(seqTransEncoderLayer,
                                                        num_layers=num_layers)
        elif self.opt.trans == 't2mgpt':
            self.seqTransEncoder = TransformerEncoder(nn.Sequential(*[TransformerEncoderLayer(embed_dim=self.latent_dim, 
                                            n_head=num_heads, 
                                            drop_out_rate=dropout, 
                                            dim_feedforward=ff_size) for _ in range(num_layers)]))
        else:
            raise Exception('Type of opt.trans '+self.opt.trans+' is not supported')
        
        self.encode_action = partial(F.one_hot, num_classes=self.num_actions)

        # if self.cond_mode != 'no_cond':
        if self.cond_mode == 'text':
            self.cond_emb = nn.Linear(self.clip_dim, self.latent_dim)
        elif self.cond_mode == 'action':
            self.cond_emb = nn.Linear(self.num_actions, self.latent_dim)
        elif self.cond_mode == 'uncond':
            self.cond_emb = nn.Identity()
        else:
            raise KeyError("Unsupported condition mode!!!")


        self.mask_latent = nn.Parameter(torch.zeros(1, 1, self.ae_dim))

        self.apply(self.__init_weights)

        '''
        Preparing frozen weights
        '''

        if self.cond_mode == 'text':
            print('Loading CLIP...')
            self.clip_version = clip_version
            self.clip_model = self.load_and_freeze_clip(clip_version)

        self.noise_schedule = cosine_schedule

        # --------------------------------------------------------------------------
        # DiffMLPs
        print('Loading DiffMLPs...')
        self.diffmlps_model='SiT-XL'
        self.DiffMLPs = DiffMLPs_models[self.diffmlps_model](target_channels=self.ae_dim, z_channels=self.latent_dim)
        self.diffmlps_batch_mul = 4

    def load_and_freeze_token_emb(self, codebook):
        '''
        :param codebook: (c, d)
        :return:
        '''
        assert self.training, 'Only necessary in training mode'
        c, d = codebook.shape
        self.token_emb.weight = nn.Parameter(torch.cat([codebook, torch.zeros(size=(2, d), device=codebook.device)], dim=0)) #add two dummy tokens, 0 vectors
        self.token_emb.requires_grad_(False)

        print("Token embedding initialized!")

    def __init_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            module.weight.data.normal_(mean=0.0, std=0.02)
            if isinstance(module, nn.Linear) and module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.LayerNorm):
            if hasattr(module, 'bias') and module.bias is not None:
                module.bias.data.zero_()
            if hasattr(module, 'weight') and module.weight is not None:
                module.weight.data.fill_(1.0)

    def parameters_wo_clip(self):
        return [p for name, p in self.named_parameters() if not name.startswith('clip_model.')]

    def load_and_freeze_clip(self, clip_version):
        clip_model, clip_preprocess = clip.load(clip_version, device='cpu',
                                                jit=False)  # Must set jit=False for training
        # Cannot run on cpu
        clip.model.convert_weights(
            clip_model)  # Actually this line is unnecessary since clip by default already on float16
        # Date 0707: It's necessary, only unecessary when load directly to gpu. Disable if need to run on cpu

        # Freeze CLIP weights
        clip_model.eval()
        for p in clip_model.parameters():
            p.requires_grad = False

        return clip_model

    def encode_text(self, raw_text):
        device = next(self.parameters()).device
        text = clip.tokenize(raw_text, truncate=True).to(device)
        feat_clip_text = self.clip_model.encode_text(text).float()
        return feat_clip_text

    def mask_cond(self, cond, force_mask=False):
        bs, d =  cond.shape
        if force_mask:
            return torch.zeros_like(cond)
        elif self.training and self.cond_drop_prob > 0.:
            mask = torch.bernoulli(torch.ones(bs, device=cond.device) * self.cond_drop_prob).view(bs, 1)
            return cond * (1. - mask)
        else:
            return cond

    def _process_motion_tokens(self, motion_ids, m_lens):
        """
        处理motion token：应用dropout（训练时）
        :param motion_ids: (b, l) 或 (b, l, q) 的离散token索引
        :param m_lens: (b,) 每个序列的有效长度
        :return: 处理后的motion token或None
        """
        if not self.training:
            return motion_ids  # 推理时不做处理
        
        device = motion_ids.device
        if motion_ids.dim() == 2:
            b, seq_len = motion_ids.shape
        elif motion_ids.dim() == 3:
            b, seq_len, _ = motion_ids.shape
        else:
            raise ValueError(f"Unexpected motion_ids shape: {motion_ids.shape}")
        
        #  Motion Token Dropout (70%概率完全丢弃)
        if random() < self.motion_token_dropout_prob:
            return None
        return motion_ids

    def _compute_cond_positions_mask(self, mask, m_lens):
        """
        计算条件位置掩码，确定latent序列中哪些位置作为条件位置

        :param mask: (b, l) 当前训练步的掩码位置，True表示被掩码的位置
        :param m_lens: (b,) 每个序列的有效长度
        :return: (b, l) 布尔张量，True表示条件位置（未被掩码的已知位置）
        """
        if not self.use_dual_causal or not self.training:
            return None
        
        batch_size, seq_len = mask.shape
        
        # 条件位置 = 有效位置 & 未被掩码的位置
        valid_mask = lengths_to_mask(m_lens, seq_len)  # (b, l) True表示有效位置
        cond_positions_mask = valid_mask & (~mask)     # (b, l) True表示条件位置
        
        return cond_positions_mask

    def trans_forward(self, latents, motion_ids, cond, padding_mask, force_mask=False, m_lens=None, current_mask=None):
        '''
        :param latents: (b, l, d)
        :param motion_ids: (b, l) 或 (b, l, q)，离散token（每步 Q 个码本）
        :param cond: (b, cond_dim)
        :param padding_mask: (b, l) => True for padding positions
        :param current_mask: (b, l) => True for masked positions (用于双重因果掩码)
        '''
        cond = self.mask_cond(cond, force_mask=force_mask)         # (b, cond_dim)
        cond_emb = self.cond_emb(cond)                             # (b, latent_dim)
        
        x = self.input_process(latents)                            # (l, b, latent_dim)
        x = self.position_enc(x)                                   # (l, b, latent_dim)
        x = x.transpose(0,1)# (b, l, latent_dim)
        
        # 构造cond侧：将文本cond压成一个step，与motion token序列拼接
        cond_seq = cond_emb.unsqueeze(1)                           # (b, 1, latent_dim) 作为[TXT] cond
        cond_pad_mask = torch.zeros(cond_seq.size(0), 1, dtype=torch.bool, device=cond_seq.device)  # (b,1)

        if motion_ids is not None:
            motion_ids = self._process_motion_tokens(motion_ids, m_lens)
            if motion_ids is not None:
                # motion token -> embedding
                if motion_ids.dim() == 2:
                    # (b, l)
                    mt = self.token_emb(motion_ids)                    # (b, l, d)
                elif motion_ids.dim() == 3:
                    # (b, l, q)
                    bsz, lc, q = motion_ids.shape
                    mt = self.token_emb(motion_ids)                    # (b, l, q, d)
                    # per-codebook learnable bias
                    if q <= self.codebook_type_emb.num_embeddings:
                        cb_idx = torch.arange(q, device=motion_ids.device)
                        cb_bias = self.codebook_type_emb(cb_idx)       # (q, d)
                    else:
                        w = self.codebook_type_emb.weight              # (max_q, d)
                        rep = math.ceil(q / w.size(0))
                        cb_bias = w.repeat(rep, 1)[:q]                 # (q, d)
                    cb_bias = cb_bias.view(1, 1, q, -1)                # (1,1,q,d)
                    mt = mt + cb_bias                                  # (b, l, q, d)
                    mt = mt.mean(dim=2)                                # (b, l, d)
                else:
                    raise ValueError(f"Unexpected motion_ids shape: {motion_ids.shape}")

                mt = mt.transpose(0, 1)                                # (lc, b, d)
                mt = self.position_enc(mt)                             # (lc, b, d)
                mt = mt.transpose(0, 1)                                # (b, lc, d)
                # 拼接 [TXT] + motion_tokens
                cond_seq = torch.cat([cond_seq, mt], dim=1)            # (b, 1+lc, d)

                lc = motion_ids.size(1)
                processed_lens = torch.min(m_lens, torch.tensor(lc, device=m_lens.device))
                mt_mask = ~lengths_to_mask(processed_lens, lc)  # (b, lc)
                cond_pad_mask = torch.cat([cond_pad_mask, mt_mask], dim=1)  # (b, 1+lc)

        cond_positions_mask = None
        if self.use_dual_causal and current_mask is not None:
            cond_positions_mask = self._compute_cond_positions_mask(current_mask, m_lens)

        if self.use_dual_causal and cond_positions_mask is not None:
            output = self.seqTransEncoder(
                x, cond_seq, padding_mask, cond_pad_mask, cond_positions_mask
            )
        else:
            output = self.seqTransEncoder(
                x, cond_seq, padding_mask, cond_pad_mask
            )
        return output


    def forward(self, latents, motion_token, y, m_lens):
        '''
        :param latents: (b, l, d)
        :param motion_token: (b, l) 或 (b, l, q)，离散token（每步 Q 个码本）
        :param y: raw text for cond_mode=text, (b, ) for cond_mode=action
        :m_lens: (b,)
        :return:
        '''
        #1.数据预处理
        b, l, d = latents.shape
        device = latents.device

        #2.创建有效位置掩码
        non_pad_mask = lengths_to_mask(m_lens, l) #(b, l)
        latents = torch.where(non_pad_mask.unsqueeze(-1), latents, torch.zeros_like(latents))

        #3.设置训练目标
        target = latents.clone().detach() 
        input = latents.clone()

        force_mask = False
        if self.cond_mode == 'text':
            with torch.no_grad():
                cond_vector = self.encode_text(y)
        elif self.cond_mode == 'action':
            cond_vector = self.enc_action(y).to(device).float()
        elif self.cond_mode == 'uncond':
            cond_vector = torch.zeros(b, self.latent_dim).float().to(device)
            force_mask = True
        else:
            raise NotImplementedError("Unsupported condition mode!!!")

        #4.随机时间步和掩码概率
        rand_time = uniform((b,), device=device)  # [0,1]均匀分布
        rand_mask_probs = cosine_schedule(rand_time)  # 余弦调度
        num_masked = (l * rand_mask_probs).round().clamp(min=1)

        # 5. 三层掩码策略
        batch_randperm = torch.rand((b, l), device=device).argsort(dim=-1)
        mask = batch_randperm < num_masked.unsqueeze(-1)
        mask &= non_pad_mask  # 只在有效位置掩码

        # 5.1 随机噪声替换 (10%)
        mask_rlatents = get_mask_subset_prob(mask, 0.1)
        rand_latents = torch.randn_like(input)
        input = torch.where(mask_rlatents.unsqueeze(-1), rand_latents, input)

        # 5.2 MASK token替换 (88%)
        mask_mlatents = get_mask_subset_prob(mask & ~mask_rlatents, 0.88)
        input = torch.where(mask_mlatents.unsqueeze(-1), self.mask_latent.repeat(b, l, 1), input)

        # 6. Transformer前向传播
        z = self.trans_forward(input, motion_token, cond_vector, ~non_pad_mask, force_mask, m_lens, current_mask=mask)

        # 7. DiffMLPs扩散损失
        target = target.reshape(b * l, -1).repeat(self.diffmlps_batch_mul, 1)
        z = z.reshape(b * l, -1).repeat(self.diffmlps_batch_mul, 1)
        mask = mask.reshape(b * l).repeat(self.diffmlps_batch_mul)
        
        target = target[mask]
        z = z[mask]
        loss = self.DiffMLPs(z=z, target=target)
        
        return loss

    
    def forward_with_CFG(self, latents, cond_vector, padding_mask, cfg=3, mask=None, force_mask=False, motion_token=None, m_lens=None):
        # 添加current_mask参数
        logits = self.trans_forward(latents, motion_token, cond_vector, padding_mask, force_mask=force_mask, m_lens=m_lens, current_mask=mask)
        if cfg != 1:
            aux_logits = self.trans_forward(latents, motion_token, cond_vector, padding_mask, force_mask=True, m_lens=m_lens, current_mask=mask)
            mixed_logits = torch.cat([logits, aux_logits], dim=0)
        else:
            mixed_logits = logits
        b, l, d = mixed_logits.size()
        if mask is not None:
            mask2 = torch.cat([mask, mask], dim=0).reshape(b * l)
            mixed_logits = (mixed_logits.reshape(b * l, d))[mask2]
        else:
            mixed_logits = mixed_logits.reshape(b * l, d)
        output = self.DiffMLPs.sample(mixed_logits, 1, cfg)
        if cfg != 1:
            scaled_logits, _ = output.chunk(2, dim=0)
        else:
            scaled_logits = output
        if mask is not None:
            latents = latents.reshape(b//2 * l, self.ae_dim)
            latents[mask.reshape(b//2 * l)] = scaled_logits
            scaled_logits = latents.reshape(b//2, l, self.ae_dim)

        return scaled_logits

    @torch.no_grad()
    @eval_decorator
    def generate(self,
                 conds,
                 m_lens,
                 timesteps: int,
                 cond_scale: int,
                 temperature=1,
                 force_mask=False,
                 hard_pseudo_reorder=False
                 ):
        
        device = next(self.parameters()).device
        l = max(m_lens)
        b = len(m_lens)

        if self.cond_mode == 'text':
            with torch.no_grad():
                cond_vector = self.encode_text(conds)
        elif self.cond_mode == 'action':
            cond_vector = self.enc_action(conds).to(device)
        elif self.cond_mode == 'uncond':
            cond_vector = torch.zeros(b, self.latent_dim).float().to(device)
        else:
            raise NotImplementedError("Unsupported condition mode!!!")

        padding_mask = ~lengths_to_mask(m_lens, l)

        latents = torch.where(padding_mask.unsqueeze(-1), torch.zeros(b, l, self.ae_dim).to(device),
                          self.mask_latent.repeat(b, l, 1))
        masked_rand_schedule = torch.where(padding_mask, 1e5, torch.rand_like(padding_mask, dtype=torch.double))

        for timestep, steps_until_x0 in zip(torch.linspace(0, 1, timesteps, device=device), reversed(range(timesteps))):
            rand_mask_prob = cosine_schedule(timestep)
            num_masked = torch.round(rand_mask_prob * m_lens).clamp(min=1)
            sorted_indices = masked_rand_schedule.argsort(dim=1)
            ranks = sorted_indices.argsort(dim=1)
            is_mask = (ranks < num_masked.unsqueeze(-1))

            latents = torch.where(is_mask.unsqueeze(-1), self.mask_latent.repeat(b, l, 1), latents)
            logits = self.forward_with_CFG(latents, cond_vector=cond_vector, padding_mask=padding_mask,
                                                  cfg=cond_scale, mask=is_mask, force_mask=force_mask)
            latents = torch.where(is_mask.unsqueeze(-1), logits, latents)

            masked_rand_schedule = masked_rand_schedule.masked_fill(~is_mask, 1e5)

        latents = torch.where(padding_mask.unsqueeze(-1), torch.zeros_like(latents), latents)
        return latents.permute(0,2,1)
    
    @torch.no_grad()
    @eval_decorator
    def edit(self,
             conds,
             latents,
             m_lens,
             timesteps: int,
             cond_scale: int,
             temperature=1,
             force_mask=False,
             edit_mask=None,
             padding_mask=None,
             hard_pseudo_reorder=False,
             ):

        device = next(self.parameters()).device
        l = latents.shape[-1]

        if self.cond_mode == 'text':
            with torch.no_grad():
                cond_vector = self.encode_text(conds)
        elif self.cond_mode == 'action':
            cond_vector = self.enc_action(conds).to(device)
        elif self.cond_mode == 'uncond':
            cond_vector = torch.zeros(1, self.latent_dim).float().to(device)
        else:
            raise NotImplementedError("Unsupported condition mode!!!")

        if padding_mask == None:
            padding_mask = ~lengths_to_mask(m_lens, l)

        if edit_mask == None:
            mask_free = True
            latents = torch.where(padding_mask.unsqueeze(-1), torch.zeros(latents.shape[0], l, self.ae_dim).to(device),
                                  latents.permute(0, 2, 1))
            edit_mask = torch.ones_like(padding_mask)
            edit_mask = edit_mask & ~padding_mask
            edit_len = edit_mask.sum(dim=-1)
            masked_rand_schedule = torch.where(edit_mask, torch.rand_like(edit_mask, dtype=torch.float), 1e5)
        else:
            mask_free = False
            edit_mask = edit_mask & ~padding_mask
            edit_len = edit_mask.sum(dim=-1)
            latents = torch.where(padding_mask.unsqueeze(-1), torch.zeros(latents.shape[0], l, self.ae_dim).to(device),
                              latents.permute(0, 2, 1))
            latents = torch.where(edit_mask.unsqueeze(-1),
                              self.mask_latent.repeat(latents.shape[0], l, 1), latents)
            masked_rand_schedule = torch.where(edit_mask, torch.rand_like(edit_mask, dtype=torch.float), 
                                               torch.tensor(1e5, dtype=torch.float, device=device))

        for timestep, steps_until_x0 in zip(torch.linspace(0, 1, timesteps, device=device), reversed(range(timesteps))):
            rand_mask_prob = 0.16 if mask_free else cosine_schedule(timestep)
            num_masked = torch.round(rand_mask_prob * edit_len).clamp(min=1)
            sorted_indices = masked_rand_schedule.argsort(dim=1)
            ranks = sorted_indices.argsort(dim=1)
            is_mask = (ranks < num_masked.unsqueeze(-1))

            latents = torch.where(is_mask.unsqueeze(-1), self.mask_latent.repeat(latents.shape[0], latents.shape[1], 1), latents)
            logits = self.forward_with_CFG(latents, cond_vector=cond_vector, padding_mask=padding_mask,
                                                  cfg=cond_scale, mask=is_mask, force_mask=force_mask, m_lens=m_lens)
            latents = torch.where(is_mask.unsqueeze(-1), logits, latents)

            masked_rand_schedule = masked_rand_schedule.masked_fill(~is_mask, 1e5)

        latents = torch.where(edit_mask.unsqueeze(-1), latents, latents)
        latents = torch.where(padding_mask.unsqueeze(-1), torch.zeros_like(latents), latents)
        return latents.permute(0,2,1)