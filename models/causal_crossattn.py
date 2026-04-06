import torch
import torch.nn as nn
import math

class AdaLayerNorm(nn.Module):
    def __init__(self, dim, cond_dim):
        super().__init__()
        self.norm = nn.LayerNorm(dim, elementwise_affine=False)
        self.scale = nn.Linear(cond_dim, dim)
        self.shift = nn.Linear(cond_dim, dim)
    
    def forward(self, x, cond):
        # x: (batch, seq_len, dim)
        # cond: (batch, cond_dim)
        scale = self.scale(cond).unsqueeze(1)  # (batch, 1, dim)
        shift = self.shift(cond).unsqueeze(1)  # (batch, 1, dim)
        return self.norm(x) * (1 + scale) + shift

# ---------- 双重因果掩码自注意力 ----------
class StrictDualCausalSelfAttention(nn.Module):
    """
    严格双重因果掩码自注意力：
    - 保持严格的时序因果关系（不能看到未来）
    - 条件位置只能看到过去的条件位置
    - 生成位置遵循标准因果约束，但可以看到过去的条件位置
    输入 x 的 shape 为 (seq_len, batch, embed_dim)
    key_padding_mask: (batch, seq_len) - True 表示 padding
    cond_positions_mask: (batch, seq_len) - True 表示条件位置
    """
    def __init__(self, embed_dim, num_heads, dropout=0.0):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.dropout = dropout
        self.allow_gen_see_future_cond = False 
        
        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        self.out_proj = nn.Linear(embed_dim, embed_dim)
        self.dropout_layer = nn.Dropout(dropout)
        
    def forward(self, x, key_padding_mask=None, cond_positions_mask=None):
        """
        x: (seq_len, batch, embed_dim) - latent序列
        key_padding_mask: (batch, seq_len) - True=PAD
        cond_positions_mask: (batch, seq_len) - True=条件位置
        """
        seq_len, batch_size, embed_dim = x.shape
        device = x.device
        
        # 转换为 (batch, seq_len, embed_dim) 以便处理
        x = x.transpose(0, 1)  # (batch, seq_len, embed_dim)
        
        # 计算 Q, K, V
        Q = self.q_proj(x)
        K = self.k_proj(x)
        V = self.v_proj(x)
        
        # Reshape for multi-head attention
        head_dim = embed_dim // self.num_heads
        Q = Q.view(batch_size, seq_len, self.num_heads, head_dim).transpose(1, 2)  
        K = K.view(batch_size, seq_len, self.num_heads, head_dim).transpose(1, 2)  
        V = V.view(batch_size, seq_len, self.num_heads, head_dim).transpose(1, 2)  
        
        # 计算attention scores
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(head_dim)  
        
        # 创建严格双重因果mask
        attn_mask = self._create_strict_dual_causal_mask(seq_len, batch_size, cond_positions_mask, device)
        
        # 扩展mask到所有heads
        attn_mask = attn_mask.unsqueeze(1).expand(-1, self.num_heads, -1, -1)  
        
        # 应用attention mask
        scores = scores.masked_fill(attn_mask, float('-inf'))
        
        # 应用padding mask
        if key_padding_mask is not None:
            key_padding_mask = key_padding_mask.unsqueeze(1).unsqueeze(2)
            scores = scores.masked_fill(key_padding_mask, float('-inf'))
        
        # Softmax
        attn_weights = torch.softmax(scores, dim=-1)
        attn_weights = self.dropout_layer(attn_weights)
        
        # 应用attention到values
        out = torch.matmul(attn_weights, V)  
        
        # 重新组合heads
        out = out.transpose(1, 2).contiguous().view(batch_size, seq_len, embed_dim)
        
        # 输出投影
        out = self.out_proj(out)
        
        # 转换回 (seq_len, batch, embed_dim)
        out = out.transpose(0, 1)
        
        return out, attn_weights.mean(dim=1)  

    def _create_strict_dual_causal_mask(self, seq_len, batch_size, cond_positions_mask, device):
        """
        向量化版本：条件位置全局可见
        """
        if cond_positions_mask is None:
            base_causal_mask = torch.triu(torch.ones(seq_len, seq_len, device=device, dtype=torch.bool), diagonal=1)
            return base_causal_mask.unsqueeze(0).expand(batch_size, -1, -1)
        
        # 创建位置索引
        i_idx = torch.arange(seq_len, device=device).view(-1, 1)  # (seq_len, 1)
        j_idx = torch.arange(seq_len, device=device).view(1, -1)  # (1, seq_len)
        
        # 基础因果约束
        future_mask = j_idx > i_idx  # (seq_len, seq_len)
        
        # 扩展维度
        cond_i = cond_positions_mask.unsqueeze(-1)  # (batch, seq_len, 1)
        cond_j = cond_positions_mask.unsqueeze(-2)  # (batch, 1, seq_len)
        
        # 条件位置之间的连接性（忽略时序）
        cond_to_cond = cond_i & cond_j  # (batch, seq_len, seq_len) 条件位置互相可见
        
        # 条件位置看非条件位置的约束
        cond_to_gen = cond_i & (~cond_j)  # (batch, seq_len, seq_len) 条件不能看生成
        
        # 基础因果mask扩展到batch维度
        future_mask_batch = future_mask.unsqueeze(0).expand(batch_size, -1, -1)
        
        # 最终掩码：
        # 1. 应用时序约束，但条件位置之间例外
        # 2. 条件位置不能看生成位置
        dual_mask = (future_mask_batch & (~cond_to_cond)) | cond_to_gen

        # 可选：允许 生成→条件 看未来条件（只对生成行、条件列解除屏蔽）
        if getattr(self, "allow_gen_see_future_cond", False):
            gen_i = (~cond_positions_mask).unsqueeze(-1)   # (b, l, 1)
            allow = gen_i & cond_j                         # 生成行、条件列
            dual_mask = dual_mask & (~allow)
        
        return dual_mask

# ---------- 双重因果掩码跨注意力 ----------
class StrictDualCausalCrossAttention(nn.Module):
    """
    双重因果掩码跨注意力：
    - Query: latent序列，有条件位置和生成位置
    - Key/Value: text + motion token序列
    - Text token全局可见
    - Motion token与latent保持相同的掩码位置
    - 应用双重因果掩码：条件位置只能看条件位置，生成位置遵循因果约束
    """
    def __init__(self, embed_dim, num_heads, dropout=0.0):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.dropout = dropout
        self.allow_gen_see_future_cond = False
        
        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        self.out_proj = nn.Linear(embed_dim, embed_dim)
        self.dropout_layer = nn.Dropout(dropout)
        
    def forward(self, q, kv, key_padding_mask=None, cond_padding_mask=None, 
                cond_positions_mask=None):
        """
        q: (seq_len_q, batch, embed_dim) - latent序列
        kv: (seq_len_kv, batch, embed_dim) - text + motion token序列
        key_padding_mask: (batch, seq_len_q) - latent的padding掩码
        cond_padding_mask: (batch, seq_len_kv) - text+motion token的padding掩码
        cond_positions_mask: (batch, seq_len_q) - latent的条件位置掩码
        """
        seq_len_q, batch_size, embed_dim = q.shape
        seq_len_kv = kv.shape[0]
        device = q.device
        
        # 转换为 (batch, seq_len, embed_dim) 以便处理
        q = q.transpose(0, 1)   # (batch, seq_len_q, embed_dim)
        kv = kv.transpose(0, 1) # (batch, seq_len_kv, embed_dim)
        
        # 计算 Q, K, V
        Q = self.q_proj(q)
        K = self.k_proj(kv)
        V = self.v_proj(kv)
        
        # Reshape for multi-head attention
        head_dim = embed_dim // self.num_heads
        Q = Q.view(batch_size, seq_len_q, self.num_heads, head_dim).transpose(1, 2)
        K = K.view(batch_size, seq_len_kv, self.num_heads, head_dim).transpose(1, 2)
        V = V.view(batch_size, seq_len_kv, self.num_heads, head_dim).transpose(1, 2)
        
        # 计算attention scores
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(head_dim)
        
        # 创建双重因果跨注意力mask
        attn_mask = self._create_dual_causal_cross_mask(
            seq_len_q, seq_len_kv, batch_size, cond_positions_mask, device
        )
        
        # 扩展mask到所有heads
        attn_mask = attn_mask.unsqueeze(1).expand(-1, self.num_heads, -1, -1)
        
        # 应用attention mask
        scores = scores.masked_fill(attn_mask, float('-inf'))
        
        # 应用padding mask
        if cond_padding_mask is not None:
            cond_padding_mask = cond_padding_mask.unsqueeze(1).unsqueeze(2)
            scores = scores.masked_fill(cond_padding_mask, float('-inf'))
        
        # Softmax
        attn_weights = torch.softmax(scores, dim=-1)
        attn_weights = self.dropout_layer(attn_weights)
        
        # 应用attention到values
        out = torch.matmul(attn_weights, V)
        
        # 重新组合heads
        out = out.transpose(1, 2).contiguous().view(batch_size, seq_len_q, embed_dim)
        
        # 输出投影
        out = self.out_proj(out)
        
        # 转换回 (seq_len, batch, embed_dim)
        out = out.transpose(0, 1)
        
        return out, attn_weights.mean(dim=1)
    

    def _create_dual_causal_cross_mask(self, seq_len_q, seq_len_kv, batch_size, 
                                       cond_positions_mask, device):
        """
        创建双重因果跨注意力掩码 - 向量化版本
        
        参数:
        seq_len_q: latent序列长度
        seq_len_kv: text + motion token序列长度 (1 + motion_len)
        cond_positions_mask: (batch, seq_len_q) latent的条件位置
        
        返回: (batch, seq_len_q, seq_len_kv) True=mask掉
        
        逻辑:
        - Text token (位置0) 全局可见
        - Motion token与latent共用条件位置掩码
        - 条件位置之间全局可见（忽略时序）
        - 生成位置遵循因果约束
        """
        if cond_positions_mask is None:
            # 如果没有条件位置信息，使用标准因果掩码
            return self._create_standard_causal_cross_mask(seq_len_q, seq_len_kv, batch_size, device)
        
        motion_len = seq_len_kv - 1  # 减去text token
        
        # 创建位置索引矩阵
        i_idx = torch.arange(seq_len_q, device=device).view(-1, 1)  # (seq_len_q, 1)
        j_idx = torch.arange(motion_len, device=device).view(1, -1)  # (1, motion_len)
        
        # 基础因果约束: i < j (latent位置不能看motion token的未来位置)
        causal_constraint = i_idx < j_idx  # (seq_len_q, motion_len)
        
        # 扩展到batch维度
        causal_constraint = causal_constraint.unsqueeze(0).expand(batch_size, -1, -1)  # (batch, seq_len_q, motion_len)
        
        # latent的条件位置掩码，motion token共用相同的条件位置
        # 因为motion token与latent序列长度相同，可以直接使用相同的掩码
        latent_cond_i = cond_positions_mask.unsqueeze(-1)  # (batch, seq_len_q, 1)
        motion_cond_j = cond_positions_mask.unsqueeze(-2)  # (batch, 1, seq_len_q)
        
        # 只取motion token部分的条件位置（前motion_len个位置）
        motion_cond_j = motion_cond_j[:, :, :motion_len]  # (batch, 1, motion_len)
        
        # 条件位置之间的连接性（忽略时序约束）
        cond_to_cond = latent_cond_i & motion_cond_j  # (batch, seq_len_q, motion_len)
        
        # 条件位置不能看生成位置的约束
        cond_to_gen = latent_cond_i & (~motion_cond_j)  # (batch, seq_len_q, motion_len)
        
        # 对motion token部分应用双重因果掩码
        motion_mask = (causal_constraint & (~cond_to_cond)) | cond_to_gen

        # 可选：允许 生成→条件 看未来条件（生成行 + 条件列 解除屏蔽）
        if getattr(self, "allow_gen_see_future_cond", False):
            gen_i = (~cond_positions_mask).unsqueeze(-1)       # (b, l_q, 1)
            allow = gen_i & motion_cond_j
            motion_mask = motion_mask & (~allow)
        
        # 构建完整的跨注意力掩码
        full_mask = torch.zeros(batch_size, seq_len_q, seq_len_kv, device=device, dtype=torch.bool)
        
        # Text token (位置0) 全局可见
        full_mask[:, :, 0] = False
        
        # Motion token部分 (位置1到seq_len_kv-1) 应用双重因果掩码
        full_mask[:, :, 1:] = motion_mask
        
        return full_mask

    
    def _create_standard_causal_cross_mask(self, seq_len_q, seq_len_kv, batch_size, device):
        """创建标准因果跨注意力掩码（fallback）- 向量化版本"""
        # 创建位置索引
        i_idx = torch.arange(seq_len_q, device=device).view(-1, 1)  # (seq_len_q, 1)
        j_idx = torch.arange(seq_len_kv - 1, device=device).view(1, -1)  # (1, motion_len)
        
        # Text token全局可见，motion token遵循标准因果约束
        mask = torch.zeros(seq_len_q, seq_len_kv, device=device, dtype=torch.bool)
        
        # Text token (位置0) 全局可见
        mask[:, 0] = False
        
        # Motion token部分使用因果掩码 (i < j 被mask掉)
        causal_mask = i_idx < j_idx  # (seq_len_q, motion_len)
        mask[:, 1:] = causal_mask
        
        return mask.unsqueeze(0).expand(batch_size, -1, -1)

# ---------- 因果自注意力 ----------
class CausalSelfAttention(nn.Module):
    """
    因果自注意力（mask future tokens）
    输入 x 的 shape 为 (seq_len, batch, embed_dim)
    key_padding_mask: (batch, seq_len) - True 表示 padding
    """
    def __init__(self, embed_dim, num_heads, dropout=0.0):
        super().__init__()
        self.attn = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, key_padding_mask=None):
        seq_len = x.size(0)
        # causal mask: True = mask out future positions (upper triangular, excluding diagonal)
        attn_mask = torch.triu(torch.ones(seq_len, seq_len, device=x.device, dtype=torch.bool), diagonal=1)
        out, attn_weights = self.attn(x, x, x, attn_mask=attn_mask, key_padding_mask=key_padding_mask)
        out = self.dropout(out)
        return out, attn_weights

# ---------- 因果跨注意力 ----------
class CausalCrossAttention(nn.Module):
    """
    跨模态因果注意力：
      q: (seq_len_t, batch, d)
      cond: (seq_len_c, batch, d) or (batch, d)
    行为：
      - 若 seq_len_c == seq_len_t：对 cond 使用 causal mask（target pos i 只能看 cond positions <= i）
      - 否则：使用无因果掩码的普通 cross-attn（兼容全局 cond embedding）
    key_padding_mask: (batch, seq_len_t) for Q (not used in MultiheadAttention call here)
    cond_padding_mask: (batch, seq_len_c) for cond (用于 MultiheadAttention 的 key_padding_mask 参数)
    """
    def __init__(self, embed_dim, num_heads, dropout=0.0):
        super().__init__()
        self.attn = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout)
        self.dropout = nn.Dropout(dropout)

    def forward(self, q, cond, key_padding_mask=None, cond_padding_mask=None):
        # q: (seq_len_t, batch, d)
        # cond: (seq_len_c, batch, d) or (batch, d)
        if cond.dim() == 2:
            cond = cond.unsqueeze(0)  # (1, b, d)
        seq_len_t = q.size(0)
        seq_len_c = cond.size(0)

        attn_mask = None
        # 只对motion token部分使用因果掩码，文本部分可以全局可见
        if seq_len_c == seq_len_t:
            # 如果长度相等，使用标准因果掩码
            attn_mask = torch.triu(torch.ones(seq_len_t, seq_len_c, device=q.device, dtype=torch.bool), diagonal=1)
        elif seq_len_c > 1:
            # 如果有多个条件token（文本+motion），对motion部分使用因果掩码
            # 假设第一个是文本token，其余是motion token
            attn_mask = torch.zeros(seq_len_t, seq_len_c, device=q.device, dtype=torch.bool)
            if seq_len_c > 1:  # 有motion token
                motion_len = seq_len_c - 1
                target_len = min(seq_len_t, motion_len)
                # 对motion token部分应用因果掩码
                causal_part = torch.triu(torch.ones(target_len, motion_len, device=q.device, dtype=torch.bool), diagonal=1)
                attn_mask[:target_len, 1:] = causal_part

        out, attn_weights = self.attn(q, cond, cond, attn_mask=attn_mask, key_padding_mask=cond_padding_mask)
        out = self.dropout(out)
        return out, attn_weights

# ---------- 使用严格双重因果掩码的 Transformer Block ----------
class StrictDualCausalTransformerBlock(nn.Module):
    def __init__(self, latent_dim, cond_dim, n_head, ff_size, dropout):
        super().__init__()
        # 使用严格双重因果掩码自注意力
        self.self_attn = StrictDualCausalSelfAttention(latent_dim, n_head, dropout=dropout)
        # 使用双重因果掩码跨注意力
        self.cross_attn = StrictDualCausalCrossAttention(latent_dim, n_head, dropout=dropout)

        self.adaln1 = AdaLayerNorm(latent_dim, cond_dim)
        self.adaln2 = AdaLayerNorm(latent_dim, cond_dim) 
        self.adaln3 = AdaLayerNorm(latent_dim, cond_dim)

        self.dropout = nn.Dropout(dropout)

        self.ff = nn.Sequential(
            nn.Linear(latent_dim, ff_size),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ff_size, latent_dim),
            nn.Dropout(dropout)
        )

    def forward(self, x, cond, padding_mask=None, cond_padding_mask=None, cond_positions_mask=None):
        """
        x: (b, l, d) - latent序列
        cond: (b, cond_dim) or (b, c_len, cond_dim) - 条件序列 
        padding_mask: (b, l) True=PAD
        cond_padding_mask: (b, c_len) True=PAD
        cond_positions_mask: (b, l) True=条件位置，用于严格双重因果掩码
        返回: (b, l, d)
        """
        b, l, d = x.shape
        
        # 提取文本条件用于AdaLN（使用第一个位置，即[TXT] token）
        if cond.dim() == 3:
            text_cond = cond[:, 0, :]  # (b, d)
        else:
            text_cond = cond  # (b, d)
        
        x = x.transpose(0, 1)  # (l, b, d)

        # Self-attn with AdaLN and 严格双重因果掩码
        residual = x.transpose(0, 1)  # (b, l, d) for AdaLN
        x_ln = self.adaln1(residual, text_cond)
        x_ln = x_ln.transpose(0, 1)  # back to (l, b, d)
        
        # 使用严格双重因果掩码自注意力
        x_attn, _ = self.self_attn(x_ln, key_padding_mask=padding_mask, cond_positions_mask=cond_positions_mask)
        x = residual.transpose(0, 1) + self.dropout(x_attn)

        # Cross-attn with AdaLN and 双重因果掩码
        residual = x.transpose(0, 1)  # (b, l, d)
        x_ln = self.adaln2(residual, text_cond)
        x_ln = x_ln.transpose(0, 1)  # (l, b, d)
        
        # 处理条件序列的格式转换
        if cond.dim() == 2:
            cond_cross = cond.unsqueeze(0).transpose(0, 1)  # (1, b, d)
        else:  # cond.dim() == 3
            cond_cross = cond.transpose(0, 1)  # (c_len, b, d)
        # 使用双重因果掩码跨注意力
        x_cross, _ = self.cross_attn(
            x_ln, cond_cross, 
            cond_padding_mask=cond_padding_mask,
            cond_positions_mask=cond_positions_mask
        )
        x = residual.transpose(0, 1) + self.dropout(x_cross)

        # FFN with AdaLN (保持不变)
        x = x.transpose(0, 1)  # (b, l, d)
        residual = x
        x_ln = self.adaln3(x, text_cond)
        x = residual + self.ff(x_ln)

        return x

# ---------- 因果 Cross-Attn Transformer Block ----------
class CausalCrossAttnTransformerBlock(nn.Module):
    def __init__(self, latent_dim, cond_dim, n_head, ff_size, dropout):
        super().__init__()
        # 使用因果自注意力（自回归场景下通常需要）
        self.self_attn = CausalSelfAttention(latent_dim, n_head, dropout=dropout)
        # 因果跨注意力
        self.cross_attn = CausalCrossAttention(latent_dim, n_head, dropout=dropout)

        self.adaln1 = AdaLayerNorm(latent_dim, cond_dim)
        self.adaln2 = AdaLayerNorm(latent_dim, cond_dim) 
        self.adaln3 = AdaLayerNorm(latent_dim, cond_dim)

        self.dropout = nn.Dropout(dropout)

        self.ff = nn.Sequential(
            nn.Linear(latent_dim, ff_size),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ff_size, latent_dim),
            nn.Dropout(dropout)
        )


    def forward(self, x, cond, padding_mask=None, cond_padding_mask=None):
        """
        x: (b, l, d)
        cond: (b, cond_dim) or (b, c_len, cond_dim)
        padding_mask: (b, l) True=PAD
        cond_padding_mask: (b, c_len) True=PAD
        返回: (b, l, d)
        """
        b, l, d = x.shape
        # cond_proj = self.cond_proj(cond)  # (b, cond_dim) -> (b, latent_dim)
        # 提取文本条件用于AdaLN（使用第一个位置，即[TXT] token）
        if cond.dim() == 3:
            text_cond = cond[:, 0, :]  # (b, d) 使用第一个位置作为全局条件
        else:
            text_cond = cond  # (b, d)
        
        x = x.transpose(0, 1)  # (l, b, d)

        # Self-attn with AdaLN
        residual = x.transpose(0, 1)  # (b, l, d) for AdaLN
        x_ln = self.adaln1(residual, text_cond)
        x_ln = x_ln.transpose(0, 1)  # back to (l, b, d)
        x_attn, _ = self.self_attn(x_ln, key_padding_mask=padding_mask)
        x = residual.transpose(0, 1) + self.dropout(x_attn)

        # Cross-attn with AdaLN
        residual = x.transpose(0, 1)  # (b, l, d)
        x_ln = self.adaln2(residual, text_cond)
        x_ln = x_ln.transpose(0, 1)  # (l, b, d)
        
        # 处理条件序列的格式转换
        if cond.dim() == 2:
            cond_cross = cond.unsqueeze(0).transpose(0, 1)  # (1, b, d)
        else:  # cond.dim() == 3
            cond_cross = cond.transpose(0, 1)  # (c_len, b, d)
        x_cross, _ = self.cross_attn(x_ln, cond_cross, cond_padding_mask=cond_padding_mask)
        x = residual.transpose(0, 1) + self.dropout(x_cross)

        # FFN with AdaLN
        x = x.transpose(0, 1)  # (b, l, d)
        residual = x
        x_ln = self.adaln3(x, text_cond)
        x = residual + self.ff(x_ln)

        return x

# ---------- 因果 Cross-Attn Transformer（多层） ----------
class CausalCrossAttnTransformer(nn.Module):
    def __init__(self, num_layers, latent_dim, cond_dim, n_head, ff_size, dropout, max_len=200):
        super().__init__()
        self.layers = nn.ModuleList([
            CausalCrossAttnTransformerBlock(latent_dim, cond_dim, n_head, ff_size, dropout)
            for _ in range(num_layers)
        ])
        # 可学习的位置编码 (broadcastable)
        self.pos_emb = nn.Parameter(torch.randn(1, max_len, latent_dim))

    def forward(self, x, cond, padding_mask=None, cond_padding_mask=None):
        """
        x: (b, l, latent_dim)
        cond: (b, cond_dim) or (b, c_len, cond_dim)
        padding_mask: (b, l) True=PAD
        cond_padding_mask: (b, c_len) True=PAD
        返回: (b, l, latent_dim)
        """
        seq_len = x.size(1)
        x = x + self.pos_emb[:, :seq_len]  # 广播添加位置编码

        for layer in self.layers:
            x = layer(x, cond, padding_mask=padding_mask, cond_padding_mask=cond_padding_mask)
        return x

# ---------- 严格双重因果掩码 Transformer（多层） ----------
class StrictDualCausalTransformer(nn.Module):
    def __init__(self, num_layers, latent_dim, cond_dim, n_head, ff_size, dropout, max_len=200):
        super().__init__()
        self.layers = nn.ModuleList([
            StrictDualCausalTransformerBlock(latent_dim, cond_dim, n_head, ff_size, dropout)
            for _ in range(num_layers)
        ])
        self.pos_emb = nn.Parameter(torch.randn(1, max_len, latent_dim))

    def forward(self, x, cond, padding_mask=None, cond_padding_mask=None, cond_positions_mask=None):
        """
        x: (b, l, latent_dim) - latent序列
        cond: (b, cond_dim) or (b, c_len, cond_dim) - 条件序列
        padding_mask: (b, l) True=PAD
        cond_padding_mask: (b, c_len) True=PAD
        cond_positions_mask: (b, l) True=条件位置，用于严格双重因果掩码
        返回: (b, l, latent_dim)
        """
        seq_len = x.size(1)
        x = x + self.pos_emb[:, :seq_len]

        for layer in self.layers:
            x = layer(x, cond, padding_mask=padding_mask, 
                     cond_padding_mask=cond_padding_mask,
                     cond_positions_mask=cond_positions_mask)
        return x
