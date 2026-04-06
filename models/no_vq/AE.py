import torch
import torch.nn as nn

#################################################################################
#                                        VAE                                   #
#################################################################################
class VAE(nn.Module):
    def __init__(self,
                 input_width=66,
                 latent_dim=512,  # VAE的潜在空间维度
                 output_emb_width=512,
                 down_t=2,
                 stride_t=2,
                 width=512,
                 depth=3,
                 dilation_growth_rate=3,
                 activation='relu',
                 norm=None):

        super().__init__()
        self.latent_dim = latent_dim
        self.output_emb_width = output_emb_width
        
        # VAE配置
        self.beta = 1.0e-4 # KL散度权重

        # 标准VAE
        self.encoder = Encoder(input_width, output_emb_width, down_t, stride_t, width, depth,
                            dilation_growth_rate, activation=activation, norm=norm)
        
        # VAE的均值和方差层
        self.mu = nn.Linear(output_emb_width, latent_dim)
        self.logvar = nn.Linear(output_emb_width, latent_dim)
        
        # 解码前的投影层
        self.decode_layer = nn.Linear(latent_dim, output_emb_width)
        
        self.decoder = Decoder(input_width, output_emb_width, down_t, stride_t, width, depth,
                               dilation_growth_rate, activation=activation, norm=norm)

    def preprocess(self, x):
        # (bs, T, Jx3) -> (bs, Jx3, T)
        x = x.permute(0, 2, 1).float()
        return x

    def reparameterize(self, mu, logvar):
        """VAE重参数化技巧"""
        mu = torch.clamp(mu, -2.0, 2.0)
        logvar = torch.clamp(logvar, -8.0, 1.0)
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def kl_divergence(self, mu, logvar, free_bits_per_dim=0.01):
        """修复的KL散度计算"""
        mu = torch.clamp(mu, -2.0, 2.0)
        logvar = torch.clamp(logvar, -8.0, 1.0)
        # 计算每个维度的KL
        kl_per_dim = -0.5 * (1 + logvar - mu.pow(2) - logvar.exp())  # (B, latent_dim, T)
        
        # Free bits: 每个维度允许少量免费KL
        kl_per_dim_clamped = torch.clamp(kl_per_dim - free_bits_per_dim, min=0.0)
        
        # 求和并平均
        kl_loss = kl_per_dim_clamped.sum(dim=1).mean()
        
        return kl_loss

    def encode(self, x):
        """编码到潜在空间"""
        N, T, _ = x.shape
        x_in = self.preprocess(x)       # (B, C, T)
        x_encoder = self.encoder(x_in)  # (B, C, T')
        
        # 这里先保留时间维度 T'，再求 mu, logvar
        x_encoder = x_encoder.permute(0, 2, 1)  # (B, T', C)
        mu = self.mu(x_encoder)                 # (B, T', D)
        logvar = self.logvar(x_encoder)         # (B, T', D)

        z = self.reparameterize(mu, logvar)     # (B, T', D)
        # print(f"Debug: z shape = {z.shape}")  # 调试信息

        return z, mu, logvar

    def decode(self, z):
        """从潜在空间解码"""
        # print(f"Debug: z shape = {z.shape}")  # 调试信息
        # print(f"Debug: decode_layer expects input dim = {self.decode_layer.in_features}")
        # # z的形状应该是 (B, T', D)
        # if len(z.shape) == 2:
        #     z = z.unsqueeze(1)
        # elif len(z.shape) == 3 and z.shape[-1] != self.latent_dim:
        #     # 如果最后一维不是latent_dim，可能维度顺序错误
        #     print(f"Warning: unexpected z shape {z.shape}, expected last dim = {self.latent_dim}")
        z = z.permute(0, 2, 1) # b d t -> b t d 
        # 先投影
        decoded_feat = self.decode_layer(z)     # (B, T', emb)

        # 转换回 (B, emb, T')
        decoded_feat = decoded_feat.permute(0, 2, 1)

        # 解码
        x_out = self.decoder(decoded_feat)      # (B, J*3, T')
        # x_out = x_out.permute(0, 2, 1)          # (B, T', J*3)

        return x_out

    def forward(self, x):
        # 编码
        z, mu, logvar = self.encode(x)
        
        # 解码
        x_out = self.decode(z)
        
        # KL损失
        kl_loss = self.kl_divergence(mu, logvar)
        
        return x_out, self.beta * kl_loss, kl_loss

    def sample(self, batch_size, device):
        """从先验分布采样"""
        z = torch.randn(batch_size, self.latent_dim, device=device)
        return self.decode_latent(z)

    # ================= 熵计算方法 =================
    
    def compute_position_entropy(self, x, method='gaussian'):
        """
        计算潜在序列每个时间步的熵
        
        Args:
            x: 输入动作序列 (B, T, input_width)
            method: 'gaussian', 'empirical', 'kl_based'
        
        Returns:
            position_entropy: (B, T) 每个位置的熵值
        """
        self.eval()
        with torch.no_grad():
            z, mu, logvar = self.encode(x)  # (B, latent_dim, T)
            
            if method == 'gaussian':
                # 基于高斯分布的理论熵
                entropy = self._gaussian_entropy(mu, logvar)
            elif method == 'empirical':
                # 基于多次采样的经验熵
                entropy = self._empirical_entropy(mu, logvar, n_samples=100)
            elif method == 'kl_based':
                # 基于KL散度的熵（相对于先验的信息量）
                entropy = self._kl_based_entropy(mu, logvar)
            else:
                raise ValueError(f"Unknown method: {method}")
                
            return entropy
    
    def _gaussian_entropy(self, mu, logvar):
        """
        计算高斯分布的理论熵: H = 0.5 * log(2πe * σ²)
        Args:
            mu: (B, T', D)
            logvar: (B, T', D)
        Returns:
            entropy: (B, T')
        """
        mu = torch.clamp(mu, -2.0, 2.0)
        logvar = torch.clamp(logvar, -8.0, 1.0)
        dim_entropy = 0.5 * (logvar + 2.8378770664093453)
        # 修复：对潜在维度D求和，得到每个时间步的熵 (B, T')
        position_entropy = dim_entropy.sum(dim=-1)
        return position_entropy
#################################################################################
#                                         AE                                    #
#################################################################################
class AE(nn.Module):
    def __init__(self, input_width=66, output_emb_width=512, down_t=2, stride_t=2, width=512, depth=3,
                 dilation_growth_rate=3, activation='relu', norm=None):
        super().__init__()
        self.output_emb_width = output_emb_width
        self.encoder = Encoder(input_width, output_emb_width, down_t, stride_t, width, depth,
                               dilation_growth_rate, activation=activation, norm=norm)
        self.decoder = Decoder(input_width, output_emb_width, down_t, stride_t, width, depth,
                               dilation_growth_rate, activation=activation, norm=norm)

    def preprocess(self, x):
        x = x.permute(0, 2, 1).float()
        return x

    def encode(self, x):
        x_in = self.preprocess(x)
        x_encoder = self.encoder(x_in)
        return x_encoder

    def forward(self, x):
        x_in = self.preprocess(x) # (B,T,66) -> (B,66,T)
        x_encoder = self.encoder(x_in) # (B,512,T//4)
        x_out = self.decoder(x_encoder) #(B, T, 66)
        return x_out

    def decode(self, x):
        x_out = self.decoder(x)
        return x_out

#################################################################################
#                                      AE Zoos                                  #
#################################################################################
def ae(**kwargs):
    return AE(output_emb_width=512, down_t=2, stride_t=2, width=512, depth=3,
                 dilation_growth_rate=3, activation='relu', norm=None, **kwargs)

def vae(**kwargs):
    return VAE(output_emb_width=512, latent_dim=512, down_t=2, stride_t=2, width=512, depth=3,
                 dilation_growth_rate=3, activation='relu', norm=None, **kwargs)

AE_models = {
    'AE_Model': ae,
    'VAE_Model': vae
}

#################################################################################
#                          跳跃连接编码器和解码器                                #
#################################################################################
class EncoderWithSkip(nn.Module):
    def __init__(self, input_emb_width=3, output_emb_width=512, down_t=2, stride_t=2, width=512, depth=3,
                 dilation_growth_rate=3, activation='relu', norm=None):
        super().__init__()
        self.down_t = down_t
        filter_t, pad_t = stride_t * 2, stride_t // 2
        
        # 初始卷积
        self.init_conv = nn.Sequential(
            nn.Conv1d(input_emb_width, width, 3, 1, 1),
            nn.ReLU()
        )
        
        # 下采样块
        self.down_blocks = nn.ModuleList()
        for i in range(down_t):
            block = nn.Sequential(
                nn.Conv1d(width, width, filter_t, stride_t, pad_t),
                Resnet1D(width, depth, dilation_growth_rate, activation=activation, norm=norm),
            )
            self.down_blocks.append(block)
        
        # 最终卷积
        self.final_conv = nn.Conv1d(width, output_emb_width, 3, 1, 1)

    def forward(self, x):
        skip_connections = []
        
        # 初始卷积
        x = self.init_conv(x)
        skip_connections.append(x.clone())
        
        # 下采样并保存跳跃连接
        for i, block in enumerate(self.down_blocks):
            x = block(x)
            # 除了最后一层，都保存跳跃连接
            if i < len(self.down_blocks) - 1:
                skip_connections.append(x.clone())
        
        x = self.final_conv(x)
        
        return x, skip_connections


class DecoderWithSkip(nn.Module):
    def __init__(self, input_emb_width=3, output_emb_width=512, down_t=2, stride_t=2, width=512, depth=3,
                 dilation_growth_rate=3, activation='relu', norm=None):
        super().__init__()
        self.down_t = down_t
        
        # 初始卷积
        self.init_conv = nn.Sequential(
            nn.Conv1d(output_emb_width, width, 3, 1, 1),
            nn.ReLU()
        )
        
        # 上采样块
        self.up_blocks = nn.ModuleList()
        self.skip_convs = nn.ModuleList()
        
        for i in range(down_t):
            # 跳跃连接融合层
            if i > 0:  # 第一个块不需要融合跳跃连接
                skip_conv = nn.Conv1d(width * 2, width, 1, 1, 0)
            else:
                skip_conv = nn.Identity()
            self.skip_convs.append(skip_conv)
            
            # 上采样块
            block = nn.Sequential(
                Resnet1D(width, depth, dilation_growth_rate, reverse_dilation=True, activation=activation, norm=norm),
                nn.Upsample(scale_factor=2, mode='nearest'),
                nn.Conv1d(width, width, 3, 1, 1)
            )
            self.up_blocks.append(block)
        
        # 最终输出层
        self.final_layers = nn.Sequential(
            nn.Conv1d(width, width, 3, 1, 1),
            nn.ReLU(),
            nn.Conv1d(width, input_emb_width, 3, 1, 1)
        )

    def forward(self, x, skip_connections):
        x = self.init_conv(x)
        
        # 反向使用跳跃连接
        for i, (up_block, skip_conv) in enumerate(zip(self.up_blocks, self.skip_convs)):
            # 上采样
            x = up_block(x)
            
            # 融合跳跃连接
            if skip_connections is not None and i > 0:
                skip_idx = len(skip_connections) - i  # 反向索引
                if skip_idx >= 0 and skip_idx < len(skip_connections):
                    skip = skip_connections[skip_idx]
                    
                    # 确保尺寸匹配
                    if skip.size(-1) != x.size(-1):
                        skip = F.interpolate(skip, size=x.size(-1), mode='nearest')
                    
                    # 拼接并融合
                    x = torch.cat([x, skip], dim=1)
                    x = skip_conv(x)
        
        x = self.final_layers(x)
        return x.permute(0, 2, 1)

#################################################################################
#                                 Inner Architectures                           #
#################################################################################
class Encoder(nn.Module):
    def __init__(self, input_emb_width=3, output_emb_width=512, down_t=2, stride_t=2, width=512, depth=3,
                 dilation_growth_rate=3, activation='relu', norm=None):
        super().__init__()
        blocks = []
        filter_t, pad_t = stride_t * 2, stride_t // 2
        blocks.append(nn.Conv1d(input_emb_width, width, 3, 1, 1))
        blocks.append(nn.ReLU())

        for i in range(down_t):
            input_dim = width
            block = nn.Sequential(
                nn.Conv1d(input_dim, width, filter_t, stride_t, pad_t),
                Resnet1D(width, depth, dilation_growth_rate, activation=activation, norm=norm),
            )
            blocks.append(block)
        blocks.append(nn.Conv1d(width, output_emb_width, 3, 1, 1))
        self.model = nn.Sequential(*blocks)

    def forward(self, x):
        return self.model(x)


class Decoder(nn.Module):
    def __init__(self, input_emb_width=3, output_emb_width=512, down_t=2, stride_t=2, width=512, depth=3,
                 dilation_growth_rate=3, activation='relu', norm=None):
        super().__init__()
        blocks = []
        blocks.append(nn.Conv1d(output_emb_width, width, 3, 1, 1))
        blocks.append(nn.ReLU())
        for i in range(down_t):
            out_dim = width
            block = nn.Sequential(
                Resnet1D(width, depth, dilation_growth_rate, reverse_dilation=True, activation=activation, norm=norm),
                nn.Upsample(scale_factor=2, mode='nearest'),
                nn.Conv1d(width, out_dim, 3, 1, 1)
            )
            blocks.append(block)
        blocks.append(nn.Conv1d(width, width, 3, 1, 1))
        blocks.append(nn.ReLU())
        blocks.append(nn.Conv1d(width, input_emb_width, 3, 1, 1))
        self.model = nn.Sequential(*blocks)

    def forward(self, x):
        x = self.model(x)
        return x.permute(0, 2, 1)


class Resnet1D(nn.Module):
    def __init__(self, n_in, n_depth, dilation_growth_rate=1, reverse_dilation=True, activation='relu', norm=None):
        super().__init__()
        blocks = [ResConv1DBlock(n_in, n_in, dilation=dilation_growth_rate ** depth, activation=activation, norm=norm)
                  for depth in range(n_depth)]
        if reverse_dilation:
            blocks = blocks[::-1]

        self.model = nn.Sequential(*blocks)

    def forward(self, x):
        return self.model(x)


class nonlinearity(nn.Module):
    def __init(self):
        super().__init__()

    def forward(self, x):
        return x * torch.sigmoid(x)


class ResConv1DBlock(nn.Module):
    def __init__(self, n_in, n_state, dilation=1, activation='silu', norm=None, dropout=0.2):
        super(ResConv1DBlock, self).__init__()
        padding = dilation
        self.norm = norm

        if norm == "LN":
            self.norm1 = nn.LayerNorm(n_in)
            self.norm2 = nn.LayerNorm(n_in)
        elif norm == "GN":
            self.norm1 = nn.GroupNorm(num_groups=32, num_channels=n_in, eps=1e-6, affine=True)
            self.norm2 = nn.GroupNorm(num_groups=32, num_channels=n_in, eps=1e-6, affine=True)
        elif norm == "BN":
            self.norm1 = nn.BatchNorm1d(num_features=n_in, eps=1e-6, affine=True)
            self.norm2 = nn.BatchNorm1d(num_features=n_in, eps=1e-6, affine=True)
        else:
            self.norm1 = nn.Identity()
            self.norm2 = nn.Identity()

        if activation == "relu":
            self.activation1 = nn.ReLU()
            self.activation2 = nn.ReLU()

        elif activation == "silu":
            self.activation1 = nonlinearity()
            self.activation2 = nonlinearity()

        elif activation == "gelu":
            self.activation1 = nn.GELU()
            self.activation2 = nn.GELU()

        self.conv1 = nn.Conv1d(n_in, n_state, 3, 1, padding, dilation)
        self.conv2 = nn.Conv1d(n_state, n_in, 1, 1, 0, )
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x_orig = x
        if self.norm == "LN":
            x = self.norm1(x.transpose(-2, -1))
            x = self.activation1(x.transpose(-2, -1))
        else:
            x = self.norm1(x)
            x = self.activation1(x)

        x = self.conv1(x)

        if self.norm == "LN":
            x = self.norm2(x.transpose(-2, -1))
            x = self.activation2(x.transpose(-2, -1))
        else:
            x = self.norm2(x)
            x = self.activation2(x)

        x = self.conv2(x)
        x = self.dropout(x)
        x = x + x_orig
        return x
