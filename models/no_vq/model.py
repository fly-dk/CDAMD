import torch
import torch.nn as nn
import torch.nn.functional as F
from models.vq.encdec import Encoder, Decoder
from utils.humanml_utils import HML_UPPER_BODY_MASK, HML_LOWER_BODY_MASK, UPPER_JOINT_Y_MASK
import clip
class VAEModel(nn.Module):
    def __init__(self,
                 opt,
                 input_width=66,
                 code_dim=512,
                 output_emb_width=512,
                 down_t=3,
                 stride_t=2,
                 width=512,
                 depth=3,
                 dilation_growth_rate=3,
                 activation='relu',
                 norm=None,
                 clip_version=None):
        super().__init__()

        self.code_dim = code_dim

        self.encoder = Encoder(input_width, output_emb_width, down_t, stride_t, width, depth,
                               dilation_growth_rate, activation=activation, norm=norm)

        # VAE bottleneck
        self.fc_mu = nn.Conv1d(output_emb_width, code_dim, kernel_size=1)
        self.fc_logvar = nn.Conv1d(output_emb_width, code_dim, kernel_size=1)

        # Decoder
        self.decoder = Decoder(input_width, output_emb_width, down_t, stride_t, width, depth,
                               dilation_growth_rate, activation=activation, norm=norm)

        print('Loading CLIP...')
        self.clip_version = clip_version
        self.clip_model = self.load_and_freeze_clip(clip_version)

    def preprocess(self, x):
        return x.permute(0, 2, 1).float()  # (bs, T, D) -> (bs, D, T)

    def postprocess(self, x):
        return x.permute(0, 2, 1)  # (bs, D, T) -> (bs, T, D)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x):
        x_in = self.preprocess(x)               # (bs, D, T)
        enc_feat = self.encoder(x_in)           # (bs, C, T)

        mu = self.fc_mu(enc_feat)               # (bs, code_dim, T)
        logvar = self.fc_logvar(enc_feat)       # (bs, code_dim, T)
        z = self.reparameterize(mu, logvar)     # (bs, code_dim, T)

        x_out = self.decoder(z)                 # (bs, D, T)
        # x_out = self.postprocess(x_out)         # (bs, T, D)

        # KL divergence
        kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1)  # sum over C
        kl_loss = kl_loss.mean()  # average over batch & time

        return x_out, kl_loss
    

