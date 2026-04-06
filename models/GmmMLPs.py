import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class GmmMLPs(nn.Module):
    def __init__(self, z_dim, target_dim, num_components=5, hidden_dim=512):
        """
        GMM-based predictor: predicts GMM parameters (mean, std, weight) given latent z.
        Args:
            z_dim: latent vector dimension
            target_dim: target vector dimension
            num_components: number of GMM components
            hidden_dim: hidden size of MLP
        """
        super().__init__()
        self.num_components = num_components
        self.target_dim = target_dim

        self.fc = nn.Sequential(
            nn.Linear(z_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )

        self.mean_head = nn.Linear(hidden_dim, num_components * target_dim)
        self.logvar_head = nn.Linear(hidden_dim, num_components * target_dim)
        self.logits_head = nn.Linear(hidden_dim, num_components)

    def forward(self, z, target, mask=None):
        """
        Compute GMM negative log-likelihood loss.
        Args:
            z: (B, D_z)
            target: (B, D_target)
            mask: (B,) or (B, 1) optional binary mask
        Returns:
            loss: scalar NLL loss
        """
        h = self.fc(z)

        B = z.size(0)
        K = self.num_components
        D = self.target_dim

        means = self.mean_head(h).view(B, K, D)
        logvars = self.logvar_head(h).view(B, K, D)
        stds = torch.exp(0.5 * logvars)  # (B, K, D)
        weights = F.softmax(self.logits_head(h), dim=-1)  # (B, K)

        target = target.unsqueeze(1)  # (B, 1, D)

        # log probability under each Gaussian
        log_prob = -0.5 * (((target - means) / stds) ** 2 + math.log(2 * math.pi) + logvars)  # (B, K, D)
        log_prob = log_prob.sum(dim=-1)  # (B, K)

        weighted_log_prob = log_prob + torch.log(weights + 1e-8)  # (B, K)
        log_sum = torch.logsumexp(weighted_log_prob, dim=1)  # (B,)

        nll = -log_sum  # (B,)

        if mask is not None:
            nll = (nll * mask).sum() / mask.sum()
        else:
            nll = nll.mean()
        return nll

    @torch.no_grad()
    def sample(self, z, temperature=1.0, cfg=1.0):
        """
        Sample from the predicted GMM.
        Args:
            z: (B, D_z)
            temperature: scaling std for diversity
            cfg: classifier-free guidance scale (optional; if < 1, used for mixing uncond & cond)
        Returns:
            samples: (B, D_target)
        """
        h = self.fc(z)
        B = z.size(0)
        K = self.num_components
        D = self.target_dim

        means = self.mean_head(h).view(B, K, D)
        logvars = self.logvar_head(h).view(B, K, D)
        stds = torch.exp(0.5 * logvars) * temperature
        weights = F.softmax(self.logits_head(h), dim=-1)  # (B, K)

        # Sample component indices
        cat_dist = torch.distributions.Categorical(weights)
        comp_idx = cat_dist.sample()  # (B,)

        # Index into means and stds
        chosen_means = torch.gather(means, 1, comp_idx.unsqueeze(-1).unsqueeze(-1).expand(-1, 1, D)).squeeze(1)
        chosen_stds = torch.gather(stds, 1, comp_idx.unsqueeze(-1).unsqueeze(-1).expand(-1, 1, D)).squeeze(1)

        # Sample from selected Gaussian
        eps = torch.randn_like(chosen_stds)
        sample = chosen_means + eps * chosen_stds  # (B, D)

        return sample
