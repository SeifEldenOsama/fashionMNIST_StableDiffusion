import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
from ..config import config

class ConditionalDenoiseDiffusion(nn.Module):
    def __init__(self, eps_model, n_steps=config.N_STEPS, device=None):
        super().__init__()
        self.eps_model = eps_model
        self.device = device if device is not None else torch.device("cpu")

        self.beta = torch.linspace(0.0001, 0.02, n_steps).to(self.device)
        self.alpha = 1 - self.beta
        self.alpha_bar = torch.cumprod(self.alpha, dim=0)
        self.n_steps = n_steps
        self.sqrt_alpha_bar = torch.sqrt(self.alpha_bar)
        self.sqrt_one_minus_alpha_bar = torch.sqrt(1 - self.alpha_bar)

        self.alphas_cumprod_prev = F.pad(self.alpha_bar[:-1], (1, 0), value=1.0)
        self.post_variance = self.beta * (1. - self.alphas_cumprod_prev) / (1. - self.alpha_bar)

    def q_sample(self, x0, t, eps=None):
        if eps is None:
            eps = torch.randn_like(x0)
        a_bar = self.sqrt_alpha_bar[t].reshape(-1, 1, 1, 1)
        one_minus = self.sqrt_one_minus_alpha_bar[t].reshape(-1, 1, 1, 1)
        return a_bar * x0 + one_minus * eps

    def p_sample(self, xt, t, c=None):
        if not isinstance(t, torch.Tensor):
            t = torch.tensor([t] * xt.shape[0], device=xt.device, dtype=torch.long)

        eps_theta = self.eps_model(xt, t, c)

        alpha_t = self.alpha[t].reshape(-1, 1, 1, 1)
        alpha_bar_t = self.alpha_bar[t].reshape(-1, 1, 1, 1)
        alpha_bar_t_prev = self.alphas_cumprod_prev[t].reshape(-1, 1, 1, 1)

        x0_pred = (xt - self.sqrt_one_minus_alpha_bar[t].reshape(-1, 1, 1, 1) * eps_theta) / self.sqrt_alpha_bar[t].reshape(-1, 1, 1, 1)
        x0_pred = torch.clamp(x0_pred, -1., 1.)

        mean = (alpha_bar_t_prev.sqrt() * self.beta[t].reshape(-1, 1, 1, 1) / (1. - alpha_bar_t)) * x0_pred + \
               (alpha_t.sqrt() * (1. - alpha_bar_t_prev) / (1. - alpha_bar_t)) * xt

        variance = self.post_variance[t].reshape(-1, 1, 1, 1)

        if t[0] > 0:
            noise = torch.randn_like(xt)
            return mean + torch.sqrt(variance) * noise
        else:
            return mean

    def sample(self, shape, device, c=None):
        x = torch.randn(shape, device=device)
        for t in tqdm(reversed(range(self.n_steps)), desc="Sampling"):
            x = self.p_sample(x, t, c)
        return x

    def loss(self, x0, labels=None):
        batch_size = x0.shape[0]
        t = torch.randint(0, self.n_steps, (batch_size,), device=x0.device, dtype=torch.long)
        eps = torch.randn_like(x0)
        xt = self.q_sample(x0, t, eps)
        eps_theta = self.eps_model(xt, t, labels)
        return F.mse_loss(eps, eps_theta)
