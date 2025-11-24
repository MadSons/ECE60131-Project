import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class CVAE_Predictor(nn.Module):
    """
    Conditional VAE for trajectory prediction:
        p(future | past, scene) = ∫ p(future | past, scene, z) p(z | past, scene) dz

    - Encoder: q(z | past, future, scene)
    - Conditional prior: p(z | past, scene)
    - Decoder: p(future | past, scene, z)

    This implementation predicts a single future trajectory given z
    (you get multi-modality by sampling multiple z's per past).
    """

    def __init__(
        self,
        past_len=20,
        future_len=40,
        dim_embedding_key=48,
        latent_dim=16,
        use_scene=True,
        use_cuda=True,
    ):
        super().__init__()

        self.past_len = past_len
        self.future_len = future_len
        self.dim_embedding_key = dim_embedding_key
        self.latent_dim = latent_dim
        self.use_scene = use_scene
        self.use_cuda = use_cuda

        channel_in = 2
        channel_out = 16
        dim_kernel = 3
        input_gru = channel_out

        # -------- Trajectory encoders (MANTRA-style) --------
        self.conv_past = nn.Conv1d(channel_in, channel_out, dim_kernel, stride=1, padding=1)
        self.conv_fut = nn.Conv1d(channel_in, channel_out, dim_kernel, stride=1, padding=1)

        self.encoder_past = nn.GRU(input_gru, dim_embedding_key, 1, batch_first=True)
        self.encoder_fut = nn.GRU(input_gru, dim_embedding_key, 1, batch_first=True)

        # -------- Scene encoder (similar spirit to IRM) --------
        if self.use_scene:
            # scene_one_hot: [B, H=180, W=180, C=4] -> permute to [B, C, H, W]
            self.convScene_1 = nn.Sequential(
                nn.Conv2d(4, 8, kernel_size=5, stride=2, padding=2),  # [B, 8, 90, 90]
                nn.ReLU(),
                nn.BatchNorm2d(8),
            )
            self.convScene_2 = nn.Sequential(
                nn.Conv2d(8, 16, kernel_size=5, stride=2, padding=2),  # [B, 16, 45, 45]
                nn.ReLU(),
                nn.BatchNorm2d(16),
            )
            # Flatten spatial to sequence of length 45*45 with 16 channels
            self.RNN_scene = nn.GRU(16, dim_embedding_key, 1, batch_first=True)
        else:
            self.convScene_1 = None
            self.convScene_2 = None
            self.RNN_scene = None

        # -------- Inference network q(z | past, future, scene) --------
        # q_input_dim = h_past + h_future (+ h_scene)
        enc_in_dim = dim_embedding_key + dim_embedding_key
        if self.use_scene:
            enc_in_dim += dim_embedding_key

        self.fc_q = nn.Linear(enc_in_dim, 2 * latent_dim)  # -> [mu_q, logvar_q]

        # -------- Conditional prior p(z | past, scene) --------
        # p_input_dim = h_past (+ h_scene)
        prior_in_dim = dim_embedding_key
        if self.use_scene:
            prior_in_dim += dim_embedding_key

        self.fc_prior = nn.Linear(prior_in_dim, 2 * latent_dim)  # -> [mu_p, logvar_p]

        # -------- Decoder p(future | past, scene, z) --------
        # GRU hidden state is initialized from (h_past, h_scene, z)
        dec_in_dim = dim_embedding_key + (dim_embedding_key if self.use_scene else 0) + latent_dim
        self.fc_dec_init = nn.Linear(dec_in_dim, 2 * dim_embedding_key)  # hidden size = 2 * D
        self.decoder = nn.GRU(2 * dim_embedding_key, 2 * dim_embedding_key, 1, batch_first=False)

        # Output layer to predict displacement
        self.FC_output = nn.Linear(2 * dim_embedding_key, 2)

        self.relu = nn.ReLU()

        self.reset_parameters()

    def reset_parameters(self):
        nn.init.kaiming_normal_(self.conv_past.weight)
        nn.init.kaiming_normal_(self.conv_fut.weight)
        nn.init.zeros_(self.conv_past.bias)
        nn.init.zeros_(self.conv_fut.bias)

        for gru in [self.encoder_past, self.encoder_fut]:
            nn.init.kaiming_normal_(gru.weight_ih_l0)
            nn.init.kaiming_normal_(gru.weight_hh_l0)
            nn.init.zeros_(gru.bias_ih_l0)
            nn.init.zeros_(gru.bias_hh_l0)

        if self.use_scene:
            nn.init.kaiming_normal_(self.convScene_1[0].weight)
            nn.init.zeros_(self.convScene_1[0].bias)
            nn.init.kaiming_normal_(self.convScene_2[0].weight)
            nn.init.zeros_(self.convScene_2[0].bias)

            nn.init.kaiming_normal_(self.RNN_scene.weight_ih_l0)
            nn.init.kaiming_normal_(self.RNN_scene.weight_hh_l0)
            nn.init.zeros_(self.RNN_scene.bias_ih_l0)
            nn.init.zeros_(self.RNN_scene.bias_hh_l0)

        for lin in [self.fc_q, self.fc_prior, self.fc_dec_init, self.FC_output]:
            nn.init.kaiming_normal_(lin.weight)
            nn.init.zeros_(lin.bias)

        nn.init.kaiming_normal_(self.decoder.weight_ih_l0)
        nn.init.kaiming_normal_(self.decoder.weight_hh_l0)
        nn.init.zeros_(self.decoder.bias_ih_l0)
        nn.init.zeros_(self.decoder.bias_hh_l0)
        
        

    # ------------------------------------------------------------------
    # Encoders
    # ------------------------------------------------------------------

    def encode_past(self, past):
        # past: [B, T_past, 2]
        x = past.transpose(1, 2)       # [B, 2, T]
        x = self.relu(self.conv_past(x))
        x = x.transpose(1, 2)          # [B, T, C]
        _, h_past = self.encoder_past(x)
        return h_past.squeeze(0)       # [B, D]

    def encode_future(self, future):
        # future: [B, T_future, 2]
        x = future.transpose(1, 2)
        x = self.relu(self.conv_fut(x))
        x = x.transpose(1, 2)
        _, h_fut = self.encoder_fut(x)
        return h_fut.squeeze(0)        # [B, D]

    def encode_scene(self, scene_one_hot):
        # scene_one_hot: [B, H=180, W=180, 4]
        if not self.use_scene:
            return None
        x = scene_one_hot.permute(0, 3, 1, 2).contiguous()  # [B, 4, H, W]
        x = self.convScene_1(x)                             # [B, 8, 90, 90]
        x = self.convScene_2(x)                             # [B,16, 45, 45]
        B, C, H, W = x.shape
        x = x.view(B, C, H * W).transpose(1, 2)             # [B, 45*45, 16]
        _, h_scene = self.RNN_scene(x)
        return h_scene.squeeze(0)                           # [B, D]

    # ------------------------------------------------------------------
    # Latent distributions
    # ------------------------------------------------------------------

    def inference_q(self, past, future, scene_one_hot):
        """
        q(z | past, future, scene) -> mu_q, logvar_q, h_past, h_scene
        """
        h_past = self.encode_past(past)
        h_fut = self.encode_future(future)
        h_scene = self.encode_scene(scene_one_hot) if self.use_scene else None

        if self.use_scene:
            h = torch.cat([h_past, h_fut, h_scene], dim=1)
        else:
            h = torch.cat([h_past, h_fut], dim=1)

        stats = self.fc_q(h)
        mu_q, logvar_q = stats.chunk(2, dim=1)
        return mu_q, logvar_q, h_past, h_scene

    def prior_p(self, h_past, h_scene):
        """
        p(z | past, scene) – conditional prior.
        h_past: [B, D]
        h_scene: [B, D] or None
        returns mu_p, logvar_p
        """
        if self.use_scene:
            h = torch.cat([h_past, h_scene], dim=1)
        else:
            h = h_past

        stats = self.fc_prior(h)
        mu_p, logvar_p = stats.chunk(2, dim=1)
        return mu_p, logvar_p

    @staticmethod
    def reparameterize(mu, logvar):
        """
        z = mu + sigma * eps, eps ~ N(0, I)
        """
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    # ------------------------------------------------------------------
    # Decoder
    # ------------------------------------------------------------------

    def decode(self, past, h_past, h_scene, z):
        B = past.size(0)

        if self.use_scene:
            dec_ctx = torch.cat([h_past, h_scene, z], dim=1)
        else:
            dec_ctx = torch.cat([h_past, z], dim=1)

        # Initialize GRU hidden state
        dec_init = self.fc_dec_init(dec_ctx)   # [B, 96]
        dec_init = dec_init.unsqueeze(0)       # [1, B, 96]

        # GRU input should be [1, B, 96]
        input_dec = torch.zeros(1, B, 2 * self.dim_embedding_key, device=past.device)

        present = past[:, -1, :].unsqueeze(1)  # [B, 1, 2]
        prediction = []

        state_dec = dec_init

        for t in range(self.future_len):
            out, state_dec = self.decoder(input_dec, state_dec)  # [1, B, 96]
            disp = self.FC_output(out)                           # [1, B, 2]
            coords_next = present + disp.squeeze(0).unsqueeze(1) # [B, 1, 2]
            prediction.append(coords_next)
            present = coords_next
            input_dec = torch.zeros_like(input_dec)

        return torch.cat(prediction, dim=1)

    # ------------------------------------------------------------------
    # Loss & forward helpers
    # ------------------------------------------------------------------

    @staticmethod
    def kl_divergence(mu_q, logvar_q, mu_p, logvar_p):
        """
        KL(q || p) for diagonal Gaussians:
          q = N(mu_q, sigma_q^2 I), p = N(mu_p, sigma_p^2 I)
        """
        # log(σ_p^2 / σ_q^2) + (σ_q^2 + (μ_q - μ_p)^2)/σ_p^2 - 1
        # summed over latent_dim
        var_q = torch.exp(logvar_q)
        var_p = torch.exp(logvar_p)
        term1 = logvar_p - logvar_q
        term2 = (var_q + (mu_q - mu_p) ** 2) / var_p
        kl = 0.5 * torch.sum(term1 + term2 - 1.0, dim=1)  # [B]
        return kl.mean()

    def compute_loss(self, past, future, scene_one_hot, beta=1.0):
        """
        Compute ELBO loss with β-annealing:
            L = E_q[||future - f(past, scene, z)||^2] + β * KL(q || p)

        :param past: [B, T_past, 2]
        :param future: [B, T_future, 2]
        :param scene_one_hot: [B, H, W, 4]
        :param beta: scalar weight for KL (β-annealing)
        """
        mu_q, logvar_q, h_past, h_scene = self.inference_q(past, future, scene_one_hot)
        mu_p, logvar_p = self.prior_p(h_past, h_scene)
        z = self.reparameterize(mu_q, logvar_q)

        future_pred = self.decode(past, h_past, h_scene, z)

        recon_loss = F.mse_loss(future_pred, future, reduction="mean")
        kl_loss = self.kl_divergence(mu_q, logvar_q, mu_p, logvar_p)

        loss = recon_loss + beta * kl_loss
        return loss, recon_loss.detach(), kl_loss.detach()

    def forward(self, past, future, scene_one_hot, beta=1.0):
        """
        For training: return loss terms.
        """
        return self.compute_loss(past, future, scene_one_hot, beta)

    # ------------------------------------------------------------------
    # Sampling (generation) using the conditional prior
    # ------------------------------------------------------------------

    def sample(self, past, scene_one_hot, num_samples=5):
        """
        Generate multiple trajectory samples from the conditional prior:

            z ~ p(z | past, scene)
            future ~ p(future | past, scene, z)

        :param past: [B, T_past, 2]
        :param scene_one_hot: [B, H, W, 4]
        :param num_samples: K
        :return: [B, K, T_future, 2]
        """
        self.eval()
        with torch.no_grad():
            h_past = self.encode_past(past)
            h_scene = self.encode_scene(scene_one_hot) if self.use_scene else None
            mu_p, logvar_p = self.prior_p(h_past, h_scene)

            # sample K z's per batch element
            B = past.size(0)
            futures = []
            for _ in range(num_samples):
                z = self.reparameterize(mu_p, logvar_p)
                fut = self.decode(past, h_past, h_scene, z)  # [B, T_future, 2]
                futures.append(fut.unsqueeze(1))  # [B, 1, T_future, 2]
            futures = torch.cat(futures, dim=1)  # [B, K, T_future, 2]
        return futures