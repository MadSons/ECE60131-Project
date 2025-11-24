import torch
import torch.nn as nn
import torch.nn.functional as F


class TrajectoryCVAE(nn.Module):
    """
    Conditional VAE for trajectory prediction:
        p(future | past, scene) with latent z.

    Training:
        q(z | past, future, scene)  (encoder)
        p(future | past, scene, z)  (decoder)

    Inference / generation:
        sample z ~ N(0, I) (or p(z | past, scene) ~ N(mu, I) using encoder without future)
        decode future autoregressively.
    """

    def __init__(self, settings):
        super().__init__()

        self.use_cuda = settings["use_cuda"]
        self.dim_embedding_key = settings["dim_embedding_key"]
        self.past_len = settings["past_len"]
        self.future_len = settings["future_len"]
        self.z_dim = settings.get("z_dim", 16)

        # === Trajectory encoders (similar to MANTRA AE) ===
        channel_in = 2
        channel_out = 16
        dim_kernel = 3

        self.conv_past = nn.Conv1d(channel_in, channel_out, dim_kernel, stride=1, padding=1)
        self.conv_fut = nn.Conv1d(channel_in, channel_out, dim_kernel, stride=1, padding=1)

        self.encoder_past = nn.GRU(channel_out, self.dim_embedding_key, 1, batch_first=True)
        self.encoder_fut = nn.GRU(channel_out, self.dim_embedding_key, 1, batch_first=True)

        # === Scene encoder (small CNN + GRU-ish collapse) ===
        # scene_one_hot shape: (B, H, W, 4) → permute to (B, 4, H, W)
        self.convScene_1 = nn.Sequential(
            nn.Conv2d(4, 8, kernel_size=5, stride=2, padding=2),
            nn.ReLU(),
            nn.BatchNorm2d(8),
        )
        self.convScene_2 = nn.Sequential(
            nn.Conv2d(8, 16, kernel_size=5, stride=2, padding=2),
            nn.ReLU(),
            nn.BatchNorm2d(16),
        )
        # Global pooling + FC to dim_embedding_key
        self.fc_scene = nn.Linear(16, self.dim_embedding_key)

        # === Latent encoder: (past_enc, future_enc, scene_enc) → (mu, logvar) ===
        enc_in_dim = self.dim_embedding_key * 3
        self.fc_mu = nn.Linear(enc_in_dim, self.z_dim)
        self.fc_logvar = nn.Linear(enc_in_dim, self.z_dim)
        
        # === Conditional Prior: p(z | past, scene) ===
        prior_in_dim = self.dim_embedding_key * 2  # past_enc + scene_enc
        self.fc_prior_mu = nn.Linear(prior_in_dim, self.z_dim)
        self.fc_prior_logvar = nn.Linear(prior_in_dim, self.z_dim)

        # === Decoder ===
        # We'll concatenate (past_enc, scene_enc, z) as a condition vector
        self.cond_dim = self.dim_embedding_key * 2 + self.z_dim

        self.decoder = nn.GRU(self.cond_dim, self.cond_dim, 1, batch_first=False)
        self.FC_output = nn.Linear(self.cond_dim, 2)  # displacement

        self.relu = nn.ReLU()
        self._reset_parameters()

    def _reset_parameters(self):
        # Kaiming init where sensible; zeros for biases
        for m in [self.conv_past, self.conv_fut]:
            nn.init.kaiming_normal_(m.weight)
            nn.init.zeros_(m.bias)

        for gru in [self.encoder_past, self.encoder_fut, self.decoder]:
            for name, param in gru.named_parameters():
                if "weight" in name:
                    nn.init.kaiming_normal_(param)
                elif "bias" in name:
                    nn.init.zeros_(param)

        for m in [self.fc_scene, self.fc_mu, self.fc_logvar, self.FC_output]:
            nn.init.kaiming_normal_(m.weight)
            nn.init.zeros_(m.bias)

    # ---- Encoders ----

    def encode_traj(self, past, future=None):
        """
        Encode past and (optionally) future trajectories.
        past:   (B, T_p, 2)
        future: (B, T_f, 2) or None
        returns:
            past_enc:  (B, D)
            future_enc:(B, D) if future is not None else zeros
        """
        # Past
        x_p = past.transpose(1, 2)          # (B, 2, T_p)
        x_p = self.relu(self.conv_past(x_p))
        x_p = x_p.transpose(1, 2)          # (B, T_p, C)
        _, h_p = self.encoder_past(x_p)    # h_p: (1, B, D)
        past_enc = h_p.squeeze(0)          # (B, D)

        if future is not None:
            x_f = future.transpose(1, 2)   # (B, 2, T_f)
            x_f = self.relu(self.conv_fut(x_f))
            x_f = x_f.transpose(1, 2)      # (B, T_f, C)
            _, h_f = self.encoder_fut(x_f) # (1, B, D)
            future_enc = h_f.squeeze(0)
        else:
            future_enc = torch.zeros_like(past_enc)

        return past_enc, future_enc

    def encode_scene(self, scene_one_hot):
        """
        scene_one_hot: (B, H, W, 4)
        returns scene_enc: (B, D)
        """
        x = scene_one_hot.permute(0, 3, 1, 2)  # (B, 4, H, W)
        x = self.convScene_1(x)                # (B, 8, H', W')
        x = self.convScene_2(x)                # (B, 16, H'', W'')
        # Global average pooling over H'', W''
        x = x.mean(dim=[2, 3])                 # (B, 16)
        x = self.fc_scene(x)                   # (B, D)
        return self.relu(x)

    def encode_latent(self, past_enc, future_enc, scene_enc):
        """
        Build q(z | past, future, scene) and sample z.
        """
        h = torch.cat([past_enc, future_enc, scene_enc], dim=-1)  # (B, 3D)
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        return mu, logvar
    
    def encode_prior(self, past_enc, scene_enc):
        """
        Compute p(z | past, scene).
        Returns mu_prior, logvar_prior.
        """
        h = torch.cat([past_enc, scene_enc], dim=-1)  # (B, 2D)
        mu_prior = self.fc_prior_mu(h)
        logvar_prior = self.fc_prior_logvar(h)
        return mu_prior, logvar_prior

    def reparameterize(self, mu, logvar):
        if self.training:
            std = torch.exp(0.5 * logvar)
            eps = torch.randn_like(std)
            return mu + eps * std
        else:
            # At eval time, you can use mean or still sample for diversity
            return mu

    # ---- Decoder ----

    def decode(self, past, z, past_enc, scene_enc):
        """
        Decode future trajectory autoregressively given z, past_enc, scene_enc.
        past:      (B, T_p, 2)
        z:         (B, z_dim)
        past_enc:  (B, D)
        scene_enc: (B, D)

        returns:
            pred: (B, T_f, 2)
        """
        B = past.size(0)
        device = past.device

        cond = torch.cat([past_enc, scene_enc, z], dim=-1)  # (B, cond_dim)

        zero_padding = torch.zeros(1, B, self.cond_dim, device=device)
        present = past[:, -1, :2].unsqueeze(1)  # (B, 1, 2)

        input_dec = cond.unsqueeze(0)  # (1, B, cond_dim)
        state_dec = zero_padding

        preds = []

        for _ in range(self.future_len):
            out, state_dec = self.decoder(input_dec, state_dec)  # out: (1, B, cond_dim)
            disp_next = self.FC_output(out)                     # (1, B, 2)
            coords_next = present + disp_next.squeeze(0).unsqueeze(1)  # (B, 1, 2)
            preds.append(coords_next)
            present = coords_next
            input_dec = zero_padding  # only condition at first step

        pred = torch.cat(preds, dim=1)  # (B, T_f, 2)
        return pred

    # ---- Main forward ----
    def forward(self, past, future, scene_one_hot):
        """
        Training forward:
            Compute posterior q(z|past,future,scene)
            Compute prior     p(z|past,scene)
            Sample z ~ q
        Returns:
            pred, mu_post, logvar_post, mu_prior, logvar_prior
        """
        past_enc, future_enc = self.encode_traj(past, future)
        scene_enc = self.encode_scene(scene_one_hot)

        # Posterior q(z|past,future,scene)
        mu_post, logvar_post = self.encode_latent(past_enc, future_enc, scene_enc)
        z = self.reparameterize(mu_post, logvar_post)

        # Prior p(z|past,scene)
        mu_prior, logvar_prior = self.encode_prior(past_enc, scene_enc)

        # Decode with z
        pred = self.decode(past, z, past_enc, scene_enc)

        return pred, mu_post, logvar_post, mu_prior, logvar_prior


    @torch.no_grad()
    def sample(self, past, scene_one_hot, K=5):
        """
        Generate samples using the conditional prior:
            z ~ p(z | past, scene)
        """
        self.eval()
        B = past.size(0)
        device = past.device

        past_enc, _ = self.encode_traj(past, future=None)
        scene_enc = self.encode_scene(scene_one_hot)

        mu_prior, logvar_prior = self.encode_prior(past_enc, scene_enc)
        std_prior = torch.exp(0.5 * logvar_prior)

        samples = []

        for _ in range(K):
            eps = torch.randn_like(std_prior)
            z = mu_prior + eps * std_prior   # sample from conditional prior
            pred = self.decode(past, z, past_enc, scene_enc)
            samples.append(pred.unsqueeze(1))

        return torch.cat(samples, dim=1)  # (B, K, T_f, 2)

