import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class CVAE_MDN_Predictor(nn.Module):
    """
    Conditional Variational Autoencoder with Mixture Density Network for trajectory prediction.
    Combines the strength of CVAEs for learning latent representations with MDN for multimodal outputs.
    """
    def __init__(self, settings):
        super(CVAE_MDN_Predictor, self).__init__()
        
        self.name_model = 'CVAE_MDN'
        self.use_cuda = settings["use_cuda"]
        self.past_len = settings["past_len"]
        self.future_len = settings["future_len"]
        self.latent_dim = settings.get("latent_dim", 32)
        self.hidden_dim = settings.get("hidden_dim", 128)
        self.num_mixtures = settings.get("num_mixtures", 5)  # Number of Gaussian components
        
        channel_in = 2
        channel_out = 16
        dim_kernel = 3
        
        # Past trajectory encoder (shared for both inference and generation)
        self.conv_past = nn.Conv1d(channel_in, channel_out, dim_kernel, stride=1, padding=1)
        self.encoder_past = nn.GRU(channel_out, self.hidden_dim, 1, batch_first=True)
        
        # Future trajectory encoder (only used during training for inference network)
        self.conv_fut = nn.Conv1d(channel_in, channel_out, dim_kernel, stride=1, padding=1)
        self.encoder_fut = nn.GRU(channel_out, self.hidden_dim, 1, batch_first=True)
        
        # Scene encoder (optional, for context)
        self.use_scene = settings.get("use_scene", True)
        if self.use_scene:
            self.convScene_1 = nn.Sequential(
                nn.Conv2d(4, 8, kernel_size=5, stride=2, padding=2),
                nn.ReLU(),
                nn.BatchNorm2d(8)
            )
            self.convScene_2 = nn.Sequential(
                nn.Conv2d(8, 16, kernel_size=5, stride=1, padding=2),
                nn.ReLU(),
                nn.BatchNorm2d(16)
            )
            #self.scene_fc = nn.Linear(16 * self.scene_size * self.scene_size, self.hidden_dim)
            self.scene_fc = nn.Linear(16, self.hidden_dim)
            
        # Inference network q(z|x,y): encodes to latent distribution given past and future
        context_dim = self.hidden_dim * 2  # past + future
        if self.use_scene:
            context_dim += self.hidden_dim  # add scene
            
        self.fc_mu = nn.Linear(context_dim, self.latent_dim)
        self.fc_logvar = nn.Linear(context_dim, self.latent_dim)
        
        # Prior network p(z|x): encodes to latent distribution given only past
        prior_dim = self.hidden_dim
        if self.use_scene:
            prior_dim += self.hidden_dim
            
        self.fc_prior_mu = nn.Linear(prior_dim, self.latent_dim)
        self.fc_prior_logvar = nn.Linear(prior_dim, self.latent_dim)
        
        # Decoder network p(y|z,x): decodes latent + past to future trajectory
        decoder_input_dim = self.latent_dim + self.hidden_dim
        if self.use_scene:
            decoder_input_dim += self.hidden_dim
            
        self.decoder_fc = nn.Linear(decoder_input_dim, self.hidden_dim)
        self.decoder_gru = nn.GRU(self.hidden_dim, self.hidden_dim, 1, batch_first=False)
        
        # Mixture Density Network output
        # For each mixture: mean_x, mean_y, sigma_x, sigma_y, rho (correlation), pi (weight)
        self.mdn_fc = nn.Linear(self.hidden_dim, self.num_mixtures * 6)
        
        # Activation functions
        self.relu = nn.ReLU()
        self.tanh = nn.Tanh()
        
        # Initialize weights
        self.reset_parameters()
        
    def reset_parameters(self):
        """Kaiming initialization for better gradient flow"""
        for m in self.modules():
            if isinstance(m, nn.Conv1d) or isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.GRU):
                for name, param in m.named_parameters():
                    if 'weight_ih' in name:
                        nn.init.kaiming_normal_(param)
                    elif 'weight_hh' in name:
                        nn.init.orthogonal_(param)
                    elif 'bias' in name:
                        nn.init.zeros_(param)
    
    def encode_past(self, past):
        """Encode past trajectory"""
        past = torch.transpose(past, 1, 2)
        past_embed = self.relu(self.conv_past(past))
        past_embed = torch.transpose(past_embed, 1, 2)
        output_past, state_past = self.encoder_past(past_embed)
        return state_past.squeeze(0)
    
    def encode_future(self, future):
        """Encode future trajectory (only used during training)"""
        future = torch.transpose(future, 1, 2)
        future_embed = self.relu(self.conv_fut(future))
        future_embed = torch.transpose(future_embed, 1, 2)
        output_fut, state_fut = self.encoder_fut(future_embed)
        return state_fut.squeeze(0)
    
    def encode_scene(self, scene):
        """Encode scene context"""
        if not self.use_scene:
            return None
        scene = scene.permute(0, 3, 1, 2)  # (B, H, W, C) -> (B, C, H, W)
        scene_1 = self.convScene_1(scene)
        scene_2 = self.convScene_2(scene_1)
        #scene_flat = scene_2.reshape(scene_2.size(0), -1)
        #scene_embed = self.relu(self.scene_fc(scene_flat))
        scene_pooled = scene_2.mean(dim=[2, 3])  # Global average pooling
        scene_embed = self.relu(self.scene_fc(scene_pooled))
        return scene_embed
    
    def reparameterize(self, mu, logvar):
        """Reparameterization trick for VAE"""
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std
    
    def inference_network(self, past_embed, future_embed, scene_embed=None):
        """
        Inference network q(z|x,y): predicts latent distribution given past and future
        Used only during training
        """
        if scene_embed is not None:
            context = torch.cat([past_embed, future_embed, scene_embed], dim=1)
        else:
            context = torch.cat([past_embed, future_embed], dim=1)
        
        mu = self.fc_mu(context)
        logvar = self.fc_logvar(context)
        return mu, logvar
    
    def prior_network(self, past_embed, scene_embed=None):
        """
        Prior network p(z|x): predicts latent distribution given only past
        Used during both training and inference
        """
        if scene_embed is not None:
            context = torch.cat([past_embed, scene_embed], dim=1)
        else:
            context = past_embed
        
        mu = self.fc_prior_mu(context)
        logvar = self.fc_prior_logvar(context)
        return mu, logvar
    
    def decode(self, z, past_embed, scene_embed=None):
        """
        Decoder p(y|z,x): generates future trajectory from latent code and past
        """
        if scene_embed is not None:
            decoder_input = torch.cat([z, past_embed, scene_embed], dim=1)
        else:
            decoder_input = torch.cat([z, past_embed], dim=1)
        
        hidden = self.relu(self.decoder_fc(decoder_input))
        hidden = hidden.unsqueeze(0)  # Add sequence dimension for GRU
        
        # Decode each timestep
        predictions = []
        present = torch.zeros(hidden.size(1), 2).to(hidden.device)
        
        for t in range(self.future_len):
            output, hidden = self.decoder_gru(hidden, hidden)
            
            # MDN output
            mdn_params = self.mdn_fc(output.squeeze(0))
            
            # Sample from mixture
            coords = self.sample_from_mdn(mdn_params)
            coords_next = present + coords
            predictions.append(coords_next.unsqueeze(1))
            present = coords_next
        
        return torch.cat(predictions, dim=1), mdn_params
    
    def sample_from_mdn(self, mdn_params):
        """Sample coordinates from Mixture Density Network"""
        batch_size = mdn_params.size(0)
        
        # Split parameters
        params = mdn_params.view(batch_size, self.num_mixtures, 6)
        
        mu_x = params[:, :, 0]
        mu_y = params[:, :, 1]
        sigma_x = torch.exp(params[:, :, 2])  # Ensure positive
        sigma_y = torch.exp(params[:, :, 3])
        rho = self.tanh(params[:, :, 4])  # Correlation in [-1, 1]
        pi_logits = params[:, :, 5]
        
        # Sample mixture component
        pi = F.softmax(pi_logits, dim=1)
        mixture_idx = torch.multinomial(pi, 1).squeeze(1)
        
        # Gather parameters for selected mixture
        batch_idx = torch.arange(batch_size, device=mdn_params.device)
        mu_x_sel = mu_x[batch_idx, mixture_idx]
        mu_y_sel = mu_y[batch_idx, mixture_idx]
        sigma_x_sel = sigma_x[batch_idx, mixture_idx]
        sigma_y_sel = sigma_y[batch_idx, mixture_idx]
        rho_sel = rho[batch_idx, mixture_idx]
        
        # Sample from bivariate Gaussian
        z = torch.randn(batch_size, 2, device=mdn_params.device)
        x = mu_x_sel + sigma_x_sel * z[:, 0]
        y = mu_y_sel + sigma_y_sel * (rho_sel * z[:, 0] + torch.sqrt(1 - rho_sel**2) * z[:, 1])
        
        return torch.stack([x, y], dim=1)
    
    def forward(self, past, future=None, scene=None, num_samples=1):
        """
        Forward pass
        Training: uses inference network with future
        Testing: uses prior network, samples multiple trajectories
        """
        
        # Encode inputs
        past_embed = self.encode_past(past)
        scene_embed = self.encode_scene(scene) if scene is not None else None
        
        if self.training and future is not None:
            # Training mode: use inference network
            future_embed = self.encode_future(future)
            
            # Get posterior q(z|x,y)
            mu_post, logvar_post = self.inference_network(past_embed, future_embed, scene_embed)
            z = self.reparameterize(mu_post, logvar_post)
            
            # Get prior p(z|x)
            mu_prior, logvar_prior = self.prior_network(past_embed, scene_embed)
            
            # Decode
            pred, mdn_params = self.decode(z, past_embed, scene_embed)
            
            return pred, mu_post, logvar_post, mu_prior, logvar_prior, mdn_params
        
        else:
            # Inference mode: sample from prior
            predictions = []
            
            for _ in range(num_samples):
                mu_prior, logvar_prior = self.prior_network(past_embed, scene_embed)
                z = self.reparameterize(mu_prior, logvar_prior)
                pred, _ = self.decode(z, past_embed, scene_embed)
                predictions.append(pred.unsqueeze(1))
            
            # Shape: (batch, num_samples, future_len, 2)
            return torch.cat(predictions, dim=1)
    
    def compute_loss(self, pred, future, mu_post, logvar_post, mu_prior, logvar_prior, mdn_params, beta=1.0):
        """
        Compute CVAE loss: reconstruction + KL divergence
        """
        # Reconstruction loss (MSE)
        recon_loss = F.mse_loss(pred, future, reduction='mean')
        
        # KL divergence between posterior and prior
        kl_loss = -0.5 * torch.sum(
            1 + logvar_post - logvar_prior 
            - ((mu_post - mu_prior).pow(2) + logvar_post.exp()) / logvar_prior.exp()
        ) / pred.size(0)
        
        # Total loss with beta weighting for KL
        total_loss = recon_loss + beta * kl_loss
        
        return total_loss, recon_loss, kl_loss