import os
import matplotlib.pyplot as plt
import datetime
import io
from PIL import Image
from torchvision.transforms import ToTensor
import json
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tensorboardX import SummaryWriter
import dataset_invariance
from model_cvae_mdn import CVAE_MDN_Predictor
import tqdm
import argparse


class TrainerCVAE:
    def __init__(self, config):
        """
        Trainer class for CVAE-MDN trajectory prediction model
        :param config: configuration parameters
        """
        # Device setup
        self.device = torch.device(f'cuda:{config.device}' if torch.cuda.is_available() and config.cuda else 'cpu')
        if torch.cuda.is_available() and config.cuda:
            torch.cuda.set_device(config.device)
        
        print(f'Using device: {self.device}')
        
        # Create test folder
        self.name_test = str(datetime.datetime.now())[:19].replace(' ', '_').replace(':', '-')
        self.folder_tensorboard = 'runs/runs-cvae/'
        self.folder_test = 'training/training_cvae/' + self.name_test + '_' + config.info
        if not os.path.exists(self.folder_test):
            os.makedirs(self.folder_test)
        self.folder_test = self.folder_test + '/'
        
        print(f'Saving results to: {self.folder_test}')
        
        self.file = open(self.folder_test + "details.txt", "w")
        
        print('Creating dataset...')
        tracks = json.load(open(config.dataset_file))
        self.dim_clip = 180
        
        self.data_train = dataset_invariance.TrackDataset(
            tracks,
            len_past=config.past_len,
            len_future=config.future_len,
            train=True,
            dim_clip=self.dim_clip
        )
        
        self.train_loader = DataLoader(
            self.data_train,
            batch_size=config.batch_size,
            num_workers=config.num_workers,
            pin_memory=True,
            shuffle=True
        )
        
        self.data_test = dataset_invariance.TrackDataset(
            tracks,
            len_past=config.past_len,
            len_future=config.future_len,
            train=False,
            dim_clip=self.dim_clip
        )
        
        self.test_loader = DataLoader(
            self.data_test,
            batch_size=config.batch_size,
            num_workers=config.num_workers,
            pin_memory=True,
            shuffle=False
        )
        print(f'Dataset created: {len(self.data_train)} train samples, {len(self.data_test)} test samples')
        
        # Model settings
        self.settings = {
            "batch_size": config.batch_size,
            "use_cuda": config.cuda,
            "past_len": config.past_len,
            "future_len": config.future_len,
            "latent_dim": config.latent_dim,
            "hidden_dim": config.hidden_dim,
            "num_mixtures": config.num_mixtures,
            "use_scene": config.use_scene
        }
        
        self.max_epochs = config.max_epochs
        self.num_predictions = config.num_predictions
        
        # Create model
        print('Creating model...')
        self.model = CVAE_MDN_Predictor(self.settings).to(self.device)
        print(f'Model created with {sum(p.numel() for p in self.model.parameters())} parameters')
        
        # Optimizer with learning rate scheduling
        self.opt = torch.optim.Adam(self.model.parameters(), lr=config.learning_rate)
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.opt, mode='min', factor=0.5, patience=5
        )
        
        # KL annealing parameters
        self.beta_start = config.beta_start
        self.beta_end = config.beta_end
        self.beta_anneal_epochs = config.beta_anneal_epochs
        
        self.iterations = 0
        self.start_epoch = 0
        self.config = config
        self.best_val_loss = float('inf')
        
        # Write details to file
        self.write_details()
        self.file.close()
        
        # Tensorboard summary
        self.writer = SummaryWriter(self.folder_tensorboard + self.name_test + '_' + config.info)
        self.writer.add_text('Training Configuration', f'model: {self.model.name_model}', 0)
        self.writer.add_text('Training Configuration', f'train size: {len(self.data_train)}', 0)
        self.writer.add_text('Training Configuration', f'test size: {len(self.data_test)}', 0)
        self.writer.add_text('Training Configuration', f'batch_size: {config.batch_size}', 0)
        self.writer.add_text('Training Configuration', f'learning_rate: {config.learning_rate}', 0)
        self.writer.add_text('Training Configuration', f'latent_dim: {config.latent_dim}', 0)
        self.writer.add_text('Training Configuration', f'num_mixtures: {config.num_mixtures}', 0)
    
    def write_details(self):
        """Serialize configuration parameters to file"""
        self.file.write(f'Model: CVAE-MDN\n')
        self.file.write(f'Past length: {self.config.past_len}\n')
        self.file.write(f'Future length: {self.config.future_len}\n')
        self.file.write(f'Train size: {len(self.data_train)}\n')
        self.file.write(f'Test size: {len(self.data_test)}\n')
        self.file.write(f'Batch size: {self.config.batch_size}\n')
        self.file.write(f'Learning rate: {self.config.learning_rate}\n')
        self.file.write(f'Latent dim: {self.config.latent_dim}\n')
        self.file.write(f'Hidden dim: {self.config.hidden_dim}\n')
        self.file.write(f'Num mixtures: {self.config.num_mixtures}\n')
        self.file.write(f'Use scene: {self.config.use_scene}\n')
        self.file.write(f'Beta annealing: {self.beta_start} -> {self.beta_end} over {self.beta_anneal_epochs} epochs\n')
    
    def get_beta(self, epoch):
        """Get beta value for KL annealing"""
        if epoch >= self.beta_anneal_epochs:
            return self.beta_end
        return self.beta_start + (self.beta_end - self.beta_start) * (epoch / self.beta_anneal_epochs)
    
    def draw_track(self, past, future, pred=None, index_tracklet=0, num_epoch=0, train=False):
        """Plot and save trajectory to tensorboard"""
        fig = plt.figure(figsize=(8, 8))
        past = past.cpu().numpy()
        future = future.cpu().numpy()
        
        plt.plot(past[:, 0], past[:, 1], c='blue', marker='o', markersize=3, label='Past', linewidth=2)
        plt.plot(future[:, 0], future[:, 1], c='green', marker='o', markersize=3, label='Ground Truth', linewidth=2)
        
        if pred is not None:
            pred = pred.cpu().numpy()
            # Plot multiple predictions with varying transparency
            for i in range(pred.shape[0]):
                alpha = 1.0 - (i / pred.shape[0]) * 0.7
                label = 'Predictions' if i == 0 else None
                plt.plot(pred[i, :, 0], pred[i, :, 1], color='red', 
                        linewidth=1, marker='o', markersize=1, alpha=alpha, label=label)
        
        plt.legend()
        plt.axis('equal')
        plt.grid(True, alpha=0.3)
        plt.title(f'Epoch {num_epoch}, Sample {index_tracklet}')
        plt.xlabel('X (meters)')
        plt.ylabel('Y (meters)')
        
        # Save to tensorboard
        buf = io.BytesIO()
        plt.savefig(buf, format='jpeg', bbox_inches='tight')
        buf.seek(0)
        image = Image.open(buf)
        image = ToTensor()(image).unsqueeze(0)
        
        if train:
            self.writer.add_image(f'Image_train/track_{index_tracklet}', image.squeeze(0), num_epoch)
        else:
            self.writer.add_image(f'Image_test/track_{index_tracklet}', image.squeeze(0), num_epoch)
        
        plt.close(fig)
    
    def fit(self):
        """Main training loop"""
        print('\n' + '='*50)
        print('Starting training...')
        print('='*50 + '\n')
        
        for epoch in range(self.start_epoch, self.max_epochs):
            print(f'\n----- Epoch: {epoch + 1}/{self.max_epochs} -----')
            
            # Get current beta for KL annealing
            beta = self.get_beta(epoch)
            print(f'Beta (KL weight): {beta:.4f}')
            
            # Train one epoch
            train_loss, train_recon, train_kl = self._train_single_epoch(beta)
            print(f'Train Loss: {train_loss:.4f} (Recon: {train_recon:.4f}, KL: {train_kl:.4f})')
            
            # Log to tensorboard
            self.writer.add_scalar('loss_epoch/total', train_loss, epoch)
            self.writer.add_scalar('loss_epoch/reconstruction', train_recon, epoch)
            self.writer.add_scalar('loss_epoch/kl', train_kl, epoch)
            self.writer.add_scalar('beta', beta, epoch)
            self.writer.add_scalar('learning_rate', self.opt.param_groups[0]['lr'], epoch)
            
            # Evaluate every 5 epochs
            if (epoch + 1) % 5 == 0:

                print('Evaluating on test set...')
                dict_metrics_test = self.evaluate(self.test_loader, epoch + 1, train=False)
                
                # Print metrics
                print(f"Test  - ADE: {dict_metrics_test['eucl_mean']:.3f}m, FDE@4s: {dict_metrics_test['horizon40s']:.3f}m")
                
                # Test metrics
                self.writer.add_scalar('accuracy_test/ADE', dict_metrics_test['eucl_mean'], epoch)
                self.writer.add_scalar('accuracy_test/FDE_1s', dict_metrics_test['horizon10s'], epoch)
                self.writer.add_scalar('accuracy_test/FDE_2s', dict_metrics_test['horizon20s'], epoch)
                self.writer.add_scalar('accuracy_test/FDE_3s', dict_metrics_test['horizon30s'], epoch)
                self.writer.add_scalar('accuracy_test/FDE_4s', dict_metrics_test['horizon40s'], epoch)
                
                # Learning rate scheduling
                self.scheduler.step(dict_metrics_test['eucl_mean'])
                
                # Save best model
                val_loss = dict_metrics_test['eucl_mean']
                if val_loss < self.best_val_loss:
                    self.best_val_loss = val_loss
                    model_path = self.folder_test + f'model_cvae_best_{self.name_test}.pt'
                    torch.save(self.model.state_dict(), model_path)
                    print(f'✓ Best model saved with ADE: {val_loss:.4f}m')
                
                # Save checkpoint every 20 epochs
                if (epoch + 1) % 20 == 0:
                    checkpoint_path = self.folder_test + f'model_cvae_epoch_{epoch+1}_{self.name_test}.pt'
                    torch.save(self.model.state_dict(), checkpoint_path)
                    print(f'✓ Checkpoint saved at epoch {epoch+1}')
        
        # Save final model
        final_path = self.folder_test + f'model_cvae_final_{self.name_test}.pt'
        torch.save(self.model.state_dict(), final_path)
        print('\n' + '='*50)
        print('Training complete!')
        print(f'Best model ADE: {self.best_val_loss:.4f}m')
        print(f'Results saved to: {self.folder_test}')
        print('='*50)
        
        self.writer.close()
    
    def _train_single_epoch(self, beta):
        """Training loop for one epoch"""
        self.model.train()
        total_loss = 0
        total_recon = 0
        total_kl = 0
        
        for step, (index, past, future, presents, angle_presents, videos, 
                   vehicles, number_vec, scene, scene_one_hot) in enumerate(tqdm.tqdm(self.train_loader)):
            
            self.iterations += 1
            past = past.to(self.device)
            future = future.to(self.device)
            scene_one_hot = scene_one_hot.to(self.device) if self.config.use_scene else None
            
            self.opt.zero_grad()
            
            # Forward pass
            pred, mu_post, logvar_post, mu_prior, logvar_prior, mdn_params = \
                self.model(past, future, scene_one_hot)
            
            # Compute loss
            loss, recon_loss, kl_loss = self.model.compute_loss(
                pred, future, mu_post, logvar_post, mu_prior, logvar_prior, mdn_params, beta
            )
            
            # Backward pass
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            self.opt.step()
            
            # Accumulate losses
            total_loss += loss.item()
            total_recon += recon_loss.item()
            total_kl += kl_loss.item()
            
            # Log to tensorboard every 100 iterations
            if self.iterations % 100 == 0:
                self.writer.add_scalar('loss/total', loss.item(), self.iterations)
                self.writer.add_scalar('loss/reconstruction', recon_loss.item(), self.iterations)
                self.writer.add_scalar('loss/kl', kl_loss.item(), self.iterations)
        
        return (total_loss / len(self.train_loader), 
                total_recon / len(self.train_loader),
                total_kl / len(self.train_loader))
    
    def evaluate(self, loader, epoch=0, train=False):
        """Evaluate model on dataset"""
        self.model.eval()
        
        eucl_mean = horizon10s = horizon20s = horizon30s = horizon40s = 0
        ADE_1s = ADE_2s = ADE_3s = 0
        
        num_samples = self.num_predictions
        
        with torch.no_grad():
            for step, (index, past, future, presents, angle_presents, videos,
                      vehicles, number_vec, scene, scene_one_hot) in enumerate(tqdm.tqdm(loader)):
                
                past = past.to(self.device)
                future = future.to(self.device)
                scene_one_hot = scene_one_hot.to(self.device) if self.config.use_scene else None
                
                # Generate multiple predictions
                pred = self.model(past, scene=scene_one_hot, num_samples=num_samples)
                
                # Compute best prediction errors (min over samples)
                future_rep = future.unsqueeze(1).repeat(1, num_samples, 1, 1)
                distances = torch.norm(pred - future_rep, dim=3)
                mean_distances = torch.mean(distances, dim=2)
                index_min = torch.argmin(mean_distances, dim=1)
                min_distances = distances[torch.arange(len(index_min)), index_min]
                
                # Accumulate metrics
                eucl_mean += torch.sum(torch.mean(min_distances, 1))
                ADE_1s += torch.sum(torch.mean(min_distances[:, :10], 1))
                ADE_2s += torch.sum(torch.mean(min_distances[:, :20], 1))
                ADE_3s += torch.sum(torch.mean(min_distances[:, :30], 1))
                horizon10s += torch.sum(min_distances[:, 9])
                horizon20s += torch.sum(min_distances[:, 19])
                horizon30s += torch.sum(min_distances[:, 29])
                horizon40s += torch.sum(min_distances[:, 39])
                
                # Draw first sample of first batch
                if step == 0:
                    self.draw_track(past[0], future[0], pred[0], 
                                  index_tracklet=step, num_epoch=epoch, train=train)
        
        # Compute averages
        dict_metrics = {
            'eucl_mean': eucl_mean / len(loader.dataset),
            'ADE_1s': ADE_1s / len(loader.dataset),
            'ADE_2s': ADE_2s / len(loader.dataset),
            'ADE_3s': ADE_3s / len(loader.dataset),
            'horizon10s': horizon10s / len(loader.dataset),
            'horizon20s': horizon20s / len(loader.dataset),
            'horizon30s': horizon30s / len(loader.dataset),
            'horizon40s': horizon40s / len(loader.dataset)
        }
        
        return dict_metrics


# Main entry point
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train CVAE-MDN trajectory predictor')
    
    # Dataset parameters
    parser.add_argument('--dataset_file', type=str, default='kitti_dataset.json',
                       help='Path to dataset JSON file')
    
    # Training parameters
    parser.add_argument('--device', type=int, default=0,
                       help='GPU device ID')
    parser.add_argument('--cuda', action='store_true', default=False,
                       help='Use CUDA (add this flag to enable GPU)')
    parser.add_argument('--batch_size', type=int, default=32,
                       help='Batch size for training')
    parser.add_argument('--max_epochs', type=int, default=100,
                       help='Maximum number of training epochs')
    parser.add_argument('--learning_rate', type=float, default=0.001,
                       help='Initial learning rate')
    parser.add_argument('--num_workers', type=int, default=4,
                       help='Number of data loading workers')
    
    # Trajectory parameters
    parser.add_argument('--past_len', type=int, default=20,
                       help='Number of past trajectory points (2 seconds)')
    parser.add_argument('--future_len', type=int, default=40,
                       help='Number of future trajectory points (4 seconds)')
    
    # Model architecture parameters
    parser.add_argument('--latent_dim', type=int, default=32,
                       help='Dimension of latent space')
    parser.add_argument('--hidden_dim', type=int, default=128,
                       help='Hidden dimension for GRU layers')
    parser.add_argument('--num_mixtures', type=int, default=5,
                       help='Number of Gaussian mixtures in MDN')
    parser.add_argument('--num_predictions', type=int, default=10,
                       help='Number of trajectory predictions to generate')
    
    # Scene context
    parser.add_argument('--use_scene', action='store_true', default=False,
                       help='Use scene context (add this flag to enable)')
    
    # KL annealing parameters
    parser.add_argument('--beta_start', type=float, default=0.0,
                       help='Initial beta value for KL term')
    parser.add_argument('--beta_end', type=float, default=1.0,
                       help='Final beta value for KL term')
    parser.add_argument('--beta_anneal_epochs', type=int, default=20,
                       help='Number of epochs for beta annealing')
    
    # Experiment info
    parser.add_argument('--info', type=str, default='cvae_mdn',
                       help='Experiment name/description')
    
    config = parser.parse_args()
    
    # Print configuration
    print('\n' + '='*50)
    print('CVAE-MDN Trajectory Predictor - Training')
    print('='*50)
    print(f'Dataset: {config.dataset_file}')
    print(f'Device: {"cuda:" + str(config.device) if config.cuda else "cpu"}')
    print(f'Batch size: {config.batch_size}')
    print(f'Epochs: {config.max_epochs}')
    print(f'Learning rate: {config.learning_rate}')
    print(f'Latent dim: {config.latent_dim}')
    print(f'Hidden dim: {config.hidden_dim}')
    print(f'Num mixtures: {config.num_mixtures}')
    print(f'Use scene: {config.use_scene}')
    print(f'Beta annealing: {config.beta_start} -> {config.beta_end} over {config.beta_anneal_epochs} epochs')
    print(f'Experiment: {config.info}')
    print('='*50 + '\n')
    
    # Create trainer and start training
    trainer = TrainerCVAE(config)
    trainer.fit()