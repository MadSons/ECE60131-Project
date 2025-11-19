import os
import matplotlib.pyplot as plt
import matplotlib.pylab as pl
from matplotlib.colors import LinearSegmentedColormap
import datetime
import cv2
import numpy as np
import json
import torch
from torch.utils.data import DataLoader
import dataset_invariance
import tqdm


class EvaluatorCVAE:
    def __init__(self, config):
        """
        Evaluation class for CVAE-MDN trajectory predictor
        :param config: configuration parameters
        """
        self.device = torch.device(f'cuda:{config.device}' if torch.cuda.is_available() and config.cuda else 'cpu')
        torch.cuda.set_device(config.device) if torch.cuda.is_available() and config.cuda else None
        
        # Create output folder
        self.name_test = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        self.folder_test = 'test/' + self.name_test + '_' + config.info + '/'
        os.makedirs(self.folder_test, exist_ok=True)
        
        print('Creating dataset...')
        self.dim_clip = 180
        tracks = json.load(open(config.dataset_file))
        
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
            num_workers=8,
            shuffle=False
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
            num_workers=8,
            shuffle=False
        )
        print('Dataset created')
        
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
        
        # Load model
        print(f'Loading model from {config.model}...')
        from model_cvae_mdn import CVAE_MDN_Predictor
        self.model = CVAE_MDN_Predictor(self.settings).to(self.device)
        self.model.load_state_dict(torch.load(config.model, map_location=self.device))
        self.model.eval()
        print('Model loaded successfully')
        
        self.config = config
    
    def evaluate(self):
        """Run evaluation on test set"""
        print('\nEvaluating model...')
        dict_metrics = self._evaluate_loader(self.test_loader, save_images=self.config.save_images)
        self.save_results(dict_metrics)
        
        print('\n===== RESULTS =====')
        print(f"ADE (Average Displacement Error): {dict_metrics['eucl_mean']:.3f}m")
        print(f"FDE @ 1s: {dict_metrics['horizon10s']:.3f}m")
        print(f"FDE @ 2s: {dict_metrics['horizon20s']:.3f}m")
        print(f"FDE @ 3s: {dict_metrics['horizon30s']:.3f}m")
        print(f"FDE @ 4s: {dict_metrics['horizon40s']:.3f}m")
        print(f"ADE @ 1s: {dict_metrics['ADE_1s']:.3f}m")
        print(f"ADE @ 2s: {dict_metrics['ADE_2s']:.3f}m")
        print(f"ADE @ 3s: {dict_metrics['ADE_3s']:.3f}m")
        print('=' * 50)
    
    def _evaluate_loader(self, loader, save_images=False):
        """
        Evaluate model on a data loader
        :param loader: PyTorch DataLoader
        :param save_images: whether to save visualization images
        :return: dictionary of metrics
        """
        eucl_mean = ADE_1s = ADE_2s = ADE_3s = 0
        horizon10s = horizon20s = horizon30s = horizon40s = 0
        
        num_samples = self.config.num_predictions
        
        with torch.no_grad():
            for step, (index, past, future, presents, angle_presents, videos,
                      vehicles, number_vec, scene, scene_one_hot) in enumerate(tqdm.tqdm(loader)):
                
                past = past.to(self.device)
                future = future.to(self.device)
                scene_one_hot = scene_one_hot.to(self.device) if self.config.use_scene else None
                
                # Generate multiple trajectory predictions
                pred = self.model(past, scene=scene_one_hot, num_samples=num_samples)
                
                # Compute errors for best prediction
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
                
                # Save visualizations
                if save_images:
                    for i in range(len(past)):
                        vid = videos[i]
                        vec = vehicles[i]
                        num_vec = number_vec[i]
                        index_track = index[i].numpy()
                        angle = angle_presents[i].cpu()
                        
                        horizon_dist = [
                            round(min_distances[i, 9].item(), 3),
                            round(min_distances[i, 19].item(), 3),
                            round(min_distances[i, 29].item(), 3),
                            round(min_distances[i, 39].item(), 3)
                        ]
                        
                        # Create folder structure
                        if not os.path.exists(self.folder_test + vid):
                            os.makedirs(self.folder_test + vid)
                        video_path = self.folder_test + vid + '/'
                        if not os.path.exists(video_path + vec + num_vec):
                            os.makedirs(video_path + vec + num_vec)
                        vehicle_path = video_path + vec + num_vec + '/'
                        
                        self.draw_track(
                            past[i], future[i], scene[i], pred[i],
                            angle, vid, vec + num_vec,
                            index_tracklet=index_track,
                            path=vehicle_path,
                            horizon_dist=horizon_dist
                        )
        
        # Compute average metrics
        dict_metrics = {
            'eucl_mean': round((eucl_mean / len(loader.dataset)).item(), 3),
            'ADE_1s': round((ADE_1s / len(loader.dataset)).item(), 3),
            'ADE_2s': round((ADE_2s / len(loader.dataset)).item(), 3),
            'ADE_3s': round((ADE_3s / len(loader.dataset)).item(), 3),
            'horizon10s': round((horizon10s / len(loader.dataset)).item(), 3),
            'horizon20s': round((horizon20s / len(loader.dataset)).item(), 3),
            'horizon30s': round((horizon30s / len(loader.dataset)).item(), 3),
            'horizon40s': round((horizon40s / len(loader.dataset)).item(), 3)
        }
        
        return dict_metrics
    
    def draw_track(self, past, future, scene, pred, angle, video_id, vec_id,
                   index_tracklet, path, horizon_dist):
        """
        Visualize and save trajectory predictions
        :param past: observed trajectory
        :param future: ground truth future
        :param scene: scene image
        :param pred: predicted trajectories (num_samples, future_len, 2)
        :param angle: rotation angle
        :param video_id: video identifier
        :param vec_id: vehicle identifier
        :param index_tracklet: trajectory index
        :param path: save path
        :param horizon_dist: horizon distances for title
        """
        angle = float(angle)
        
        # Color map for scene
        colors = [(0, 0, 0), (0.87, 0.87, 0.87), (0.54, 0.54, 0.54), 
                 (0.49, 0.33, 0.16), (0.29, 0.57, 0.25)]
        cmap_name = 'scene_cmap'
        cm = LinearSegmentedColormap.from_list(cmap_name, colors, N=5)
        
        fig = plt.figure(figsize=(10, 10))
        plt.imshow(scene, cmap=cm)
        
        # Color gradient for predictions
        pred_colors = pl.cm.Reds(np.linspace(1, 0.3, pred.shape[0]))
        
        # Rotation matrix to restore original orientation
        matRot_track = cv2.getRotationMatrix2D((0, 0), -angle, 1)
        
        # Transform past and future
        past = cv2.transform(past.cpu().numpy().reshape(-1, 1, 2), matRot_track).squeeze()
        future = cv2.transform(future.cpu().numpy().reshape(-1, 1, 2), matRot_track).squeeze()
        
        # Convert to scene coordinates
        past_scene = past * 2 + self.dim_clip
        future_scene = future * 2 + self.dim_clip
        
        # Plot past
        plt.plot(past_scene[:, 0], past_scene[:, 1], c='blue', 
                linewidth=2, marker='o', markersize=2, label='Past')
        
        # Plot predictions
        if pred is not None:
            for i_p in reversed(range(pred.shape[0])):
                pred_i = cv2.transform(pred[i_p].cpu().numpy().reshape(-1, 1, 2), matRot_track).squeeze()
                pred_scene = pred_i * 2 + self.dim_clip
                label = 'Predictions' if i_p == pred.shape[0] - 1 else None
                plt.plot(pred_scene[:, 0], pred_scene[:, 1], 
                        color=pred_colors[i_p], linewidth=1, 
                        marker='o', markersize=1, alpha=0.6, label=label)
        
        # Plot ground truth
        plt.plot(future_scene[:, 0], future_scene[:, 1], c='green',
                linewidth=2, marker='o', markersize=2, label='Ground Truth')
        
        plt.title(f'Video: {video_id}, Vehicle: {vec_id}\n' +
                 f'FDE 1s: {horizon_dist[0]}m, 2s: {horizon_dist[1]}m, ' +
                 f'3s: {horizon_dist[2]}m, 4s: {horizon_dist[3]}m')
        plt.legend()
        plt.axis('equal')
        
        # Save figure
        plt.savefig(path + video_id + '_' + vec_id + '_' + str(index_tracklet).zfill(3) + '.png', 
                   dpi=150, bbox_inches='tight')
        plt.close(fig)
    
    def save_results(self, dict_metrics):
        """
        Save evaluation results to file
        :param dict_metrics: dictionary of metrics
        """
        with open(self.folder_test + "results.txt", "w") as f:
            f.write("=" * 50 + "\n")
            f.write("CVAE-MDN Trajectory Prediction Evaluation\n")
            f.write("=" * 50 + "\n\n")
            
            f.write(f"Model: {self.config.model}\n")
            f.write(f"Dataset: {self.config.dataset_file}\n")
            f.write(f"Test split: {self.data_test.ids_split_test}\n")
            f.write(f"Number of predictions: {self.config.num_predictions}\n")
            f.write(f"Train size: {len(self.data_train)}\n")
            f.write(f"Test size: {len(self.data_test)}\n\n")
            
            f.write("=" * 50 + "\n")
            f.write("METRICS\n")
            f.write("=" * 50 + "\n")
            f.write(f"Average Displacement Error (ADE): {dict_metrics['eucl_mean']}m\n")
            f.write(f"ADE @ 1s: {dict_metrics['ADE_1s']}m\n")
            f.write(f"ADE @ 2s: {dict_metrics['ADE_2s']}m\n")
            f.write(f"ADE @ 3s: {dict_metrics['ADE_3s']}m\n\n")
            
            f.write(f"Final Displacement Error @ 1s: {dict_metrics['horizon10s']}m\n")
            f.write(f"Final Displacement Error @ 2s: {dict_metrics['horizon20s']}m\n")
            f.write(f"Final Displacement Error @ 3s: {dict_metrics['horizon30s']}m\n")
            f.write(f"Final Displacement Error @ 4s: {dict_metrics['horizon40s']}m\n")
        
        print(f'\nResults saved to {self.folder_test}results.txt')


# Evaluation configuration
if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Evaluate CVAE-MDN trajectory predictor')
    parser.add_argument('--model', type=str, required=True, 
                       help='Path to trained model')
    parser.add_argument('--dataset_file', type=str, default='kitti_dataset.json')
    parser.add_argument('--device', type=int, default=0)
    parser.add_argument('--cuda', action='store_true', default=True)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--past_len', type=int, default=20)
    parser.add_argument('--future_len', type=int, default=40)
    parser.add_argument('--latent_dim', type=int, default=32)
    parser.add_argument('--hidden_dim', type=int, default=128)
    parser.add_argument('--num_mixtures', type=int, default=5)
    parser.add_argument('--num_predictions', type=int, default=5,
                       help='Number of trajectory samples to generate')
    parser.add_argument('--use_scene', action='store_true', default=False)
    parser.add_argument('--save_images', action='store_true', default=False,
                       help='Save visualization images')
    parser.add_argument('--info', type=str, default='cvae_eval')
    
    config = parser.parse_args()
    
    evaluator = EvaluatorCVAE(config)
    evaluator.evaluate()