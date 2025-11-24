import os
import argparse
import datetime
import json
import time

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import tqdm
import cv2

import dataset_invariance
from models.cvae_predictor import CVAE_Predictor

import matplotlib.pyplot as plt
import matplotlib.pylab as pl
from matplotlib.colors import LinearSegmentedColormap


class ValidatorCVAE:
    """
    Evaluator for a CVAE trajectory predictor on the KITTI / MANTRA-style dataset.

    - Loads dataset_invariance.TrackDataset
    - Loads a trained CVAE_Predictor
    - Draws K samples per past trajectory
    - Computes best-of-K ADE/FDE metrics (same style as MANTRA)
    """

    def __init__(self, config):
        self.config = config

        # Device setup
        self.device = torch.device(
            f"cuda:{config.device}" if torch.cuda.is_available() and config.cuda else "cpu"
        )
        if torch.cuda.is_available() and config.cuda:
            torch.cuda.set_device(config.device)

        # Output folder
        self.name_test = str(datetime.datetime.now())[:19].replace(' ', '_').replace(':', '-')
        self.folder_test = os.path.join(
            "test",
            f"{self.name_test}_{config.info}"
        )
        os.makedirs(self.folder_test, exist_ok=True)

        # Dataset
        print("Creating dataset...")
        with open(config.dataset_file, "r") as f:
            tracks = json.load(f)

        self.dim_clip = 180
        self.data_train = dataset_invariance.TrackDataset(
            tracks,
            len_past=config.past_len,
            len_future=config.future_len,
            train=True,
            dim_clip=self.dim_clip,
        )

        self.data_test = dataset_invariance.TrackDataset(
            tracks,
            len_past=config.past_len,
            len_future=config.future_len,
            train=False,
            dim_clip=self.dim_clip,
        )

        self.train_loader = DataLoader(
            self.data_train,
            batch_size=config.batch_size,
            num_workers=config.num_workers,
            shuffle=True,
            pin_memory=True,
        )

        self.test_loader = DataLoader(
            self.data_test,
            batch_size=config.batch_size,
            num_workers=config.num_workers,
            shuffle=False,
            pin_memory=True,
        )
        print(f"Dataset created. Train size: {len(self.data_train)}, Test size: {len(self.data_test)}")

        # Build or load model
        self.model = self._build_and_load_model()
        
        # Print Model Parameter Count
        total_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        print(f"Total trainable parameters in model: {total_params}")

        # For distances (just using torch.norm directly, but keep pairwise if needed)
        self.EuclDistance = nn.PairwiseDistance(p=2)
        
        # ----------------------------------------------------------------------
        # Choose which test samples to visualize (10 random)
        # ----------------------------------------------------------------------
        num_examples = 5
        total = len(self.data_test)
        self.example_ids = set(
            np.random.choice(total, size=min(num_examples, total), replace=False)
        )

    # -------------------------------------------------------------------------
    # Model loading
    # -------------------------------------------------------------------------
    def _build_and_load_model(self):
        """
        Load the model from a checkpoint. Handles both:
        - full pickled nn.Module
        - state_dict, in which case we reconstruct the CVAE_Predictor.
        """
        print(f"Loading model from: {self.config.model}")
        loaded_obj = torch.load(self.config.model, map_location="cpu", weights_only=False)

        if isinstance(loaded_obj, torch.nn.Module):
            # Model object was pickled directly
            model = loaded_obj
            model.to(self.device)
            print("Loaded full nn.Module from checkpoint.")
            return model

        elif isinstance(loaded_obj, dict) and all(
            isinstance(v, torch.Tensor) for v in loaded_obj.values()
        ):
            # Raw state_dict: reconstruct CVAE_Predictor using config hyperparams
            settings = {
                "use_cuda": self.config.cuda,
                "dim_embedding_key": self.config.dim_embedding_key,
                "latent_dim": self.config.latent_dim,
                "past_len": self.config.past_len,
                "future_len": self.config.future_len,
                "scene_channels": 4,  # from TrackDataset one-hot (background/street/sidewalk/veg)
            }
            model = CVAE_Predictor(settings)
            model.load_state_dict(loaded_obj, strict=False)
            model.to(self.device)
            print("Reconstructed CVAE_Predictor from state_dict.")
            return model

        else:
            raise RuntimeError(
                f"Unrecognized object in torch.load({self.config.model!r}): "
                "expected an nn.Module or a state_dict"
            )
            
    def draw_track(
        self,
        past,
        future,
        scene_track,
        preds=None,
        angle=0.0,
        path="",
        name="example.png",
        horizon_dist=None,
    ):
        """
        Plot past, future, and ALL predicted trajectories (K=5 predicted futures).
        """

        angle = float(angle)

        # Scene colormap (same as MANTRA)
        colors_bg = [
            (0, 0, 0),
            (0.87, 0.87, 0.87),
            (0.54, 0.54, 0.54),
            (0.49, 0.33, 0.16),
            (0.29, 0.57, 0.25),
        ]
        cm = LinearSegmentedColormap.from_list("scene_cmap", colors_bg, N=5)

        fig = plt.figure()
        plt.imshow(scene_track, cmap=cm)

        # Rotation matrix (undo padding rotation)
        matRot = cv2.getRotationMatrix2D((0, 0), -angle, 1)

        # Rotate + scale coordinates
        def transform(coords):
            coords_np = coords.cpu().numpy()
            coords_np = cv2.transform(coords_np.reshape(-1, 1, 2), matRot).squeeze()
            return coords_np * 2 + self.dim_clip

        # Past + future
        past_scene = transform(past)
        future_scene = transform(future)

        plt.plot(past_scene[:, 0], past_scene[:, 1],
                c="blue", linewidth=1, marker="o", markersize=1, label="past")

        plt.plot(future_scene[:, 0], future_scene[:, 1],
                c="green", linewidth=1, marker="o", markersize=1, label="future")

        # Predictions: plot all K trajectories
        if preds is not None:
            K = preds.shape[0]
            color_map = pl.cm.Reds(np.linspace(1, 0.3, K))

            for k in range(K):
                pred_scene = transform(preds[k])
                plt.plot(
                    pred_scene[:, 0],
                    pred_scene[:, 1],
                    color=color_map[k],
                    linewidth=1,
                    marker="o",
                    markersize=1,
                    label=f"pred_{k}",
                    alpha=0.9,
                )

        if horizon_dist is not None:
            plt.title(
                "FDE 1s: {:.3f}  FDE 2s: {:.3f}  FDE 3s: {:.3f}  FDE 4s: {:.3f}".format(
                    *horizon_dist
                )
            )

        plt.axis("equal")
        plt.legend(loc="upper left", fontsize=6)
        plt.savefig(os.path.join(path, name), dpi=150, bbox_inches="tight")
        plt.close(fig)


    # -------------------------------------------------------------------------
    # Evaluation
    # -------------------------------------------------------------------------
    def evaluate(self, loader):
        """
        Evaluate the model with best-of-K ADE/FDE metrics.

        Assumes:
            pred = model.sample(past, scene_one_hot, num_samples)
            -> pred shape: [B, K, T, 2]
        """
        self.model.eval()
        num_samples = self.config.num_samples
        future_len = self.config.future_len

        # Metric accumulators
        eucl_mean = 0.0
        ADE_1s = ADE_2s = ADE_3s = 0.0
        horizon10s = horizon20s = horizon30s = horizon40s = 0.0

        n_total = len(loader.dataset)
        global_counter = 0  # counts samples across batches
        saved_count = 0

        with torch.no_grad():
            for (
                index,
                past,
                future,
                presents,
                angle_presents,
                videos,
                vehicles,
                number_vec,
                scene,
                scene_one_hot,
            ) in tqdm.tqdm(loader):

                past = past.to(self.device)                       # [B, T_p, 2]
                future = future.to(self.device)                   # [B, T_f, 2]
                scene_one_hot = scene_one_hot.to(self.device)     # [B, H, W, 4]

                # ------------------------------------------------------------------
                # Sampling from the CVAE
                # ------------------------------------------------------------------
                # IMPORTANT:
                # We assume your model has a method:
                #   sample(past, scene_one_hot, num_samples) -> [B, K, T, 2]
                # If your name/signature differs, adapt this line accordingly.
                # ------------------------------------------------------------------
                pred = self.model.sample(
                    past, scene_one_hot, num_samples=num_samples
                )  # [B, K, T, 2]

                if pred.dim() != 4:
                    raise RuntimeError(
                        f"Expected pred shape [B, K, T, 2], got {list(pred.shape)}"
                    )

                B, K, T, _ = pred.shape
                assert T == future_len, f"Future length mismatch: pred T={T}, config={future_len}"

                # Compute distances to GT
                future_rep = future.unsqueeze(1).repeat(1, K, 1, 1)  # [B, K, T, 2]
                distances = torch.norm(pred - future_rep, dim=3)      # [B, K, T]
                distances_mean = torch.mean(distances, dim=2)         # [B, K] (ADE per sample)
                index_min = torch.argmin(distances_mean, dim=1)       # [B]

                # Best sample per trajectory
                best_distances = distances[torch.arange(B), index_min]  # [B, T]

                # ADE over entire horizon
                eucl_mean += torch.sum(torch.mean(best_distances[:, :future_len], dim=1)).item()

                # ADE partial horizons
                ADE_1s += torch.sum(torch.mean(best_distances[:, :10], dim=1)).item()
                ADE_2s += torch.sum(torch.mean(best_distances[:, :20], dim=1)).item()
                ADE_3s += torch.sum(torch.mean(best_distances[:, :30], dim=1)).item()

                # FDE horizons (assuming 10 steps = 1s)
                horizon10s += torch.sum(best_distances[:, 9]).item()
                horizon20s += torch.sum(best_distances[:, 19]).item()
                horizon30s += torch.sum(best_distances[:, 29]).item()
                horizon40s += torch.sum(best_distances[:, 39]).item()
                
                batch_size = past.shape[0]
                for i in range(batch_size):
                    if global_counter in self.example_ids and saved_count < 10:

                        best_k = index_min[i].item()
                        best_pred = pred[i, best_k]                   # [40,2]
                        all_preds = pred[i]                           # [5,40,2]
                        d = torch.norm(best_pred - future[i], dim=1)  # [40]

                        def fd(idx):
                            return round(d[min(idx, 39)].item(), 3)

                        horizon_dist = [fd(9), fd(19), fd(29), fd(39)]

                        angle = angle_presents[i].cpu().item()
                        name = f"example_{saved_count:03d}.png"

                        self.draw_track(
                            past[i].cpu(),
                            future[i].cpu(),
                            scene[i].numpy(),
                            preds=all_preds.cpu(),
                            angle=angle,
                            path=self.folder_test,
                            name=name,
                            horizon_dist=horizon_dist,
                        )

                        saved_count += 1
                    global_counter += 1

        # Normalize by dataset size
        metrics = {
            "eucl_mean": eucl_mean / n_total,
            "ADE_1s": ADE_1s / n_total,
            "ADE_2s": ADE_2s / n_total,
            "ADE_3s": ADE_3s / n_total,
            "horizon10s": horizon10s / n_total,
            "horizon20s": horizon20s / n_total,
            "horizon30s": horizon30s / n_total,
            "horizon40s": horizon40s / n_total,
        }

        return metrics

    # -------------------------------------------------------------------------
    # Save results
    # -------------------------------------------------------------------------
    def save_results(self, metrics):
        """
        Write metrics to a text file in self.folder_test.
        """
        results_path = os.path.join(self.folder_test, "results.txt")
        with open(results_path, "w") as f:
            f.write("CVAE TEST RESULTS\n")
            f.write(f"model: {self.config.model}\n")
            f.write(f"split test: {self.data_test.ids_split_test}\n")
            f.write(f"num_predictions (samples): {self.config.num_samples}\n")
            f.write(f"TRAIN size: {len(self.data_train)}\n")
            f.write(f"TEST size: {len(self.data_test)}\n\n")

            f.write(f"error 1s (FDE 1s): {metrics['horizon10s']:.4f} m\n")
            f.write(f"error 2s (FDE 2s): {metrics['horizon20s']:.4f} m\n")
            f.write(f"error 3s (FDE 3s): {metrics['horizon30s']:.4f} m\n")
            f.write(f"error 4s (FDE 4s): {metrics['horizon40s']:.4f} m\n\n")

            f.write(f"ADE 1s: {metrics['ADE_1s']:.4f} m\n")
            f.write(f"ADE 2s: {metrics['ADE_2s']:.4f} m\n")
            f.write(f"ADE 3s: {metrics['ADE_3s']:.4f} m\n")
            f.write(f"ADE 4s (full): {metrics['eucl_mean']:.4f} m\n")

        print(f"Saved results to {results_path}")


# --------------------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------------------
def parse_args():
    parser = argparse.ArgumentParser(
        description="Evaluate CVAE trajectory predictor (best-of-K ADE/FDE)."
    )

    # Data / model paths
    parser.add_argument(
        "--dataset_file",
        type=str,
        default="kitti_dataset.json",
        help="Path to KITTI-style trajectory JSON (same as training).",
    )
    parser.add_argument(
        "--model",
        type=str,
        default='training\\training_cvae\\2025-11-21_13-44-27_cvae_cond_prior_beta\\cvae_best.pt',
        help="Path to trained CVAE model (.pt / .pth).",
    )
    parser.add_argument(
        "--info",
        type=str,
        default="cvae_eval",
        help="Info string appended to test folder name.",
    )

    # Model hyperparams (only needed if loading from state_dict)
    parser.add_argument(
        "--dim_embedding_key",
        type=int,
        default=48,
        help="Embedding dimension used for past/scene encoders.",
    )
    parser.add_argument(
        "--latent_dim",
        type=int,
        default=16,
        help="Latent dimension z.",
    )
    parser.add_argument(
        "--past_len",
        type=int,
        default=20,
        help="Length of past trajectory.",
    )
    parser.add_argument(
        "--future_len",
        type=int,
        default=40,
        help="Length of future trajectory.",
    )

    # Evaluation settings
    parser.add_argument(
        "--batch_size",
        type=int,
        default=64,
        help="Evaluation batch size.",
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=4,
        help="Number of DataLoader workers.",
    )
    parser.add_argument(
        "--num_samples",
        type=int,
        default=5,
        help="Number of trajectory samples per agent (K for best-of-K).",
    )

    # Device
    parser.add_argument(
        "--cuda",
        action="store_true",
        help="Use CUDA if available.",
    )
    parser.add_argument(
        "--device",
        type=int,
        default=0,
        help="CUDA device index.",
    )

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    print("===== CVAE Evaluation =====")
    print(f"Model: {args.model}")
    print(f"Dataset: {args.dataset_file}")
    print(f"Batch size: {args.batch_size}, num_samples: {args.num_samples}")

    validator = ValidatorCVAE(args)

    start = time.time()
    metrics = validator.evaluate(validator.test_loader)
    end = time.time()

    print("Evaluation complete in {:.1f} s".format(end - start))
    print("--- Metrics (meters) ---")
    for k, v in metrics.items():
        print(f"{k}: {v:.4f}")

    validator.save_results(metrics)
