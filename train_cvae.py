# trainer_cvae.py

import os
import json
import datetime
import tqdm
import torch
from torch.utils.data import DataLoader
from types import SimpleNamespace
from tensorboardX import SummaryWriter

import dataset_invariance
from models.cvae_predictor import CVAE_Predictor


class TrainerCVAE:
    def __init__(self, config):

        self.config = config

        # ---- Device ----
        self.device = torch.device(
            f"cuda:{config.device}" if torch.cuda.is_available() and config.cuda else "cpu"
        )
        if torch.cuda.is_available() and config.cuda:
            torch.cuda.set_device(config.device)

        # ---- Folders ----
        self.name_run = str(datetime.datetime.now())[:19].replace(" ", "_").replace(":", "-")
        self.folder_runs = "runs/runs-cvae/"
        self.folder_test = f"training/training_cvae/{self.name_run}_{config.info}"
        os.makedirs(self.folder_test, exist_ok=True)
        self.folder_test = self.folder_test + "/"
        self.save_details_file(self.folder_test, config)

        # ---- Dataset ----
        print("Creating dataset...")
        tracks = json.load(open(config.dataset_file))
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
            num_workers=0,
            pin_memory=True,
            shuffle=True,
        )
        self.test_loader = DataLoader(
            self.data_test,
            batch_size=config.batch_size,
            num_workers=0,
            pin_memory=True,
            shuffle=False,
        )
        print(f"Train size: {len(self.data_train)}, Test size: {len(self.data_test)}")

        # ---- Model ----
        self.model = CVAE_Predictor(
            past_len=config.past_len,
            future_len=config.future_len,
            dim_embedding_key=config.dim_embedding_key,
            latent_dim=config.latent_dim,
            use_scene=True,
            use_cuda=config.cuda,
        ).to(self.device)

        # ---- Optimizer ----
        self.opt = torch.optim.Adam(
            self.model.parameters(),
            lr=config.learning_rate,
            weight_decay=config.weight_decay,
        )

        self.max_epochs = config.max_epochs
        self.beta_max = config.beta_max
        self.beta_warmup_epochs = config.beta_warmup_epochs
        self.num_samples_eval = config.num_samples_eval
        self.eval_every = config.eval_every

        self.iterations = 0
        self.best_val_ade = float("inf")

        # ---- TensorBoard ----
        self.writer = SummaryWriter(self.folder_runs + self.name_run + "_" + config.info)
        self.writer.add_text("config/summary", str(config), 0)
        
        # Print Prameter Count
        total_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        print(f"Model has {total_params} trainable parameters.")
        
    def save_details_file(self, folder_path, config):
        """
        Writes a details.txt file containing all config values.
        """
        path = os.path.join(folder_path, "details.txt")
        with open(path, "w") as f:
            f.write("Training Configuration\n")
            f.write("----------------------\n\n")
            for k, v in config.__dict__.items():
                f.write(f"{k}: {v}\n")

    # ---------------------------------------------------------
    def beta_schedule(self, epoch):
        if self.beta_warmup_epochs <= 0:
            return self.beta_max
        frac = min(1.0, float(epoch + 1) / float(self.beta_warmup_epochs))
        return self.beta_max * frac

    # ---------------------------------------------------------
    def fit(self):
        for epoch in range(self.max_epochs):

            beta = self.beta_schedule(epoch)
            print(f"----- Epoch {epoch} | beta={beta:.4f} -----")

            train_loss, train_recon, train_kl = self._train_single_epoch(beta)
            print(
                f"Train loss: {train_loss:.4f} | recon: {train_recon:.4f} | KL: {train_kl:.4f}"
            )

            self.writer.add_scalar("train/loss", train_loss, epoch)
            self.writer.add_scalar("train/recon", train_recon, epoch)
            self.writer.add_scalar("train/kl", train_kl, epoch)
            self.writer.add_scalar("train/beta", beta, epoch)

            # ---- Evaluation ----
            if (epoch + 1) % self.eval_every == 0:
                print("Evaluating on test set...")
                metrics = self.evaluate(self.test_loader, self.num_samples_eval)
                ade_4s = metrics["ADE_4s"]

                print("Test metrics:", {k: float(v) for k, v in metrics.items()})

                for k, v in metrics.items():
                    self.writer.add_scalar(f"test/{k}", float(v), epoch)

                if ade_4s < self.best_val_ade:
                    self.best_val_ade = ade_4s
                    path = os.path.join(self.folder_test, "cvae_best.pt")
                    torch.save(self.model.state_dict(), path)
                    print(f"Saved NEW BEST model to {path}")

        # ---- Save final ----
        path = os.path.join(self.folder_test, "cvae_final.pt")
        torch.save(self.model.state_dict(), path)
        print(f"Saved final model to {path}")

    # ---------------------------------------------------------
    def _train_single_epoch(self, beta):
        self.model.train()

        total_loss = 0.0
        total_recon = 0.0
        total_kl = 0.0
        num_batches = 0

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
        ) in tqdm.tqdm(self.train_loader):

            past = past.to(self.device)
            future = future.to(self.device)
            scene_one_hot = scene_one_hot.to(self.device)

            self.opt.zero_grad()
            loss, recon_loss, kl_loss = self.model.compute_loss(
                past, future, scene_one_hot, beta=beta
            )
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            self.opt.step()

            self.writer.add_scalar("train/iter_loss", loss.item(), self.iterations)
            self.iterations += 1

            total_loss += loss.item()
            total_recon += recon_loss.item()
            total_kl += kl_loss.item()
            num_batches += 1

        return (
            total_loss / num_batches,
            total_recon / num_batches,
            total_kl / num_batches,
        )

    # ---------------------------------------------------------
    def evaluate(self, loader, num_samples=5):
        """
        Best-of-K ADE/FDE evaluation.
        """
        self.model.eval()
        device = self.device

        ADE_1s = ADE_2s = ADE_3s = ADE_4s = 0.0
        FDE_1s = FDE_2s = FDE_3s = FDE_4s = 0.0

        N = len(loader.dataset)

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

                past = past.to(device)
                future = future.to(device)
                scene_one_hot = scene_one_hot.to(device)

                samples = self.model.sample(past, scene_one_hot, num_samples=num_samples)

                future_rep = future.unsqueeze(1).expand_as(samples)
                distances = torch.norm(samples - future_rep, dim=3)

                mean_distances = distances.mean(dim=2)
                idx_best = torch.argmin(mean_distances, dim=1)
                best_dist = distances[torch.arange(past.size(0)), idx_best]

                ADE_1s += torch.mean(best_dist[:, :10]).item() * past.size(0)
                ADE_2s += torch.mean(best_dist[:, :20]).item() * past.size(0)
                ADE_3s += torch.mean(best_dist[:, :30]).item() * past.size(0)
                ADE_4s += torch.mean(best_dist[:, :40]).item() * past.size(0)

                FDE_1s += torch.mean(best_dist[:,  9]).item() * past.size(0)
                FDE_2s += torch.mean(best_dist[:, 19]).item() * past.size(0)
                FDE_3s += torch.mean(best_dist[:, 29]).item() * past.size(0)
                FDE_4s += torch.mean(best_dist[:, 39]).item() * past.size(0)

        metrics = {
            "ADE_1s": ADE_1s / N,
            "ADE_2s": ADE_2s / N,
            "ADE_3s": ADE_3s / N,
            "ADE_4s": ADE_4s / N,
            "FDE_1s": FDE_1s / N,
            "FDE_2s": FDE_2s / N,
            "FDE_3s": FDE_3s / N,
            "FDE_4s": FDE_4s / N,
        }
        return {k: torch.tensor(v) for k, v in metrics.items()}


# ===============================================================
# MAIN ENTRY POINT
# ===============================================================

if __name__ == "__main__":

    config = SimpleNamespace(
        # ----- Dataset -----
        dataset_file="kitti_dataset.json",

        # ----- Model -----
        past_len=20,
        future_len=40,
        dim_embedding_key=48,
        latent_dim=4,

        # ----- Training -----
        batch_size=32,
        learning_rate=1e-4,
        max_epochs=200,
        cuda=True,
        device=0,
        weight_decay=5e-5,

        # ----- β-annealing -----
        beta_max=0.8,
        beta_warmup_epochs=80,

        # ----- Evaluation -----
        num_samples_eval=5,
        eval_every=5,

        # ----- Info ----- 
        info="cvae_cond_prior_beta",
    )

    trainer = TrainerCVAE(config)
    trainer.fit()
