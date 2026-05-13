import torch
import torch.nn.functional as F
import os
import time
import torch.optim as optim
from collections import OrderedDict
from os.path import join as pjoin
from torch.nn.utils import clip_grad_norm_
from utils.utils import print_current_loss_decomp
import matplotlib.pyplot as plt
from networks.nn import MotionVQVAE
from torch.optim.lr_scheduler import CosineAnnealingLR

class Logger(object):
  def __init__(self, log_dir):
    # self.writer = tf.summary.create_file_writer(log_dir)
    pass

  def scalar_summary(self, tag, value, step):
    #   with self.writer.as_default():
    #       tf.summary.scalar(tag, value, step=step)
    #       self.writer.flush()
    pass

class MotionTrainer(object):

    def __init__(self, args, movement_enc, movement_dec):
        self.opt = args
        self.movement_enc = movement_enc
        self.movement_dec = movement_dec
        self.device = args.device

        if args.is_train:
            self.logger = Logger(args.log_dir)
            self.sml1_criterion = torch.nn.SmoothL1Loss()
            self.l1_criterion = torch.nn.L1Loss()
            self.mse_criterion = torch.nn.MSELoss()


class MotionVQVAETrainer(object):
    def __init__(self, args, vqvae: MotionVQVAE):
        self.opt = args
        self.vqvae = vqvae
        self.device = args.device

        if args.is_train:
            self.logger = Logger(args.log_dir)

    @staticmethod
    def zero_grad(opt_list):
        for opt in opt_list:
            opt.zero_grad()

    @staticmethod
    def clip_norm(network_list, max_norm=0.5):
        for network in network_list:
            clip_grad_norm_(network.parameters(), max_norm)

    @staticmethod
    def step(opt_list):
        for opt in opt_list:
            opt.step()

    def save_loss_data(self, history):

        os.makedirs(self.opt.experiment_dir, exist_ok=True)
        epochs = range(1, len(history["train_loss"]) + 1)

        loss_pairs = [
            ("loss", "Total Loss"),
            ("loss_rec", "Reconstruction Loss"),
            ("loss_vq", "VQ Loss"),
            ("loss_codebook", "Codebook Loss"),
            ("loss_commit", "Commitment Loss"),
        ]

        for key, title in loss_pairs:
            plt.figure(figsize=(8, 5))
            plt.plot(epochs, history[f"train_{key}"], label=f"train_{key}")
            plt.plot(epochs, history[f"val_{key}"], label=f"val_{key}")
            plt.xlabel("Epoch")
            plt.ylabel("Loss")
            plt.title(title)
            plt.legend()
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.savefig(pjoin(self.opt.experiment_dir, f"{key}.png"))
            plt.close()

        plt.figure(figsize=(10, 6))
        for key, title in loss_pairs:
            plt.plot(epochs, history[f"train_{key}"], label=f"train_{key}")
            plt.plot(epochs, history[f"val_{key}"], linestyle="--", label=f"val_{key}")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.title("All Losses")
        plt.legend(fontsize=8, ncol=2)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(pjoin(self.opt.experiment_dir, "all_losses.png"))
        plt.close()

    def forward(self, batch_data):
        motions = batch_data
        self.motions_by_part = motions['motion_parts'].detach().to(self.device).float()

        self.outputs = self.vqvae(self.motions_by_part)

        self.recon_motions_by_part = self.outputs["x_recon"]
        self.loss = self.outputs["loss"]
        self.loss_rec = self.outputs["recon_loss"]
        self.loss_vq = self.outputs["vq_loss"]
        self.loss_codebook = self.outputs["codebook_loss"]
        self.loss_commit = self.outputs["commitment_loss"]

    def update(self):
        self.zero_grad([self.opt_vqvae])
        self.loss.backward()
        self.clip_norm([self.vqvae], 0.5)
        self.step([self.opt_vqvae])
        self.scheduler_vqvae.step()

        loss_logs = OrderedDict()
        loss_logs["loss"] = self.loss.item()
        loss_logs["loss_rec"] = self.loss_rec.item()
        loss_logs["loss_vq"] = self.loss_vq.item()
        loss_logs["loss_codebook"] = self.loss_codebook.item()
        loss_logs["loss_commit"] = self.loss_commit.item()
        return loss_logs

    def save(self, file_name, ep, total_it, history = None):
        state = {
            "vqvae": self.vqvae.state_dict(),
            "opt_vqvae": self.opt_vqvae.state_dict(),
            "scheduler_vqvae": self.scheduler_vqvae.state_dict(),
            "ep": ep,
            "total_it": total_it,
            "history": history
        }
        torch.save(state, file_name)

    def resume(self, model_dir):
        checkpoint = torch.load(model_dir, map_location=self.device)
        self.vqvae.load_state_dict(checkpoint["vqvae"])
        self.opt_vqvae.load_state_dict(checkpoint["opt_vqvae"])
        self.scheduler_vqvae.load_state_dict(checkpoint["scheduler_vqvae"])
        return checkpoint["ep"], checkpoint["total_it"], checkpoint["history"]

    def train(self, train_dataloader, val_dataloader, plot_eval = None):
        self.vqvae.to(self.device)
        self.opt_vqvae = optim.Adam(self.vqvae.parameters(), lr=self.opt.lr)
        start_time = time.time()
        total_iters = self.opt.max_epoch * len(train_dataloader)
        self.scheduler_vqvae = CosineAnnealingLR(self.opt_vqvae, T_max = total_iters, eta_min = 1e-5)

        history = {
            "train_loss": [],
            "train_loss_rec": [],
            "train_loss_vq": [],
            "train_loss_codebook": [],
            "train_loss_commit": [],
            "val_loss": [],
            "val_loss_rec": [],
            "val_loss_vq": [],
            "val_loss_codebook": [],
            "val_loss_commit": [],
        }
        
        print("Number of epochs:", self.opt.max_epoch)

        epoch = 0
        it = 0
        if self.opt.is_continue:
            model_dir = pjoin(self.opt.model_dir, "latest.tar")
            epoch, it, history = self.resume(model_dir)
            print(f'Resuming training from previous checkpoint at epoch {epoch}')

        start_time = time.time()
        print("Iters Per Epoch, Training: %04d, Validation: %03d" %
              (len(train_dataloader), len(val_dataloader)))

        val_loss = 0
        logs = OrderedDict()

        while epoch < self.opt.max_epoch:
            train_loss_sum = 0.0
            train_rec_sum = 0.0
            train_vq_sum = 0.0
            train_codebook_sum = 0.0
            train_commit_sum = 0.0
            train_steps = 0
            for i, batch_data in enumerate(train_dataloader):
                self.vqvae.train()
                self.forward(batch_data)
                log_dict = self.update()

                train_loss_sum += self.loss.item()
                train_rec_sum += self.loss_rec.item()
                train_vq_sum += self.loss_vq.item()
                train_codebook_sum += self.loss_codebook.item()
                train_commit_sum += self.loss_commit.item()
                train_steps += 1

                for k, v in log_dict.items():
                    if k not in logs:
                        logs[k] = v
                    else:
                        logs[k] += v

                it += 1

                if it % self.opt.log_every == 0:
                    mean_loss = OrderedDict({"val_loss": val_loss})
                    self.logger.scalar_summary("val_loss", val_loss, it)

                    for tag, value in logs.items():
                        self.logger.scalar_summary(tag, value / self.opt.log_every, it)
                        mean_loss[tag] = value / self.opt.log_every

                    logs = OrderedDict()
                    #print_current_loss_decomp(start_time, it, total_iters, mean_loss, epoch, i)

                if it % self.opt.save_latest == 0:
                    self.save(pjoin(self.opt.model_dir, "latest.tar"), epoch, it, history)

            #epoch += 1

            train_loss_avg = train_loss_sum / max(train_steps, 1)
            train_rec_avg = train_rec_sum / max(train_steps, 1)
            train_vq_avg = train_vq_sum / max(train_steps, 1)
            train_codebook_avg = train_codebook_sum / max(train_steps, 1)
            train_commit_avg = train_commit_sum / max(train_steps, 1)

            history["train_loss"].append(train_loss_avg)
            history["train_loss_rec"].append(train_rec_avg)
            history["train_loss_vq"].append(train_vq_avg)
            history["train_loss_codebook"].append(train_codebook_avg)
            history["train_loss_commit"].append(train_commit_avg)

            #print("Validation time:")
            val_loss = 0
            val_rec_loss = 0
            val_vq_loss = 0
            val_codebook_loss = 0
            val_commit_loss = 0

            with torch.no_grad():
                self.vqvae.eval()
                for i, batch_data in enumerate(val_dataloader):
                    self.forward(batch_data)

                    val_loss += self.loss.item()
                    val_rec_loss += self.loss_rec.item()
                    val_vq_loss += self.loss_vq.item()
                    val_codebook_loss += self.loss_codebook.item()
                    val_commit_loss += self.loss_commit.item()

            denom = max(len(val_dataloader), 1)
            val_loss /= denom
            val_rec_loss /= denom
            val_vq_loss /= denom
            val_codebook_loss /= denom
            val_commit_loss /= denom

            history["val_loss"].append(val_loss)
            history["val_loss_rec"].append(val_rec_loss)
            history["val_loss_vq"].append(val_vq_loss)
            history["val_loss_codebook"].append(val_codebook_loss)
            history["val_loss_commit"].append(val_commit_loss)
            
            
            if epoch % self.opt.save_every_e == 0:
                self.save(pjoin(self.opt.model_dir, "E%04d.tar" % epoch), epoch, total_it=it, history = history)

            if epoch % self.opt.eval_every_e == 0:
                print("Epoch:", epoch)
                print(
                    "Train Loss: %.5f Reconstruction Loss: %.5f "
                    "VQ Loss: %.5f Codebook Loss: %.5f Commitment Loss: %.5f"
                    % (train_loss_avg, train_rec_avg, train_vq_avg, train_codebook_avg, train_commit_avg)
                )
                print(
                    "Validation Loss: %.5f Reconstruction Loss: %.5f "
                    "VQ Loss: %.5f Codebook Loss: %.5f Commitment Loss: %.5f"
                    % (val_loss, val_rec_loss, val_vq_loss, val_codebook_loss, val_commit_loss)
                )
                data = torch.cat([self.recon_motions_by_part, self.motions_by_part], dim=0).detach().cpu().numpy()
                save_dir = pjoin(self.opt.eval_dir, "E%04d" % epoch)
                os.makedirs(save_dir, exist_ok=True)
                #plot_eval(data, save_dir)
                self.save_loss_data(history = history)
            
            epoch += 1
        
        self.save_loss_data(history = history)
