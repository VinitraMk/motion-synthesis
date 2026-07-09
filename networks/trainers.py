import torch
import torch.nn.functional as F
import os
import time
import torch.optim as optim
from collections import OrderedDict
from os.path import join as pjoin
from torch.nn.utils import clip_grad_norm_
from utils.utils import print_current_loss_decomp, cpu_deepcopy_state, move_state_to_device
import matplotlib.pyplot as plt
from networks.nn import MotionVQVAE, DiT, MotionVAE
from torch.optim.lr_scheduler import CosineAnnealingLR
from networks.autoencoder_modules import MovementConvEncoder, MovementConvDecoder
from utils.pretrained_model_utils import get_pretrained_vae, get_pretrained_text_encoder
from networks.transformer_modules import TextTokenEncoder
import numpy as np
from data_utils.motion_processor import recover_from_ric
import json

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
        if torch.isnan(self.loss):
            print("NaN loss before backward, skipping step")
            return OrderedDict()
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

        # loss value init
        train_loss_avg = 0
        train_rec_avg = 0
        train_vq_avg = 0
        train_codebook_avg = 0
        train_commit_avg = 0
        val_loss = 0
        val_rec_loss = 0
        val_vq_loss = 0
        val_codebook_loss = 0
        val_commit_loss = 0

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

                '''
                if it % self.opt.log_every == 0:
                    mean_loss = OrderedDict({"val_loss": val_loss})
                    self.logger.scalar_summary("val_loss", val_loss, it)

                    for tag, value in logs.items():
                        self.logger.scalar_summary(tag, value / self.opt.log_every, it)
                        mean_loss[tag] = value / self.opt.log_every

                    logs = OrderedDict()
                    #print_current_loss_decomp(start_time, it, total_iters, mean_loss, epoch, i)
                '''

                if it % self.opt.save_latest == 0:
                    self.save(pjoin(self.opt.model_dir, "latest.tar"), ep = epoch, total_it = it, history = history)

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
                #data = torch.cat([self.recon_motions_by_part, self.motions_by_part], dim=0).detach().cpu().numpy()
                save_dir = pjoin(self.opt.eval_dir, "E%04d" % epoch)
                os.makedirs(save_dir, exist_ok=True)
                #plot_eval(data, save_dir)
                self.save_loss_data(history = history)
            
            epoch += 1

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
        
        self.save_loss_data(history = history)

class MotionVAETrainer(object):
    def __init__(self, args, vae: MotionVAE):
        self.opt = args
        self.vae = vae
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
            ("loss_kl", "KL Loss"),
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
        self.motions = motions['motion'].detach().to(self.device).float()
        motion_masks = batch_data['motion_mask'].to(self.device).float()
        beta_max = self.opt.kl_beta_max
        warmup_step = self.opt.kl_warmup_step
        beta_t = min(beta_max, beta_max * (self.it / warmup_step))
        #additive_masks = (1 - motion_masks) * (-1e-5)
        if torch.isnan(self.motions).any():
            print("NaN in motions")
        if torch.isnan(motion_masks).any():
            print("NaN in motion_masks")

        self.outputs = self.vae(self.motions[:,:,:-4], key_padding_mask=motion_masks, beta = beta_t)
        
        self.recon_motions = self.outputs["x_recon"]
        self.loss = self.outputs["loss"]
        self.loss_rec = self.outputs["recon_loss"]
        self.loss_kl = self.outputs["kl_loss"]
        
        if torch.isnan(self.recon_motions).any():
            print("NaN in pred")
            
        if torch.isnan(self.loss).any():
            print("NaN in loss")

    def update(self):
        if torch.isnan(self.loss):
            print("NaN loss before backward, skipping step")
            return OrderedDict()
        self.zero_grad([self.opt_vae])
        self.loss.backward()
        self.clip_norm([self.vae], 0.5)
        self.step([self.opt_vae])
        self.scheduler_vae.step()

        loss_logs = OrderedDict()
        loss_logs["loss"] = self.loss.item()
        loss_logs["loss_rec"] = self.loss_rec.item()
        loss_logs["loss_kl"] = self.loss_kl.item()
        return loss_logs

    def save(self, file_name, ep, total_it, history = None, best_model_state = None):
        state = {
            "vae": self.vae.state_dict() if best_model_state == None else best_model_state,
            "opt_vae": self.opt_vae.state_dict(),
            "scheduler_vae": self.scheduler_vae.state_dict(),
            "ep": ep,
            "total_it": total_it,
            "history": history
        }
        torch.save(state, file_name)

    def resume(self, model_dir):
        checkpoint = torch.load(model_dir, map_location=self.device)
        self.vae.load_state_dict(checkpoint["vae"])
        self.opt_vae.load_state_dict(checkpoint["opt_vae"])
        self.scheduler_vae.load_state_dict(checkpoint["scheduler_vae"])
        return checkpoint["ep"], checkpoint["total_it"], checkpoint["history"]

    def train(self, train_dataloader, val_dataloader, plot_eval = None):
        self.vae.to(self.device)
        self.opt_vae = optim.Adam(self.vae.parameters(), lr=self.opt.lr)
        start_time = time.time()
        total_iters = self.opt.max_epoch * len(train_dataloader)
        self.scheduler_vae = CosineAnnealingLR(self.opt_vae, T_max = total_iters, eta_min = 1e-5)

        history = {
            "train_loss": [],
            "train_loss_rec": [],
            "train_loss_kl": [],
            "val_loss": [],
            "val_loss_rec": [],
            "val_loss_kl": [],
        }
        
        print("Number of epochs:", self.opt.max_epoch)

        self.epoch = 0
        self.it = 0
        if self.opt.is_continue:
            model_dir = pjoin(self.opt.model_dir, "latest.tar")
            self.epoch, _, history = self.resume(model_dir)
            self.it = (self.epoch + 1) * len(train_dataloader)
            print(f'Resuming training from previous checkpoint at epoch {self.epoch}')

        print("Iters Per Epoch, Training: %04d, Validation: %03d" %
              (len(train_dataloader), len(val_dataloader)))

        val_loss = 0

        # loss value init
        train_loss_avg = 0
        train_rec_avg = 0
        train_kl_avg = 0
        val_loss = 0
        val_rec_loss = 0
        val_kl_loss = 0
        patience = self.opt.patience
        min_delta = self.opt.min_loss_delta
        best_val = float('inf')
        best_state = None
        epochs_without_improve = 0

        while self.epoch < self.opt.max_epoch:
            train_loss_sum = 0.0
            train_rec_sum = 0.0
            train_kl_sum = 0.0
            train_steps = 0
            for _, batch_data in enumerate(train_dataloader):
                self.vae.train()
                self.forward(batch_data)

                train_loss_sum += self.loss.item()
                train_rec_sum += self.loss_rec.item()
                train_kl_sum += self.loss_kl.item()
                #print(f"Epoch: {self.epoch}, Iter: {it}, Loss: {self.loss.item():.5f}, Rec Loss: {self.loss_rec.item():.5f}, KL Loss: {self.loss_kl.item():.5f}")
                train_steps += 1

                self.it += 1

                if self.it % self.opt.save_latest == 0:
                    self.save(pjoin(self.opt.model_dir, "tmp.tar"), ep = self.epoch, total_it = self.it, history = history)

            train_loss_avg = train_loss_sum / max(train_steps, 1)
            train_rec_avg = train_rec_sum / max(train_steps, 1)
            train_kl_avg = train_kl_sum / max(train_steps, 1)

            history["train_loss"].append(train_loss_avg)
            history["train_loss_rec"].append(train_rec_avg)
            history["train_loss_kl"].append(train_kl_avg)

            #print("Validation time:")
            val_loss = 0
            val_rec_loss = 0
            val_kl_loss = 0

            with torch.no_grad():
                self.vae.eval()
                for i, batch_data in enumerate(val_dataloader):
                    self.forward(batch_data)

                    val_loss += self.loss.item()
                    val_rec_loss += self.loss_rec.item()
                    val_kl_loss += self.loss_kl.item()

            denom = max(len(val_dataloader), 1)
            val_loss /= denom
            val_rec_loss /= denom
            val_kl_loss /= denom

            history["val_loss"].append(val_loss)
            history["val_loss_rec"].append(val_rec_loss)
            history["val_loss_kl"].append(val_kl_loss)

            if os.path.exists(pjoin(self.opt.model_dir, "tmp.tar")):
                try:
                    model_ckpt = torch.load(pjoin(self.opt.model_dir, "tmp.tar"), map_location="cpu")
                    self.save(pjoin(self.opt.model_dir, "latest.tar"), ep = self.epoch, total_it = it, history = history)
                    del model_ckpt
                except Exception as e:
                    print(f"Failed to load checkpoint from {pjoin(self.opt.model_dir, 'tmp.tar')}. Skipping save to latest.tar. Error: {e}")
                os.remove(pjoin(self.opt.model_dir, "tmp.tar")) # removing tar if latest stable is saved

            if best_val - val_loss > min_delta:
                best_val = val_loss
                best_state = cpu_deepcopy_state(self.vae.state_dict())
                epochs_without_improve = 0
            else:
                epochs_without_improve += 1

            if epochs_without_improve >= patience:
                print(f"Early stopping at epoch {self.epoch}, best val {best_val:.4f}")
                self.save(pjoin(self.opt.model_dir, "latest.tar"), ep = self.epoch, total_it=it, history = history, best_model_state=best_state)
                self.save_loss_data(history = history)
                break

            
            if self.epoch % self.opt.save_every_e == 0:
                self.save(pjoin(self.opt.model_dir, "E%04d.tar" % self.epoch), self.epoch, total_it=it, history = history)

            if self.epoch % self.opt.eval_every_e == 0:
                print("Epoch:", self.epoch)
                print(
                    "Train Loss: %.5f Reconstruction Loss: %.5f "
                    "KL Loss: %.5f"
                    % (train_loss_avg, train_rec_avg, train_kl_avg)
                )
                print(
                    "Validation Loss: %.5f Reconstruction Loss: %.5f "
                    "KL Loss: %.5f"
                    % (val_loss, val_rec_loss, val_kl_loss)
                )
                save_dir = pjoin(self.opt.eval_dir, "E%04d" % self.epoch)
                os.makedirs(save_dir, exist_ok=True)
                self.save_loss_data(history = history)
            
            self.epoch += 1

        print("Epoch:", self.epoch)
        print(
            "Train Loss: %.5f Reconstruction Loss: %.5f "
            "KL Loss: %.5f"
            % (train_loss_avg, train_rec_avg, train_kl_avg)
        )
        print(
            "Validation Loss: %.5f Reconstruction Loss: %.5f "
            "KL Loss: %.5f"
            % (val_loss, val_rec_loss, val_kl_loss)
        )
        
        self.save_loss_data(history = history)



class MotionShortcutDiTTrainer(object):
    def __init__(self,
            args,
            dit: DiT,
            autoencoder_type: str
        ):
        self.opt = args
        self.dit = dit 
        self.device = args.device

        if args.is_train:
            self.logger = Logger(args.log_dir)

        self.dit = dit.to(self.device)

        weight_decay = 1e-3
        param_groups = self._get_param_groups(model = self.dit, weight_decay=weight_decay)

        self.opt_dit = torch.optim.AdamW(
            param_groups,
            lr=args.lr,
            betas=(0.9, 0.999),
            weight_decay=1e-3
        )

        self.vae = None
        #self.text_encoder = get_pretrained_text_encoder(self.device)
        self.text_encoder = TextTokenEncoder(device = self.device).to(self.device)
        self.text_encoder.eval()

        self.mean = np.load(pjoin(self.opt.meta_dir, 'mean.npy'))
        self.std = np.load(pjoin(self.opt.meta_dir, 'std.npy'))

        with open(pjoin(self.opt.meta_dir, 'part_mapping.json'), 'r') as f:
            mapping = json.load(f)

        self.joints_num = mapping['joints_num']

        self._init_vae(autoencoder_type)

    def _get_param_groups(self, model: DiT, weight_decay: float = 1e-4):
        decay_params = []
        no_decay_params = []

        for name, param in model.named_parameters():
            if not param.requires_grad:
                continue
            # 1D params are usually biases or norm weights -> no weight decay
            if param.ndim == 1 or name.endswith(".bias"):
                no_decay_params.append(param)
            else:
                decay_params.append(param)

        return [
            {"params": decay_params, "weight_decay": weight_decay},
            {"params": no_decay_params, "weight_decay": 0.0},
        ]

    def _init_vae(self, autoencoder_type: str):
        if autoencoder_type == "pretrained_vae":
            self.encoder, self.decoder = get_pretrained_vae(self.opt.model_dir)
            self.encoder.to(self.device)
            self.decoder.to(self.device)
        else:
            raise ValueError(f"Unknown vae_name: {autoencoder_type}")
        
    def _compute_foot_contact_loss(self, predicted_joints, target_joints, foot_ids = [7, 8], height_thresh = 0.08, vel_thresh = 0.01):

        B, T, J, _ = predicted_joints.shape

        feet_pred = predicted_joints[:, :, foot_ids, :]
        feet_target = target_joints[:, :, foot_ids, :]

        target_vel = feet_target[:, 1:] - feet_target[:, :-1]
        pred_vel = feet_pred[:, 1:] - feet_pred[:, :-1]
        pred_speed = pred_vel.norm(dim = -1)
        target_speed = target_vel.norm(dim = -1)
        #root_vel_loss = F.mse_loss(pred_vel, target_vel)
        root_vel_loss = F.smooth_l1_loss(pred_vel, target_vel)

        #target_speed_xy = torch.norm(target_vel[..., [0, 2]], dim = -1)
        target_height = feet_target[..., 1]
        contact = (target_height < height_thresh).float()

        pred_speed_xy = torch.norm(pred_vel[..., [0, 2]], dim = -1)

        foot_pos_error = torch.norm(feet_pred - feet_target, dim = -1)
        num = contact.sum()
        if num < 1:
            return predicted_joints.new_zeros(()), predicted_joints.new_zeros(()), root_vel_loss

        contact_loss = (contact[:, :-1, :] * (pred_speed_xy ** 2)).sum() / num
        foot_pos_loss = (contact * foot_pos_error).sum() / num

        return contact_loss, foot_pos_loss, root_vel_loss

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

    def denormalize_motion(self, motion):
        return motion * self.std + self.mean

    def save_loss_data(self, history):

        os.makedirs(self.opt.experiment_dir, exist_ok=True)
        epochs = range(1, len(history["train_loss"]) + 1)

        loss_pairs = [
            ("loss", "Total Loss"),
            ("mse_loss", "MSE Loss"),
            ("contrastive_loss", "Contrastive Loss"),
            ("contact_loss", "Contact Loss"),
            ("feet_pos_loss", "Feet Position Loss"),
            ("root_vel_loss", "Root Velocity Loss")
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
        self.dit.train()
        motions = batch_data['motion'].to(self.device).float()
        texts = batch_data['text']
        motion_masks = batch_data['motion_mask'].to(self.device).float()
        stride = 4
        T_enc = motion_masks.shape[0] // stride
        motion_masks_enc = motion_masks[:, ::stride].clone().float()
        #motion_masks = motion_masks.unsqueeze(-1)
        motion_masks_enc = motion_masks_enc.unsqueeze(-1)

        with torch.no_grad():
            self.latents = self.encoder(motions[:, :, :-4])
            #text_emb = self.text_encoder.encode(
                #texts,
                #convert_to_tensor = True,
                #device = str(self.device)
            #).float()
            self.text_tokens, self.text_mask = self.text_encoder.encode_tokens(texts)

        self.noise = torch.randn_like(self.latents)
        B = self.latents.shape[0]
        self.t = torch.rand(B, device = self.device)
        t_view = self.t.view(B, 1, 1)

        self.xt = (1.0 - t_view) * self.noise + t_view * self.latents
        self.d = torch.zeros(B, device = self.device)
        self.target = self.latents - self.noise
        # Check inputs to DiT
        if torch.isnan(self.xt).any():
            print("NaN in xt")
        if torch.isnan(self.text_tokens).any():
            print("NaN in text_tokens")
        if torch.isnan(self.target).any():
            print("NaN in target")
        self.pred = self.dit(
            self.xt,
            self.t,
            self.d,
            self.text_tokens,
            text_mask = self.text_mask
        )
        # Check output of DiT and loss
        if torch.isnan(self.pred).any():
            print("NaN in pred")


        #mse loss
        #print('pred shape', self.pred.shape, motion_masks_enc.shape)
        valid_pred = self.pred * motion_masks_enc
        lambda_feat = 0.3
        self.mse_loss = F.mse_loss(valid_pred, self.target)

        #contrastive loss term
        pred_flat = valid_pred.flatten(start_dim = 1).float()
        pred_flat = F.normalize(pred_flat, dim = 1)
        sim = pred_flat @ pred_flat.t()

        diff_mask = torch.tensor(
            [[texts[i] != texts[j] for j in range(B)] for i in range(B)],
            device = self.device,
            dtype = torch.bool
        )
        off_diag_idx = ~torch.eye(B, dtype = torch.bool, device = self.device)
        valid_els = off_diag_idx & diff_mask
        true_count = valid_els.sum().item()

        margin = 0.2
        if true_count > 0:
            sim = sim[valid_els]
            sim = sim.clamp(min = -1.0, max = 1.0)
            margin = 0.2
            diff = sim - margin
            diff = diff.clamp(min = -1.0, max = 1.0)
            self.contrastive_loss = F.relu(diff).mean()
        else:
            self.contrastive_loss = sim.new_zeros(())
        lambda_contrast = 0.05 # lambda contrastive text loss

        # foot contact component

        target_motions = self.decoder(self.target)
        pred_motions = self.decoder(valid_pred)
        target_motions = self.denormalize_motion(target_motions.detach().cpu())
        pred_motions = self.denormalize_motion(pred_motions.detach().cpu())
        target_motions_jts = recover_from_ric(target_motions.float(), self.joints_num)
        pred_motions_jts = recover_from_ric(pred_motions.float(), self.joints_num)
        #print('joints shape', target_motions_jts.shape, pred_motions_jts.shape)

        self.contact_loss, self.foot_pos_loss, self.root_vel_loss = self._compute_foot_contact_loss(
            predicted_joints=pred_motions_jts,
            target_joints=target_motions_jts
        )
        #self.contact_loss = F.mse_loss(self.pred[..., 259:263], self.target[..., 259:263])
        lambda_foot = 0.05 # lambda contact
        lambda_pos = 0.05 # lambda foot position
        lambda_root = 0.1 # lambda root velocity


        # total loss computation
        self.loss = (lambda_feat * self.mse_loss) + (lambda_contrast * self.contrastive_loss) + (lambda_foot * self.contact_loss) + (lambda_pos * self.foot_pos_loss) + (lambda_root * self.root_vel_loss)


    def update(self):
        if torch.isnan(self.loss):
            print("NaN loss before backward, skipping step")
            return OrderedDict()
        self.zero_grad([self.opt_dit])
        self.loss.backward()
        self.clip_norm([self.dit], 0.5)
        self.step([self.opt_dit])
        self.scheduler_dit.step()

        #loss_logs = OrderedDict()
        #loss_logs["loss"] = self.loss.detach().item()
        #loss_logs['mse_loss'] = self.mse_loss.detach().item()
        #loss_logs['contrastive_loss'] = self.contrastive_loss.detach().item()
        #return loss_logs

    def save(self, file_name, ep, train_batch_index, total_it, history = None, best_model_state = None):
        state = {
            "dit": best_model_state if best_model_state != None else cpu_deepcopy_state(self.dit.state_dict()),
            "opt_dit": cpu_deepcopy_state(self.opt_dit.state_dict()),
            "scheduler_dit": cpu_deepcopy_state(self.scheduler_dit.state_dict()),
            "ep": ep,
            "total_it": total_it,
            "train_batch_index": train_batch_index,
            "history": history
        }
        torch.save(state, file_name)

    def resume(self, model_dir):
        checkpoint = torch.load(model_dir, map_location=self.device)
        self.dit.load_state_dict(checkpoint["dit"])
        move_state_to_device(self.dit, self.device)
        self.opt_dit.load_state_dict(checkpoint["opt_dit"])
        move_state_to_device(self.opt_dit, self.device)
        self.scheduler_dit.load_state_dict(checkpoint["scheduler_dit"])
        move_state_to_device(self.scheduler_dit, self.device)
        return checkpoint["ep"], checkpoint["train_batch_index"], checkpoint["total_it"], checkpoint["history"]

    def train(self, train_dataloader, val_dataloader, plot_eval = None):
        self.dit.to(self.device)
        total_iters = self.opt.max_epoch * len(train_dataloader)

        self.scheduler_dit = CosineAnnealingLR(self.opt_dit, T_max = total_iters, eta_min = 1e-5)

        history = {
            "train_loss": [],
            "train_mse_loss": [],
            "train_contrastive_loss": [],
            "train_contact_loss": [],
            "train_feet_pos_loss": [],
            "train_root_vel_loss": [],
            "val_loss": [],
            "val_mse_loss": [],
            "val_contrastive_loss": [],
            "val_contact_loss": [],
            "val_feet_pos_loss": [],
            "val_root_vel_loss": []
        }
        
        print("Number of epochs:", self.opt.max_epoch)
        print("Number of steps:", total_iters)

        epoch = 0
        it = 0
        train_batch_index = -1
        B = len(train_dataloader)
        if self.opt.is_continue:
            model_dir = pjoin(self.opt.model_dir, "tmp.tar")
            try:
                epoch, train_batch_index, it, history = self.resume(model_dir)
                #print(f'Resuming training from previous checkpoint at epoch {epoch} from batch {train_batch_index} and iteration {it}')
            except Exception as e:
                print(f"Failed to load checkpoint from {model_dir}, trying last stable checkpoint. Error: {e}")
                model_dir = pjoin(self.opt.model_dir, "latest.tar")
                try:
                    epoch, train_batch_index, it, history = self.resume(model_dir)
                    #print(f'Resuming training from previous stable checkpoint at epoch {epoch} from batch {train_batch_index} and iteration {it}')
                except Exception as e:
                    print(f"Failed to load checkpoint from {model_dir}. Starting training from scratch. Error: {e}")

            if train_batch_index == B-1:
                # this is to handle resuming from a checkpoint that ended at max_epochs in a previous run and user decided to extend it.
                epoch += 1
                train_batch_index = -1
            print(f'Resuming training from previous checkpoint at epoch {epoch} from batch {train_batch_index} and iteration {it}')

        print("Iters Per Epoch, Training: %04d, Validation: %03d\n" %
              (len(train_dataloader), len(val_dataloader)))

        val_loss = 0
        logs = OrderedDict()

        # loss value init
        train_loss = 0
        val_loss = 0
        val_mse_loss_avg = 0
        val_contrastive_loss_avg = 0
        best_val = float("inf")
        best_state = None
        patience = self.opt.patience
        epochs_without_improve = 0
        min_delta = self.opt.min_loss_delta

        while epoch < self.opt.max_epoch:
            
            train_loss = 0.0
            train_mse_loss_sum = 0.0
            train_contrastive_loss_sum = 0.0
            train_contact_loss_sum = 0.0
            train_root_loss_sum = 0.0
            train_feet_pos_sum = 0.0
            train_steps = 0
            for i, batch_data in enumerate(train_dataloader):
                '''
                if train_batch_index != -1 and i <= train_batch_index:
                    continue
                '''
                self.dit.train()
                self.forward(batch_data)
                self.update()

                train_loss += self.loss.detach().cpu().item()
                train_contrastive_loss_sum += self.contrastive_loss.detach().cpu().item()
                train_mse_loss_sum += self.mse_loss.detach().cpu().item()
                train_contact_loss_sum += self.contact_loss.detach().cpu().item()
                train_root_loss_sum += self.root_vel_loss.detach().cpu().item()
                train_feet_pos_sum += self.foot_pos_loss.detach().cpu().item()

                train_steps += 1

                it += 1
                '''

                for k, v in log_dict.items():
                    if k not in logs:
                        logs[k] = v
                    else:
                        logs[k] += v

                if it % self.opt.log_every == 0:
                    mean_loss = OrderedDict({"val_loss": val_loss})
                    self.logger.scalar_summary("val_loss", val_loss, it)

                    for tag, value in logs.items():
                        self.logger.scalar_summary(tag, value / self.opt.log_every, it)
                        mean_loss[tag] = value / self.opt.log_every

                    logs = OrderedDict()
                    #print_current_loss_decomp(start_time, it, total_iters, mean_loss, epoch, i)
                '''

                if it % self.opt.save_latest == 0:
                    self.save(pjoin(self.opt.model_dir, "tmp.tar"), ep = epoch, train_batch_index = i, total_it = it, history = history)

            #epoch += 1

            train_loss = train_loss / max(train_steps, 1)
            train_mse_loss_sum = train_mse_loss_sum / max(train_steps, 1)
            train_contrastive_loss_sum = train_contrastive_loss_sum / max(train_steps, 1)
            train_contact_loss_sum = train_contact_loss_sum / max(train_steps, 1)
            train_feet_pos_sum = train_feet_pos_sum / max(train_steps, 1)
            train_root_loss_sum = train_root_loss_sum / max(train_steps, 1)

            history["train_loss"].append(train_loss)
            history['train_mse_loss'].append(train_mse_loss_sum)
            history['train_contrastive_loss'].append(train_contrastive_loss_sum)
            history['train_contact_loss'].append(train_contact_loss_sum)
            history['train_feet_pos_loss'].append(train_feet_pos_sum)
            history['train_root_vel_loss'].append(train_root_loss_sum)

            #print("Validation time:")
            val_loss = 0
            val_contrastive_loss_avg = 0
            val_mse_loss_avg = 0
            val_contact_loss_avg = 0
            val_feet_poss_loss_avg = 0
            val_root_vel_loss_avg = 0

            with torch.no_grad():
                self.dit.eval()
                for i, batch_data in enumerate(val_dataloader):
                    self.forward(batch_data)

                    val_loss += self.loss.item()
                    val_contrastive_loss_avg += self.contrastive_loss.item()
                    val_mse_loss_avg += self.mse_loss.item()
                    val_contact_loss_avg += self.contact_loss.item()
                    val_feet_poss_loss_avg += self.foot_pos_loss.item()
                    val_root_vel_loss_avg += self.root_vel_loss.item()

            denom = max(len(val_dataloader), 1)
            val_loss /= denom
            val_contrastive_loss_avg /= denom
            val_mse_loss_avg /= denom
            val_feet_poss_loss_avg /= denom
            val_root_vel_loss_avg /= denom
            val_contact_loss_avg /= denom
            history["val_loss"].append(val_loss)
            history["val_contrastive_loss"].append(val_contrastive_loss_avg)
            history["val_mse_loss"].append(val_mse_loss_avg)
            history["val_contact_loss"].append(val_contact_loss_avg)
            history["val_feet_pos_loss"].append(val_feet_poss_loss_avg)
            history["val_root_vel_loss"].append(val_root_vel_loss_avg)

            if os.path.exists(pjoin(self.opt.model_dir, "tmp.tar")):
                try:
                    model_ckpt = torch.load(pjoin(self.opt.model_dir, "tmp.tar"), map_location="cpu")
                    self.save(pjoin(self.opt.model_dir, "latest.tar"), ep = epoch, train_batch_index = -1, total_it = it, history = history)
                    del model_ckpt
                except Exception as e:
                    print(f"Failed to load checkpoint from {pjoin(self.opt.model_dir, 'tmp.tar')}. Skipping save to latest.tar. Error: {e}")
                os.remove(pjoin(self.opt.model_dir, "tmp.tar")) # removing tar if latest stable is saved

            if best_val - val_loss > min_delta:
                best_val = val_loss
                best_state = cpu_deepcopy_state(self.dit.state_dict())
                epochs_without_improve = 0
            else:
                epochs_without_improve += 1

            if epochs_without_improve >= patience:
                print(f"Early stopping at epoch {epoch}, best val {best_val:.4f}")
                self.save(pjoin(self.opt.model_dir, "latest.tar"), ep = epoch, train_batch_index=-1, total_it=it, history = history, best_model_state=best_state)
                self.save_loss_data(history = history)
                break
            

            if epoch % self.opt.save_every_e == 0:
                self.save(pjoin(self.opt.model_dir, "E%04d.tar" % epoch), ep = epoch, train_batch_index=-1, total_it=it, history = history)

            if epoch % self.opt.log_every == 0:
                print("Epoch:", epoch)
                print(
                    "Train Loss: %.5f"
                    % (train_loss)
                )
                print(
                    "Validation Loss: %.5f"
                    % (val_loss)
                )

            if epoch % self.opt.eval_every_e == 0:
                #data = torch.cat([self.recon_motions_by_part, self.motions_by_part], dim=0).detach().cpu().numpy()
                save_dir = pjoin(self.opt.eval_dir, "E%04d" % epoch)
                os.makedirs(save_dir, exist_ok=True)
                #plot_eval(data, save_dir)
                self.save_loss_data(history = history)
            
            epoch += 1

        print("Epoch:", epoch)
        print(
            "Train Loss: %.5f"
            % (train_loss)
        )
        print(
            "Validation Loss: %.5f"
            % (val_loss)
        )
        
        self.save_loss_data(history = history)



class MotionDiTTrainer(object):
    def __init__(self,
            args,
            dit: DiT,
            autoencoder_type: str,
            num_train_timesteps = 1000,
            beta_start = 1e-4,
            beta_end = 2e-2,
            prediction_type = "epsilon",
            grad_clip = None
        ):
        self.opt = args
        self.dit = dit 
        self.device = args.device
        self.num_train_timesteps = num_train_timesteps
        self.prediction_type = prediction_type
        self.grad_clip = grad_clip

        if args.is_train:
            self.logger = Logger(args.log_dir)

        self.dit = dit.to(self.device)

        weight_decay = 1e-3
        param_groups = self._get_param_groups(model = self.dit, weight_decay=weight_decay)

        self.opt_dit = torch.optim.AdamW(
            param_groups,
            lr=args.lr,
            betas=(0.9, 0.999),
            weight_decay=1e-3
        )

        self.vae = None
        self.text_encoder = TextTokenEncoder(model_name="clip_text", device = self.device).to(self.device)
        self.text_encoder.eval()

        self.mean = np.load(pjoin(self.opt.meta_dir, 'mean.npy'))
        self.std = np.load(pjoin(self.opt.meta_dir, 'std.npy'))

        with open(pjoin(self.opt.meta_dir, 'part_mapping.json'), 'r') as f:
            mapping = json.load(f)

        self.joints_num = mapping['joints_num']

        self._init_vae(autoencoder_type)

        betas = torch.linspace(beta_start, beta_end, num_train_timesteps, dtype = torch.float32)
        alphas = 1.0 - betas
        alphas_cumprod = torch.cumprod(alphas, dim = 0)
        alphas_cumprod_prev = torch.cat(
            [torch.tensor([1.0], dtype = torch.float32), alphas_cumprod[:-1]], dim = 0
        )
        self.betas = betas.to(self.device)
        self.alphas = alphas.to(self.device)
        self.alphas_cumprod = alphas_cumprod.to(self.device)
        self.alphas_cumprod_prev = alphas_cumprod_prev.to(self.device)
        self.sqrt_alphas_cumprod = torch.sqrt(alphas_cumprod).to(self.device)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - alphas_cumprod).to(self.device)
        self.sqrt_recip_alphas = torch.sqrt(1.0 / alphas).to(self.device)
        self.posterior_variance = (
            betas * (1.0 - alphas_cumprod_prev) / (1.0 - alphas_cumprod)
        ).to(self.device)

    def _get_param_groups(self, model: DiT, weight_decay: float = 1e-4):
        decay_params = []
        no_decay_params = []

        for name, param in model.named_parameters():
            if not param.requires_grad:
                continue
            # 1D params are usually biases or norm weights -> no weight decay
            if param.ndim == 1 or name.endswith(".bias"):
                no_decay_params.append(param)
            else:
                decay_params.append(param)

        return [
            {"params": decay_params, "weight_decay": weight_decay},
            {"params": no_decay_params, "weight_decay": 0.0},
        ]

    def _init_vae(self, autoencoder_type: str):
        if autoencoder_type == "pretrained_vae":
            self.encoder, self.decoder = get_pretrained_vae(self.opt.model_dir)
            self.encoder.to(self.device)
            self.decoder.to(self.device)
        else:
            raise ValueError(f"Unknown vae_name: {autoencoder_type}")
        
    def _compute_foot_contact_loss(self, predicted_joints, target_joints, foot_ids = [7, 8], height_thresh = 0.08, vel_thresh = 0.01):

        B, T, J, _ = predicted_joints.shape

        root_pred = predicted_joints[:, :, 0, :]
        root_target = target_joints[:, :, 0, :]

        target_vel = root_target[:, 1:] - root_target[:, :-1]
        pred_vel = root_pred[:, 1:] - root_pred[:, :-1]
        pred_speed = pred_vel.norm(dim = -1)
        target_speed = target_vel.norm(dim = -1)
        #root_vel_loss = F.mse_loss(pred_vel, target_vel)
        root_vel_loss = F.smooth_l1_loss(pred_speed, target_speed)

        pos_loss = F.mse_loss(predicted_joints, target_joints)

        #target_speed_xy = torch.norm(target_vel[..., [0, 2]], dim = -1)
        feet_target = target_joints[:, :, foot_ids, :]
        feet_pred = predicted_joints[:, :, foot_ids, :]
        pred_feet_vel = feet_pred[:, 1:] - feet_pred[:, :-1]
        target_height = feet_target[..., 1]
        contact = (target_height < height_thresh).float()

        pred_speed_xy = torch.norm(pred_feet_vel[..., [0, 2]], dim = -1)
        
        #print('target height: ', contact.size(), target_height.size(), feet_target.size(), pred_speed_xy.size())

        #foot_pos_error = torch.norm(feet_pred - feet_target, dim = -1)
        num = contact.sum()
        if num < 1:
            return predicted_joints.new_zeros(()), predicted_joints.new_zeros(()), root_vel_loss

        contact_loss = (contact[:, :-1, :] * (pred_speed_xy ** 2)).sum() / num
        #foot_pos_loss = (contact * foot_pos_error).sum() / num

        return contact_loss, pos_loss, root_vel_loss
    
    def _extract(self, a, t, x_shape):
        out = a.gather(0, t)
        return out.view(t.shape[0], *((1,) * (len(x_shape) - 1)))

    def _q_sample(self, x_start, t, noise=None):
        if noise is None:
            noise = torch.randn_like(x_start)

        sqrt_alphas_cumprod_t = self._extract(self.sqrt_alphas_cumprod, t, x_start.shape)
        sqrt_one_minus_alphas_cumprod_t = self._extract(
            self.sqrt_one_minus_alphas_cumprod, t, x_start.shape
        )

        return sqrt_alphas_cumprod_t * x_start + sqrt_one_minus_alphas_cumprod_t * noise

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

    def predict_x0_from_eps(self, x_t, eps_pred, sqrt_alphas_cumprod, sqrt_one_minus_alphas_cumprod, t):
        a = self._extract(sqrt_alphas_cumprod, t, x_t.shape)
        b = self._extract(sqrt_one_minus_alphas_cumprod, t, x_t.shape)
        return (x_t - b * eps_pred) / (a + 1e-8)

    def denormalize_motion(self, motion):
        return motion * self.std + self.mean

    def save_loss_data(self, history):

        os.makedirs(self.opt.experiment_dir, exist_ok=True)
        epochs = range(1, len(history["train_loss"]) + 1)

        loss_pairs = [
            ("loss", "Total Loss"),
            ("mse_loss", "MSE Loss"),
            ("pad_loss", "Padding Loss"),
            ("root_vel_loss", "Root Velocity Loss")
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
        self.dit.train()
        motions = batch_data['motion'].to(self.device).float()
        texts = batch_data['text']
        motion_masks = batch_data['motion_mask'].to(self.device).float()
        stride = 4
        motion_masks_enc = motion_masks[:, ::stride].clone().float()
        #motion_masks = motion_masks.unsqueeze(-1)
        motion_masks_enc = motion_masks_enc.unsqueeze(-1)

        with torch.no_grad():
            self.latents = self.encoder(motions[:, :, :-4])
            self.text_unpooled_embeddings, self.text_pooled_embeddings, self.text_mask = self.text_encoder.encode_tokens(texts)
            #print('text embeddings: ', self.text_pooled_embeddings.shape, self.text_unpooled_embeddings.shape, self.text_mask.shape)

        B = self.latents.shape[0]
        t = torch.randint(
            0, self.num_train_timesteps, (B, ), device = self.device, dtype = torch.long
        )
        d = torch.zeros_like(t)
        noise = torch.randn_like(self.latents)
        x_t = self._q_sample(x_start = self.latents, t = t, noise = noise)

        if self.prediction_type == "epsilon":
            self.target = noise
        elif self.prediction_type == "x0":
            self.target = self.latents

        # Check inputs to DiT
        if torch.isnan(x_t).any():
            print("NaN in xt")
        if torch.isnan(self.text_pooled_embeddings).any():
            print("NaN in text_embeddings")
        if torch.isnan(self.target).any():
            print("NaN in target")

        self.pred = self.dit(
            x_t,
            t,
            d,
            self.text_pooled_embeddings,
            self.text_unpooled_embeddings,
            text_mask = self.text_mask
        )
        # Check output of DiT and loss
        if torch.isnan(self.pred).any():
            print("NaN in pred")
        
        masked_pred = self.pred * motion_masks_enc
        masked_target = self.target * motion_masks_enc

        x0_pred = self.predict_x0_from_eps(x_t, self.pred, self.sqrt_alphas_cumprod, self.sqrt_one_minus_alphas_cumprod, t)
        motion_pred = self.decoder(x0_pred)
        gt_motions = self.denormalize_motion(motions.detach().cpu())
        pred_motions = self.denormalize_motion(motion_pred.detach().cpu())
        gt_motions_jts = recover_from_ric(gt_motions.float(), self.joints_num)
        pred_motions_jts = recover_from_ric(pred_motions.float(), self.joints_num)
        #lambda_pos = 0.05
        lambda_vel = 0.05
        #lambda_contact = 0.01
        #print('jts shape: ', pred_motions_jts.shape, gt_motions_jts.shape)
        _, _, self.root_vel_loss = self._compute_foot_contact_loss(pred_motions_jts, gt_motions_jts)
        

        pad_mask = 1.0 - motion_masks_enc
        pad_pred = self.pred * pad_mask
        self.pad_loss = (pad_pred ** 2).mean()
        lambda_pad = 0.1

        # mse_loss
        self.mse_loss = F.mse_loss(masked_pred, masked_target)

        self.loss = self.mse_loss + (lambda_pad * self.pad_loss) + (lambda_vel * self.root_vel_loss)


    def update(self):
        if torch.isnan(self.loss):
            print("NaN loss before backward, skipping step")
            return OrderedDict()
        self.zero_grad([self.opt_dit])
        self.loss.backward()
        self.clip_norm([self.dit], 0.5)
        self.step([self.opt_dit])
        self.scheduler_dit.step()

        #loss_logs = OrderedDict()
        #loss_logs["loss"] = self.loss.detach().item()
        #loss_logs['mse_loss'] = self.mse_loss.detach().item()
        #loss_logs['contrastive_loss'] = self.contrastive_loss.detach().item()
        #return loss_logs

    def save(self, file_name, ep, train_batch_index, total_it, history = None, best_model_state = None):
        state = {
            "dit": best_model_state if best_model_state != None else cpu_deepcopy_state(self.dit.state_dict()),
            "opt_dit": cpu_deepcopy_state(self.opt_dit.state_dict()),
            "scheduler_dit": cpu_deepcopy_state(self.scheduler_dit.state_dict()),
            "ep": ep,
            "total_it": total_it,
            "train_batch_index": train_batch_index,
            "history": history
        }
        torch.save(state, file_name)

    def resume(self, model_dir):
        checkpoint = torch.load(model_dir, map_location=self.device)
        self.dit.load_state_dict(checkpoint["dit"])
        move_state_to_device(self.dit, self.device)
        self.opt_dit.load_state_dict(checkpoint["opt_dit"])
        move_state_to_device(self.opt_dit, self.device)
        self.scheduler_dit.load_state_dict(checkpoint["scheduler_dit"])
        move_state_to_device(self.scheduler_dit, self.device)
        return checkpoint["ep"], checkpoint["train_batch_index"], checkpoint["total_it"], checkpoint["history"]

    def train(self, train_dataloader, val_dataloader, plot_eval = None):
        self.dit.to(self.device)
        total_iters = self.opt.max_epoch * len(train_dataloader)

        self.scheduler_dit = CosineAnnealingLR(self.opt_dit, T_max = total_iters, eta_min = 1e-5)

        history = {
            "train_loss": [],
            "train_mse_loss": [],
            "train_pad_loss": [],
            #"train_contact_loss": [],
            #"train_pos_loss": [],
            "train_root_vel_loss": [],
            "val_loss": [],
            "val_mse_loss": [],
            "val_pad_loss": [],
            #"val_contact_loss": [],
            #"val_pos_loss": [],
            "val_root_vel_loss": [],
        }
        
        print("Number of epochs:", self.opt.max_epoch)
        print("Number of steps:", total_iters)

        epoch = 0
        it = 0
        train_batch_index = -1
        B = len(train_dataloader)
        if self.opt.is_continue:
            model_dir = pjoin(self.opt.model_dir, "tmp.tar")
            try:
                epoch, train_batch_index, it, history = self.resume(model_dir)
                #print(f'Resuming training from previous checkpoint at epoch {epoch} from batch {train_batch_index} and iteration {it}')
            except Exception as e:
                print(f"Failed to load checkpoint from {model_dir}, trying last stable checkpoint. Error: {e}")
                model_dir = pjoin(self.opt.model_dir, "latest.tar")
                try:
                    epoch, train_batch_index, it, history = self.resume(model_dir)
                    #print(f'Resuming training from previous stable checkpoint at epoch {epoch} from batch {train_batch_index} and iteration {it}')
                except Exception as e:
                    print(f"Failed to load checkpoint from {model_dir}. Starting training from scratch. Error: {e}")

            if train_batch_index == B-1:
                # this is to handle resuming from a checkpoint that ended at max_epochs in a previous run and user decided to extend it.
                epoch += 1
                train_batch_index = -1
            print(f'Resuming training from previous checkpoint at epoch {epoch} from batch {train_batch_index} and iteration {it}')

        print("Iters Per Epoch, Training: %04d, Validation: %03d\n" %
              (len(train_dataloader), len(val_dataloader)))

        val_loss = 0
        logs = OrderedDict()

        # loss value init
        train_loss = 0
        val_loss = 0
        val_mse_loss_avg = 0
        val_pad_loss_avg = 0
        val_contact_loss_avg = 0
        val_pos_loss_avg = 0
        val_root_vel_loss_avg = 0
        best_val = float("inf")
        best_state = None
        patience = self.opt.patience
        epochs_without_improve = 0
        min_delta = self.opt.min_loss_delta

        while epoch < self.opt.max_epoch:
            
            train_loss = 0.0
            train_mse_loss_sum = 0.0
            train_pad_loss_sum = 0.0
            train_contact_loss_sum = 0.0
            train_pos_loss_sum = 0.0
            train_root_vel_loss_sum = 0.0
            train_steps = 0
            for i, batch_data in enumerate(train_dataloader):
                '''
                if train_batch_index != -1 and i <= train_batch_index:
                    continue
                '''
                self.dit.train()
                self.forward(batch_data)
                self.update()

                train_loss += self.loss.detach().cpu().item()
                train_mse_loss_sum += self.mse_loss.detach().cpu().item()
                train_pad_loss_sum += self.pad_loss.detach().cpu().item()
                #train_contact_loss_sum += self.contact_loss.detach().cpu().item()
                #train_pos_loss_sum += self.pos_loss.detach().cpu().item()
                train_root_vel_loss_sum += self.root_vel_loss.detach().cpu().item()

                train_steps += 1

                it += 1

                if it % self.opt.save_latest == 0:
                    self.save(pjoin(self.opt.model_dir, "tmp.tar"), ep = epoch, train_batch_index = i, total_it = it, history = history)

            #epoch += 1

            train_loss = train_loss / max(train_steps, 1)
            train_mse_loss_sum = train_mse_loss_sum / max(train_steps, 1)
            train_pad_loss_sum = train_pad_loss_sum / max(train_steps, 1)
            train_contact_loss_sum = train_contact_loss_sum / max(train_steps, 1)
            train_pos_loss_sum  = train_pos_loss_sum / max(train_steps, 1)
            train_root_vel_loss_sum = train_root_vel_loss_sum / max(train_steps, 1)

            history["train_loss"].append(train_loss)
            history['train_mse_loss'].append(train_mse_loss_sum)
            history['train_pad_loss'].append(train_pad_loss_sum)
            #history['train_contact_loss'].append(train_contact_loss_sum)
            #history['train_pos_loss'].append(train_pos_loss_sum)
            history['train_root_vel_loss'].append(train_root_vel_loss_sum)

            #print("Validation time:")
            val_loss = 0
            val_mse_loss_avg = 0
            val_pad_loss_avg = 0
            val_contact_loss_avg = 0
            val_pos_loss_avg = 0
            val_root_vel_loss_avg = 0

            with torch.no_grad():
                self.dit.eval()
                for i, batch_data in enumerate(val_dataloader):
                    self.forward(batch_data)

                    val_loss += self.loss.item()
                    val_mse_loss_avg = self.mse_loss.item()
                    val_pad_loss_avg = self.pad_loss.item()
                    #val_contact_loss_avg = self.contact_loss.item()
                    #val_pos_loss_avg = self.pos_loss.item()
                    val_root_vel_loss_avg = self.root_vel_loss.item()


            denom = max(len(val_dataloader), 1)
            val_loss /= denom
            val_mse_loss_avg /= denom
            val_pad_loss_avg /= denom
            history["val_loss"].append(val_loss)
            history["val_mse_loss"].append(val_mse_loss_avg)
            history["val_pad_loss"].append(val_pad_loss_avg)
            #history["val_contact_loss"].append(val_contact_loss_avg)
            #history["val_pos_loss"].append(val_pos_loss_avg)
            history["val_root_vel_loss"].append(val_root_vel_loss_avg)

            if os.path.exists(pjoin(self.opt.model_dir, "tmp.tar")):
                try:
                    model_ckpt = torch.load(pjoin(self.opt.model_dir, "tmp.tar"), map_location="cpu")
                    self.save(pjoin(self.opt.model_dir, "latest.tar"), ep = epoch, train_batch_index = -1, total_it = it, history = history)
                    del model_ckpt
                except Exception as e:
                    print(f"Failed to load checkpoint from {pjoin(self.opt.model_dir, 'tmp.tar')}. Skipping save to latest.tar. Error: {e}")
                os.remove(pjoin(self.opt.model_dir, "tmp.tar")) # removing tar if latest stable is saved

            if best_val - val_loss > min_delta:
                best_val = val_loss
                best_state = cpu_deepcopy_state(self.dit.state_dict())
                epochs_without_improve = 0
            else:
                epochs_without_improve += 1

            if epochs_without_improve >= patience:
                print(f"Early stopping at epoch {epoch}, best val {best_val:.4f}")
                self.save(pjoin(self.opt.model_dir, "latest.tar"), ep = epoch, train_batch_index=-1, total_it=it, history = history, best_model_state=best_state)
                self.save_loss_data(history = history)
                break
            

            if epoch % self.opt.save_every_e == 0:
                self.save(pjoin(self.opt.model_dir, "E%04d.tar" % epoch), ep = epoch, train_batch_index=-1, total_it=it, history = history)

            if epoch % self.opt.log_every == 0:
                print("Epoch:", epoch)
                print(
                    "Train Loss: %.5f"
                    % (train_loss)
                )
                print(
                    "Validation Loss: %.5f"
                    % (val_loss)
                )

            if epoch % self.opt.eval_every_e == 0:
                #data = torch.cat([self.recon_motions_by_part, self.motions_by_part], dim=0).detach().cpu().numpy()
                save_dir = pjoin(self.opt.eval_dir, "E%04d" % epoch)
                os.makedirs(save_dir, exist_ok=True)
                #plot_eval(data, save_dir)
                self.save_loss_data(history = history)
            
            epoch += 1

        print("Epoch:", epoch)
        print(
            "Train Loss: %.5f"
            % (train_loss)
        )
        print(
            "Validation Loss: %.5f"
            % (val_loss)
        )
        
        self.save_loss_data(history = history)
