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
from networks.nn import MotionVQVAE, DiT
from torch.optim.lr_scheduler import CosineAnnealingLR
from networks.autoencoder_modules import MovementConvEncoder, MovementConvDecoder
from utils.pretrained_model_utils import get_pretrained_vae, get_pretrained_text_encoder
from networks.transformer_modules import TextTokenEncoder
from copy import deepcopy

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


class MotionDiTTrainer(object):
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
            ("mse_loss", "MSE Loss"),
            ("contrastive_loss", "Contrastive Loss")
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
        self.mse_loss = F.mse_loss(self.pred, self.target)
        self.contrastive_loss = torch.tensor(0.0)

        #contrastive loss term
        pred_flat = self.pred.flatten(start_dim = 1).float()
        pred_flat = F.normalize(pred_flat, dim = 1)
        sim = pred_flat @ pred_flat.t()

        diff_mask = torch.tensor(
            [[texts[i] != texts[j] for j in range(B)] for i in range(B)],
            device = self.device,
            dtype = torch.bool
        )
        off_diag_idx = ~torch.eye(B, dtype = torch.bool, device = self.device)
        print('diff text in batch', diff_mask.sum().item())
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
        lambda_contrast = 0.05
        print('mse loss vs contrastive loss', self.mse_loss, self.contrastive_loss)
        self.loss = self.mse_loss + lambda_contrast * self.contrastive_loss


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
            "val_loss": [],
            "val_mse_loss": [],
            "val_contrastive_loss": []
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
        train_loss_avg = 0
        train_mse_loss_avg = 0
        train_contrastive_loss_avg = 0
        val_loss = 0
        val_mse_loss_avg = 0
        val_contrastive_loss_avg = 0
        best_val = float("inf")
        best_state = None
        patience = 10
        epochs_without_improve = 0
        min_delta = 1e-3

        while epoch < self.opt.max_epoch:
            
            train_loss_sum = 0.0
            train_mse_loss_sum = 0.0
            train_contrastive_loss_sum = 0.0
            train_steps = 0
            for i, batch_data in enumerate(train_dataloader):
                '''
                if train_batch_index != -1 and i <= train_batch_index:
                    continue
                '''
                self.dit.train()
                self.forward(batch_data)
                self.update()

                train_loss_sum += self.loss.detach().cpu().item()
                train_contrastive_loss_sum += self.contrastive_loss.detach().cpu().item()
                train_mse_loss_sum += self.mse_loss.detach().cpu().item()

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

            train_loss_avg = train_loss_sum / max(train_steps, 1)
            train_mse_loss_avg = train_mse_loss_sum / max(train_steps, 1)
            train_contrastive_loss_avg = train_contrastive_loss_sum / max(train_steps, 1)

            history["train_loss"].append(train_loss_avg)
            history['train_mse_loss'].append(train_mse_loss_avg)
            history['train_contrastive_loss'].append(train_contrastive_loss_avg)

            #print("Validation time:")
            val_loss = 0
            val_contrastive_loss_avg = 0
            val_mse_loss_avg = 0

            with torch.no_grad():
                self.dit.eval()
                for i, batch_data in enumerate(val_dataloader):
                    self.forward(batch_data)

                    val_loss += self.loss.item()
                    val_contrastive_loss_avg += self.contrastive_loss.item()
                    val_mse_loss_avg += self.mse_loss.item()

            denom = max(len(val_dataloader), 1)
            val_loss /= denom
            val_contrastive_loss_avg /= denom
            val_mse_loss_avg /= denom
            history["val_loss"].append(val_loss)
            history["val_contrastive_loss"].append(val_contrastive_loss_avg)
            history["val_mse_loss"].append(val_mse_loss_avg)

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
                    % (train_loss_avg)
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
            % (train_loss_avg)
        )
        print(
            "Validation Loss: %.5f"
            % (val_loss)
        )
        
        self.save_loss_data(history = history)

