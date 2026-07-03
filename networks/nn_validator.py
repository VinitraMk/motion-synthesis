import csv
import math
from modulefinder import test
import os
from dataclasses import dataclass
from typing import Any, Dict, List
import json
import numpy as np
import torch
import random
from utils.utils import ensure_dir
from utils.nn_utils import move_batch_to_device
from data_utils.motion_processor import render_skeleton_animation, HUMANML3D_SKELETON_EDGES, render_skeleton_single_animation
from data_utils.motion_processor import recover_from_ric
from os.path import join as pjoin
from networks.autoencoder_modules import MovementConvDecoder, MovementConvEncoder
from utils.pretrained_model_utils import get_pretrained_vae, get_pretrained_text_encoder
from networks.transformer_modules import TextTokenEncoder

# this will just validate whether latents from training/validation sample and close to those samples
# reconstruct without errors
class VQVAEValidator:

    def __init__(self, opt, vqvae, train_dataloader, val_dataloader, samples_to_test: int = 5, video_dir_name: str = 'videos', tensors_dir_name: str = 'tensors', metrics_dir_name: str = 'metrics'):
        self.opt = opt
        self.vqvae = vqvae
        self.vqvae.eval()
        self.val_dataloader = val_dataloader
        self.train_dataloader = train_dataloader
        self.samples_to_test = samples_to_test
        self.sampling_seed = 42

        self.videos_dir = pjoin(opt.output_dir, f'{video_dir_name}')
        self.tensors_dir = pjoin(opt.output_dir, f'{tensors_dir_name}')
        self.metrics_dir = pjoin(opt.output_dir, f'{metrics_dir_name}')
        ensure_dir(self.videos_dir)
        ensure_dir(self.tensors_dir)
        ensure_dir(self.metrics_dir)
        self.device = torch.device("cpu")

        self.mean = np.load(pjoin(opt.meta_dir, 'mean.npy'))
        self.std = np.load(pjoin(opt.meta_dir, 'std.npy'))
        self.d_part_max = 60

        with open(pjoin(opt.meta_dir, 'part_mapping.json'), 'r') as f:
            mapping = json.load(f)

        self.part_names = mapping['part_names']
        self.d_part_max = mapping['d_part_max']
        self.joints_num = mapping['joints_num']
        self.part_feature_indices = {
            k: np.array(v, dtype=np.int64)
            for k, v in mapping['part_feature_indices'].items()
        }
        self.motion_dim = self.mean.shape[0]
        self.pretrained_movementenc, self.pretrained_movementdec = get_pretrained_vae(self.opt.checkpoints_dir) 
        self.pretrained_movementenc.to(self.device)
        self.pretrained_movementdec.to(self.device)

        
    @torch.no_grad()
    def _compute_metrics(self, x: torch.Tensor, x_recon: torch.Tensor) -> Dict[str, float]:
        diff = x_recon - x
        l1 = diff.abs().mean().item()
        mse = (diff ** 2).mean().item()
        rmse = float(np.sqrt(mse))
        max_abs = diff.abs().max().item()
        return {
            'l1': l1,
            'mse': mse,
            'rmse': rmse,
            'max_abs': max_abs,
        }

    def motion_to_parts(self, motion):
        T, D = motion.shape
        P = len(self.part_names)
        motion_parts = np.zeros((T, P, self.d_part_max), dtype=np.float32)

        for p, part_name in enumerate(self.part_names):
            idxs = self.part_feature_indices[part_name]
            part_feat = motion[:, idxs]
            motion_parts[:, p, :part_feat.shape[1]] = part_feat

        return motion_parts.astype(np.float32)

    def motion_parts_to_full_motion(self, motion_parts):
        """
        motion_parts:
            torch.Tensor or np.ndarray
            shape (1, T, P, D_part_max) or (T, P, D_part_max)

        returns:
            np.ndarray of shape (T, motion_dim)  # usually (T, 263)
            still normalized
        """
        if torch.is_tensor(motion_parts):
            motion_parts = motion_parts.detach().cpu().numpy()

        if motion_parts.ndim == 4:
            motion_parts = motion_parts[0]   # -> (T, P, D_part_max)

        T, P, Dp = motion_parts.shape
        full_motion = np.zeros((T, self.motion_dim), dtype=np.float32)

        for p, part_name in enumerate(self.part_names):
            idxs = self.part_feature_indices[part_name]
            part_feat = motion_parts[:, p, :len(idxs)]
            full_motion[:, idxs] = part_feat

        return full_motion

    def denormalize_motion(self, motion):
        return motion * self.std + self.mean

    def _get_result_from_vqvae(self, x: torch.Tensor, clip_id: str, sample_id: str, sample_text: str, full_text: str = "", recon_caption: str = "", x_baseline: torch.Tensor = None):
        #x = batch_motion_parts[random_sample_idx].unsqueeze(0).float()
        out = self.vqvae.forward(x)
        x_recon = out['x_recon']
        metrics = self._compute_metrics(x, x_recon)
        base_matrics = {}

        if x_baseline != None:
            self.pretrained_movementenc.eval()
            self.pretrained_movementdec.eval()
            enc_out = self.pretrained_movementenc(x_baseline[...,:-4])
            x_base_recon = self.pretrained_movementdec(enc_out)
            base_matrics = self._compute_metrics(x_baseline, x_base_recon)
        #clip_id = clip_ids[random_sample_idx]
        #sample_id = f'{dataset_type}_{clip_id}_{random_sample_idx}'

        #print(type(metrics['l1']), type(out['loss']))
        row = {
            'clip_id': clip_id,
            'snippet_id': sample_id,
            'shape': tuple(x.shape),
            'l1': metrics['l1'],
            'mse': metrics['mse'],
            'rmse': metrics['rmse'],
            'max_abs': metrics['max_abs'],
            'base_l1': base_matrics.get('l1', None),
            'base_mse': base_matrics.get('mse', None),
            'base_rmse': base_matrics.get('rmse', None),
            'base_max_abs': base_matrics.get('max_abs', None),
            'recon_loss_model': out['recon_loss'].item(),
            'total_loss_model': out['loss'].item(),
            'vq_loss_model': out['vq_loss'].item(),
            'codebook_loss_model': out['codebook_loss'].item(),
            'commitment_loss_model': out['commitment_loss'].item(),
            'full_text': full_text,
            'sample_text': sample_text
        }

        # save output tensors
        torch.save({
            'snippet_id': sample_id,
            'x': x.detach().cpu(),
            'x_base_recon': x_base_recon.detach().cpu() if x_baseline is not None else None,
            'x_recon': x_recon.detach().cpu(),
        }, os.path.join(self.tensors_dir, f'{sample_id}.pt'))


        # extract video

        try:
            full_motion_gt = self.motion_parts_to_full_motion(x.detach().cpu())
            #print('full motion gt in vqvae: ', full_motion_gt[..., self.missing_parts_indices])
            full_motion_gt = self.denormalize_motion(full_motion_gt)
            full_motion_recon = self.motion_parts_to_full_motion(x_recon.detach().cpu())
            full_motion_recon = self.denormalize_motion(full_motion_recon)

            joints_gt = recover_from_ric(torch.from_numpy(full_motion_gt).float(), self.joints_num)
            joints_recon = recover_from_ric(torch.from_numpy(full_motion_recon).float(), self.joints_num)
            joints_baseline = None

            if x_baseline != None:
                full_motion_base_recon = self.denormalize_motion(x_base_recon[0].detach().cpu())
                joints_baseline = recover_from_ric(full_motion_base_recon.detach().cpu().float(), self.joints_num)


            if joints_gt is not None and joints_recon is not None:
                #print('joints gt, recon', joints_gt.shape, joints_recon.shape, self.videos_dir)
                video_path = render_skeleton_animation(
                    joints_gt=joints_gt,
                    joints_recon=joints_recon,
                    skeleton_edges=HUMANML3D_SKELETON_EDGES,
                    output_path_no_ext=pjoin(self.videos_dir, sample_id),
                    clip_id = sample_id,
                    joints_baseline=joints_baseline,
                    text = sample_text,
                    fps=20,
                    save_mp4=True
                )
                if video_path != "" or video_path != None:
                    print('Saved video file at: ', video_path)
                row["video_path"] = video_path or ""
                row['video_error'] = ""
        except Exception as exc:
            print('Exception in getting joints: ', exc)
            row["video_path"] = ""
            row["video_error"] = str(exc)

        return row
    
    def validate_dataset(self, dataloader, dataset_type = 'train'):

        #self.vqvae.eval()
        rows: List[Dict[str, Any]] = []
        rng = random.Random(self.sampling_seed)
        num_batches = len(dataloader)
        k = min(self.samples_to_test, num_batches)
        batch_indices = rng.sample(range(num_batches), k)

        for bi, batch in enumerate(dataloader):

            if bi in batch_indices: 
                batch = move_batch_to_device(batch, self.device)
                batch_motion_parts = batch['motion_parts']
                batch_motion = batch['motion']
                batch_size = batch_motion_parts.size(0)

                batch_text = batch['text']
                #print('batch texts: ', batch_text)
                clip_ids = batch['file_id']
                random_sample_idx = rng.sample(range(batch_size), 1)[0]
                clip_id = clip_ids[random_sample_idx]
                sample_id = f'{dataset_type}_{clip_id}_{random_sample_idx}'

                #rng = random.Random(100)
                #print('batch length: ', batch_size, len(self.val_dataloader.dataset))
                #sample_indices = rng.sample(range(batch_size), 10)
                x = batch_motion_parts[random_sample_idx].unsqueeze(0).float()
                x_motion = batch_motion[random_sample_idx].unsqueeze(0).float()
                row = self._get_result_from_vqvae(
                    x = x,
                    clip_id = clip_ids[random_sample_idx],
                    sample_id = sample_id,
                    sample_text = batch_text[random_sample_idx],
                    full_text = batch_text[random_sample_idx],
                    recon_caption='Part-Aware VQVAE',
                    x_baseline=x_motion
                )
                
                rows.append(row)

        metrics_path = pjoin(self.metrics_dir, 'train_metrics.json' if dataset_type == 'train' else 'val_metrics.json')
        with open(metrics_path, "w") as f:
            json.dump(rows, f, indent=4)

        return rows
    
    @torch.no_grad()
    def validate_interpolated_samples(self):

        #self.vqvae.eval()
        drows: List[Dict[str, Any]] = []
        rows: List[Dict[str, Any]] = []

        rng = random.Random(self.sampling_seed)
        sample_indices = rng.sample(range(len(self.val_dataloader.dataset)), 2)
        sample_a, sample_b = self.val_dataloader.dataset[sample_indices[0]], self.val_dataloader.dataset[sample_indices[1]]

        x_a = sample_a['motion_parts']
        x_b = sample_b['motion_parts']
        x_a_m = sample_a['motion']
        x_b_m = sample_a['motion']

        if not torch.is_tensor(x_a):
            x_a = torch.tensor(x_a)
            x_a_m = torch.tensor(x_a_m)
        if not torch.is_tensor(x_b):
            x_b = torch.tensor(x_b)
            x_b_m = torch.tensor(x_b_m)

        x_a = x_a.unsqueeze(0).float().to(self.device)  # (1, T, P, Dp_max)
        x_b = x_b.unsqueeze(0).float().to(self.device)
        x_a_m = x_a_m.unsqueeze(0).float().to(self.device) # (1, T, D)
        x_b_m = x_b_m.unsqueeze(0).float().to(self.device)

        z_a = self.vqvae.encode(x_a)
        z_b = self.vqvae.encode(x_b)
        z_a_m = self.pretrained_movementenc(x_a_m[...,:-4])
        z_b_m = self.pretrained_movementenc(x_b_m[...,:-4])


        data_interpolated_samples = {
            'sample_idx_a': sample_indices[0],
            'sample_idx_b': sample_indices[1],
            'file_id_a': sample_a.get('file_id', ''),
            'file_id_b': sample_b.get('file_id', ''),
            'text_a': sample_a.get('text', ''),
            'text_b': sample_b.get('text', ''),
            'alphas': [],
            'motion_parts': [],
            'motions': []
        }

        interpolated_samples = {
            'sample_idx_a': sample_indices[0],
            'sample_idx_b': sample_indices[1],
            'file_id_a': sample_a.get('file_id', ''),
            'file_id_b': sample_b.get('file_id', ''),
            'text_a': sample_a.get('text', ''),
            'text_b': sample_b.get('text', ''),
            'alphas': [],
            'motion_parts': [],
            'motions': []
        }

        for alpha in np.linspace(0.0, 1.0, self.samples_to_test):
            x_interp = (1.0 - alpha) * x_a + alpha * x_b
            x_interp_m = (1.0 - alpha) * x_a_m + alpha * x_b_m
            data_interpolated_samples['alphas'].append(float(alpha))
            data_interpolated_samples['motion_parts'].append(x_interp.detach().cpu())
            data_interpolated_samples['motions'].append(x_interp_m.detach().cpu())

            z_interp = (1.0 - alpha) * z_a + alpha * z_b
            z_interp_m = (1.0 - alpha) * z_a_m + alpha * z_b_m
            z_q, _, _, _, _ = self.vqvae.quantize(z_interp)
            x_interp = self.vqvae.decode(z_q)
            x_interp_m = self.pretrained_movementdec(z_interp_m)
            interpolated_samples['alphas'].append(float(alpha))
            interpolated_samples['motion_parts'].append(x_interp.detach().cpu())
            interpolated_samples['motions'].append(x_interp_m.detach().cpu())

        clip_id = f"{interpolated_samples['file_id_a']}_{interpolated_samples['file_id_b']}"
        for si in range(self.samples_to_test):
            x = data_interpolated_samples['motion_parts'][si]
            x_motion = data_interpolated_samples['motions'][si]
            row = self._get_result_from_vqvae(
                x = x,
                clip_id=clip_id,
                sample_id=f"data_interpolated_{clip_id}_{si}",
                sample_text=f"Interpolated sample {si} (data space)",
                full_text=f"Interpolation of - {interpolated_samples['text_a']} : {interpolated_samples['text_b']}",
                recon_caption='Part-Aware VQVAE',
                x_baseline=x_motion
            )
            drows.append(row)

            x = interpolated_samples['motion_parts'][si]
            x_motion = interpolated_samples['motions'][si]
            row = self._get_result_from_vqvae(
                x = x,
                clip_id=clip_id,
                sample_id=f"interpolated_{clip_id}_{si}",
                sample_text=f"Interpolated sample {si} (latent space)",
                full_text=f"Interpolation of - {interpolated_samples['text_a']} : {interpolated_samples['text_b']}",
                x_baseline=x_motion
            )
            rows.append(row)

        metrics_path = pjoin(self.metrics_dir, 'data_interpolated_samples_metrics.json')
        with open(metrics_path, "w") as f:
            json.dump(drows, f, indent=4)

        metrics_path = pjoin(self.metrics_dir, 'interpolated_samples_metrics.json')
        with open(metrics_path, "w") as f:
            json.dump(rows, f, indent=4)

        return drows, rows


    @torch.no_grad()
    def validate_random_uniform_samples(self):
        #self.vqvae.eval()
        rows: List[Dict[str, Any]] = []

        rng = random.Random(self.sampling_seed)
        sample_idx = rng.randrange(len(self.val_dataloader.dataset))
        sample = self.val_dataloader.dataset[sample_idx]

        x = sample['motion_parts']
        if not torch.is_tensor(x):
            x = torch.tensor(x)

        x = x.unsqueeze(0).float().to(self.device)  # (1, T, P, D)

        # infer latent index shape from a real example
        z_e = self.vqvae.encode(x)
        _, code_indices, _, _, _ = self.vqvae.quantize(z_e)

        uniform_random_samples = {
            'sample_idx_ref': sample_idx,
            'file_id_ref': sample.get('file_id', ''),
            'text_ref': sample.get('text', ''),
            'motion_parts': [],
            'indices': []
        }

        for _ in range(self.samples_to_test):
            rand_indices = torch.randint(
                low=0,
                high=512,
                size=code_indices.shape,
                device=self.device
            )

            x_sample = self.vqvae.decode_from_indices(rand_indices)

            uniform_random_samples['indices'].append(rand_indices.detach().cpu())
            uniform_random_samples['motion_parts'].append(x_sample.detach().cpu())

        clip_id = uniform_random_samples['file_id_ref']
        for si, x in enumerate(uniform_random_samples['motion_parts']):
            sample_text = f"Uniform random sample {si} - {uniform_random_samples['text_ref']}"
            row = self._get_result_from_vqvae(
                x = x,
                clip_id=clip_id,
                sample_id=f"usampled_{clip_id}_{si}",
                sample_text=sample_text,
                full_text=sample_text,
                recon_caption='Part-Aware VQVAE'
            )
            rows.append(row)

        metrics_path = pjoin(self.metrics_dir, 'random_uniform_samples_metrics.json')
        with open(metrics_path, "w") as f:
            json.dump(rows, f, indent=4)
        return rows

    def validate(self):
        
        # test for memorization of samples
        self.validate_dataset(self.train_dataloader, 'train')
        self.validate_dataset(self.val_dataloader, 'val')
        self.validate_interpolated_samples()
        self.validate_random_uniform_samples()
        
        #return train_eval_results, val_eval_results, interpolation_results, uniform_sampling_results

# this will just validate whether latents from training/validation sample and close to those samples
# reconstruct without errors
class DiffusionValidator:

    def __init__(self, opt, dit, val_dataloader, test_dataloader,
            test_type: str = "val", samples_to_test: int = 5,
            video_dir_name: str = 'videos', tensors_dir_name: str = 'tensors', metrics_dir_name: str = 'metrics',
            beta_start = 1e-4, beta_end = 0.02, prediction_type = "epsilon", num_train_timesteps: int = 1000, num_inference_steps: int = 1000):
        self.opt = opt
        self.dit = dit
        self.dit.eval()
        self.val_dataloader = val_dataloader
        self.test_dataloader = test_dataloader
        self.test_type = test_type
        self.samples_to_test = samples_to_test
        self.sampling_seed = 42
        self.num_train_timesteps = num_train_timesteps
        self.num_inference_steps = num_inference_steps
        self.prediction_type = prediction_type

        self.videos_dir = pjoin(opt.output_dir, f'{video_dir_name}')
        self.tensors_dir = pjoin(opt.output_dir, f'{tensors_dir_name}')
        self.metrics_dir = pjoin(opt.output_dir, f'{metrics_dir_name}')
        self.simple_test_split_file = pjoin(opt.data_root, 'simple_test.txt')
        ensure_dir(self.videos_dir)
        ensure_dir(self.tensors_dir)
        ensure_dir(self.metrics_dir)
        self.device = torch.device("cpu")

        self.mean = np.load(pjoin(opt.meta_dir, 'mean.npy'))
        self.std = np.load(pjoin(opt.meta_dir, 'std.npy'))
        self.d_part_max = 60

        with open(pjoin(opt.meta_dir, 'part_mapping.json'), 'r') as f:
            mapping = json.load(f)

        self.part_names = mapping['part_names']
        self.d_part_max = mapping['d_part_max']
        self.joints_num = mapping['joints_num']
        self.part_feature_indices = {
            k: np.array(v, dtype=np.int64)
            for k, v in mapping['part_feature_indices'].items()
        }
        self.motion_dim = self.mean.shape[0]

        self._init_pretrained_models()
        self._build_schedule(beta_start, beta_end, self.num_train_timesteps)
        self.text_proj = torch.nn.Linear(768, 512)
        #self.motion_proj = torch.nn.Linear(263, 256)
        

    def _init_pretrained_models(self):
        self.pretrained_movementenc, self.pretrained_movementdec = get_pretrained_vae(self.opt.model_dir) 
        self.pretrained_movementenc.to(self.device)
        self.pretrained_movementdec.to(self.device)

        #self.text_embedder = TextTokenEncoder(device = self.device)
        self.text_embedder, self.text_tokenizer = get_pretrained_text_encoder(model = 'clip_text', device = self.device)

        self.text_embedder.eval()
        self.pretrained_movementenc.eval()
        self.pretrained_movementdec.eval()

    def _build_schedule(self, beta_start, beta_end, num_train_timesteps):
        betas = torch.linspace(beta_start, beta_end, num_train_timesteps, dtype=torch.float32, device=self.device)
        alphas = 1.0 - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)
        alphas_cumprod_prev = torch.cat(
            [torch.tensor([1.0], dtype=torch.float32, device=self.device), alphas_cumprod[:-1]],
            dim=0
        )

        self.betas = betas
        self.alphas = alphas
        self.alphas_cumprod = alphas_cumprod
        self.alphas_cumprod_prev = alphas_cumprod_prev

        self.sqrt_alphas_cumprod = torch.sqrt(alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - alphas_cumprod)
        self.sqrt_recip_alphas = torch.sqrt(1.0 / alphas)

        self.posterior_variance = (
            betas * (1.0 - alphas_cumprod_prev) / (1.0 - alphas_cumprod)
        )

    def _extract(self, a, t, x_shape):
        out = a.gather(0, t)
        return out.view(t.shape[0], *((1,) * (len(x_shape) - 1)))

    @torch.no_grad()
    def _p_sample(self, x, t, text_tokens, text_mask=None):
        B = x.shape[0]
        t_batch = torch.full((B,), t, device=self.device, dtype=torch.long)
        d = torch.zeros_like(t_batch)

        model_pred = self.dit(x, t_batch, d, text_tokens, text_mask)

        if self.prediction_type == "epsilon":
            betas_t = self._extract(self.betas, t_batch, x.shape)
            sqrt_one_minus_alphas_cumprod_t = self._extract(
                self.sqrt_one_minus_alphas_cumprod, t_batch, x.shape
            )
            sqrt_recip_alphas_t = self._extract(self.sqrt_recip_alphas, t_batch, x.shape)

            model_mean = sqrt_recip_alphas_t * (
                x - betas_t * model_pred / sqrt_one_minus_alphas_cumprod_t
            )
        elif self.prediction_type == "x0":
            betas_t = self._extract(self.betas, t_batch, x.shape)
            alphas_t = self._extract(self.alphas, t_batch, x.shape)
            alphas_cumprod_t = self._extract(self.alphas_cumprod, t_batch, x.shape)
            alphas_cumprod_prev_t = self._extract(self.alphas_cumprod_prev, t_batch, x.shape)

            coef1 = betas_t * torch.sqrt(alphas_cumprod_prev_t) / (1.0 - alphas_cumprod_t)
            coef2 = (1.0 - alphas_cumprod_prev_t) * torch.sqrt(alphas_t) / (1.0 - alphas_cumprod_t)
            model_mean = coef1 * model_pred + coef2 * x
        else:
            raise ValueError(f"Unsupported prediction_type: {self.prediction_type}")

        if t == 0:
            return model_mean
        else:
            posterior_variance_t = self._extract(self.posterior_variance, t_batch, x.shape)
            noise = torch.randn_like(x)
            return model_mean + torch.sqrt(posterior_variance_t) * noise

    @torch.no_grad()
    def _sample(self, text_tokens, seq_len, latent_dim, text_mask=None, batch_size=1):
        self.dit.eval()

        text_tokens = text_tokens.to(self.device)
        if text_mask is not None:
            text_mask = text_mask.to(self.device)

        x = torch.randn(batch_size, seq_len, latent_dim, device=self.device)
        full_t = self.num_train_timesteps
        timesteps = np.linspace(0, full_t - 1, self.num_inference_steps)
        timesteps = list(np.round(timesteps).astype(int))

        for t in reversed(timesteps):
            x = self._p_sample(x, t, text_tokens, text_mask)

        return x

    @torch.no_grad()
    def _compute_metrics(self, x: torch.Tensor, x_recon: torch.Tensor) -> Dict[str, float]:
        diff = x_recon - x
        l1 = diff.abs().mean().item()
        mse = (diff ** 2).mean().item()
        rmse = float(np.sqrt(mse))
        max_abs = diff.abs().max().item()
        return {
            'l1': l1,
            'mse': mse,
            'rmse': rmse,
            'max_abs': max_abs,
        }
    
    @torch.no_grad()
    def _compute_dataset_metrics(
        self,
        gen_feats: torch.Tensor,
        gt_feats: torch.Tensor,
        text_feats: torch.Tensor,
    ) -> Dict[str, float]:
        """
        gen_feats: [N, T, D] motion features for generated motions
        gt_feats:  [N, T, D] motion features for ground-truth motions
        text_feats: [N, D_text] text features for captions
        """
        if gen_feats.dim() == 3:
            gen_feats = gen_feats.mean(dim=1)
            gt_feats = gt_feats.mean(dim=1)
        # FID: Fréchet distance between generated and GT motion feature distributions
        mu_gen = gen_feats.mean(dim=0)
        mu_gt = gt_feats.mean(dim=0)
        cov_gen = torch.cov(gen_feats.T)
        cov_gt = torch.cov(gt_feats.T)

        mean_diff = (mu_gen - mu_gt).unsqueeze(0)
        mean_term = (mean_diff @ mean_diff.T).item()
        # Small epsilon for numerical stability
        eps = 1e-6
        cov_gen_eps = cov_gen + eps * torch.eye(cov_gen.size(0), device=cov_gen.device)
        cov_gt_eps = cov_gt + eps * torch.eye(cov_gt.size(0), device=cov_gt.device)
        cov_prod = cov_gen_eps @ cov_gt_eps
        cov_prod = cov_prod.cpu()
        evals, _ = torch.linalg.eig(cov_prod)
        
        evals = evals.real.clamp(min = 0)
        sqrt_evals = torch.sqrt(evals)

        trace_sqrt = sqrt_evals.sum().item()
        trace_gen = torch.trace(cov_gen_eps).item()
        trace_gt = torch.trace(cov_gt_eps).item()
        fid = mean_term + trace_gen + trace_gt - (2.0 * trace_sqrt)

        # TODO: skipping text-motion retrieval metrics for now 
        '''
        # R-Precision: text-motion retrieval
        # cosine similarity between text_feats and motion_feats (generated)
        t = torch.nn.functional.normalize(self.dit.encode_text_to_motion_space(text_feats), dim=-1)    # [N, 512]
        #m = torch.nn.functional.normalize(self.motion_proj(gen_feats), dim=-1) # [N, 512]
        text_norm = torch.nn.functional.normalize(t, dim=-1)
        motion_norm = torch.nn.functional.normalize(gen_feats, dim=-1)
        sim = text_norm @ motion_norm.T  # [N, N]

        # For each text, rank motions
        ranks = sim.argsort(dim=1, descending=True)
        gt_indices = torch.arange(sim.size(0), device=sim.device)

        def r_precision_at_k(k: int, gt_indices: torch.Tensor) -> float:
            topk = ranks[:, :k]
            correct = (topk == gt_indices.unsqueeze(1)).any(dim=1).float()
            return correct.mean().item()

        r1 = r_precision_at_k(1, gt_indices[:1])
        r2 = r_precision_at_k(2, gt_indices[:2])
        r3 = r_precision_at_k(3, gt_indices[:3])

        # MM-Dist: mean distance between matched text-motion embeddings
        # (using generated motion features)
        mm_dist = (1.0 - (text_norm * motion_norm.unsqueeze(1)).sum(dim=-1)).mean().item()
        '''

        # Diversity: average L2 distance between random pairs of generated motion features
        if gen_feats.size(0) > 1:
            idx = torch.randperm(gen_feats.size(0), device=gen_feats.device)
            half = idx.numel() // 2
            a = gen_feats[idx[:half]]
            b = gen_feats[idx[half:half * 2]]
            diversity = (a - b).norm(dim=-1).mean().item()
        else:
            diversity = 0.0

        return {
            "fid": fid,
            #"r_precision_top1": r1,
            #"r_precision_top2": r2,
            #"r_precision_top3": r3,
            #"mm_dist": mm_dist,
            "diversity": diversity,
        }

    def motion_to_parts(self, motion):
        T, D = motion.shape
        P = len(self.part_names)
        motion_parts = np.zeros((T, P, self.d_part_max), dtype=np.float32)

        for p, part_name in enumerate(self.part_names):
            idxs = self.part_feature_indices[part_name]
            part_feat = motion[:, idxs]
            motion_parts[:, p, :part_feat.shape[1]] = part_feat

        return motion_parts.astype(np.float32)

    def motion_parts_to_full_motion(self, motion_parts):
        """
        motion_parts:
            torch.Tensor or np.ndarray
            shape (1, T, P, D_part_max) or (T, P, D_part_max)

        returns:
            np.ndarray of shape (T, motion_dim)  # usually (T, 263)
            still normalized
        """
        if torch.is_tensor(motion_parts):
            motion_parts = motion_parts.detach().cpu().numpy()

        if motion_parts.ndim == 4:
            motion_parts = motion_parts[0]   # -> (T, P, D_part_max)

        T, P, Dp = motion_parts.shape
        full_motion = np.zeros((T, self.motion_dim), dtype=np.float32)

        for p, part_name in enumerate(self.part_names):
            idxs = self.part_feature_indices[part_name]
            part_feat = motion_parts[:, p, :len(idxs)]
            full_motion[:, idxs] = part_feat

        return full_motion

    def denormalize_motion(self, motion):
        return motion * self.std + self.mean

    @torch.no_grad()
    def _get_result_from_dit(self, test_type: str, x: torch.Tensor, clip_ids: List[str], sample_texts: List[str], save_samples_test: bool = False):
        #x = batch_motion_parts[random_sample_idx].unsqueeze(0).float()
        #text_tokens, text_masks = self.text_embedder.encode_tokens(
            #sample_texts
        #)
        inputs = self.text_tokenizer(sample_texts, return_tensors="pt", padding=True, truncation=True)
        inputs = {name: tensor.to(self.device) for name, tensor in inputs.items()}
        text_embeddings = self.text_embedder(**inputs).last_hidden_state
        text_masks = inputs['attention_mask']
        B = min(x.shape[0], self.samples_to_test)
        latents = self._sample(
            text_tokens=text_embeddings,
            seq_len=self.opt.max_motion_length//4,
            latent_dim = 512,
            text_mask = text_masks,
            batch_size = x.shape[0]
        )
        x_recons = self.pretrained_movementdec(latents)
        print('Reconstructed motion shape:', x_recons.shape, x.shape, text_embeddings.shape)
        ds_metrics = self._compute_dataset_metrics(x_recons, x, text_embeddings)

        # extract video
        if test_type == "val" or save_samples_test:
            for i in range(B):
                gt_motion = x[i]
                recon_motion = x_recons[i]
                clip_id = f'{test_type}_{clip_ids[i]}'
                sample_text = sample_texts[i]
                try:
                    #full_motion_gt = self.motion_parts_to_full_motion(x.detach().cpu())
                    #print('full motion gt in vqvae: ', full_motion_gt[..., self.missing_parts_indices])
                    full_motion_gt = self.denormalize_motion(gt_motion.detach().cpu().numpy())
                    #full_motion_recon = self.motion_parts_to_full_motion(x_recon.detach().cpu())
                    full_motion_recon = self.denormalize_motion(recon_motion.detach().cpu().numpy())

                    joints_gt = recover_from_ric(torch.from_numpy(full_motion_gt).float(), self.joints_num)
                    joints_recon = recover_from_ric(torch.from_numpy(full_motion_recon).float(), self.joints_num)


                    if joints_gt is not None and joints_recon is not None:
                        #print('joints gt, recon', joints_gt.shape, joints_recon.shape, self.videos_dir)
                        video_path = render_skeleton_animation(
                            joints_gt=joints_gt,
                            joints_recon=joints_recon,
                            skeleton_edges=HUMANML3D_SKELETON_EDGES,
                            output_path_no_ext=pjoin(self.videos_dir, clip_id),
                            clip_id = clip_id,
                            text = sample_text,
                            recon_caption='Baseline Diffusion',
                            fps=120,
                            save_mp4=True
                        )
                        if video_path != "" or video_path != None:
                            print('Saved video file at: ', video_path)
                except Exception as exc:
                    print('inside except: ', exc)
        return ds_metrics
    
    @torch.no_grad()
    def _get_prompts_results_from_dit(self, prompts: List[str]):
        #x = batch_motion_parts[random_sample_idx].unsqueeze(0).float()
        random.shuffle(prompts)
        inputs = self.text_tokenizer(prompts, return_tensors="pt", padding=True, truncation=True)
        inputs = {name: tensor.to(self.device) for name, tensor in inputs.items()}
        text_embeddings = self.text_embedder(**inputs).last_hidden_state
        text_masks = inputs['attention_mask']
        B = min(text_embeddings.shape[0], self.samples_to_test)
        latents = self._sample(
            text_tokens=text_embeddings[:B, ...],
            seq_len=self.opt.max_motion_length//4,
            latent_dim = 512,
            text_mask = text_masks[:B, ...],
            batch_size = B
        )
        x_recons = self.pretrained_movementdec(latents)
        print('Reconstructed motion shape:', x_recons.shape, text_embeddings.shape)

        # extract video
        for i in range(B):
            recon_motion = x_recons[i]
            sample_text = prompts[i]
            clip_id = f'prompt_{i}'
            try:
                #full_motion_gt = self.motion_parts_to_full_motion(x.detach().cpu())
                #print('full motion gt in vqvae: ', full_motion_gt[..., self.missing_parts_indices])
                full_motion_recon = self.denormalize_motion(recon_motion.detach().cpu().numpy())
                joints_recon = recover_from_ric(torch.from_numpy(full_motion_recon).float(), self.joints_num)

                video_path = render_skeleton_single_animation(
                    joints_recon=joints_recon,
                    output_path_no_ext=pjoin(self.videos_dir, clip_id),
                    clip_id=clip_id,
                    text=sample_text,
                    recon_caption='Baseline Diffusion',
                    fps=120,
                    save_gif_fallback=True
                )
                if joints_recon is not None:
                    print('joints recon', joints_recon.shape, self.videos_dir)
                    if video_path != "" or video_path != None:
                        print('Saved video file at: ', video_path)
            except Exception as exc:
                print('inside except: ', exc)

    
    def validate_dataset(self, dataloader = None):

        ds_metrics = {}

        if self.test_type in ['test', 'val'] and dataloader != None:
            rng = random.Random(self.sampling_seed)
            num_batches = len(dataloader)
            random_batch_idx = rng.sample(range(num_batches), 1)[0]
            for bi, batch in enumerate(dataloader):

                if self.test_type == 'val' and bi == random_batch_idx:
                    batch = move_batch_to_device(batch, self.device)
                    batch_motion_parts = batch['motion_parts'][:self.samples_to_test,...]
                    batch_motion = batch['motion'][:self.samples_to_test,...]
                    batch_size = batch_motion_parts.size(0)

                    batch_text = batch['text'][:self.samples_to_test]
                    #print('batch texts: ', batch_text)
                    clip_ids = batch['file_id'][:self.samples_to_test]
                    #clip_id = clip_ids[random_sample_idx]
                    #pret_sample_id = f'prevae_val_{clip_id}'

                    x_motion = batch_motion.float()
                    ds_metrics = self._get_result_from_dit(
                        test_type=self.test_type,
                        x = x_motion,
                        clip_ids=clip_ids,
                        sample_texts=batch_text
                    )

                elif self.test_type == 'test':
                    batch = move_batch_to_device(batch, self.device)
                    batch_motion_parts = batch['motion_parts']
                    batch_motion = batch['motion']

                    batch_text = batch['text']
                    #print('batch texts: ', batch_text)
                    clip_ids = batch['file_id']

                    x_motion = batch_motion.float()
                    ds_metrics_batch = self._get_result_from_dit(
                        test_type = self.test_type,
                        x = x_motion,
                        clip_ids=clip_ids,
                        sample_texts=batch_text,
                        save_samples_test=True
                    )
                    if ds_metrics == {}:
                        ds_metrics = ds_metrics_batch
                    else:
                        for key in ds_metrics_batch:
                            if key in ds_metrics:
                                ds_metrics[key] += ds_metrics_batch[key]
                            else:
                                ds_metrics[key] = ds_metrics_batch[key]
            if self.test_type == 'test':
                ds_metrics = { k: v / num_batches for k, v in ds_metrics.items() }
            metrics_path = pjoin(self.metrics_dir, f'{self.test_type}_metrics.json')
            with open(metrics_path, "w") as f:
                json.dump(ds_metrics, f, indent=4)
        else:
            simple_prompts = []
            with open(self.simple_test_split_file, 'r') as f:
                simple_prompts = f.readlines()
            assert len(simple_prompts) != 0, "Simple test prompts file is empty!"

            self._get_prompts_results_from_dit(prompts = simple_prompts)
                
        
        

    def validate(self):
        
        # test for memorization of samples
        if self.test_type == "val":
            self.validate_dataset(self.val_dataloader)
        else:
            self.validate_dataset(self.test_dataloader)

        #return train_eval_results, val_eval_results, interpolation_results, uniform_sampling_results
