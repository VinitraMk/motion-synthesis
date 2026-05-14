import csv
import math
import os
from dataclasses import dataclass
from typing import Any, Dict, List
import json
import numpy as np
import torch
import random
from utils.utils import ensure_dir
from utils.nn_utils import move_batch_to_device
from data_utils.motion_processor import render_skeleton_animation, HUMANML3D_SKELETON_EDGES
from data_utils.motion_processor import recover_from_ric
from os.path import join as pjoin


class VQVAEValidator:

    def __init__(self, opt, vqvae, train_dataloader, val_dataloader, samples_to_test: int = 5, video_dir_name: str = 'videos', tensors_dir_name: str = 'tensors', metrics_dir_name: str = 'metrics'):
        self.opt = opt
        self.vqvae = vqvae
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

    def _get_result_from_vqvae(self, x: torch.Tensor, random_sample_idx: int, clip_id: str, sample_id: str, sample_text: str, full_text: str = ""):

        #x = batch_motion_parts[random_sample_idx].unsqueeze(0).float()
        out = self.vqvae.forward(x)
        x_recon = out['x_recon']
        metrics = self._compute_metrics(x, x_recon)
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
            'x_recon': x_recon.detach().cpu(),
        }, os.path.join(self.tensors_dir, f'{sample_id}.pt'))


        # extract video

        try:
            full_motion_gt = self.motion_parts_to_full_motion(x.detach().cpu())
            full_motion_gt = self.denormalize_motion(full_motion_gt)
            full_motion_recon = self.motion_parts_to_full_motion(x_recon.detach().cpu())
            full_motion_recon = self.denormalize_motion(full_motion_recon)

            joints_gt = recover_from_ric(torch.from_numpy(full_motion_gt).float(), self.joints_num)
            joints_recon = recover_from_ric(torch.from_numpy(full_motion_recon).float(), self.joints_num)


            if joints_gt is not None and joints_recon is not None:
                #print('joints gt, recon', joints_gt.shape, joints_recon.shape, self.videos_dir)
                video_path = render_skeleton_animation(
                    joints_gt=joints_gt,
                    joints_recon=joints_recon,
                    skeleton_edges=HUMANML3D_SKELETON_EDGES,
                    output_path_no_ext=pjoin(self.videos_dir, sample_id),
                    clip_id = sample_id,
                    text = sample_text,
                    fps=20,
                    save_mp4=True
                )
                if video_path != "" or video_path != None:
                    print('Saved video file at: ', video_path)
                row["video_path"] = video_path or ""
                row['video_error'] = ""
        except Exception as exc:
            #print('inside except')
            row["video_path"] = ""
            row["video_error"] = str(exc)

        return row
    
    def validate_dataset(self, dataloader, dataset_type = 'train'):

        self.vqvae.eval()
        rows: List[Dict[str, Any]] = []
        saved_visuals = 0
        rng = random.Random(self.sampling_seed)
        num_batches = len(dataloader)
        k = min(self.samples_to_test, num_batches)
        batch_indices = rng.sample(range(num_batches), k)

        for bi, batch in enumerate(dataloader):

            if bi in batch_indices: 
                batch = move_batch_to_device(batch, self.device)
                batch_motion_parts = batch['motion_parts']
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
                row = self._get_result_from_vqvae(
                    x = x,
                    random_sample_idx = random_sample_idx,
                    clip_id = clip_ids[random_sample_idx],
                    sample_id = sample_id,
                    sample_text = batch_text[random_sample_idx],
                    full_text = batch_text[random_sample_idx]
                )
                
                rows.append(row)

        metrics_path = pjoin(self.metrics_dir, 'train_metrics.json' if dataset_type == 'train' else 'val_metrics.json')
        with open(metrics_path, "w") as f:
            json.dump(rows, f, indent=4)

        return rows
    
    @torch.no_grad()
    def validate_interpolated_samples(self):

        self.vqvae.eval()
        rows: List[Dict[str, Any]] = []

        rng = random.Random(self.sampling_seed)
        sample_indices = rng.sample(range(len(self.val_dataloader.dataset)), 2)
        sample_a, sample_b = self.val_dataloader.dataset[sample_indices[0]], self.val_dataloader.dataset[sample_indices[1]]

        x_a = sample_a['motion_parts']
        x_b = sample_b['motion_parts']

        if not torch.is_tensor(x_a):
            x_a = torch.tensor(x_a)
        if not torch.is_tensor(x_b):
            x_b = torch.tensor(x_b)

        x_a = x_a.unsqueeze(0).float().to(self.device)  # (1, T, P, D)
        x_b = x_b.unsqueeze(0).float().to(self.device)

        z_a = self.vqvae.encode(x_a)
        z_b = self.vqvae.encode(x_b)

        interpolated_samples = {
            'sample_idx_a': sample_indices[0],
            'sample_idx_b': sample_indices[1],
            'file_id_a': sample_a.get('file_id', ''),
            'file_id_b': sample_b.get('file_id', ''),
            'text_a': sample_a.get('text', ''),
            'text_b': sample_b.get('text', ''),
            'alphas': [],
            'motion_parts': []
        }

        for alpha in np.linspace(0.0, 1.0, self.samples_to_test):
            z_interp = (1.0 - alpha) * z_a + alpha * z_b

            z_q, _, _, _, _ = self.vqvae.quantize(z_interp)
            x_interp = self.vqvae.decode(z_q)

            interpolated_samples['alphas'].append(float(alpha))
            interpolated_samples['motion_parts'].append(x_interp.detach().cpu())

        clip_id = f"{interpolated_samples['file_id_a']}_{interpolated_samples['file_id_b']}"
        for si, x in enumerate(interpolated_samples['motion_parts']):
            row = self._get_result_from_vqvae(
                x = x,
                random_sample_idx=si,
                clip_id=clip_id,
                sample_id=f"interpolated_{clip_id}_{si}",
                sample_text=f"Interpolated sample {si}",
                full_text=f"Interpolation of - {interpolated_samples['text_a']} : {interpolated_samples['text_b']}"
            )
            rows.append(row)

        metrics_path = pjoin(self.metrics_dir, 'interpolated_samples_metrics.json')
        with open(metrics_path, "w") as f:
            json.dump(rows, f, indent=4)

        return rows


    @torch.no_grad()
    def validate_random_uniform_samples(self):
        self.vqvae.eval()
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
                random_sample_idx=si,
                clip_id=clip_id,
                sample_id=f"usampled_{clip_id}_{si}",
                sample_text=sample_text,
                full_text=sample_text
            )
            rows.append(row)

        metrics_path = pjoin(self.metrics_dir, 'random_uniform_samples_metrics.json')
        with open(metrics_path, "w") as f:
            json.dump(rows, f, indent=4)
        return rows

    def validate(self):
        
        # test for memorization of samples
        train_eval_results = self.validate_dataset(self.train_dataloader, 'train')
        val_eval_results = self.validate_dataset(self.val_dataloader, 'val')
        interpolation_results = self.validate_interpolated_samples()
        uniform_sampling_results = self.validate_random_uniform_samples()
        
        #return train_eval_results, val_eval_results, interpolation_results, uniform_sampling_results
