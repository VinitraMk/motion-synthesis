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

    def __init__(self, opt, vqvae, val_dataloader, video_dir_name: str = 'videos', tensors_dir_name: str = 'tensors', metrics_dir_name: str = 'metrics'):
        self.opt = opt
        self.vqvae = vqvae
        self.val_dataloader = val_dataloader

        self.videos_dir = pjoin(opt.output_dir, f'{video_dir_name}')
        self.tensors_dir = pjoin(opt.output_dir, f'{tensors_dir_name}')
        self.metrics_json_path = pjoin(opt.output_dir, f'{metrics_dir_name}/metrics.json')
        ensure_dir(self.videos_dir)
        ensure_dir(self.tensors_dir)
        ensure_dir(pjoin(opt.output_dir, metrics_dir_name))
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

    def validate(self):
        
        self.vqvae.eval()
        rows: List[Dict[str, Any]] = []
        saved_visuals = 0

        for bi, batch in enumerate(self.val_dataloader):
            #if bi > 1:
                #break

            batch = move_batch_to_device(batch, self.device)
            batch_motion_parts = batch['motion_parts']
            batch_text = batch['text']
            #print('batch texts: ', batch_text)
            clip_ids = batch['file_id']
            batch_size = batch_motion_parts.size(0)

            #rng = random.Random(100)
            #print('batch length: ', batch_size, len(self.val_dataloader.dataset))
            #sample_indices = rng.sample(range(batch_size), 10)


            for sample_idx in range(batch_size):
                x = batch_motion_parts[sample_idx].unsqueeze(0).float()
                out = self.vqvae.forward(x)
                x_recon = out['x_recon']
                metrics = self._compute_metrics(x, x_recon)
                clip_id = clip_ids[sample_idx]
                sample_id = f'{clip_id}_{sample_idx}'

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
                }

                # save output tensors
                torch.save({
                    'clip_id': clip_id,
                    'x': x.detach().cpu(),
                    'x_recon': x_recon.detach().cpu(),
                }, os.path.join(self.tensors_dir, f'{clip_id}.pt'))


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
                            text = batch_text[sample_idx],
                            fps=20,
                            save_mp4=True
                        )
                        #print('video path: ', video_path)
                        row["video_path"] = video_path or ""
                        row['video_error'] = ""
                        saved_visuals += 1
                except Exception as exc:
                    #print('inside except')
                    row["video_path"] = ""
                    row["video_error"] = str(exc)
                rows.append(row)

        with open(self.metrics_json_path, "w") as f:
            json.dump(rows, f, indent=4)

        return rows
