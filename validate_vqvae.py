import os
import random
from typing import Dict, Any, List

import numpy as np
import torch


"""
Reusable held-out VQ-VAE validation script.

Purpose:
- Load a frozen VQ-VAE checkpoint.
- Run encode/decode reconstruction on untouched validation clips.
- Save per-clip metrics and optional recon tensors for later qualitative review.

Assumptions you will adapt in your repo:
- A MotionVQVAE class exists and can be imported.
- A validation dataset/dataloader can be built and yields dicts containing:
    batch['motion_parts'] with shape (B, T, P, Dp_max) or compatible.
- The model forward returns a dict with at least:
    out['x_recon'], out['loss'], out['recon_loss']
- Optional code indices may be available as out['encoding_indices'] or similar.

Usage example:
python validate_vqvae_recon.py \
  --checkpoint path/to/latest.tar \
  --num_clips 15 \
  --batch_size 1 \
  --device cuda \
  --out_dir output/vqvae_validation
"""


def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


@torch.no_grad()
def compute_metrics(x: torch.Tensor, x_recon: torch.Tensor) -> Dict[str, float]:
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


def move_batch_to_device(batch: Dict[str, Any], device: torch.device) -> Dict[str, Any]:
    out = {}
    for k, v in batch.items():
        if torch.is_tensor(v):
            out[k] = v.to(device)
        else:
            out[k] = v
    return out


def select_batch_items(batch: Dict[str, Any], idx: int) -> Dict[str, Any]:
    out = {}
    for k, v in batch.items():
        if torch.is_tensor(v) and v.size(0) > idx:
            out[k] = v[idx:idx+1]
        else:
            out[k] = v
    return out


def get_clip_id(batch: Dict[str, Any], idx: int) -> str:
    for key in ['name', 'names', 'clip_name', 'clip_id', 'id', 'ids']:
        if key in batch:
            v = batch[key]
            if isinstance(v, (list, tuple)) and len(v) > idx:
                return str(v[idx])
            return str(v)
    return f'clip_{idx:04d}'


def build_model_and_loader(args):
    """
    ADAPT THIS FUNCTION IN YOUR REPO.

    Expected return:
        model, val_loader
    """
    raise NotImplementedError(
        'Implement build_model_and_loader(args) to return your MotionVQVAE model and validation DataLoader.'
    )


@torch.no_grad()
def run_validation(model, val_loader, device, num_clips: int, out_dir: str, save_tensors: bool = True):
    os.makedirs(out_dir, exist_ok=True)
    tensors_dir = os.path.join(out_dir, 'recon_tensors')
    if save_tensors:
        os.makedirs(tensors_dir, exist_ok=True)

    model.eval()
    rows: List[Dict[str, Any]] = []
    saved = 0

    for batch in val_loader:
        if saved >= num_clips:
            break

        batch = move_batch_to_device(batch, device)
        bsz = batch['motion_parts'].size(0)

        for i in range(bsz):
            if saved >= num_clips:
                break

            one = select_batch_items(batch, i)
            x = one['motion_parts'].float()
            out = model(x) if not isinstance(one, dict) else model(one['motion_parts'].float())

            if isinstance(out, dict):
                x_recon = out['x_recon']
                recon_loss = out.get('recon_loss', None)
                total_loss = out.get('loss', None)
                encoding_indices = out.get('encoding_indices', None)
            else:
                raise ValueError('Model forward must return a dict with x_recon.')

            metrics = compute_metrics(x, x_recon)
            clip_id = get_clip_id(batch, i)

            row = {
                'clip_id': clip_id,
                'shape': tuple(x.shape),
                'l1': metrics['l1'],
                'mse': metrics['mse'],
                'rmse': metrics['rmse'],
                'max_abs': metrics['max_abs'],
                'recon_loss_model': float(recon_loss.item()) if recon_loss is not None else np.nan,
                'total_loss_model': float(total_loss.item()) if total_loss is not None else np.nan,
            }
            rows.append(row)

            if save_tensors:
                save_path = os.path.join(tensors_dir, f'{saved:03d}_{clip_id}.pt')
                payload = {
                    'clip_id': clip_id,
                    'x': x.detach().cpu(),
                    'x_recon': x_recon.detach().cpu(),
                    'metrics': metrics,
                }
                if encoding_indices is not None:
                    payload['encoding_indices'] = (
                        encoding_indices.detach().cpu()
                        if torch.is_tensor(encoding_indices)
                        else encoding_indices
                    )
                torch.save(payload, save_path)

            saved += 1

    csv_path = os.path.join(out_dir, 'reconstruction_metrics.csv')
    with open(csv_path, 'w') as f:
        f.write('clip_id,shape,l1,mse,rmse,max_abs,recon_loss_model,total_loss_model\n')
        for r in rows:
            f.write(
                f"{r['clip_id']},\"{r['shape']}\",{r['l1']:.8f},{r['mse']:.8f},"
                f"{r['rmse']:.8f},{r['max_abs']:.8f},{r['recon_loss_model']:.8f},"
                f"{r['total_loss_model']:.8f}\n"
            )

    if rows:
        mean_l1 = float(np.mean([r['l1'] for r in rows]))
        mean_mse = float(np.mean([r['mse'] for r in rows]))
        mean_rmse = float(np.mean([r['rmse'] for r in rows]))
        summary_path = os.path.join(out_dir, 'summary.txt')
        with open(summary_path, 'w') as f:
            f.write(f'num_clips={len(rows)}\n')
            f.write(f'mean_l1={mean_l1:.8f}\n')
            f.write(f'mean_mse={mean_mse:.8f}\n')
            f.write(f'mean_rmse={mean_rmse:.8f}\n')
            f.write('notes=Use recon_tensors/*.pt for later qualitative visualization in your repo.\n')

    return rows


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', type=str, required=False, default='path/to/latest.tar')
    parser.add_argument('--num_clips', type=int, default=15)
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--out_dir', type=str, default='output/vqvae_validation')
    parser.add_argument('--save_tensors', action='store_true')
    args = parser.parse_args()

    set_seed(args.seed)
    device = torch.device(args.device)

    model, val_loader = build_model_and_loader(args)
    model = model.to(device)

    rows = run_validation(
        model=model,
        val_loader=val_loader,
        device=device,
        num_clips=args.num_clips,
        out_dir=args.out_dir,
        save_tensors=args.save_tensors,
    )

    print(f'Saved validation outputs for {len(rows)} clips to: {args.out_dir}')
