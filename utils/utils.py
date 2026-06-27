import math
import time
import os
import torch

def print_current_loss_decomp(start_time, niter_state, total_niters, losses, epoch=None, inner_iter=None):

    def as_minutes(s):
        m = math.floor(s / 60)
        s -= m * 60
        return '%dm %ds' % (m, s)

    def time_since(since, percent):
        now = time.time()
        s = now - since
        es = s / percent
        rs = es - s
        return '%s (- %s)' % (as_minutes(s), as_minutes(rs))

    print('epoch: %03d inner_iter: %5d' % (epoch, inner_iter), end=" ")
    # now = time.time()
    message = '%s niter: %07d completed: %3d%%)'%(time_since(start_time, niter_state / total_niters), niter_state, niter_state / total_niters * 100)
    for k, v in losses.items():
        message += ' %s: %.4f ' % (k, v)
    print(message)


def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)

def cpu_deepcopy_state(obj):
    """
    Recursively copy a state-like object to CPU RAM.

    - Tensors -> detached clone on CPU
    - dict -> recursively copied dict
    - list/tuple -> recursively copied sequence
    - other python objects -> returned as-is
    """
    if torch.is_tensor(obj):
        return obj.detach().cpu().clone()
    elif isinstance(obj, dict):
        return {k: cpu_deepcopy_state(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [cpu_deepcopy_state(v) for v in obj]
    elif isinstance(obj, tuple):
        return tuple(cpu_deepcopy_state(v) for v in obj)
    else:
        return obj
    
def move_state_to_device(obj, device):
    """
    Recursively move a state-like object to `device`.

    - Tensors -> .to(device)
    - dict -> recursively processed dict
    - list/tuple -> recursively processed sequence
    - other Python objects -> returned as-is
    """
    if torch.is_tensor(obj):
        return obj.to(device)
    elif isinstance(obj, dict):
        return {k: move_state_to_device(v, device) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [move_state_to_device(v, device) for v in obj]
    elif isinstance(obj, tuple):
        return tuple(move_state_to_device(v, device) for v in obj)
    else:
        return obj
    