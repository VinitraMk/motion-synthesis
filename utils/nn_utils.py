from torch import nn
import math
import time
from typing import Dict, Any
import torch
import math

def init_weight(m):
    if isinstance(m, nn.Conv1d) or isinstance(m, nn.Linear) or isinstance(m, nn.ConvTranspose1d):
        nn.init.xavier_normal_(m.weight)
        # m.bias.data.fill_(0.01)
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)


def move_batch_to_device(batch: Dict[str, Any], device: torch.device) -> Dict[str, Any]:
    out = {}
    for k, v in batch.items():
        out[k] = v.to(device) if torch.is_tensor(v) else v
    return out


class CyclicBetaScheduler:
    def __init__(
        self,
        total_steps: int,
        num_cycles: int = 3,
        beta_max: float = 0.01,
        schedule: str = "cosine",   # "linear" or "cosine"
        start_step: int = 0,
    ):
        assert total_steps > 0
        assert num_cycles > 0
        assert beta_max >= 0.0
        assert schedule in {"linear", "cosine"}

        self.total_steps = total_steps
        self.num_cycles = num_cycles
        self.beta_max = beta_max
        self.schedule = schedule

        self.cycle_steps = max(1, total_steps // num_cycles)
        self.step_num = start_step
        self.beta = self._compute_beta(self.step_num)

    def _compute_beta(self, step: int) -> float:
        cycle_pos = (step % self.cycle_steps) / self.cycle_steps

        if self.schedule == "linear":
            return self.beta_max * cycle_pos

        if self.schedule == "cosine":
            return self.beta_max * 0.5 * (1.0 - math.cos(math.pi * cycle_pos))

        raise ValueError(f"Unknown schedule: {self.schedule}")

    def step(self) -> float:
        self.step_num += 1
        self.beta = self._compute_beta(self.step_num)
        return self.beta

    def get_beta(self) -> float:
        return self.beta

    def state_dict(self):
        return {
            "total_steps": self.total_steps,
            "num_cycles": self.num_cycles,
            "beta_max": self.beta_max,
            "schedule": self.schedule,
            "cycle_steps": self.cycle_steps,
            "step_num": self.step_num,
            "beta": self.beta,
        }

    def load_state_dict(self, state_dict):
        self.total_steps = state_dict["total_steps"]
        self.num_cycles = state_dict["num_cycles"]
        self.beta_max = state_dict["beta_max"]
        self.schedule = state_dict["schedule"]
        self.cycle_steps = state_dict["cycle_steps"]
        self.step_num = state_dict["step_num"]
        self.beta = state_dict["beta"]