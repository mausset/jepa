import numpy as np
import math

from torch.optim.lr_scheduler import LRScheduler


def cosine_schedule(
    step, total_steps, start_val, end_val, warmup_pct=0.1, cooldown_pct=0.3
):
    warmup_steps = int(warmup_pct * total_steps)
    cooldown_steps = int(cooldown_pct * total_steps)
    anneal_steps = total_steps - warmup_steps - cooldown_steps

    if anneal_steps <= 0:
        return end_val

    if step < warmup_steps:
        return start_val
    elif step >= (total_steps - cooldown_steps):
        return end_val
    else:
        progress = (step - warmup_steps) / anneal_steps
        cosine_factor = 0.5 * (1 + np.cos(progress * np.pi))
        return end_val + (start_val - end_val) * cosine_factor


class LinearSchedule(object):
    def __init__(self, start, end, duration, start_step=0):
        self.start = start
        self.end = end
        self.duration = duration
        self.start_step = start_step

    def __call__(self, t):
        t = max(t - self.start_step, 0)
        if self.start > self.end:
            return max(
                self.start + (self.end - self.start) * t / self.duration, self.end
            )
        return min(self.start + (self.end - self.start) * t / self.duration, self.end)


class TrapezoidSchedule(LRScheduler):
    def __init__(
        self,
        optimizer,
        warmup_steps: int,
        start_lr: float,
        ref_lr: float,
        total_steps: int,
        cooldown_frac: float,
        final_lr_frac: float,
        last_epoch: int = -1,
    ):
        self.optimizer = optimizer
        self.start_lr = start_lr
        self.ref_lr = ref_lr

        self.warmup_steps = warmup_steps
        self.total_steps = max(1, total_steps)
        self.cooldown_frac = cooldown_frac
        self.final_lr_frac = final_lr_frac

        # Derived quantities
        self.cooldown_steps = int(self.cooldown_frac * self.total_steps)
        self.cooldown_steps = max(
            0, min(self.cooldown_steps, self.total_steps - self.warmup_steps)
        )

        self.plateau_steps = max(
            0, self.total_steps - self.warmup_steps - self.cooldown_steps
        )

        self.final_lr = self.final_lr_frac * self.ref_lr

        self._step_count = 0.0
        self.last_epoch = last_epoch

        super().__init__(optimizer, last_epoch)

    def get_lr(self):  # type: ignore
        step = self._step_count

        # 1) Warmup: start_lr -> ref_lr
        if step < self.warmup_steps:
            progress = float(step) / float(max(1, self.warmup_steps))
            new_lr = self.start_lr + progress * (self.ref_lr - self.start_lr)

        # 2) Plateau: constant ref_lr
        elif step < self.warmup_steps + self.plateau_steps:
            new_lr = self.ref_lr

        # 3) Linear cooldown: ref_lr -> final_lr
        elif step < self.total_steps and self.cooldown_steps > 0:
            cooldown_step = step - self.warmup_steps - self.plateau_steps
            progress = float(cooldown_step) / float(max(1, self.cooldown_steps))
            progress = min(max(progress, 0.0), 1.0)
            new_lr = self.ref_lr + progress * (self.final_lr - self.ref_lr)

        # After total_steps, stay at final_lr
        else:
            new_lr = self.final_lr

        return [new_lr for _ in self.optimizer.param_groups]

    def state_dict(self):
        return {"_step_count": self._step_count, "last_epoch": self.last_epoch}

    def load_state_dict(self, state_dict):
        self._step_count = state_dict["_step_count"]
        self.last_epoch = state_dict["last_epoch"]


class WarmupCosineSchedule(LRScheduler):
    def __init__(
        self,
        optimizer,
        warmup_steps,
        start_lr,
        ref_lr,
        total_steps,
        last_epoch=-1,
        final_lr=0.0,
    ):
        self.optimizer = optimizer
        self.start_lr = start_lr
        self.ref_lr = ref_lr
        self.final_lr = final_lr
        self.warmup_steps = warmup_steps
        self.total_steps = total_steps - warmup_steps
        self._step_count = 0.0
        self.last_epoch = -1

        super().__init__(optimizer, last_epoch)

    def get_lr(self):  # type: ignore
        if self._step_count < self.warmup_steps:
            progress = float(self._step_count) / float(max(1, self.warmup_steps))
            new_lr = self.start_lr + progress * (self.ref_lr - self.start_lr)
        else:
            # -- progress after warmup
            progress = float(self._step_count - self.warmup_steps) / float(
                max(1, self.total_steps)
            )
            new_lr = max(
                self.final_lr,
                self.final_lr
                + (self.ref_lr - self.final_lr)
                * 0.5
                * (1.0 + math.cos(math.pi * progress)),
            )

        return [new_lr for _ in self.optimizer.param_groups]

    def state_dict(self):
        return {"_step_count": self._step_count, "last_epoch": self.last_epoch}

    def load_state_dict(self, state_dict):
        self._step_count = state_dict["_step_count"]
        self.last_epoch = state_dict["last_epoch"]


class CosineWDSchedule(object):
    def __init__(self, optimizer, ref_wd, total_steps, final_wd=0.0):
        self.optimizer = optimizer
        self.ref_wd = ref_wd
        self.final_wd = final_wd
        self.total_steps = total_steps
        self._step = 0.0

    def step(self):
        self._step += 1
        progress = self._step / self.total_steps
        new_wd = self.final_wd + (self.ref_wd - self.final_wd) * 0.5 * (
            1.0 + math.cos(math.pi * progress)
        )

        if self.final_wd <= self.ref_wd:
            new_wd = max(self.final_wd, new_wd)
        else:
            new_wd = min(self.final_wd, new_wd)

        for group in self.optimizer.param_groups:
            if ("WD_exclude" not in group) or not group["WD_exclude"]:
                group["weight_decay"] = new_wd
        return new_wd

    def state_dict(self):
        return {"_step": self._step}

    def load_state_dict(self, state_dict):
        self._step = state_dict["_step"]
