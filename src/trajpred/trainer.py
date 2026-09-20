"""Config-driven training loop shared by every trained stage."""

import copy
import csv
import random
from pathlib import Path
from typing import Callable

import numpy as np
import torch
from torch import Tensor, nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from trajpred.dataset import AV2Dataset, collate
from trajpred.env import pick_device, processed_dir
from trajpred.evaluation import to_device
from trajpred.metrics import min_ade, min_fde

# (model output, batch) -> scalar loss
LossFn = Callable[[tuple[Tensor, Tensor], dict[str, Tensor]], Tensor]


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


class Trainer:
    def __init__(self, model: nn.Module, loss_fn: LossFn, cfg: dict):
        seed_everything(cfg["seed"])
        self.cfg = cfg
        self.device = pick_device()
        self.model = model.to(self.device)
        self.loss_fn = loss_fn
        self.ckpt_dir = Path(cfg["ckpt_dir"])

        train_set = AV2Dataset(cfg.get("processed_train") or processed_dir("train"))
        val_set = AV2Dataset(cfg.get("processed_val") or processed_dir("val"))
        self.train_loader = DataLoader(
            train_set, batch_size=cfg["batch_size"], shuffle=True, collate_fn=collate, drop_last=True
        )
        self.val_loader = DataLoader(val_set, batch_size=256, collate_fn=collate)

        self.opt = torch.optim.AdamW(
            model.parameters(), lr=cfg["lr"], weight_decay=cfg["weight_decay"]
        )
        steps = cfg["epochs"] * (len(self.train_loader) // cfg["grad_accum"])
        self.sched = torch.optim.lr_scheduler.OneCycleLR(
            self.opt, max_lr=cfg["lr"], total_steps=steps, pct_start=0.1
        )

    def overfit_one_batch(self, steps: int = 1000, n: int = 16) -> tuple[float, float]:
        """Memorize n scenes with a throwaway copy of the model.

        Returns (initial loss, final loss). If a model + loss can't drive this
        toward zero, something is broken (shapes, masks, gradient flow) and a
        real training run would only hide it behind hours of compute.
        """
        model = copy.deepcopy(self.model).train()
        opt = torch.optim.AdamW(model.parameters(), lr=self.cfg["lr"])
        batch = to_device(next(iter(self.train_loader)), self.device)
        batch = {k: v[:n] for k, v in batch.items()}
        first = last = float("nan")
        for step in range(steps):
            loss = self.loss_fn(model(batch), batch)
            opt.zero_grad()
            loss.backward()
            opt.step()
            last = loss.item()
            if step == 0:
                first = last
        return first, last

    @torch.no_grad()
    def validate(self) -> dict[str, float]:
        self.model.eval()
        total = {"loss": 0.0, "minADE": 0.0, "minFDE": 0.0}
        n = 0
        for batch in self.val_loader:
            batch = to_device(batch, self.device)
            traj, logits = self.model(batch)
            b = len(traj)
            total["loss"] += self.loss_fn((traj, logits), batch).item() * b
            total["minADE"] += min_ade(traj, batch["future"]).sum().item()
            total["minFDE"] += min_fde(traj, batch["future"]).sum().item()
            n += b
        return {k: v / n for k, v in total.items()}

    def fit(self) -> Path:
        """Train for cfg['epochs']; keep the checkpoint with the best val minFDE."""
        self.ckpt_dir.mkdir(parents=True, exist_ok=True)
        best_path = self.ckpt_dir / "best.pt"
        best = float("inf")
        accum = self.cfg["grad_accum"]
        with open(self.ckpt_dir / "log.csv", "w", newline="") as f:
            log = csv.writer(f)
            log.writerow(["epoch", "train_loss", "val_loss", "val_minADE", "val_minFDE", "lr"])
            for epoch in range(1, self.cfg["epochs"] + 1):
                self.model.train()
                running, seen = 0.0, 0
                bar = tqdm(self.train_loader, desc=f"epoch {epoch}", leave=False)
                for i, batch in enumerate(bar):
                    batch = to_device(batch, self.device)
                    loss = self.loss_fn(self.model(batch), batch)
                    (loss / accum).backward()
                    if (i + 1) % accum == 0:
                        nn.utils.clip_grad_norm_(self.model.parameters(), 5.0)
                        self.opt.step()
                        self.sched.step()
                        self.opt.zero_grad()
                    running += loss.item()
                    seen += 1
                self.opt.zero_grad()  # drop any partial accumulation at epoch end

                val = self.validate()
                lr = self.sched.get_last_lr()[0]
                log.writerow(
                    [epoch, running / seen, val["loss"], val["minADE"], val["minFDE"], lr]
                )
                f.flush()
                improved = val["minFDE"] < best
                if improved:
                    best = val["minFDE"]
                    torch.save(
                        {"model": self.model.state_dict(), "epoch": epoch, "val": val, "cfg": self.cfg},
                        best_path,
                    )
                print(
                    f"epoch {epoch:3d}  train {running / seen:.4f}  val {val['loss']:.4f}  "
                    f"minADE {val['minADE']:.3f}  minFDE {val['minFDE']:.3f}"
                    + ("  *" if improved else ""),
                    flush=True,
                )
        return best_path
