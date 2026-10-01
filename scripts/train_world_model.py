#!/usr/bin/env python3
"""Train the optional GRU world model on synthetic trajectories.

Requires the optional learn extra:  pip install ".[learn]"

The model learns residual box dynamics from short windows of past boxes and is
loaded by setting ``predict.backend: torch`` and ``predict.torch_model`` in the
config. Without it, the pipeline falls back to the Kalman predictor, so this
script is entirely optional.
"""

from __future__ import annotations

import argparse
import sys


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Train the AeroNetra-Orion GRU world model")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--samples", type=int, default=4000)
    parser.add_argument("--out", default="outputs/world_model.pt")
    args = parser.parse_args(argv)

    try:
        import numpy as np
        import torch
        import torch.nn as nn
    except ImportError:
        print("PyTorch is required: pip install \"aeronetra-orion[learn]\"", file=sys.stderr)
        return 1

    from orion.predict import TorchWorldModel

    past, future = 5, 30
    rng = np.random.default_rng(0)

    # Synthesise constant-velocity trajectories with small noise.
    def make_batch(n):
        cx = rng.uniform(0, 640, n)
        cy = rng.uniform(0, 480, n)
        vx = rng.uniform(-120, 120, n)
        vy = rng.uniform(-120, 120, n)
        w = rng.uniform(18, 50, n)
        h = rng.uniform(14, 40, n)
        vw = rng.uniform(-20, 20, n)
        seq = []
        for t in range(past + future):
            x = np.stack([cx + vx * t, cy + vy * t, w + vw * t, h + vw * t], axis=1)
            seq.append(x)
        return np.stack(seq, axis=1)  # (n, past+future, 4)

    model = TorchWorldModel(past=past, future=future).net
    optim = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = nn.MSELoss()

    for epoch in range(args.epochs):
        data = make_batch(args.samples)
        ref = data[:, past - 1]
        cx, cy = (ref[:, 0] + ref[:, 2]) / 2, (ref[:, 1] + ref[:, 3]) / 2
        w = np.maximum(1e-3, ref[:, 2] - ref[:, 0])
        h = np.maximum(1e-3, ref[:, 3] - ref[:, 1])
        norm = data.copy()
        norm[:, :, [0, 2]] = (norm[:, :, [0, 2]] - cx[:, None]) / w[:, None]
        norm[:, :, [1, 3]] = (norm[:, :, [1, 3]] - cy[:, None]) / h[:, None]
        past_in = torch.tensor(norm[:, :past], dtype=torch.float32)
        target = torch.tensor(norm[:, past:], dtype=torch.float32)
        pred = model(past_in)
        loss = loss_fn(pred, target)
        optim.zero_grad()
        loss.backward()
        optim.step()
        if epoch % 5 == 0 or epoch == args.epochs - 1:
            print(f"epoch {epoch:3d}  loss {loss.item():.5f}")

    import os

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    torch.save(model.state_dict(), args.out)
    print(f"saved {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
