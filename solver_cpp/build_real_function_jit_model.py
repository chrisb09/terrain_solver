#!/usr/bin/env python3
import argparse
from pathlib import Path

import torch


class RealFunctionModel(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.register_buffer("quarter", torch.tensor(0.25, dtype=torch.float32))

    def forward(self, x_water: torch.Tensor, x_terrain: torch.Tensor) -> torch.Tensor:
        if x_water.dim() != 4 or x_terrain.dim() != 4:
            raise RuntimeError("Expected 4D tensors with shape [B, 1, 3, 3]")
        if x_water.size(1) != 1 or x_terrain.size(1) != 1:
            raise RuntimeError("Expected channel dimension C=1")
        if x_water.size(2) != 3 or x_water.size(3) != 3:
            raise RuntimeError("Expected x_water shape [B, 1, 3, 3]")
        if x_terrain.size(2) != 3 or x_terrain.size(3) != 3:
            raise RuntimeError("Expected x_terrain shape [B, 1, 3, 3]")

        center_w = x_water[:, 0, 1, 1]
        center_t = x_terrain[:, 0, 1, 1]
        this_total = center_w + center_t
        new_value = center_w.clone()

        nz = 0
        nx = 1
        neighbor_total = x_water[:, 0, nz, nx] + x_terrain[:, 0, nz, nx]
        diff = this_total - neighbor_total
        outflow = torch.minimum(center_w, torch.clamp_min(diff, 0.0)) * self.quarter
        inflow = torch.minimum(x_water[:, 0, nz, nx], torch.clamp_min(-diff, 0.0)) * self.quarter
        new_value = new_value - (outflow - inflow)

        nz = 2
        nx = 1
        neighbor_total = x_water[:, 0, nz, nx] + x_terrain[:, 0, nz, nx]
        diff = this_total - neighbor_total
        outflow = torch.minimum(center_w, torch.clamp_min(diff, 0.0)) * self.quarter
        inflow = torch.minimum(x_water[:, 0, nz, nx], torch.clamp_min(-diff, 0.0)) * self.quarter
        new_value = new_value - (outflow - inflow)

        nz = 1
        nx = 0
        neighbor_total = x_water[:, 0, nz, nx] + x_terrain[:, 0, nz, nx]
        diff = this_total - neighbor_total
        outflow = torch.minimum(center_w, torch.clamp_min(diff, 0.0)) * self.quarter
        inflow = torch.minimum(x_water[:, 0, nz, nx], torch.clamp_min(-diff, 0.0)) * self.quarter
        new_value = new_value - (outflow - inflow)

        nz = 1
        nx = 2
        neighbor_total = x_water[:, 0, nz, nx] + x_terrain[:, 0, nz, nx]
        diff = this_total - neighbor_total
        outflow = torch.minimum(center_w, torch.clamp_min(diff, 0.0)) * self.quarter
        inflow = torch.minimum(x_water[:, 0, nz, nx], torch.clamp_min(-diff, 0.0)) * self.quarter
        new_value = new_value - (outflow - inflow)

        return new_value


def main() -> None:
    parser = argparse.ArgumentParser(description="Build and save a TorchScript model implementing the exact real_function update rule")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("train_models/model_a/real_function_jit.pt"),
        help="Output path for scripted model",
    )
    parser.add_argument(
        "--device",
        choices=["cpu", "cuda"],
        default="cpu",
        help="Device used during scripting and quick self-check",
    )
    args = parser.parse_args()

    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but not available")

    device = torch.device(args.device)
    model = RealFunctionModel().to(device).eval()

    scripted = torch.jit.script(model)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    scripted.save(str(args.output))

    x_water = torch.zeros((1, 1, 3, 3), dtype=torch.float32, device=device)
    x_terrain = torch.zeros((1, 1, 3, 3), dtype=torch.float32, device=device)
    with torch.no_grad():
        y = scripted(x_water, x_terrain)

    print(f"Saved TorchScript model to: {args.output}")
    print(f"Self-check output on zero input: {y.detach().cpu().numpy()}")


if __name__ == "__main__":
    main()
