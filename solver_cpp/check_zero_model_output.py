#!/usr/bin/env python3
import argparse
from pathlib import Path

import torch

print("Imports done")

def real_function(x_water, x_terrain):
    # The actual solver uses this update logic:
    # for every of the 4 neighboring cells, we take the difference in the absolute water level min(this.water, (this.water + this.terrain) - (neighbor.water + neighbor.terrain)), and 1/4 of that difference flows from the current cell to the neighbor cell - or the other way around.
    
    new_value = x_water[1][1] # start with current cell's water level
    this_total_water = x_water[1][1] + x_terrain[1][1]
    for t in [(0, 1), (2, 1), (1, 0), (1, 2)]: # for each of the 4 neighbors
        neighbor_total_water = x_water[t] + x_terrain[t]
        diff = this_total_water - neighbor_total_water
        flow = min(x_water[1][1], max(diff, 0)) * 0.25 - min(x_water[t], max(-diff, 0)) * 0.25
        new_value -= flow
        
    return new_value


def main() -> None:
    parser = argparse.ArgumentParser(description="Run TorchScript model on 1x1x3x3 zero input and print output")
    parser.add_argument(
        "--model",
        type=Path,
        default=Path("train_models/model_a/best_model_jit.pt"),
        #default=Path("train_models/model_a/best_model_jit.pt"),
        help="Path to TorchScript model",
    )
    parser.add_argument(
        "--device",
        choices=["cpu", "cuda"],
        default="cpu",
        help="Inference device",
    )
    args = parser.parse_args()

    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but not available.")
    
    print(f"Loading model from {args.model} onto device {args.device}...")

    model = torch.jit.load(str(args.model), map_location=args.device)
    model.eval()
    
    print(f"Model loaded. Forwarding zero input through the model...")

    x_water = torch.zeros((1, 1, 3, 3), dtype=torch.float32, device=args.device)
    x_terrain = torch.zeros((1, 1, 3, 3), dtype=torch.float32, device=args.device)
    
    # water=(0 0 0 ,0 0 0 ,0 0 0 ,) terrain=(0 198 198 ,82 82 83 ,74 74 74 ,)
    
    # test water vals
    #x_water[0, 0, 0, 1] = 10
    #x_water[0, 0, 2, 1] = 10
    #x_water[0, 0, 1, 0] = 10
    #x_water[0, 0, 1, 2] = 10
    
    x_terrain[0, 0, 0, 0] = 0
    x_terrain[0, 0, 0, 1] = 198
    x_terrain[0, 0, 0, 2] = 198
    x_terrain[0, 0, 1, 0] = 82
    x_terrain[0, 0, 1, 1] = 82
    x_terrain[0, 0, 1, 2] = 83
    x_terrain[0, 0, 2, 0] = 74
    x_terrain[0, 0, 2, 1] = 74
    x_terrain[0, 0, 2, 2] = 74

    with torch.no_grad():
        out = model(x_water, x_terrain)
        print("forward args: 2")
        print("input[0]:", x_water)
        print("input[1]:", x_terrain)

    print("output shape:", tuple(out.shape))
    print("output tensor:", out)
    print("output value(s):", out.detach().cpu().numpy())
    print("Expected output value (based on the real function):", real_function(x_water.cpu().numpy()[0, 0], x_terrain.cpu().numpy()[0, 0]))


if __name__ == "__main__":
    main()
