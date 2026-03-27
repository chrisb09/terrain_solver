#!/usr/bin/env python3

from __future__ import annotations

import argparse
import os
import sys


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Prepare terrain/water input HDF5 for the mini app solver."
    )
    parser.add_argument("--input-image", required=True, help="Path to source heightmap image.")
    parser.add_argument("--output-hdf5", required=True, help="Path to output HDF5 file.")
    parser.add_argument(
        "--target-width",
        type=int,
        default=2160,
        help="Output terrain width. Must be divisible by chunk size.",
    )
    parser.add_argument(
        "--target-height",
        type=int,
        default=1080,
        help="Output terrain height. Must be divisible by chunk size.",
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=60,
        help="Chunk edge size used for decomposition metadata.",
    )
    parser.add_argument(
        "--init-mode",
        choices=["uniform", "circle", "square"],
        default="uniform",
        help="Initial water distribution mode.",
    )
    parser.add_argument(
        "--init-depth",
        type=float,
        default=0.0,
        help="Water depth value used by the selected initialization mode.",
    )
    parser.add_argument("--center-x", type=int, default=None, help="Center X for circle/square.")
    parser.add_argument("--center-z", type=int, default=None, help="Center Z for circle/square.")
    parser.add_argument("--radius", type=int, default=100, help="Radius for circle mode.")
    parser.add_argument("--half-size", type=int, default=100, help="Half-size for square mode.")
    parser.add_argument(
        "--preview-png",
        default=None,
        help="Optional quicklook render output path.",
    )
    return parser.parse_args()


def fail(message: str) -> None:
    print(f"ERROR: {message}", file=sys.stderr)
    sys.exit(1)


def import_required_modules():
    missing = []
    try:
        import numpy as np  # noqa: F401
    except ImportError:
        missing.append("numpy")
    try:
        import h5py  # noqa: F401
    except ImportError:
        missing.append("h5py")
    try:
        from PIL import Image  # noqa: F401
    except ImportError:
        missing.append("Pillow")

    if missing:
        fail(
            "Missing Python packages: "
            + ", ".join(missing)
            + ". Use a SmartSim env under /hpcwork/ro092286/smartsim/python (e.g. smartsim_cpu)."
        )


def make_water(
    np,
    width: int,
    height: int,
    mode: str,
    depth: float,
    center_x: int | None,
    center_z: int | None,
    radius: int,
    half_size: int,
):
    water = np.zeros((height, width), dtype=np.float32)
    if depth <= 0.0:
        return water

    if mode == "uniform":
        water[:, :] = depth
        return water

    cx = width // 2 if center_x is None else center_x
    cz = height // 2 if center_z is None else center_z

    zz, xx = np.indices((height, width), dtype=np.int32)
    if mode == "circle":
        mask = (xx - cx) ** 2 + (zz - cz) ** 2 <= radius**2
    elif mode == "square":
        mask = (np.abs(xx - cx) <= half_size) & (np.abs(zz - cz) <= half_size)
    else:
        fail(f"Unsupported init mode: {mode}")
    water[mask] = depth
    return water


def terrain_colormap(np, terrain):
    elevation_ground = 144
    end_of_green = 154
    end_of_brown = 185

    color_map_ground = np.zeros((256, 3), dtype=np.uint8)
    idx = np.arange(256)

    mask = idx < elevation_ground
    color_map_ground[mask] = np.column_stack([idx[mask], idx[mask], idx[mask]])

    mask = (idx >= elevation_ground) & (idx < end_of_green)
    green_value = (
        128 + 127 * (idx[mask] - elevation_ground) / (end_of_green - elevation_ground)
    ).astype(np.uint8)
    color_map_ground[mask] = np.column_stack(
        [np.zeros_like(green_value), green_value, np.zeros_like(green_value)]
    )

    mask = (idx >= end_of_green) & (idx < end_of_brown)
    brown_value = (
        255 * (idx[mask] - end_of_green) / (end_of_brown - end_of_green)
    ).astype(np.uint8)
    color_map_ground[mask] = np.column_stack(
        [brown_value, np.full_like(brown_value, 128), np.zeros_like(brown_value)]
    )

    mask = idx >= end_of_brown
    white_value = (
        200 + 55 * (idx[mask] - end_of_brown) / (256 - end_of_brown)
    ).astype(np.uint8)
    color_map_ground[mask] = np.column_stack([white_value, white_value, white_value])

    # Stretch terrain values into an 8-bit index range before applying fixed cutoffs.
    # This keeps the old threshold logic but adapts to input maps with shifted ranges.
    p_low = float(np.percentile(terrain, 1.0))
    p_high = float(np.percentile(terrain, 99.0))
    if p_high <= p_low:
        t_idx = np.clip(terrain, 0, 255).astype(np.uint8)
    else:
        t_scaled = (terrain.astype(np.float32) - p_low) * (255.0 / (p_high - p_low))
        t_idx = np.clip(t_scaled, 0, 255).astype(np.uint8)
    return color_map_ground[t_idx]


def overlay_water(np, rgb, water):
    if not np.any(water > 0):
        return rgb

    out = rgb.astype(np.float32)
    w = water.astype(np.float32)
    w_max = float(w.max()) if float(w.max()) > 0 else 1.0
    alpha = np.clip((w / w_max) * 0.8, 0.0, 0.8)

    out[:, :, 2] = out[:, :, 2] * (1.0 - alpha) + 255.0 * alpha
    out[:, :, 1] = out[:, :, 1] * (1.0 - 0.5 * alpha)
    out[:, :, 0] = out[:, :, 0] * (1.0 - 0.7 * alpha)

    return np.clip(out, 0, 255).astype(np.uint8)


def main() -> None:
    args = parse_args()
    import_required_modules()

    import h5py
    import numpy as np
    from PIL import Image

    if not os.path.exists(args.input_image):
        fail(f"Input image not found: {args.input_image}")
    if args.target_width <= 0 or args.target_height <= 0:
        fail("Target width/height must be positive.")
    if args.chunk_size <= 0:
        fail("Chunk size must be positive.")
    if args.target_width % args.chunk_size != 0 or args.target_height % args.chunk_size != 0:
        fail(
            f"Target size ({args.target_width}x{args.target_height}) must be divisible by chunk size {args.chunk_size}."
        )
    if args.radius < 0 or args.half_size < 0:
        fail("Radius/half-size must be non-negative.")

    Image.MAX_IMAGE_PIXELS = None
    src = Image.open(args.input_image)
    resized = src.resize((args.target_width, args.target_height), Image.Resampling.BILINEAR)
    resized_arr = np.asarray(resized)
    if resized_arr.ndim == 3:
        # Keep behavior aligned with old script: use first channel from ramp image.
        terrain = resized_arr[:, :, 0].astype(np.float32)
    else:
        terrain = resized_arr.astype(np.float32)
    water = make_water(
        np=np,
        width=args.target_width,
        height=args.target_height,
        mode=args.init_mode,
        depth=float(args.init_depth),
        center_x=args.center_x,
        center_z=args.center_z,
        radius=args.radius,
        half_size=args.half_size,
    )

    os.makedirs(os.path.dirname(os.path.abspath(args.output_hdf5)), exist_ok=True)
    with h5py.File(args.output_hdf5, "w") as h5f:
        h5f.create_dataset("terrain", data=terrain, dtype=np.float32)
        h5f.create_dataset("water_init", data=water, dtype=np.float32)
        h5f.attrs["nx"] = args.target_width
        h5f.attrs["nz"] = args.target_height
        h5f.attrs["chunk_w"] = args.chunk_size
        h5f.attrs["chunk_h"] = args.chunk_size
        h5f.attrs["chunks_x"] = args.target_width // args.chunk_size
        h5f.attrs["chunks_z"] = args.target_height // args.chunk_size
        h5f.attrs["source_image"] = args.input_image
        h5f.attrs["init_mode"] = args.init_mode
        h5f.attrs["init_depth"] = float(args.init_depth)

    if args.preview_png:
        rgb = terrain_colormap(np, terrain)
        rgb = overlay_water(np, rgb, water)
        preview = Image.fromarray(rgb, mode="RGB")
        os.makedirs(os.path.dirname(os.path.abspath(args.preview_png)), exist_ok=True)
        preview.save(args.preview_png)

    print(
        "Prepared dataset:",
        f"terrain={terrain.shape}",
        f"water_nonzero={int((water > 0).sum())}",
        f"output={args.output_hdf5}",
    )
    if args.preview_png:
        print(f"Preview image written to: {args.preview_png}")


if __name__ == "__main__":
    main()
