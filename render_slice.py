#!/usr/bin/env python3

import argparse
import os
import queue
import sys
import threading


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Render side-view terrain/water slice (east-west vs absolute height)."
    )
    parser.add_argument("--input-hdf5", required=True, help="Trajectory HDF5 file.")
    parser.add_argument(
        "--slice-z",
        type=int,
        default=-1,
        help="Z index of horizontal slice (default: center row).",
    )
    parser.add_argument(
        "--image-height",
        type=int,
        default=1080,
        help="Output image height in world-height units.",
    )
    parser.add_argument(
        "--height-min",
        type=float,
        default=0.0,
        help="Minimum absolute world height shown at the image bottom row.",
    )
    parser.add_argument(
        "--step",
        type=int,
        default=-1,
        help="Step index in saved timeline (default: last). Ignored when --all-steps is set.",
    )
    parser.add_argument("--all-steps", action="store_true", help="Render all saved snapshots.")
    parser.add_argument(
        "--output-image",
        default=None,
        help="Output image path for single-step mode.",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Output directory for all-step mode.",
    )
    parser.add_argument(
        "--threads",
        type=int,
        default=1,
        help="Worker threads for --all-steps rendering (default: 1).",
    )
    parser.add_argument(
        "--no-overlay-text",
        action="store_true",
        help="Disable metadata text overlay in rendered frames.",
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
        from PIL import Image, ImageDraw, ImageFont  # noqa: F401
    except ImportError:
        missing.append("Pillow")

    if missing:
        fail(
            "Missing Python packages: "
            + ", ".join(missing)
            + ". Use a SmartSim env under /hpcwork/ro092286/smartsim/python (e.g. smartsim_cpu)."
        )


def build_ground_lut(np):
    elevation_ground = 144
    end_of_green = 154
    end_of_brown = 185

    lut = np.zeros((256, 3), dtype=np.uint8)
    idx = np.arange(256)

    # Gray for low elevations
    mask = idx < elevation_ground
    lut[mask] = np.column_stack([idx[mask], idx[mask], idx[mask]])

    # Green gradient
    mask = (idx >= elevation_ground) & (idx < end_of_green)
    green = (
        128 + 127 * (idx[mask] - elevation_ground) / (end_of_green - elevation_ground)
    ).astype(np.uint8)
    lut[mask] = np.column_stack([np.zeros_like(green), green, np.zeros_like(green)])

    # Brown gradient
    mask = (idx >= end_of_green) & (idx < end_of_brown)
    brown = (
        255 * (idx[mask] - end_of_green) / (end_of_brown - end_of_green)
    ).astype(np.uint8)
    lut[mask] = np.column_stack([brown, np.full_like(brown, 128), np.zeros_like(brown)])

    # White gradient
    mask = idx >= end_of_brown
    white = (
        200 + 55 * (idx[mask] - end_of_brown) / (256 - end_of_brown)
    ).astype(np.uint8)
    lut[mask] = np.column_stack([white, white, white])
    return lut


def resolve_slice_z(slice_z: int, nz: int) -> int:
    if slice_z < 0:
        return nz // 2
    if slice_z >= nz:
        fail(f"--slice-z={slice_z} is out of bounds for nz={nz}.")
    return slice_z


def render_slice_frame(
    np,
    Image,
    ImageDraw,
    ImageFont,
    ground_lut,
    terrain_row,
    water_row,
    image_height: int,
    height_min: float,
    out_path: str,
    overlay_lines=None,
):
    nx = terrain_row.shape[0]
    canvas = np.zeros((image_height, nx, 3), dtype=np.uint8)

    terrain_h = np.floor(terrain_row).astype(np.float32)
    surface_h = np.floor(terrain_row + water_row).astype(np.float32)

    y_from_bottom = (image_height - 1) - np.arange(image_height, dtype=np.float32)[:, None]  # (H,1)
    abs_y = height_min + y_from_bottom
    terrain_mask = abs_y <= terrain_h[None, :]

    # Terrain color comes from absolute vertical level (row), not from column height.
    # This creates consistent horizontal color bands across the full slice.
    row_height_idx = np.clip(abs_y[:, 0], 0, 255).astype(np.uint8)
    row_colors = ground_lut[row_height_idx]  # (H,3)
    for c in range(3):
        canvas[:, :, c] = np.where(terrain_mask, row_colors[:, None, c], canvas[:, :, c])

    water_top = np.maximum(surface_h, terrain_h)
    water_mask = (abs_y > terrain_h[None, :]) & (abs_y <= water_top[None, :])

    # Water color is vertical within each column: brightest at the top,
    # gradually darker downwards toward the terrain.
    #water_depth_px = np.maximum(water_top - terrain_h, 1).astype(np.float32)  # (nx,)
    rel_from_top = np.clip(
        (water_top[None, :] - y_from_bottom.astype(np.float32)) / 100.0,  # scale factor for color gradient
        0.0,
        1.0,
    )  # 0 at top, 1 at bottom
    bright = 1.0 - rel_from_top

    top_r, top_g, top_b = 60.0, 160.0, 220.0
    bottom_r, bottom_g, bottom_b = 2.0, 15.0, 120.0

    water_r = (bottom_r + (top_r - bottom_r) * bright).astype(np.uint8)
    water_g = (bottom_g + (top_g - bottom_g) * bright).astype(np.uint8)
    water_b = (bottom_b + (top_b - bottom_b) * bright).astype(np.uint8)

    canvas[:, :, 0] = np.where(water_mask, water_r, canvas[:, :, 0])
    canvas[:, :, 1] = np.where(water_mask, water_g, canvas[:, :, 1])
    canvas[:, :, 2] = np.where(water_mask, water_b, canvas[:, :, 2])

    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    img = Image.fromarray(canvas, mode="RGB")
    if overlay_lines:
        draw_overlay_text(ImageDraw, ImageFont, img, overlay_lines)
    img.save(out_path)


def _decode_attr(value):
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    return value


def load_overlay_metadata(np, h5f, n_saved: int):
    if n_saved <= 0:
        return None

    def read_series(name: str, dtype):
        if name in h5f and h5f[name].shape[0] >= n_saved:
            return h5f[name][:n_saved].astype(dtype)
        return np.full((n_saved,), np.nan, dtype=dtype)

    if "step_index" in h5f and h5f["step_index"].shape[0] >= n_saved:
        step_index = h5f["step_index"][:n_saved].astype(np.int64)
    else:
        step_index = np.arange(n_saved, dtype=np.int64)

    def attr_scalar(key: str, default):
        if key not in h5f.attrs:
            return default
        value = h5f.attrs[key]
        if isinstance(value, np.ndarray):
            if value.size == 0:
                return default
            value = value.reshape(-1)[0]
        if isinstance(value, np.generic):
            value = value.item()
        return value

    attrs = h5f.attrs
    return {
        "step_index": step_index,
        "solver_type": read_series("solver_type", np.int32),
        "mass": read_series("mass", np.float64),
        "drift": read_series("drift", np.float64),
        "moved_this_step": read_series("moved_this_step", np.float64),
        "min_water": read_series("min_water", np.float32),
        "min_positive_water": read_series("min_positive_water", np.float32),
        "max_water": read_series("max_water", np.float32),
        "runtime_seconds": read_series("runtime_seconds", np.float64),
        "grid_width": int(attr_scalar("grid_width", attr_scalar("width", -1))),
        "grid_height": int(attr_scalar("grid_height", attr_scalar("height", -1))),
        "chunk_size": int(attr_scalar("chunk_size", -1)),
        "chunks_x": int(attr_scalar("chunks_x", -1)),
        "chunks_z": int(attr_scalar("chunks_z", -1)),
        "ranks_x": int(attr_scalar("ranks_x", -1)),
        "ranks_z": int(attr_scalar("ranks_z", -1)),
        "io_mode": str(_decode_attr(attr_scalar("io_mode", ""))),
        "hdf5_xfer_mode": str(_decode_attr(attr_scalar("hdf5_xfer_mode", ""))),
        "sync_mode": str(_decode_attr(attr_scalar("sync_mode", ""))),
        "slurm_nodes": int(attr_scalar("slurm_nodes", attr_scalar("nodes", -1))),
        "slurm_tasks": int(attr_scalar("slurm_tasks", attr_scalar("tasks", -1))),
        "slurm_cores_per_task": int(attr_scalar("slurm_cores_per_task", attr_scalar("cores_per_task", -1))),
        "slurm_partition": str(_decode_attr(attr_scalar("slurm_partition", attr_scalar("partition", "")))),
    }


def _fmt_runtime(seconds: float) -> str:
    if not (seconds == seconds):  # NaN
        return "n/a"
    total = int(round(max(0.0, float(seconds))))
    hh = total // 3600
    mm = (total % 3600) // 60
    ss = total % 60
    return f"{hh:02d}:{mm:02d}:{ss:02d}"


def _solver_type_label(value: int) -> str:
    if value == 2:
        return "ML"
    if value == 1:
        return "Regular"
    return "Init"


def build_overlay_lines(meta, idx: int):
    if meta is None:
        return []

    step = int(meta["step_index"][idx])
    solver_type = _solver_type_label(int(meta["solver_type"][idx]))
    mass = float(meta["mass"][idx])
    drift = float(meta["drift"][idx])
    moved = float(meta["moved_this_step"][idx])
    min_w = float(meta["min_water"][idx])
    min_pos = float(meta["min_positive_water"][idx])
    max_w = float(meta["max_water"][idx])
    runtime = float(meta["runtime_seconds"][idx])

    return [
        f"saved_idx: {idx}",
        f"step: {step}",
        f"solver_type: {solver_type}",
        f"runtime: {_fmt_runtime(runtime)}",
        f"mass: {mass:.6g}",
        f"drift: {drift:.6g}",
        f"moved_this_step: {moved:.6g}",
        f"min_water: {min_w:.6g}",
        f"min_positive_water: {min_pos:.6g}",
        f"max_water: {max_w:.6g}",
        f"grid_width: {meta['grid_width']} (height={meta['grid_height']})",
        f"chunk_size: {meta['chunk_size']} (chunks_x={meta['chunks_x']}, chunks_z={meta['chunks_z']})",
        f"ranks_x: {meta['ranks_x']} (ranks_z={meta['ranks_z']})",
        f"io_mode: {meta['io_mode']}",
        f"hdf5_xfer_mode: {meta['hdf5_xfer_mode']}",
        f"sync_mode: {meta['sync_mode']}",
        f"slurm_nodes: {meta['slurm_nodes']}",
        f"slurm_tasks: {meta['slurm_tasks']}",
        f"slurm_cores_per_task: {meta['slurm_cores_per_task']}",
        f"partition: {meta['slurm_partition']}",
    ]


def draw_overlay_text(ImageDraw, ImageFont, image, lines):
    if not lines:
        return

    draw = ImageDraw.Draw(image, "RGBA")
    font_size = max(14, int(image.width * 0.011))
    font = None
    for candidate in ("DejaVuSans.ttf", "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"):
        try:
            font = ImageFont.truetype(candidate, font_size)
            break
        except Exception:
            continue
    if font is None:
        font = ImageFont.load_default()
    margin = max(10, int(image.width * 0.007))
    padding = max(8, int(font_size * 0.35))
    line_spacing = max(2, int(font_size * 0.18))

    widths = []
    heights = []
    for line in lines:
        try:
            left, top, right, bottom = draw.textbbox((0, 0), line, font=font)
            w = right - left
            h = bottom - top
        except Exception:
            w, h = draw.textsize(line, font=font)
        widths.append(w)
        heights.append(h)

    box_w = max(widths) + 2 * padding
    box_h = sum(heights) + max(0, len(lines) - 1) * line_spacing + 2 * padding
    x0 = image.width - box_w - margin
    y0 = margin
    x1 = x0 + box_w
    y1 = y0 + box_h
    
    if y1 > image.height - margin:
        return # not enough vertical space for overlay, skip it

    draw.rectangle((x0, y0, x1, y1), fill=(0, 0, 0, 165))
    y = y0 + padding
    for line, h in zip(lines, heights):
        draw.text((x0 + padding, y), line, fill=(255, 255, 255, 255), font=font)
        y += h + line_spacing


def main():
    args = parse_args()
    import_required_modules()

    import h5py
    import numpy as np
    from PIL import Image, ImageDraw, ImageFont

    if args.image_height <= 0:
        fail("--image-height must be > 0.")
    if args.threads <= 0:
        fail("--threads must be > 0.")
    if not os.path.exists(args.input_hdf5):
        fail(f"Input file does not exist: {args.input_hdf5}")

    step = args.step
    water_row_single = None
    n_saved = 0
    steps = None
    nx = 0
    slice_z = 0
    terrain_row = None
    lut = None

    with h5py.File(args.input_hdf5, "r") as h5f:
        if "terrain" not in h5f or "water" not in h5f:
            fail("Expected datasets 'terrain' and 'water' in trajectory file.")

        terrain = h5f["terrain"][:]  # (nz,nx)
        water_ds = h5f["water"]  # (nt,nz,nx)

        if terrain.ndim != 2:
            fail(f"Expected terrain to be 2D, got shape {terrain.shape}.")
        if water_ds.ndim != 3:
            fail(f"Expected water to be 3D, got shape {water_ds.shape}.")

        n_saved, nz, nx = water_ds.shape
        if n_saved == 0:
            fail("Water trajectory has zero snapshots.")
        if terrain.shape != (nz, nx):
            fail(f"terrain shape {terrain.shape} does not match water grid {(nz, nx)}.")

        slice_z = resolve_slice_z(args.slice_z, nz)
        terrain_row = terrain[slice_z, :].astype(np.float32)
        steps = h5f["step_index"][:] if "step_index" in h5f else None
        lut = build_ground_lut(np)
        overlay_meta = None if args.no_overlay_text else load_overlay_metadata(np, h5f, n_saved)

        if not args.all_steps:
            if step < 0:
                step = n_saved - 1
            if step >= n_saved:
                fail(f"Requested step {step} exceeds last saved index {n_saved - 1}.")
            water_row_single = water_ds[step, slice_z, :].astype(np.float32)

    if args.all_steps:
        out_dir = args.output_dir
        if not out_dir:
            base = os.path.splitext(os.path.basename(args.input_hdf5))[0]
            out_dir = os.path.join(
                os.path.dirname(args.input_hdf5),
                f"{base}_slice_z{slice_z:04d}_frames",
            )
        os.makedirs(out_dir, exist_ok=True)
        progress_interval = max(1, n_saved // 20)

        def maybe_log_progress(done_count: int):
            if done_count == 1 or done_count == n_saved or (done_count % progress_interval) == 0:
                pct = 100.0 * done_count / n_saved
                print(f"[render_slice] progress: {done_count}/{n_saved} ({pct:.1f}%)")

        def render_one(i: int, water_row):
            if steps is not None:
                name = f"frame_{i:06d}_step_{int(steps[i]):06d}.png"
            else:
                name = f"frame_{i:06d}.png"
            render_slice_frame(
                np=np,
                Image=Image,
                ImageDraw=ImageDraw,
                ImageFont=ImageFont,
                ground_lut=lut,
                terrain_row=terrain_row,
                water_row=water_row,
                image_height=args.image_height,
                height_min=args.height_min,
                out_path=os.path.join(out_dir, name),
                overlay_lines=build_overlay_lines(overlay_meta, i) if overlay_meta is not None else None,
            )

        if args.threads == 1:
            print(f"[render_slice] rendering {n_saved} frames with 1 thread...")
            with h5py.File(args.input_hdf5, "r") as h5f:
                water_ds = h5f["water"]
                for i in range(n_saved):
                    render_one(i, water_ds[i, slice_z, :].astype(np.float32))
                    maybe_log_progress(i + 1)
        else:
            task_queue = queue.Queue(maxsize=max(2, args.threads * 2))
            stop_event = threading.Event()
            errors = []
            errors_lock = threading.Lock()
            progress_state = {"done_count": 0}
            done_lock = threading.Lock()
            sentinel = object()

            print(f"[render_slice] rendering {n_saved} frames with {args.threads} threads...")

            def record_error(exc: Exception):
                with errors_lock:
                    if not errors:
                        errors.append(exc)
                stop_event.set()

            def reader_thread_fn():
                try:
                    with h5py.File(args.input_hdf5, "r") as h5f:
                        water_ds = h5f["water"]
                        for i in range(n_saved):
                            if stop_event.is_set():
                                break
                            row = water_ds[i, slice_z, :].astype(np.float32)
                            task_queue.put((i, row))
                except Exception as exc:
                    record_error(exc)
                finally:
                    for _ in range(args.threads):
                        task_queue.put(sentinel)

            def worker_thread_fn():
                while True:
                    item = task_queue.get()
                    try:
                        if item is sentinel:
                            return
                        i, row = item
                        if stop_event.is_set():
                            continue
                        render_one(i, row)
                        with done_lock:
                            progress_state["done_count"] += 1
                            done_count_local = progress_state["done_count"]
                        maybe_log_progress(done_count_local)
                    except Exception as exc:
                        record_error(exc)
                    finally:
                        task_queue.task_done()

            reader = threading.Thread(target=reader_thread_fn, name="slice-reader", daemon=True)
            workers = [
                threading.Thread(target=worker_thread_fn, name=f"slice-worker-{idx}", daemon=True)
                for idx in range(args.threads)
            ]

            reader.start()
            for worker in workers:
                worker.start()

            task_queue.join()
            reader.join()
            for worker in workers:
                worker.join()

            if errors:
                raise errors[0]

        print(
            f"Rendered {n_saved} side-view full-size frames "
            f"(width={nx}, height={args.image_height}, height_min={args.height_min}, "
            f"slice_z={slice_z}, threads={args.threads}) into {out_dir}"
        )
        return

    if args.output_image is None:
        fail("--output-image is required for single-step mode.")

    render_slice_frame(
        np=np,
        Image=Image,
        ImageDraw=ImageDraw,
        ImageFont=ImageFont,
        ground_lut=lut,
        terrain_row=terrain_row,
        water_row=water_row_single,
        image_height=args.image_height,
        height_min=args.height_min,
        out_path=args.output_image,
        overlay_lines=build_overlay_lines(overlay_meta, step) if overlay_meta is not None else None,
    )
    print(
        f"Rendered side-view full-size step {step} "
        f"(width={nx}, height={args.image_height}, height_min={args.height_min}, slice_z={slice_z}) -> {args.output_image}"
    )


if __name__ == "__main__":
    main()
