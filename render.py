#!/usr/bin/env python3

import argparse
import os
import queue
import sys
import threading

from numpy import inf

max_lines = inf

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Render terrain + water snapshots from solver trajectory HDF5."
    )
    parser.add_argument("--input-hdf5", required=True, help="Trajectory HDF5 file.")
    parser.add_argument(
        "--step",
        type=int,
        default=-1,
        help="Step index in saved timeline (default: last). Ignored when --all-steps is set.",
    )
    parser.add_argument("--all-steps", action="store_true", help="Render all saved snapshots.")
    parser.add_argument("--step-range", default=None, help="Range of steps to render (e.g. '0:10' for the first ten steps or '0:100:5' for steps 0 to 99 every 5 steps). Ignored when --all-steps is set.")
    parser.add_argument("--output-png", default=None, help="Output PNG path for single step.")
    parser.add_argument("--output-dir", default=None, help="Output folder for all steps rendering.")
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


def terrain_colormap(np, terrain):
    
    elevation_ground = 144
    end_of_green = 154
    end_of_brown = 185

    color_map_ground = np.zeros((256, 3), dtype=np.uint8)
    idx = np.arange(256)

    # Gray for low elevations
    mask = idx < elevation_ground
    color_map_ground[mask] = np.column_stack([idx[mask], idx[mask], idx[mask]])

    # Green gradient for low-mid elevations
    mask = (idx >= elevation_ground) & (idx < end_of_green)
    green_value = (
        128 + 127 * (idx[mask] - elevation_ground) / (end_of_green - elevation_ground)
    ).astype(np.uint8)
    color_map_ground[mask] = np.column_stack(
        [np.zeros_like(green_value), green_value, np.zeros_like(green_value)]
    )

    # Brown gradient for mid-high elevations
    mask = (idx >= end_of_green) & (idx < end_of_brown)
    brown_value = (
        255 * (idx[mask] - end_of_green) / (end_of_brown - end_of_green)
    ).astype(np.uint8)
    color_map_ground[mask] = np.column_stack(
        [brown_value, np.full_like(brown_value, 128), np.zeros_like(brown_value)]
    )

    # White gradient for high elevations
    mask = idx >= end_of_brown
    white_value = (
        200 + 55 * (idx[mask] - end_of_brown) / (256 - end_of_brown)
    ).astype(np.uint8)
    color_map_ground[mask] = np.column_stack([white_value, white_value, white_value])
    
    
    gh_clamped = np.clip(terrain, 0, 255).astype(np.uint8)
    
    return color_map_ground[gh_clamped]


def overlay_water(np, rgb, water):
    if not np.any(water > 0):
        return rgb

    out = rgb.astype(np.float32)
        
        
    shallow_r, shallow_g, shallow_b = 60.0, 160.0, 220.0
    deep_r, deep_g, deep_b = 2.0, 15.0, 120.0
    
    
    max_depth = 100.0  # adjust based on expected water height range
    
    min_alpha = 0.6
    min_omega = 0.3
    
    alpha = np.where(water > 0, min_alpha + (1-min_alpha) * (water / max_depth), 0.0)
    omega = np.clip(np.where(alpha < 0, min_omega - (1-min_omega) * alpha, 0.0), 0.0, 1.0)
    alpha = np.clip(alpha, 0, 1)
    
    water_color_r = np.clip((1 - alpha) * shallow_r + alpha * deep_r, 0, 255).astype(np.uint8)
    water_color_g = np.clip((1 - alpha) * shallow_g + alpha * deep_g, 0, 255).astype(np.uint8)
    water_color_b = np.clip((1 - alpha) * shallow_b + alpha * deep_b, 0, 255).astype(np.uint8)
    
    # blend water color with terrain color using alpha based on water height
    out[:, :, 0] = (out[:, :, 0] * (1.0 - alpha) + water_color_r * alpha) * (1.0 - omega) + omega * 255
    out[:, :, 1] = (out[:, :, 1] * (1.0 - alpha) + water_color_g * alpha) * (1.0 - omega) + omega * 0
    out[:, :, 2] = (out[:, :, 2] * (1.0 - alpha) + water_color_b * alpha) * (1.0 - omega) + omega * 0

    return np.clip(out, 0, 255).astype(np.uint8)


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
    font_size = max(8, int(image.width * 0.011))
    font = None
    for candidate in ("DejaVuSans.ttf", "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"):
        try:
            font = ImageFont.truetype(candidate, font_size)
            break
        except Exception:
            continue
    if font is None:
        font = ImageFont.load_default()
    margin = max(0, 1+int(image.width * 0.007))
    padding = max(0, 1+int(font_size * 0.35))
    line_spacing = max(0, int(font_size * 0.18))

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
        
    last_fitting_line = -1
    
    for i, w in enumerate(widths):
        h = heights[i]
        if w > 0.5 * image.width - 2 * margin - 2 * padding:
            #print(f"Line {i} ('{lines[i]}') width {w} exceeds image width; stopping overlay text at line {i-1}.")
            break
        if sum(heights[:i+1]) > image.height - 2 * margin - 2 * padding - i * line_spacing:
            #print(f"Line {i} ('{lines[i]}') height {h} exceeds image height; stopping overlay text at line {i-1}.")
            break
        last_fitting_line = i
        
    #print(f"Overlay text: {len(lines)} lines total, {last_fitting_line + 1} fit within image dimensions.")
    
    global max_lines
    
    last_fitting_line = min(last_fitting_line + 1, max_lines)
    if last_fitting_line < max_lines:
        max_lines = last_fitting_line

    box_w = max(widths[:last_fitting_line]) + 2 * padding
    box_h = sum(heights[:last_fitting_line]) + max(0, len(lines[:last_fitting_line]) - 1) * line_spacing + 2 * padding
    x0 = image.width - box_w - margin
    y0 = margin
    x1 = x0 + box_w
    y1 = y0 + box_h

    draw.rectangle((x0, y0, x1, y1), fill=(0, 0, 0, 175))
    y = y0 + padding
    lineid = 0
    for line, h in zip(lines, heights):
        if lineid >= last_fitting_line:
            break
        lineid += 1
        draw.text((x0 + padding, y), line, fill=(255, 255, 255, 255), font=font)
        y += h + line_spacing


def render_frame(np, Image, ImageDraw, ImageFont, terrain, water, output_path, overlay_lines=None):
    rgb = terrain_colormap(np, terrain)
    rgb = overlay_water(np, rgb, water)
    img = Image.fromarray(rgb, mode="RGB")
    if overlay_lines:
        draw_overlay_text(ImageDraw, ImageFont, img, overlay_lines)
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    img.save(output_path)


def main():
    args = parse_args()
    import_required_modules()

    import h5py
    import numpy as np
    from PIL import Image, ImageDraw, ImageFont

    if not os.path.exists(args.input_hdf5):
        fail(f"Input file does not exist: {args.input_hdf5}")
    if args.threads <= 0:
        fail("--threads must be > 0.")

    with h5py.File(args.input_hdf5, "r") as h5f:
        if "terrain" not in h5f or "water" not in h5f:
            fail("Expected datasets 'terrain' and 'water' in trajectory file.")

        terrain = h5f["terrain"][:]
        water_ds = h5f["water"]
        n_saved = water_ds.shape[0]
        if n_saved == 0:
            fail("Water trajectory has zero snapshots.")
        overlay_meta = None if args.no_overlay_text else load_overlay_metadata(np, h5f, n_saved)
        
        min_step = 0
        max_step = n_saved
        step_stride = 1
        if args.step_range and not args.all_steps:
            try:
                parts = args.step_range.split(":")
                if len(parts) == 2:
                    min_step = int(parts[0]) if parts[0] else 0
                    max_step = int(parts[1]) if parts[1] else n_saved
                elif len(parts) == 3:
                    min_step = int(parts[0]) if parts[0] else 0
                    max_step = int(parts[1]) if parts[1] else n_saved
                    step_stride = int(parts[2]) if parts[2] else 1
                else:
                    raise ValueError()
            except Exception:
                fail(f"Invalid --step-range format: {args.step_range}. Expected formats like '0:10' or '0:100:5'.")
            if min_step < 0 or max_step > n_saved or min_step >= max_step or step_stride <= 0:
                fail(f"Invalid --step-range values: {args.step_range}. Must satisfy 0 <= min < max <= {n_saved} and stride > 0.")
            print(f"Rendering steps in range [{min_step}:{max_step}:{step_stride}] from {n_saved} total saved steps.")
            steps_to_render = list(range(min_step, max_step, step_stride))
        elif args.all_steps:
            print(f"Rendering all {n_saved} saved steps.")
            steps_to_render = list(range(n_saved))
        else:
            steps_to_render = None  # will render single step based on --step

        if steps_to_render is not None:
            out_dir = args.output_dir
            if not out_dir:
                base = os.path.splitext(os.path.basename(args.input_hdf5))[0]
                out_dir = os.path.join(os.path.dirname(args.input_hdf5), f"{base}_frames")
            os.makedirs(out_dir, exist_ok=True)
            steps = h5f["step_index"][:] if "step_index" in h5f else None
            num_steps_to_render = len(steps_to_render)
            progress_interval = max(1, num_steps_to_render // 20)

            def maybe_log_progress(done_count: int):
                if done_count == 1 or done_count == num_steps_to_render or (done_count % progress_interval) == 0:
                    pct = 100.0 * done_count / num_steps_to_render
                    print(f"[render] progress: {done_count}/{num_steps_to_render} ({pct:.1f}%)")

            def render_one(i: int, water_frame):
                if steps is not None:
                    out_name = f"frame_{i:06d}_step_{int(steps[i]):06d}.png"
                else:
                    out_name = f"frame_{i:06d}.png"
                out = os.path.join(out_dir, out_name)
                overlay_lines = build_overlay_lines(overlay_meta, i) if overlay_meta is not None else None
                render_frame(np, Image, ImageDraw, ImageFont, terrain, water_frame, out, overlay_lines)

            if args.threads == 1:
                print(f"[render] rendering {num_steps_to_render} frames with 1 thread...")
                done_count = 0
                for i in steps_to_render:
                    render_one(i, water_ds[i].astype(np.float32))
                    done_count += 1
                    maybe_log_progress(done_count)
            else:
                print(f"[render] rendering {num_steps_to_render} frames with {args.threads} threads...")
                task_queue = queue.Queue(maxsize=max(2, args.threads * 2))
                stop_event = threading.Event()
                errors = []
                errors_lock = threading.Lock()
                progress_state = {"done_count": 0}
                done_lock = threading.Lock()
                sentinel = object()

                def record_error(exc: Exception):
                    with errors_lock:
                        if not errors:
                            errors.append(exc)
                    stop_event.set()

                def reader_thread_fn():
                    try:
                        for i in steps_to_render:
                            if stop_event.is_set():
                                break
                            frame = water_ds[i].astype(np.float32)
                            task_queue.put((i, frame))
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
                            i, frame = item
                            if stop_event.is_set():
                                continue
                            render_one(i, frame)
                            with done_lock:
                                progress_state["done_count"] += 1
                                done_count_local = progress_state["done_count"]
                            maybe_log_progress(done_count_local)
                        except Exception as exc:
                            record_error(exc)
                        finally:
                            task_queue.task_done()

                reader = threading.Thread(target=reader_thread_fn, name="render-reader", daemon=True)
                workers = [
                    threading.Thread(target=worker_thread_fn, name=f"render-worker-{idx}", daemon=True)
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
                f"Rendered {num_steps_to_render} full-size frames "
                f"(threads={args.threads}) into {out_dir}"
            )
            return

        if args.output_png is None:
            fail("--output-png is required when rendering a single step.")

        step = args.step
        if step < 0:
            step = n_saved - 1
        if step >= n_saved:
            fail(f"Requested step {step} exceeds last saved index {n_saved - 1}.")

        overlay_lines = build_overlay_lines(overlay_meta, step) if overlay_meta is not None else None
        render_frame(np, Image, ImageDraw, ImageFont, terrain, water_ds[step], args.output_png, overlay_lines)
        print(f"Rendered full-size step {step} -> {args.output_png}")


if __name__ == "__main__":
    main()
