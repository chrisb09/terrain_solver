

# We use pytorch to train a model

# The data is in a hdf5 file in ../../external_data/circle_r30_d300_s1000000_periodic_216x108_c23mm_1n_18t_1c__rank0_gather_none/world_trajectory.h5

# The data is fp32, and has shape (1000000, 108, 216) for the "water"
# and "terrain" is a (108, 216) array of fp32 that is the same for all time steps.

# We train a simple CNN to predict the "water" at the next time step given the "water" and "terrain" at the current time step. We use a simple architecture with a few convolutional layers.

# Specifically, we expect two inputs of size 3x3, one for the "water" and one for the "terrain", and we predict a single output value for the center cell of the "water" at the next time step.

# We train one 3x3 Convolutional layer for the "water" and one 3x3 Convolutional layer for the "terrain", and we also combine the water and terrain data using a 1x1 convolutional layer, before using another 3x3 convolutional layer on it. So we have 3 different intermediate outputs, which we combine using a fully connected layer to predict the final output. We use ReLU activations after each convolutional layer and after the fully connected layer.

# We train the model using mean squared error loss and the Adam optimizer. We train for a few epochs and print the training loss after each epoch.

# We also use a DataLoader to load the data in batches, and we shuffle the data at the beginning of each epoch.
# We also use a learning rate scheduler to reduce the learning rate if the training loss does not improve for a few epochs.

# We also use a validation set to evaluate the model after each epoch, and we print the validation loss as well. We use early stopping to stop training if the validation loss does not improve for a few epochs.

# We pad the input data by wrapping around the edges. The input data is of shape (108, 216), and we pad it to (110, 218) by adding one row/column on each side, where the new row/column is a copy of the opposite edge of the original data.
# We then extract 3x3 patches from the padded data to use as input to the model. For each cell in the original data, we take the 3x3 patch centered on that cell from the padded data. We predict the value for the center cell of the "water" at the next time step, so we use the 3x3 patch of the "water" and the 3x3 patch of the "terrain" as input to the model.

import argparse
import bisect
import json
import math
import os
import sys
import time

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import Dataset, DataLoader
import h5py
import numpy as np

# ---------------------------------------------------------------------------
# CLI arguments
# ---------------------------------------------------------------------------
#DEFAULT_DATA_PATH = "../../external_data/circle_r30_d300_s1000000_periodic_216x108_c23mm_1n_18t_1c__rank0_gather_none/world_trajectory.h5"
DEFAULT_DATA_PATH = "../../external_data/circle_r300_d300_s10000_periodic_2160x1080_devel_1n_96t_1c__rank0_gather_none/world_trajectory.h5"
PREPARED_RECORD_FLOATS = 20
PREPARED_RECORD_BYTES = PREPARED_RECORD_FLOATS * 4

def parse_args():
    p = argparse.ArgumentParser(description="Train WaterCNN")
    p.add_argument("--model", choices=["watercnn", "transformer_mlp"],
                   default="watercnn",
                   help="Model architecture to train")
    p.add_argument("--cuda-sdp", choices=["auto", "math"], default="math",
                   help="Scaled-dot-product attention backend on CUDA for transformer_mlp: "
                        "'math' disables flash/mem-efficient kernels for maximum compatibility; "
                        "'auto' leaves PyTorch defaults")
    p.add_argument("--data-path",     default=DEFAULT_DATA_PATH,
                   help="Path to the HDF5 data file")
    p.add_argument("--prepared-data-path", default=None,
                   help="Path to prepared binary data (.bin file or directory with metadata.json + batch .bin files). "
                        "If set, training reads prepared data instead of HDF5.")
    p.add_argument("--ignore-prepared-counts", action="store_true",
                   help="When using --prepared-data-path, treat each deduplicated record as weight=1 instead of using stored counts")
    # Data loading strategy
    p.add_argument("--cache-mode", choices=["stream", "cache", "window"],
                   default="stream",
                   help="How to load water data: "
                        "'stream'=lazy HDF5 reads per sample (low memory), "
                        "'cache'=load all time-steps into RAM at startup (fast), "
                        "'window'=load only --window-steps time-steps at a time "
                        "(middle ground; dataset is refreshed each epoch)")
    p.add_argument("--window-steps",  type=int, default=1000,
                   help="Number of consecutive time-steps to cache at once "
                        "(only used when --cache-mode=window)")
    p.add_argument("--max-steps",     type=int, default=None,
                   help="Only consider the first N time-steps of the dataset "
                        "(e.g. 50000 if the system becomes stale afterwards). "
                        "Defaults to using all available steps.")
    # Training hyper-parameters
    p.add_argument("--batch-size",    type=int,   default=4096)
    p.add_argument("--epochs",        type=int,   default=50)
    p.add_argument("--lr",            type=float, default=1e-3)
    p.add_argument("--val-ratio",     type=float, default=0.1)
    p.add_argument("--train-split",   type=float, default=None,
                   help="Train split percentage (0-100). If set, overrides --val-ratio. "
                        "Example: --train-split 90 means 90%% train / 10%% val.")
    p.add_argument("--patience",      type=int,   default=5,
                   help="Epochs without improvement before LR reduction / early stopping")
    p.add_argument("--num-workers",   type=int,   default=None,
                   help="DataLoader worker processes (use 0 with cache modes to avoid "
                        "HDF5 multiprocess issues). Defaults to 4 or auto-adjusted based on --num-threads.")
    p.add_argument("--num-threads",   type=int,   default=None,
                   help="Number of threads for PyTorch CPU operations. "
                        "Defaults to number of CPU cores available.")
    p.add_argument("--output",        default="best_model.pt",
                    help="Path to save the best model checkpoint. "
                        "Use '{model}' placeholder or keep default for automatic model suffix.")
    p.add_argument("--inference-output", default="best_model_jit.pt",
                    help="Path to save a TorchScript inference artifact for the best model. "
                        "Use '{model}' placeholder or keep default for automatic model suffix.")
    p.add_argument("--export-field-inference", action="store_true",
                   help="Export a TorchScript field-inference model (water/terrain tiles with halo -> NxN output)")
    p.add_argument("--field-inference-output", default="best_model_field_jit.pt",
                    help="Path to save the TorchScript field-inference artifact. "
                        "Use '{model}' placeholder or keep default for automatic model suffix.")
    p.add_argument("--export-field-iter", action="store_true",
                   help="Export a TorchScript iterative field model (t+N from 3x3 chunks)")
    p.add_argument("--field-iter-output", default="best_model_field_iter_jit.pt",
                    help="Path to save the TorchScript iterative field artifact. "
                        "Use '{model}' placeholder or keep default for automatic model suffix.")
    p.add_argument("--field-iter-steps", type=int, default=None,
                   help="Number of iterative steps for field-iter export (required if --export-field-iter)")
    return p.parse_args()

H, W = 108, 216


def _append_model_suffix(path: str, model_name: str) -> str:
    base, ext = os.path.splitext(path)
    if not ext:
        ext = ".pt"
    return f"{base}_{model_name}{ext}"


def resolve_model_output_path(path: str, model_name: str, default_path: str) -> str:
    """
    Resolve output path for artifacts:
      - If '{model}' appears, replace it with model_name.
      - If path is the unchanged default, append _{model_name} before extension.
      - Otherwise keep user-provided explicit path as-is.
    """
    if "{model}" in path:
        return path.replace("{model}", model_name)
    if path == default_path:
        return _append_model_suffix(path, model_name)
    return path


def load_checkpoint(checkpoint_path, model, optimizer=None, scheduler=None, device="cpu", expected_model_name=None):
    """
    Load a checkpoint if present.

    Supports both:
    - legacy files containing only a model state_dict
    - richer checkpoint dicts with training state for resume
    """
    if not os.path.exists(checkpoint_path):
        return {
            "loaded": False,
            "start_epoch": 1,
            "best_val_loss": float("inf"),
            "epochs_no_improve": 0,
        }

    checkpoint = torch.load(checkpoint_path, map_location=device)

    # Backward compatibility with older checkpoints that only saved weights.
    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        checkpoint_model_name = checkpoint.get("model_name")
        if expected_model_name is not None and checkpoint_model_name is not None:
            if checkpoint_model_name != expected_model_name:
                return {
                    "loaded": False,
                    "start_epoch": 1,
                    "best_val_loss": float("inf"),
                    "epochs_no_improve": 0,
                    "model_mismatch": True,
                    "checkpoint_model_name": checkpoint_model_name,
                }
        try:
            model.load_state_dict(checkpoint["model_state_dict"])
        except RuntimeError as error:
            return {
                "loaded": False,
                "start_epoch": 1,
                "best_val_loss": float("inf"),
                "epochs_no_improve": 0,
                "load_error": str(error),
            }
        if optimizer is not None and "optimizer_state_dict" in checkpoint:
            optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        if scheduler is not None and "scheduler_state_dict" in checkpoint:
            scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        start_epoch = checkpoint.get("epoch", 0) + 1
        best_val_loss = checkpoint.get("best_val_loss", float("inf"))
        epochs_no_improve = checkpoint.get("epochs_no_improve", 0)
        return {
            "loaded": True,
            "start_epoch": start_epoch,
            "best_val_loss": best_val_loss,
            "epochs_no_improve": epochs_no_improve,
            "legacy_weights_only": False,
        }

    try:
        model.load_state_dict(checkpoint)
    except RuntimeError as error:
        return {
            "loaded": False,
            "start_epoch": 1,
            "best_val_loss": float("inf"),
            "epochs_no_improve": 0,
            "load_error": str(error),
        }
    return {
        "loaded": True,
        "start_epoch": 1,
        "best_val_loss": float("inf"),
        "epochs_no_improve": 0,
        "legacy_weights_only": True,
    }


def export_inference_model(model: nn.Module, output_path, device):
    """Export a TorchScript artifact for inference."""
    was_training = model.training
    model.eval()

    restore_mha_fastpath = None
    restore_flash = None
    restore_mem_efficient = None
    restore_math = None

    # Transformer inference in some cluster stacks crashes inside
    # torch._transformer_encoder_layer_fwd with CUDA invalid configuration.
    # Export with conservative kernels so TorchScript uses the stable path.
    if isinstance(model, WaterTransformerMLP):
        if hasattr(torch.backends, "mha") and hasattr(torch.backends.mha, "get_fastpath_enabled") and hasattr(torch.backends.mha, "set_fastpath_enabled"):
            restore_mha_fastpath = torch.backends.mha.get_fastpath_enabled()
            torch.backends.mha.set_fastpath_enabled(False)
        if hasattr(torch.backends, "cuda") and hasattr(torch.backends.cuda, "enable_flash_sdp"):
            if hasattr(torch.backends.cuda, "flash_sdp_enabled"):
                restore_flash = torch.backends.cuda.flash_sdp_enabled()
            if hasattr(torch.backends.cuda, "mem_efficient_sdp_enabled"):
                restore_mem_efficient = torch.backends.cuda.mem_efficient_sdp_enabled()
            if hasattr(torch.backends.cuda, "math_sdp_enabled"):
                restore_math = torch.backends.cuda.math_sdp_enabled()
            torch.backends.cuda.enable_flash_sdp(False)
            torch.backends.cuda.enable_mem_efficient_sdp(False)
            torch.backends.cuda.enable_math_sdp(True)

    example_water = torch.zeros(1, 1, 3, 3, device=device)
    example_terrain = torch.zeros(1, 1, 3, 3, device=device)
    try:
        try:
            scripted_model = torch.jit.script(model)
        except Exception:
            scripted_model = torch.jit.trace(
                model,
                (example_water, example_terrain),
                check_trace=False,
            )
        scripted_model.save(output_path)
    finally:
        if restore_mha_fastpath is not None:
            torch.backends.mha.set_fastpath_enabled(restore_mha_fastpath)
        if restore_flash is not None:
            torch.backends.cuda.enable_flash_sdp(restore_flash)
        if restore_mem_efficient is not None:
            torch.backends.cuda.enable_mem_efficient_sdp(restore_mem_efficient)
        if restore_math is not None:
            torch.backends.cuda.enable_math_sdp(restore_math)

    if was_training:
        model.train()


def export_field_inference_model(model: nn.Module, output_path, device, example_hw=(6, 6)):
    """Export a TorchScript artifact for field inference using unfold."""
    was_training = model.training
    model.eval()
    field_model = WaterCNNField(model).to(device)
    h, w = example_hw
    example_water = torch.zeros(1, 1, h, w, device=device)
    example_terrain = torch.zeros(1, 1, h, w, device=device)
    try:
        scripted_model = torch.jit.script(field_model)
    except Exception:
        scripted_model = torch.jit.trace(
            field_model,
            (example_water, example_terrain),
            check_trace=False,
        )
    scripted_model.save(output_path)
    if was_training:
        model.train()


def export_field_iter_model(model: nn.Module, output_path, device, steps, example_hw=(12, 12)):
    """Export a TorchScript artifact for iterative field inference."""
    if steps is None or steps <= 0:
        raise ValueError("field-iter steps must be a positive integer")
    was_training = model.training
    model.eval()
    field_iter_model = WaterCNNFieldIter(model, steps=steps).to(device)
    h, w = example_hw
    example_water = torch.zeros(1, 1, h, w, device=device)
    example_terrain = torch.zeros(1, 1, h, w, device=device)
    try:
        scripted_model = torch.jit.script(field_iter_model)
    except Exception:
        scripted_model = torch.jit.trace(
            field_iter_model,
            (example_water, example_terrain),
            check_trace=False,
        )
    scripted_model.save(output_path)
    if was_training:
        model.train()


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------
def periodic_pad(field: np.ndarray) -> np.ndarray:
    """Pad a (H, W) field to (H+2, W+2) with periodic (wrap-around) boundary."""
    # field shape: (H, W)
    padded = np.pad(field, pad_width=1, mode="wrap")
    return padded  # (H+2, W+2)


def extract_patches(field_padded: np.ndarray) -> np.ndarray:
    """
    Extract all 3x3 patches from a periodically-padded (H+2, W+2) field.
    Returns array of shape (H*W, 1, 3, 3).
    """
    h, w = field_padded.shape[0] - 2, field_padded.shape[1] - 2
    patches = np.lib.stride_tricks.sliding_window_view(field_padded, (3, 3))
    # patches: (H, W, 3, 3)
    patches = patches.reshape(h * w, 1, 3, 3).astype(np.float32)
    return patches


class WaterDataset(Dataset):
    """
    Three cache modes controlled by `cache_mode`:

    'stream'  – lazy per-sample HDF5 reads. Lowest memory, highest I/O cost.
    'cache'   – load all `indices` time-steps into RAM at construction.
                Fastest for repeated epochs; requires enough memory.
    'window'  – load `window_steps` randomly chosen time-steps into RAM.
                Call `refresh()` between epochs to load a fresh window.
                Good middle-ground when full data fits if subsampled.
    """

    def __init__(self, data_path: str, terrain: np.ndarray, indices: np.ndarray,
                 cache_mode: str = "stream", window_steps: int = 1000):
        self.data_path   = data_path
        self.indices     = indices
        self.cache_mode  = cache_mode
        self.window_steps = window_steps
        self.h, self.w   = terrain.shape
        self.n_cells     = self.h * self.w

        self.terrain_patches = torch.from_numpy(
            extract_patches(periodic_pad(terrain))
        )  # (H*W, 1, 3, 3) – constant across time

        # Internal state for stream mode
        self._h5_file = None
        # Internal state for cache / window modes
        self._cached_water: np.ndarray | None = None  # (N_steps, H, W)
        self._active_indices: np.ndarray | None = None  # subset of indices in cache
        self._cached_water_patches: dict[int, torch.Tensor] = {}
        self._cached_targets: dict[int, torch.Tensor] = {}

        if cache_mode == "cache":
            self._load_into_cache(indices)
        elif cache_mode == "window":
            self.refresh()

    # -- Cache helpers -------------------------------------------------------

    def _load_into_cache(self, idx_subset: np.ndarray):
        """Load the given time-step indices (and t+1) into self._cached_water."""
        # We need both t and t+1 for each index
        print(f"Loading {len(idx_subset)} time-steps into cache …")
        needed = np.union1d(idx_subset, idx_subset + 1)
        needed.sort()
        print(f"  → actually loading {len(needed)} unique time-steps (including t+1) …")
        start_load_time = time.time()
        with h5py.File(self.data_path, "r") as f:
            self._cached_water = f["water"][needed]  # contiguous read is fast
        # Map original indices → row index inside _cached_water
        print(f"  → loaded in {time.time() - start_load_time:.1f}s")
        self._idx_to_row = {int(t): i for i, t in enumerate(needed)}
        self._active_indices = idx_subset
        self._build_cached_tensors(needed)

    def _build_cached_tensors(self, needed: np.ndarray):
        """Precompute per-time-step patches and targets once for cache/window modes."""
        self._cached_water_patches = {}
        self._cached_targets = {}
        for timestep in needed:
            row = self._idx_to_row[int(timestep)]
            water = self._cached_water[row]
            water_patches = extract_patches(periodic_pad(water))
            self._cached_water_patches[int(timestep)] = torch.from_numpy(water_patches)
            self._cached_targets[int(timestep)] = torch.from_numpy(
                water.reshape(self.n_cells).astype(np.float32, copy=False)
            )

    def refresh(self):
        """
        Window mode: pick a random subset of `window_steps` time-steps and
        load them into RAM.  Call this at the start of each epoch.
        """
        if self.cache_mode != "window":
            return
        chosen = np.random.choice(
            self.indices,
            size=min(self.window_steps, len(self.indices)),
            replace=False,
        )
        chosen.sort()
        self._load_into_cache(chosen)

    # -- HDF5 lazy opener (stream mode only) --------------------------------

    def _get_h5_file(self):
        if self._h5_file is None:
            self._h5_file = h5py.File(self.data_path, "r")
        return self._h5_file

    # -- Dataset interface ---------------------------------------------------

    def __len__(self):
        if self.cache_mode in ("cache", "window"):
            return len(self._active_indices) * self.n_cells
        return len(self.indices) * self.n_cells

    def __getitem__(self, idx):
        if self.cache_mode in ("cache", "window"):
            active = self._active_indices
        else:
            active = self.indices

        t_idx, cell = divmod(idx, self.n_cells)
        t = int(active[t_idx])

        if self.cache_mode in ("cache", "window"):
            x_water = self._cached_water_patches[t][cell]
            y = self._cached_targets[t + 1][cell]
        else:
            f = self._get_h5_file()
            water_t      = f["water"][t]
            water_t_next = f["water"][t + 1]

            water_padded  = periodic_pad(water_t)
            water_patches = extract_patches(water_padded)  # (H*W, 1, 3, 3)
            x_water = torch.from_numpy(water_patches[cell])  # (1, 3, 3)
            y = torch.tensor(water_t_next[cell // self.w, cell % self.w], dtype=torch.float32)

        x_terrain = self.terrain_patches[cell]                         # (1, 3, 3)
        return x_water, x_terrain, y

    def __del__(self):
        if self._h5_file is not None:
            self._h5_file.close()


def _resolve_prepared_files(prepared_data_path: str):
    metadata = None
    if os.path.isdir(prepared_data_path):
        metadata_path = os.path.join(prepared_data_path, "metadata.json")
        if os.path.exists(metadata_path):
            with open(metadata_path, "r", encoding="utf-8") as mf:
                metadata = json.load(mf)

        if metadata is not None and "batching" in metadata and "files" in metadata["batching"]:
            batch_files = metadata["batching"]["files"]
            bin_files = [os.path.join(prepared_data_path, name) for name in batch_files]
        else:
            bin_files = [
                os.path.join(prepared_data_path, name)
                for name in sorted(os.listdir(prepared_data_path))
                if name.endswith(".bin")
            ]
    else:
        bin_files = [prepared_data_path]
        metadata_path = os.path.join(os.path.dirname(prepared_data_path), "metadata.json")
        if os.path.exists(metadata_path):
            with open(metadata_path, "r", encoding="utf-8") as mf:
                metadata = json.load(mf)

    if not bin_files:
        raise ValueError(f"No .bin files found for prepared data path: {prepared_data_path}")

    for path in bin_files:
        if not os.path.exists(path):
            raise FileNotFoundError(f"Prepared data file not found: {path}")
        if os.path.getsize(path) % PREPARED_RECORD_BYTES != 0:
            raise ValueError(
                f"Prepared data file size is not a multiple of {PREPARED_RECORD_BYTES} bytes: {path}"
            )
    return bin_files, metadata


class PreparedPairDataset(Dataset):
    """
    Dataset for prepared deduplicated binary pairs.
    Record layout per sample (80 bytes total):
      - 19 float32 values: 9x water patch + 9x terrain patch + 1x output
      - 1 int32 count (stored as raw 32-bit value in final float32 slot)
    """

    def __init__(self, prepared_data_path: str, use_counts: bool = True):
        self.prepared_data_path = prepared_data_path
        self.use_counts = use_counts
        self.bin_files, self.metadata = _resolve_prepared_files(prepared_data_path)

        self._arrays = []
        self._lengths = []
        self._offsets = [0]
        for path in self.bin_files:
            num_records = os.path.getsize(path) // PREPARED_RECORD_BYTES
            arr = np.memmap(path, dtype="<f4", mode="r", shape=(num_records, PREPARED_RECORD_FLOATS))
            self._arrays.append(arr)
            self._lengths.append(num_records)
            self._offsets.append(self._offsets[-1] + num_records)

        self.n_records = self._offsets[-1]
        self.h = None
        self.w = None
        if self.metadata is not None:
            source = self.metadata.get("source", {})
            self.h = source.get("h")
            self.w = source.get("w")

    def __len__(self):
        return self.n_records

    def __getitem__(self, idx):
        file_idx = bisect.bisect_right(self._offsets, idx) - 1
        local_idx = idx - self._offsets[file_idx]
        row = self._arrays[file_idx][local_idx]

        x_water = torch.from_numpy(np.asarray(row[0:9], dtype=np.float32).reshape(1, 3, 3))
        x_terrain = torch.from_numpy(np.asarray(row[9:18], dtype=np.float32).reshape(1, 3, 3))
        y = torch.tensor(float(row[18]), dtype=torch.float32)

        if self.use_counts:
            count_u32 = np.asarray(row[19:20], dtype=np.float32).view(np.uint32)[0]
            count_i32 = np.int32(count_u32)
            weight = torch.tensor(float(max(int(count_i32), 1)), dtype=torch.float32)
        else:
            weight = torch.tensor(1.0, dtype=torch.float32)

        return x_water, x_terrain, y, weight


class IndexSubsetDataset(Dataset):
    def __init__(self, base_dataset: Dataset, indices: np.ndarray):
        self.base_dataset = base_dataset
        self.indices = np.asarray(indices, dtype=np.int64)

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        return self.base_dataset[int(self.indices[idx])]


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------
class WaterCNN(nn.Module):
    """
    Architecture (as specified):
      branch_water  : Conv2d(1→8, 3×3)  → ReLU    output: (8, 1, 1)
      branch_terrain: Conv2d(1→8, 3×3)  → ReLU    output: (8, 1, 1)
      branch_combined:
        concat(water_in, terrain_in) → Conv2d(2→8, 1×1) → ReLU  → (8, 3, 3)
        → Conv2d(8→16, 3×3) → ReLU                               → (16,1,1)
      fc: Linear(8 + 8 + 16, 16) → ReLU → Linear(16, 1)
    """

    def __init__(self):
        super().__init__()
        # Individual branches
        self.water_conv   = nn.Conv2d(1,  8,  kernel_size=3, padding=0)
        self.terrain_conv = nn.Conv2d(1,  8,  kernel_size=3, padding=0)
        # Combined branch
        self.combine_1x1  = nn.Conv2d(2,  8,  kernel_size=1, padding=0)
        self.combine_3x3  = nn.Conv2d(8,  16, kernel_size=3, padding=0)
        # Head
        self.fc = nn.Sequential(
            nn.Linear(8 + 8 + 16, 16),
            nn.ReLU(),
            nn.Linear(16, 1),
        )
        self.relu = nn.ReLU()

    def forward(self, x_water, x_terrain):
        # x_water, x_terrain: (B, 1, 3, 3)
        out_water   = self.relu(self.water_conv(x_water))          # (B, 8,  1, 1)
        out_terrain = self.relu(self.terrain_conv(x_terrain))      # (B, 8,  1, 1)

        combined = torch.cat([x_water, x_terrain], dim=1)          # (B, 2,  3, 3)
        out_combined = self.relu(self.combine_1x1(combined))       # (B, 8,  3, 3)
        out_combined = self.relu(self.combine_3x3(out_combined))   # (B, 16, 1, 1)

        flat = torch.cat([
            out_water.flatten(1),
            out_terrain.flatten(1),
            out_combined.flatten(1),
        ], dim=1)                                                   # (B, 32)

        return self.fc(flat).squeeze(1)                            # (B,)


class WaterTransformerMLP(nn.Module):
    """
    Alternative architecture for 3x3 local forecasting.

    Treats each 3x3 cell as a token with 2 features (water, terrain), processes
    tokens with a Transformer encoder, fuses with a global MLP branch, and
    regresses a scalar output.
    """

    def __init__(self, d_model: int = 64, nhead: int = 4, num_layers: int = 3, ff_dim: int = 256, dropout: float = 0.1):
        super().__init__()
        self.input_proj = nn.Linear(2, d_model)
        self.pos_embed = nn.Parameter(torch.zeros(1, 9, d_model))
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=ff_dim,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=False,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        self.global_mlp = nn.Sequential(
            nn.Linear(18, 128),
            nn.GELU(),
            nn.Linear(128, 64),
            nn.GELU(),
        )

        self.head = nn.Sequential(
            nn.Linear(d_model + 64, 128),
            nn.GELU(),
            nn.Linear(128, 64),
            nn.GELU(),
            nn.Linear(64, 1),
        )

    def forward(self, x_water: torch.Tensor, x_terrain: torch.Tensor) -> torch.Tensor:
        # x_water, x_terrain: (B, 1, 3, 3)
        b = x_water.shape[0]
        tokens = torch.stack([x_water.squeeze(1), x_terrain.squeeze(1)], dim=-1)  # (B, 3, 3, 2)
        tokens = tokens.view(b, 9, 2)
        token_features = self.input_proj(tokens) + self.pos_embed
        token_features = self.encoder(token_features)
        pooled = token_features.mean(dim=1)

        global_features = torch.cat([x_water.flatten(1), x_terrain.flatten(1)], dim=1)
        global_features = self.global_mlp(global_features)

        fused = torch.cat([pooled, global_features], dim=1)
        return self.head(fused).squeeze(1)


def build_model(model_name: str) -> nn.Module:
    if model_name == "watercnn":
        return WaterCNN()
    if model_name == "transformer_mlp":
        return WaterTransformerMLP()
    raise ValueError(f"Unknown model architecture: {model_name}")


class WaterCNNField(nn.Module):
    """
    Field inference wrapper for WaterCNN.

    Expects water/terrain tiles with halo (B, 1, H, W) where H=W=N+2.
    Returns (B, 1, H-2, W-2) so each interior position is inferred.
    """
    def __init__(self, base_model: nn.Module):
        super().__init__()
        self.base_model = base_model

    def forward(self, water: torch.Tensor, terrain: torch.Tensor) -> torch.Tensor:
        if water.ndim != 4 or terrain.ndim != 4:
            raise ValueError("Expected water/terrain tensors with shape (B, 1, H, W)")
        if water.shape != terrain.shape:
            raise ValueError("Water and terrain tensors must have the same shape")
        if water.shape[1] != 1:
            raise ValueError("Expected single-channel inputs (C=1)")

        b, _, h, w = water.shape
        if h < 3 or w < 3:
            raise ValueError("Input height/width must be at least 3")

        # Unfold into 3x3 patches: (B, 9, L)
        water_patches = F.unfold(water, kernel_size=3, stride=1, padding=0)
        terrain_patches = F.unfold(terrain, kernel_size=3, stride=1, padding=0)

        # Reshape to (B*L, 1, 3, 3)
        l = water_patches.shape[-1]
        water_patches = water_patches.transpose(1, 2).contiguous().view(b * l, 1, 3, 3)
        terrain_patches = terrain_patches.transpose(1, 2).contiguous().view(b * l, 1, 3, 3)

        preds = self.base_model(water_patches, terrain_patches)  # (B*L,)
        preds = preds.view(b, 1, h - 2, w - 2)
        return preds


class WaterCNNFieldIter(nn.Module):
    """
    Iterative field inference wrapper for WaterCNN.

    Expects water/terrain tiles with halo (B, 1, H, W). Each iteration shrinks
    the field by 2 in each dimension. After `steps`, output is (H-2*steps, W-2*steps).
    """
    def __init__(self, base_model: nn.Module, steps: int):
        super().__init__()
        if steps <= 0:
            raise ValueError("steps must be a positive integer")
        self.base_model = base_model
        self.steps = int(steps)
        self.field_model = WaterCNNField(base_model)

    def forward(self, water: torch.Tensor, terrain: torch.Tensor) -> torch.Tensor:
        if water.ndim != 4 or terrain.ndim != 4:
            raise ValueError("Expected water/terrain tensors with shape (B, 1, H, W)")
        if water.shape != terrain.shape:
            raise ValueError("Water and terrain tensors must have the same shape")
        if water.shape[1] != 1:
            raise ValueError("Expected single-channel inputs (C=1)")

        b, _, h, w = water.shape
        max_steps = (min(h, w) - 1) // 2
        if self.steps > max_steps:
            raise ValueError(f"steps={self.steps} is too large for input size {h}x{w}")

        cur_water = water
        cur_terrain = terrain
        for _ in range(self.steps):
            # Each step predicts the next field and shrinks by 2.
            cur_water = self.field_model(cur_water, cur_terrain)
            cur_terrain = cur_terrain[:, :, 1:-1, 1:-1]
        return cur_water


# ---------------------------------------------------------------------------
# Training helpers
# ---------------------------------------------------------------------------
def run_epoch(model, loader, criterion, device, optimizer=None, distributed=False, is_main=False, epoch=None, total_epochs=None):
    training = optimizer is not None
    model.train() if training else model.eval()
    total_loss = torch.tensor(0.0, device=device)
    total_samples = torch.tensor(0.0, device=device)
    
    phase = "Train" if training else "Val"
    batch_count = 0
    log_interval = max(1, len(loader) // 10)  # Log ~10 times per epoch
    phase_start = time.time()

    ctx = torch.enable_grad() if training else torch.no_grad()
    with ctx:
        for batch_idx, batch in enumerate(loader):
            if len(batch) == 4:
                x_w, x_s, y, sample_weight = batch
                sample_weight = sample_weight.to(device)
            else:
                x_w, x_s, y = batch
                sample_weight = None

            x_w, x_s, y = x_w.to(device), x_s.to(device), y.to(device)
            pred = model(x_w, x_s)
            if sample_weight is None:
                loss = criterion(pred, y)
                weighted_loss_sum = loss.detach() * y.size(0)
                sample_count = torch.tensor(float(y.size(0)), device=device)
            else:
                per_sample_loss = F.mse_loss(pred, y, reduction="none")
                weighted_loss_sum = (per_sample_loss * sample_weight).sum()
                sample_count = sample_weight.sum().clamp_min(1.0)
                loss = weighted_loss_sum / sample_count

            if training:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            total_loss    += weighted_loss_sum.detach()
            total_samples += sample_count.detach()
            batch_count += 1
            
            # Log progress
            if batch_count % log_interval == 0:
                elapsed = time.time() - phase_start
                samples_for_rate = total_samples.detach().clone()
                if distributed:
                    dist.all_reduce(samples_for_rate, op=dist.ReduceOp.SUM)
                if is_main:
                    rate = samples_for_rate.item() / elapsed
                    if epoch is not None:
                        print(f"  [{phase}] Epoch {epoch}/{total_epochs} | Batch {batch_idx+1}/{len(loader)} | "
                              f"Samples: {samples_for_rate.item():.0f} | Loss: {(total_loss/total_samples).item():.6f} | "
                              f"Rate: {rate:.0f} samples/s", flush=True)
                    else:
                        print(f"  [{phase}] Batch {batch_idx+1}/{len(loader)} | "
                              f"Samples: {samples_for_rate.item():.0f} | Loss: {(total_loss/total_samples).item():.6f} | "
                              f"Rate: {rate:.0f} samples/s", flush=True)

    # Aggregate across all DDP ranks so every process sees the same loss
    if distributed:
        dist.all_reduce(total_loss,    op=dist.ReduceOp.SUM)
        dist.all_reduce(total_samples, op=dist.ReduceOp.SUM)

    return (total_loss / total_samples).item()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    args = parse_args()

    args.output = resolve_model_output_path(args.output, args.model, "best_model.pt")
    args.inference_output = resolve_model_output_path(args.inference_output, args.model, "best_model_jit.pt")
    args.field_inference_output = resolve_model_output_path(
        args.field_inference_output, args.model, "best_model_field_jit.pt"
    )
    args.field_iter_output = resolve_model_output_path(
        args.field_iter_output, args.model, "best_model_field_iter_jit.pt"
    )

    if args.train_split is not None:
        if not (0.0 < args.train_split < 100.0):
            raise ValueError("--train-split must be between 0 and 100 (exclusive)")
        effective_val_ratio = 1.0 - (args.train_split / 100.0)
    else:
        effective_val_ratio = args.val_ratio

    if not (0.0 < effective_val_ratio < 1.0):
        raise ValueError("Validation ratio must be between 0 and 1 (exclusive)")

    # -- DDP initialisation --------------------------------------------------
    # torchrun sets RANK / LOCAL_RANK / WORLD_SIZE automatically.
    # When running without torchrun these env vars are absent → single-process.
    # For single-node CPU training, disable DDP to avoid Gloo TCP issues.
    distributed = "RANK" in os.environ
    if distributed:
        rank       = int(os.environ["RANK"])
        local_rank = int(os.environ["LOCAL_RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
        # Check if this is single-node CPU training (no GPUs + only local ranks on one node)
        is_single_node_cpu = not torch.cuda.is_available()
        if is_single_node_cpu:
            # All ranks: disable DDP; only rank 0 continues, others exit cleanly.
            if rank == 0:
                print("Single-node CPU training detected. Disabling DDP to avoid Gloo TCP issues.", flush=True)
            else:
                sys.exit(0)
            distributed = False
            rank = local_rank = 0
            world_size = 1
        else:
            backend    = "nccl" if torch.cuda.is_available() else "gloo"
            dist.init_process_group(backend=backend)
    else:
        rank = local_rank = 0
        world_size = 1

    # Device: each process owns one GPU when available, otherwise CPU
    if torch.cuda.is_available():
        device = torch.device(f"cuda:{local_rank}")
        torch.cuda.set_device(device)
    else:
        device = torch.device("cpu")

    # -- CUDA attention backend guard --------------------------------------
    # Some cluster GPU/software stacks hit "CUDA error: invalid configuration
    # argument" inside scaled_dot_product_attention. For transformer_mlp we
    # default to math SDP backend on CUDA for stability.
    if torch.cuda.is_available() and args.model == "transformer_mlp":
        if args.cuda_sdp == "math":
            if hasattr(torch.backends, "cuda") and hasattr(torch.backends.cuda, "enable_flash_sdp"):
                torch.backends.cuda.enable_flash_sdp(False)
                torch.backends.cuda.enable_mem_efficient_sdp(False)
                torch.backends.cuda.enable_math_sdp(True)
            if rank == 0:
                print("CUDA SDP backend: math (flash/mem-efficient disabled for compatibility)", flush=True)
        elif rank == 0:
            print("CUDA SDP backend: auto (PyTorch default kernels)", flush=True)

    # -- CPU thread configuration -----------------------------------------------
    if args.num_threads is not None:
        num_threads = args.num_threads
    elif torch.cuda.is_available():
        # Keep one host thread per GPU rank unless explicitly overridden.
        num_threads = 1
    else:
        num_threads = os.cpu_count() or 1
    torch.set_num_threads(num_threads)
    
    # Set num_workers intelligently if not specified
    if args.num_workers is None:
        # Use 0 workers for cache/window modes (data already in RAM), 4 for stream mode
        num_workers = 0 if args.cache_mode != "stream" else min(4, max(1, num_threads // 4))
    else:
        num_workers = args.num_workers

    is_main = (rank == 0)  # only rank 0 prints / saves checkpoints
    if is_main:
        print(f"Distributed: {distributed}  world_size: {world_size}  device: {device}", flush=True)
        print(f"Model architecture: {args.model}", flush=True)
        print(f"Train/Val split: {100.0 * (1.0 - effective_val_ratio):.1f}% / {100.0 * effective_val_ratio:.1f}%", flush=True)
        print(f"CPU threads: {num_threads}  DataLoader workers: {num_workers}", flush=True)
        print(f"Cache mode: {args.cache_mode}"
              + (f"  window_steps: {args.window_steps}" if args.cache_mode == "window" else ""), flush=True)

    data_h = None
    data_w = None

    if args.prepared_data_path is None:
        # -- Load metadata (all ranks read terrain; it is tiny) --------------
        if is_main:
            print("Loading metadata …", flush=True)
        with h5py.File(args.data_path, "r") as f:
            if is_main:
                print("Datasets in HDF5 file:", list(f.keys()), flush=True)
            water_shape = f["water"].shape
            terrain = f["terrain"][:]

        data_h, data_w = water_shape[1], water_shape[2]
        if terrain.shape != (data_h, data_w):
            raise ValueError(
                f"Spatial shape mismatch: water has {(data_h, data_w)} but terrain has {terrain.shape}"
            )

        if is_main:
            print(f"  Water shape: {water_shape}", flush=True)
            print(f"  Terrain shape: {terrain.shape}", flush=True)

        T = water_shape[0]
        if args.max_steps is not None:
            T = min(T, args.max_steps + 1)
        all_indices = np.arange(T - 1)
        n_val = max(1, int(len(all_indices) * effective_val_ratio))
        n_train = len(all_indices) - n_val
        train_idx = all_indices[:n_train]
        val_idx = all_indices[n_train:]
    else:
        if is_main:
            print(f"Loading prepared binary data from: {args.prepared_data_path}", flush=True)
        full_prepared_ds = PreparedPairDataset(
            args.prepared_data_path,
            use_counts=not args.ignore_prepared_counts,
        )
        n_total = len(full_prepared_ds)
        n_val = max(1, int(n_total * effective_val_ratio))
        n_train = n_total - n_val
        all_indices = np.arange(n_total, dtype=np.int64)
        train_idx = all_indices[:n_train]
        val_idx = all_indices[n_train:]

        if full_prepared_ds.h is not None and full_prepared_ds.w is not None:
            data_h = full_prepared_ds.h
            data_w = full_prepared_ds.w

        if is_main:
            print(f"  Prepared unique records: {n_total}", flush=True)
            if full_prepared_ds.metadata is not None and "stats" in full_prepared_ds.metadata:
                stats = full_prepared_ds.metadata["stats"]
                raw_pairs = stats.get("raw_pairs")
                if raw_pairs is not None:
                    print(f"  Raw pair occurrences (sum counts): {raw_pairs}", flush=True)
    
    # Validate distributed setup: need enough indices per rank
    if distributed:
        indices_per_rank = len(train_idx) // world_size
        if indices_per_rank == 0:
            if is_main:
                print(f"WARNING: Only {len(train_idx)} training steps for {world_size} ranks. "
                      f"Falling back to single-process training.", flush=True)
            # Destroy old process group and run single-process
            dist.destroy_process_group()
            distributed = False
            rank = local_rank = 0
            world_size = 1
    
    if is_main:
        if args.prepared_data_path is None:
            print(f"  Train steps: {len(train_idx)}, Val steps: {len(val_idx)}", flush=True)
        else:
            print(f"  Train records: {len(train_idx)}, Val records: {len(val_idx)}", flush=True)

    # -- Datasets ------------------------------------------------------------
    # Split indices across ranks so each process only loads its own slice.
    # This keeps memory usage constant regardless of world_size.
    if distributed and world_size > 1:
        train_idx_local = train_idx[rank::world_size]
        val_idx_local   = val_idx[rank::world_size]
    else:
        train_idx_local = train_idx
        val_idx_local   = val_idx

    if args.prepared_data_path is None:
        train_ds = WaterDataset(
            args.data_path, terrain, train_idx_local,
            cache_mode=args.cache_mode, window_steps=args.window_steps
        )
        val_ds = WaterDataset(
            args.data_path, terrain, val_idx_local,
            cache_mode=args.cache_mode, window_steps=args.window_steps
        )
    else:
        train_ds = IndexSubsetDataset(full_prepared_ds, train_idx_local)
        val_ds = IndexSubsetDataset(full_prepared_ds, val_idx_local)

    # -- DataLoaders ---------------------------------------------------------
    # With per-rank index splitting above, we no longer need DistributedSampler
    # (which would re-split an already-split dataset). Use a plain DataLoader
    # with shuffle on all ranks.
    train_loader = DataLoader(train_ds, batch_size=args.batch_size,
                              shuffle=True,
                              num_workers=num_workers, pin_memory=True)
    val_loader   = DataLoader(val_ds,   batch_size=args.batch_size,
                              shuffle=False,
                              num_workers=num_workers, pin_memory=True)

    # -- Model / optimizer ---------------------------------------------------
    model     = build_model(args.model).to(device)
    if distributed and world_size > 1:
        model = DDP(model, device_ids=[local_rank] if torch.cuda.is_available() else None)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    scheduler_patience = args.patience
    if args.cache_mode == "window" and args.window_steps > 0:
        window_scale = math.sqrt(len(train_idx) / args.window_steps)
        window_scale = max(1, int(round(window_scale)))
        scheduler_patience = max(1, args.patience * window_scale)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=scheduler_patience
    )
    
    # We also want to print how many parameters the model has (weights and biases), and how much memory it would take if stored in fp32.
    num_params = sum(p.numel() for p in model.parameters())
    param_size_kb = num_params * 4 / (1024 ** 1)
    if is_main:
        print(f"Model has {num_params} parameters ({param_size_kb:.2f} KB in fp32)", flush=True)
        print(f"Parameter breakdown:", flush=True)
        for name, param in model.named_parameters():
            count = param.numel()
            size_kb = count * 4 / 1024
            print(f"  {name}: {count} params ({size_kb:.2f} KB)", flush=True)

    # -- Resume from checkpoint if available --------------------------------
    model_to_save = model.module if distributed and world_size > 1 else model
    resume_state = load_checkpoint(
        args.output,
        model_to_save,
        optimizer=optimizer,
        scheduler=scheduler,
        device=device,
        expected_model_name=args.model,
    )
    start_epoch = resume_state["start_epoch"]
    best_val_loss = resume_state["best_val_loss"]
    epochs_no_improve = resume_state["epochs_no_improve"]
    final_epoch = args.epochs

    # -- Training loop -------------------------------------------------------
    if is_main:
        if resume_state["loaded"]:
            if resume_state.get("legacy_weights_only"):
                print(f"Loaded existing model weights from {args.output}; continuing training from epoch 1.", flush=True)
                start_epoch = 1
            else:
                print(f"Resumed training from {args.output} at epoch {start_epoch}.", flush=True)
                print(f"  Restored best val loss: {best_val_loss:.6f}", flush=True)
                if start_epoch > final_epoch:
                    print(f"  Target epochs already reached ({start_epoch - 1}/{final_epoch}).", flush=True)
        elif resume_state.get("model_mismatch"):
            print(
                f"Checkpoint model mismatch: requested '{args.model}' but checkpoint was "
                f"'{resume_state.get('checkpoint_model_name')}'. Starting fresh training.",
                flush=True,
            )
        elif resume_state.get("load_error"):
            print(
                f"Checkpoint at {args.output} could not be loaded for model '{args.model}' "
                f"({resume_state['load_error']}). Starting fresh training.",
                flush=True,
            )
        print("\nStarting training …", flush=True)
        if data_h is not None and data_w is not None:
            print(f"  Data resolution: {data_h}x{data_w}", flush=True)
        if args.prepared_data_path is None:
            print(
                f"  Train steps per epoch: {len(train_idx)} (global), {len(train_idx_local)} (per-rank)",
                flush=True,
            )
            print(
                f"  Val steps per epoch: {len(val_idx)} (global), {len(val_idx_local)} (per-rank)",
                flush=True,
            )
        else:
            print(
                f"  Train records per epoch: {len(train_idx)} (global), {len(train_idx_local)} (per-rank)",
                flush=True,
            )
            print(
                f"  Val records per epoch: {len(val_idx)} (global), {len(val_idx_local)} (per-rank)",
                flush=True,
            )
        samples_per_epoch_per_rank = len(train_ds)
        samples_per_epoch_global = samples_per_epoch_per_rank * world_size
        print(
            f"  Samples per epoch: {samples_per_epoch_global} (global), {samples_per_epoch_per_rank} (per-rank)",
            flush=True,
        )
        bytes_per_record = PREPARED_RECORD_BYTES if args.prepared_data_path is not None else 4
        print(f"  Train dataset size: {len(train_ds)} samples ({len(train_ds) * bytes_per_record // (1024**3)} GB)", flush=True)
        print(f"  Val dataset size: {len(val_ds)} samples ({len(val_ds) * bytes_per_record // (1024**3)} GB)", flush=True)
        epochs_this_run = max(0, final_epoch - start_epoch + 1)
        print(f"  Batch size: {args.batch_size}, Epochs this run: {epochs_this_run}", flush=True)
        if args.export_field_inference:
            print(f"  Field inference export: {args.field_inference_output}", flush=True)
        if args.export_field_iter:
            print(f"  Field iter export: {args.field_iter_output} (steps={args.field_iter_steps})", flush=True)
        if args.cache_mode == "window" and args.window_steps > 0:
            window_scale = math.sqrt(len(train_idx) / args.window_steps)
            window_scale = max(1, int(round(window_scale)))
            eff_patience = args.patience * window_scale
            print(
                f"  Window scaling: sqrt(train_steps/window_steps)={window_scale} -> "
                f"patience={eff_patience}, early_stop={eff_patience * 2}",
                flush=True,
            )
    training_start_time = time.time()

    for epoch in range(start_epoch, final_epoch + 1):
        # For window mode: refresh cached window at the start of each epoch
        if args.prepared_data_path is None:
            train_ds.refresh()
            val_ds.refresh()

        train_loss = run_epoch(model, train_loader, criterion, device,
                               optimizer=optimizer, distributed=distributed,
                               is_main=is_main, epoch=epoch, total_epochs=final_epoch)
        val_loss   = run_epoch(model, val_loader,   criterion, device,
                               distributed=distributed,
                               is_main=is_main, epoch=epoch, total_epochs=final_epoch)
        scheduler.step(val_loss)

        if is_main:
            elapsed = time.time() - training_start_time
            completed_this_run = epoch - start_epoch + 1
            remaining_epochs = final_epoch - epoch
            eta = (elapsed / completed_this_run * remaining_epochs) if completed_this_run > 0 else 0.0
            print(f"Epoch {epoch:3d}/{final_epoch}  "
                  f"train_loss={train_loss:.6f}  val_loss={val_loss:.6f}  "
                  f"lr={optimizer.param_groups[0]['lr']:.2e}  "
                  f"improve={val_loss < best_val_loss}  "
                  f"time={elapsed:.1f}s  ETA={eta:.1f}s", flush=True)

            if val_loss < best_val_loss:
                best_val_loss     = val_loss
                epochs_no_improve = 0
                checkpoint = {
                    "epoch": epoch,
                    "model_name": args.model,
                    "model_state_dict": model_to_save.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "scheduler_state_dict": scheduler.state_dict(),
                    "best_val_loss": best_val_loss,
                    "epochs_no_improve": epochs_no_improve,
                }
                torch.save(checkpoint, args.output)
                export_ok = True
                export_error = None
                try:
                    export_inference_model(model_to_save, args.inference_output, device)
                    if args.export_field_inference:
                        export_field_inference_model(model_to_save, args.field_inference_output, device)
                    if args.export_field_iter:
                        if args.field_iter_steps is None:
                            raise ValueError("--field-iter-steps is required with --export-field-iter")
                        export_field_iter_model(
                            model_to_save,
                            args.field_iter_output,
                            device,
                            steps=args.field_iter_steps,
                        )
                except Exception as error:
                    export_ok = False
                    export_error = str(error)
                print(
                    f"  → New best val loss {best_val_loss:.6f}, "
                    f"checkpoint saved to {args.output}",
                    flush=True,
                )
                if export_ok:
                    print(f"    Inference model saved to {args.inference_output}", flush=True)
                else:
                    print(f"    WARNING: TorchScript export failed: {export_error}", flush=True)
            else:
                epochs_no_improve += 1
                early_stop_patience = args.patience * 2
                if args.cache_mode == "window" and args.window_steps > 0:
                    window_scale = math.sqrt(len(train_idx) / args.window_steps)
                    window_scale = max(1, int(round(window_scale)))
                    early_stop_patience = max(1, args.patience * 2 * window_scale)
                if epochs_no_improve >= early_stop_patience:
                    print(f"Early stopping after {epoch} epochs "
                          f"(no improvement for {epochs_no_improve} epochs).", flush=True)
                    break
        else:
            # Non-main ranks still need to track early stopping consistently;
            # broadcast the flag from rank 0.
            if val_loss < best_val_loss:
                best_val_loss     = val_loss
                epochs_no_improve = 0
            else:
                epochs_no_improve += 1

    if is_main:
        print(f"\nTraining complete. Best val loss: {best_val_loss:.6f}", flush=True)

    if distributed and world_size > 1:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
