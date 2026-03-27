# prepare_training_data.py help

This script converts trajectory data from `world_trajectory.h5` into a deduplicated binary training format.

## What it does

- Builds training pairs from `(t -> t+1)` for each grid cell.
- Input features per pair:
  - 3x3 water patch (9 fp32)
  - 3x3 terrain patch (9 fp32)
- Target per pair:
  - next-step center value (1 fp32)
- Deduplicates identical `(input, output)` pairs and stores a frequency count.

Each output record is exactly **80 bytes**:

- 19 × `float32` values (water patch + terrain patch + output)
- 1 × `int32` count

## CLI usage

```bash
python3 prepare_training_data.py \
  <path/to/world_trajectory.h5> \
  <output_dir> \
  [--max-steps N] \
  [--num-batches K] \
  [--shuffle-output] \
  [--topk M]
```

## Arguments

- `data_path` (required positional): input HDF5 file.
- `output_dir` (required positional): directory for output `.bin` files and `metadata.json`.
- `--max-steps`: use at most the first `N` time steps from the input.
- `--num-batches`: split final deduplicated output into `K` binary files.
- `--shuffle-output`: randomize final output order before writing files.
- `--topk`: number of most frequent pairs stored in metadata.

## Output files

- `pairs_batch_000.bin`, `pairs_batch_001.bin`, ...
- `metadata.json` containing:
  - format and record layout
  - source information (path, used steps, H/W)
  - deduplication stats and size estimates
  - list of output batch files
  - top frequent pairs

## Example commands

Minimal:

```bash
python3 prepare_training_data.py \
  ../../external_data/.../world_trajectory.h5 \
  ./prepared_data \
  --max-steps 100
```

Shuffled + segmented:

```bash
python3 prepare_training_data.py \
  ../../external_data/.../world_trajectory.h5 \
  ./prepared_data_shuffled \
  --max-steps 1000 \
  --num-batches 8 \
  --shuffle-output \
  --topk 20
```

## Notes

- The dedup pipeline keeps exact floating-point bit patterns, so equal pairs are merged exactly.
- Very large datasets can still need significant RAM/CPU time during preprocessing.
- For training with prepared data, use `train.py --prepared-data-path <output_dir>`.