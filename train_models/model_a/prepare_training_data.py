from __future__ import annotations

import argparse
import json
import os
import socket
import shutil
import time
from dataclasses import dataclass

import numpy as np


KEY_FLOATS = 19
RECORD_FLOATS = 20
DEDUP_RECORD_DTYPE = np.dtype([("key", np.uint32, (KEY_FLOATS,)), ("count", np.int64)])


class MPIContext:
    def __init__(self, enabled=False, rank=0, size=1, comm=None, mpi_module=None):
        self.enabled = enabled
        self.rank = rank
        self.size = size
        self.comm = comm
        self.mpi = mpi_module

    @property
    def is_root(self) -> bool:
        return self.rank == 0


def log_msg(message: str, mpi_ctx: MPIContext | None = None) -> None:
    ts = time.strftime("%Y-%m-%d %H:%M:%S")
    if mpi_ctx is None:
        print(f"[{ts}] {message}", flush=True)
        return
    print(f"[{ts}][rank {mpi_ctx.rank}] {message}", flush=True)


def get_rss_bytes() -> int | None:
    try:
        import resource

        rss_kb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        # Linux: ru_maxrss is in KB.
        return int(rss_kb) * 1024
    except Exception:
        return None


def init_mpi_context() -> MPIContext:
    try:
        from mpi4py import MPI
    except Exception:
        return MPIContext(enabled=False)

    comm = MPI.COMM_WORLD
    size = comm.Get_size()
    rank = comm.Get_rank()
    if size > 1:
        return MPIContext(enabled=True, rank=rank, size=size, comm=comm, mpi_module=MPI)
    return MPIContext(enabled=False, rank=rank, size=size, comm=comm, mpi_module=MPI)


def periodic_pad(field: np.ndarray) -> np.ndarray:
    return np.pad(field, pad_width=1, mode="wrap")


def extract_patches(field_padded: np.ndarray) -> np.ndarray:
    h, w = field_padded.shape[0] - 2, field_padded.shape[1] - 2
    patches = np.lib.stride_tricks.sliding_window_view(field_padded, (3, 3))
    return patches.reshape(h * w, 9).astype(np.float32, copy=False)


def human_bytes(num_bytes: int) -> str:
    units = ["B", "KB", "MB", "GB", "TB"]
    value = float(num_bytes)
    unit = 0
    while value >= 1024.0 and unit < len(units) - 1:
        value /= 1024.0
        unit += 1
    return f"{value:.2f} {units[unit]}"


def merge_sorted_keys_counts(
    keys_a: np.ndarray,
    counts_a: np.ndarray,
    keys_b: np.ndarray,
    counts_b: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    if keys_a.size == 0:
        return keys_b, counts_b
    if keys_b.size == 0:
        return keys_a, counts_a

    i = 0
    j = 0
    merged_len_max = len(keys_a) + len(keys_b)
    merged_keys = np.empty((merged_len_max, keys_a.shape[1]), dtype=np.uint32)
    merged_counts = np.empty((merged_len_max,), dtype=np.int64)
    k = 0

    def _row_less(row_a: np.ndarray, row_b: np.ndarray) -> bool:
        diff_idx = np.flatnonzero(row_a != row_b)
        if diff_idx.size == 0:
            return False
        first = int(diff_idx[0])
        return row_a[first] < row_b[first]

    while i < len(keys_a) and j < len(keys_b):
        ka = keys_a[i]
        kb = keys_b[j]
        if np.array_equal(ka, kb):
            merged_keys[k] = ka
            merged_counts[k] = int(counts_a[i]) + int(counts_b[j])
            k += 1
            i += 1
            j += 1
            continue

        if _row_less(ka, kb):
            merged_keys[k] = ka
            merged_counts[k] = int(counts_a[i])
            k += 1
            i += 1
        else:
            merged_keys[k] = kb
            merged_counts[k] = int(counts_b[j])
            k += 1
            j += 1

    while i < len(keys_a):
        merged_keys[k] = keys_a[i]
        merged_counts[k] = int(counts_a[i])
        k += 1
        i += 1

    while j < len(keys_b):
        merged_keys[k] = keys_b[j]
        merged_counts[k] = int(counts_b[j])
        k += 1
        j += 1

    return merged_keys[:k], merged_counts[:k]


def bucket_ids_from_keys(keys_u32: np.ndarray, num_buckets: int) -> np.ndarray:
    h = np.full((keys_u32.shape[0],), np.uint64(1469598103934665603), dtype=np.uint64)
    for col in range(KEY_FLOATS):
        value = keys_u32[:, col].astype(np.uint64, copy=False)
        h ^= value + np.uint64(0x9E3779B97F4A7C15) + (h << np.uint64(6)) + (h >> np.uint64(2))
    return (h % np.uint64(num_buckets)).astype(np.int32)


def append_step_pairs_to_raw_buckets(step_keys_u32: np.ndarray, raw_dir: str, num_buckets: int) -> int:
    bucket_ids = bucket_ids_from_keys(step_keys_u32, num_buckets)
    order = np.argsort(bucket_ids, kind="mergesort")
    sorted_bucket_ids = bucket_ids[order]
    sorted_keys = step_keys_u32[order]

    split_points = np.flatnonzero(np.diff(sorted_bucket_ids)) + 1
    chunk_starts = np.concatenate(([0], split_points))
    chunk_ends = np.concatenate((split_points, [len(sorted_bucket_ids)]))

    written = 0
    for start, end in zip(chunk_starts, chunk_ends):
        bucket_id = int(sorted_bucket_ids[start])
        bucket_path = os.path.join(raw_dir, f"bucket_{bucket_id:05d}.raw")
        try:
            with open(bucket_path, "ab") as bf:
                sorted_keys[start:end].astype(np.uint32, copy=False).tofile(bf)
        except OSError as error:
            usage = shutil.disk_usage(raw_dir)
            raise OSError(
                f"Failed writing bucket file '{bucket_path}': {error}. "
                f"Disk free at '{raw_dir}' is {human_bytes(usage.free)} "
                f"(total {human_bytes(usage.total)})."
            ) from error
        written += int(end - start)
    return written


def estimate_disk_bucket_bytes(raw_pairs: int) -> int:
    # raw bucket payload stores full 19-value key as uint32.
    raw_payload = raw_pairs * KEY_FLOATS * 4
    # rough overhead for dedup bucket files + final bins/metadata.
    overhead = int(raw_payload * 0.35)
    return raw_payload + overhead


def check_disk_space_or_raise(path: str, required_bytes: int, mpi_ctx: MPIContext) -> None:
    usage = shutil.disk_usage(path)
    free = int(usage.free)
    log_msg(
        f"Disk check at {path}: free={human_bytes(free)} total={human_bytes(usage.total)} "
        f"required_estimate={human_bytes(required_bytes)}",
        mpi_ctx,
    )
    if free < required_bytes:
        raise RuntimeError(
            f"Insufficient free disk space in '{path}'. "
            f"Need ~{human_bytes(required_bytes)}, have {human_bytes(free)}. "
            "Use a larger BeeOND/local SSD path, reduce --max-steps, or disable --disk-buckets."
        )


def dedup_raw_bucket_file(raw_path: str) -> tuple[np.ndarray, np.ndarray]:
    raw = np.fromfile(raw_path, dtype=np.uint32)
    if raw.size == 0:
        return np.empty((0, KEY_FLOATS), dtype=np.uint32), np.empty((0,), dtype=np.int64)
    if raw.size % KEY_FLOATS != 0:
        raise ValueError(f"Corrupt raw bucket file (size not divisible by {KEY_FLOATS}): {raw_path}")
    keys = raw.reshape(-1, KEY_FLOATS)
    uniq, counts = np.unique(keys, axis=0, return_counts=True)
    return uniq.astype(np.uint32, copy=False), counts.astype(np.int64, copy=False)


def write_dedup_bucket(path: str, keys_u32: np.ndarray, counts_i64: np.ndarray) -> None:
    arr = np.empty((len(keys_u32),), dtype=DEDUP_RECORD_DTYPE)
    arr["key"] = keys_u32
    arr["count"] = counts_i64
    arr.tofile(path)


def read_dedup_bucket(path: str) -> tuple[np.ndarray, np.ndarray]:
    arr = np.fromfile(path, dtype=DEDUP_RECORD_DTYPE)
    if arr.size == 0:
        return np.empty((0, KEY_FLOATS), dtype=np.uint32), np.empty((0,), dtype=np.int64)
    return arr["key"].astype(np.uint32, copy=False), arr["count"].astype(np.int64, copy=False)


def update_top_pairs(
    current_top: list[dict],
    keys_u32: np.ndarray,
    counts_i64: np.ndarray,
    topk: int,
) -> list[dict]:
    if len(keys_u32) == 0:
        return current_top
    take = min(topk, len(keys_u32))
    local_idx = np.argpartition(counts_i64, -take)[-take:]
    for idx in local_idx:
        key_f32 = keys_u32[int(idx)].view(np.float32)
        current_top.append(
            {
                "count": int(counts_i64[int(idx)]),
                "water_patch": key_f32[:9].tolist(),
                "terrain_patch": key_f32[9:18].tolist(),
                "output": float(key_f32[18]),
            }
        )
    current_top.sort(key=lambda x: x["count"], reverse=True)
    return current_top[:topk]


def write_binary_from_global_buckets(
    global_bucket_paths: list[str],
    output_dir: str,
    num_batches: int,
    shuffle_output: bool,
    topk: int,
    source_data_path: str,
    max_steps: int | None,
    stats: BuildStats,
) -> None:
    os.makedirs(output_dir, exist_ok=True)

    bucket_infos = []
    total_records = 0
    for path in global_bucket_paths:
        arr = np.fromfile(path, dtype=DEDUP_RECORD_DTYPE)
        n = int(arr.size)
        if n > 0:
            bucket_infos.append((path, n))
            total_records += n

    if total_records == 0:
        raise ValueError("No records generated")

    if shuffle_output:
        rng = np.random.default_rng(42)
        perm = rng.permutation(len(bucket_infos))
        bucket_infos = [bucket_infos[i] for i in perm]

    num_batches = max(1, int(num_batches))
    boundaries = np.linspace(0, total_records, num_batches + 1, dtype=np.int64)
    batch_paths = [os.path.join(output_dir, f"pairs_batch_{idx:03d}.bin") for idx in range(num_batches)]
    batch_handles = [open(path, "wb") for path in batch_paths]

    top_pairs: list[dict] = []
    global_pos = 0

    try:
        for path, _ in bucket_infos:
            arr = np.fromfile(path, dtype=DEDUP_RECORD_DTYPE)
            if arr.size == 0:
                continue
            keys_u32 = arr["key"].astype(np.uint32, copy=False)
            counts_i64 = arr["count"].astype(np.int64, copy=False)

            if np.any(counts_i64 > np.iinfo(np.int32).max):
                raise OverflowError(f"Count exceeds int32 range in {path}")

            top_pairs = update_top_pairs(top_pairs, keys_u32, counts_i64, topk=max(1, topk))

            if shuffle_output:
                rng = np.random.default_rng(42 + int(global_pos % 1000003))
                order = rng.permutation(len(keys_u32))
                keys_u32 = keys_u32[order]
                counts_i64 = counts_i64[order]

            records = np.empty((len(keys_u32), RECORD_FLOATS), dtype=np.float32)
            records[:, :KEY_FLOATS] = keys_u32.view(np.float32)
            records[:, KEY_FLOATS] = counts_i64.astype(np.int32, copy=False).view(np.float32)

            local_pos = 0
            while local_pos < len(records):
                absolute_pos = global_pos + local_pos
                batch_idx = int(np.searchsorted(boundaries, absolute_pos, side="right") - 1)
                batch_idx = min(max(batch_idx, 0), num_batches - 1)
                batch_end = int(boundaries[batch_idx + 1])
                take = min(len(records) - local_pos, batch_end - absolute_pos)
                if take <= 0:
                    batch_idx += 1
                    continue
                records[local_pos : local_pos + take].astype("<f4", copy=False).tofile(batch_handles[batch_idx])
                local_pos += take

            global_pos += len(records)
    finally:
        for fh in batch_handles:
            fh.close()

    batch_files = [os.path.basename(path) for path in batch_paths if os.path.getsize(path) > 0]
    raw_bytes = stats.raw_pairs * RECORD_FLOATS * 4
    dedup_bytes = total_records * RECORD_FLOATS * 4
    metadata = {
        "format": "water_pairs_v1",
        "record_layout": {
            "float32_values": 19,
            "count_int32": 1,
            "record_bytes": 80,
            "order": ["water_patch_3x3", "terrain_patch_3x3", "output", "count_i32"],
        },
        "source": {
            "data_path": source_data_path,
            "max_steps": max_steps,
            "used_steps": stats.used_steps,
            "h": stats.h,
            "w": stats.w,
        },
        "stats": {
            "raw_pairs": stats.raw_pairs,
            "unique_pairs": total_records,
            "dedup_saved_pairs": int(stats.raw_pairs - total_records),
            "dedup_ratio": float(total_records / stats.raw_pairs),
            "raw_size_bytes": int(raw_bytes),
            "dedup_size_bytes": int(dedup_bytes),
            "saved_size_bytes": int(raw_bytes - dedup_bytes),
        },
        "batching": {
            "num_batches": int(len(batch_files)),
            "shuffle_output": bool(shuffle_output),
            "files": batch_files,
        },
        "top_frequent_pairs": top_pairs,
    }

    meta_path = os.path.join(output_dir, "metadata.json")
    with open(meta_path, "w", encoding="utf-8") as mf:
        json.dump(metadata, mf, indent=2)

    print("\nBuild summary")
    print(f"  Raw pairs: {stats.raw_pairs}")
    print(f"  Unique pairs: {total_records}")
    print(f"  Saved pairs by dedup: {stats.raw_pairs - total_records}")
    print(f"  Raw data size:   {raw_bytes} bytes ({human_bytes(raw_bytes)})")
    print(f"  Dedup data size: {dedup_bytes} bytes ({human_bytes(dedup_bytes)})")
    print(f"  Saved size:      {raw_bytes - dedup_bytes} bytes ({human_bytes(raw_bytes - dedup_bytes)})")
    print(f"  Output directory: {output_dir}")
    print(f"  Batch files: {len(batch_files)}")
    print(f"  Metadata: {meta_path}")
    if top_pairs:
        most = top_pairs[0]
        print(f"  Most frequent pair count: {most['count']}")
        print(
            f"  Most frequent pair: water={most['water_patch']} | "
            f"terrain={most['terrain_patch']} | output={most['output']}",
            flush=True,
        )


@dataclass
class BuildStats:
    raw_pairs: int
    unique_pairs: int
    h: int
    w: int
    used_steps: int


def mpi_send_pairs(ctx: MPIContext, dest: int, keys: np.ndarray, counts: np.ndarray, tag_base: int = 1000) -> None:
    comm = ctx.comm
    mpi = ctx.mpi
    rows = int(len(keys))
    comm.send(rows, dest=dest, tag=tag_base)
    if rows == 0:
        return
    key_cols = int(keys.shape[1])
    comm.send(key_cols, dest=dest, tag=tag_base + 1)

    # Some MPI stacks still use 32-bit count arguments for Send/Recv calls.
    # Keep each transfer chunk below that limit to avoid MPI_ERR_ARG on large merges.
    max_rows_per_chunk = max(1, (2**31 - 1) // key_cols)
    keys_c = np.ascontiguousarray(keys, dtype=np.uint32)
    counts_c = np.ascontiguousarray(counts, dtype=np.int64)

    offset = 0
    while offset < rows:
        chunk_rows = min(max_rows_per_chunk, rows - offset)
        keys_chunk = keys_c[offset : offset + chunk_rows].reshape(-1)
        counts_chunk = counts_c[offset : offset + chunk_rows]
        comm.Send([keys_chunk, mpi.UINT32_T], dest=dest, tag=tag_base + 2)
        comm.Send([counts_chunk, mpi.INT64_T], dest=dest, tag=tag_base + 3)
        offset += chunk_rows


def mpi_recv_pairs(ctx: MPIContext, src: int, tag_base: int = 1000) -> tuple[np.ndarray, np.ndarray]:
    comm = ctx.comm
    mpi = ctx.mpi
    rows = comm.recv(source=src, tag=tag_base)
    if rows == 0:
        return np.empty((0, KEY_FLOATS), dtype=np.uint32), np.empty((0,), dtype=np.int64)
    key_cols = comm.recv(source=src, tag=tag_base + 1)
    keys_flat = np.empty(rows * key_cols, dtype=np.uint32)
    counts = np.empty(rows, dtype=np.int64)

    max_rows_per_chunk = max(1, (2**31 - 1) // key_cols)
    offset = 0
    while offset < rows:
        chunk_rows = min(max_rows_per_chunk, rows - offset)
        key_start = offset * key_cols
        key_end = (offset + chunk_rows) * key_cols
        comm.Recv([keys_flat[key_start:key_end], mpi.UINT32_T], source=src, tag=tag_base + 2)
        comm.Recv([counts[offset : offset + chunk_rows], mpi.INT64_T], source=src, tag=tag_base + 3)
        offset += chunk_rows

    return keys_flat.reshape(rows, key_cols), counts


def mpi_tree_merge_pairs(ctx: MPIContext, keys: np.ndarray, counts: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    if not ctx.enabled:
        return keys, counts

    step = 1
    while step < ctx.size:
        group_width = step * 2
        if (ctx.rank % group_width) == 0:
            partner = ctx.rank + step
            if partner < ctx.size:
                log_msg(
                    f"merge stage step={step}: waiting for rank {partner} "
                    f"(local unique={len(keys)}, local mem~{human_bytes(keys.nbytes + counts.nbytes)})",
                    ctx,
                )
                recv_keys, recv_counts = mpi_recv_pairs(ctx, partner, tag_base=1000 + step * 10)
                log_msg(
                    f"merge stage step={step}: received {len(recv_keys)} unique rows from rank {partner} "
                    f"(recv mem~{human_bytes(recv_keys.nbytes + recv_counts.nbytes)})",
                    ctx,
                )
                keys, counts = merge_sorted_keys_counts(keys, counts, recv_keys, recv_counts)
                log_msg(
                    f"merge stage step={step}: merged unique={len(keys)} "
                    f"(merged mem~{human_bytes(keys.nbytes + counts.nbytes)})",
                    ctx,
                )
        else:
            partner = ctx.rank - step
            log_msg(
                f"merge stage step={step}: sending {len(keys)} unique rows to rank {partner} "
                f"(send mem~{human_bytes(keys.nbytes + counts.nbytes)})",
                ctx,
            )
            mpi_send_pairs(ctx, partner, keys, counts, tag_base=1000 + step * 10)
            return np.empty((0, KEY_FLOATS), dtype=np.uint32), np.empty((0,), dtype=np.int64)
        step *= 2

    return keys, counts


def build_dedup_keys(
    data_path: str,
    max_steps: int | None,
    mpi_ctx: MPIContext,
    progress_chunks: int,
    log_rss: bool,
) -> tuple[np.ndarray, np.ndarray, BuildStats]:
    import h5py

    with h5py.File(data_path, "r") as f:
        water = f["water"]
        terrain = f["terrain"][:].astype(np.float32, copy=False)

        total_steps = water.shape[0]
        h, w = water.shape[1], water.shape[2]
        if terrain.shape != (h, w):
            raise ValueError(f"Terrain shape mismatch: expected {(h, w)}, got {terrain.shape}")

        if max_steps is None:
            used_steps = total_steps
        else:
            used_steps = min(total_steps, max_steps)

        if used_steps < 2:
            raise ValueError("Need at least 2 steps to build (t -> t+1) pairs")

        terrain_patches = extract_patches(periodic_pad(terrain))

        local_keys = np.empty((0, KEY_FLOATS), dtype=np.uint32)
        local_counts = np.empty((0,), dtype=np.int64)

        step_pairs = h * w
        total_pair_steps = used_steps - 1
        raw_pairs = total_pair_steps * step_pairs

        if mpi_ctx.enabled:
            split_edges = np.linspace(0, total_pair_steps, mpi_ctx.size + 1, dtype=np.int64)
            local_start = int(split_edges[mpi_ctx.rank])
            local_end = int(split_edges[mpi_ctx.rank + 1])
        else:
            local_start = 0
            local_end = total_pair_steps

        local_pair_steps = max(0, local_end - local_start)

        if mpi_ctx.is_root:
            log_msg(f"Input water shape: {water.shape}", mpi_ctx)
            log_msg(f"Terrain shape: {terrain.shape}", mpi_ctx)
            log_msg(f"Using steps: {used_steps} (pairs from 0..{used_steps - 2})", mpi_ctx)
            log_msg(f"Raw pairs before dedup: {raw_pairs}", mpi_ctx)
            if mpi_ctx.enabled:
                log_msg(f"MPI enabled: ranks={mpi_ctx.size}", mpi_ctx)

        if mpi_ctx.enabled:
            log_msg(
                f"[rank {mpi_ctx.rank}] local pair-step range: [{local_start}, {local_end}) "
                f"({local_pair_steps} step pairs)",
                mpi_ctx,
            )

        if log_rss:
            rss = get_rss_bytes()
            if rss is not None:
                log_msg(f"Initial RSS: {human_bytes(rss)}", mpi_ctx)

        progress_start_time = time.time()

        for local_t in range(local_pair_steps):
            if local_t == 0:
                water_t = water[local_start].astype(np.float32, copy=False)
                water_next = water[local_start + 1].astype(np.float32, copy=False)
            elif local_t + 1 < local_pair_steps:
                water_t = water_next
                water_next = water[local_start + local_t + 1].astype(np.float32, copy=False)
            else:
                water_t = water_next
                water_next = water[local_start + local_t + 1].astype(np.float32, copy=False)

            water_patches = extract_patches(periodic_pad(water_t))
            y = water_next.reshape(step_pairs, 1).astype(np.float32, copy=False)

            step_keys_f32 = np.concatenate([water_patches, terrain_patches, y], axis=1)
            step_keys_u32 = step_keys_f32.view(np.uint32)
            step_unique, step_counts = np.unique(step_keys_u32, axis=0, return_counts=True)

            local_keys, local_counts = merge_sorted_keys_counts(
                local_keys,
                local_counts,
                step_unique,
                step_counts.astype(np.int64, copy=False),
            )

            if (local_t + 1) % max(1, local_pair_steps // max(1, progress_chunks)) == 0 or (local_t + 1) == local_pair_steps:
                done = local_t + 1
                total = local_pair_steps
                elapsed = time.time() - progress_start_time
                eta = (elapsed / done) * (total - done) if done > 0 else 0.0
                current_unique = len(local_keys)
                rough_mem_bytes = current_unique * ((KEY_FLOATS * 4) + 8)
                dedup_bytes = local_keys.nbytes + local_counts.nbytes
                msg = (
                    f"processed {done}/{total} step pairs | "
                    f"current unique: {current_unique} | "
                    f"elapsed: {elapsed:.1f}s | eta: {eta:.1f}s | "
                    f"rough dedup mem: {human_bytes(rough_mem_bytes)} | "
                    f"dedup array mem: {human_bytes(dedup_bytes)}"
                )
                if log_rss:
                    rss = get_rss_bytes()
                    if rss is not None:
                        msg += f" | rss_max: {human_bytes(rss)}"
                log_msg(msg, mpi_ctx)

        log_msg(
            f"Local build complete: unique={len(local_keys)} "
            f"(local dedup mem~{human_bytes(local_keys.nbytes + local_counts.nbytes)})",
            mpi_ctx,
        )

    log_msg("Starting tree merge", mpi_ctx)
    global_keys, global_counts = mpi_tree_merge_pairs(mpi_ctx, local_keys, local_counts)
    if mpi_ctx.enabled:
        log_msg("Waiting at merge barrier", mpi_ctx)

    if mpi_ctx.enabled:
        mpi_ctx.comm.Barrier()

    if not mpi_ctx.enabled or mpi_ctx.is_root:
        log_msg(
            f"Global merge complete: unique={len(global_keys)} "
            f"(global dedup mem~{human_bytes(global_keys.nbytes + global_counts.nbytes)})",
            mpi_ctx,
        )

    stats = BuildStats(
        raw_pairs=raw_pairs,
        unique_pairs=len(global_keys) if mpi_ctx.is_root else 0,
        h=h,
        w=w,
        used_steps=used_steps,
    )
    return global_keys, global_counts, stats


def build_dedup_disk_buckets(
    data_path: str,
    output_dir: str,
    max_steps: int | None,
    mpi_ctx: MPIContext,
    progress_chunks: int,
    log_rss: bool,
    num_buckets: int,
    bucket_tmp_dir: str | None,
    keep_bucket_tmp: bool,
    num_batches: int,
    shuffle_output: bool,
    topk: int,
) -> None:
    import h5py

    if num_buckets <= 0:
        raise ValueError("--disk-buckets must be a positive integer")

    tmp_root = bucket_tmp_dir or os.path.join(output_dir, "_bucket_tmp")
    rank_root = os.path.join(tmp_root, f"rank_{mpi_ctx.rank:05d}")
    raw_dir = os.path.join(rank_root, "raw")
    dedup_dir = os.path.join(rank_root, "dedup")
    merged_dir = os.path.join(tmp_root, "merged")
    os.makedirs(raw_dir, exist_ok=True)
    os.makedirs(dedup_dir, exist_ok=True)
    os.makedirs(merged_dir, exist_ok=True)

    with h5py.File(data_path, "r") as f:
        water = f["water"]
        terrain = f["terrain"][:].astype(np.float32, copy=False)

        total_steps = water.shape[0]
        h, w = water.shape[1], water.shape[2]
        if terrain.shape != (h, w):
            raise ValueError(f"Terrain shape mismatch: expected {(h, w)}, got {terrain.shape}")

        used_steps = total_steps if max_steps is None else min(total_steps, max_steps)
        if used_steps < 2:
            raise ValueError("Need at least 2 steps to build (t -> t+1) pairs")

        terrain_patches = extract_patches(periodic_pad(terrain))
        step_pairs = h * w
        total_pair_steps = used_steps - 1
        raw_pairs = total_pair_steps * step_pairs

        if mpi_ctx.is_root:
            estimated_total_tmp = estimate_disk_bucket_bytes(raw_pairs)
            check_disk_space_or_raise(tmp_root, estimated_total_tmp, mpi_ctx)

        if mpi_ctx.enabled:
            split_edges = np.linspace(0, total_pair_steps, mpi_ctx.size + 1, dtype=np.int64)
            local_start = int(split_edges[mpi_ctx.rank])
            local_end = int(split_edges[mpi_ctx.rank + 1])
        else:
            local_start, local_end = 0, total_pair_steps
        local_pair_steps = max(0, local_end - local_start)

        if mpi_ctx.is_root:
            log_msg(f"Input water shape: {water.shape}", mpi_ctx)
            log_msg(f"Terrain shape: {terrain.shape}", mpi_ctx)
            log_msg(f"Using steps: {used_steps} (pairs from 0..{used_steps - 2})", mpi_ctx)
            log_msg(f"Raw pairs before dedup: {raw_pairs}", mpi_ctx)
            log_msg(f"Disk-bucket mode: enabled, buckets={num_buckets}, tmp_root={tmp_root}", mpi_ctx)

        log_msg(
            f"local pair-step range: [{local_start}, {local_end}) ({local_pair_steps} step pairs)",
            mpi_ctx,
        )

        if log_rss:
            rss = get_rss_bytes()
            if rss is not None:
                log_msg(f"Initial RSS: {human_bytes(rss)}", mpi_ctx)

        progress_start_time = time.time()
        for local_t in range(local_pair_steps):
            if local_t == 0:
                water_t = water[local_start].astype(np.float32, copy=False)
                water_next = water[local_start + 1].astype(np.float32, copy=False)
            else:
                water_t = water_next
                water_next = water[local_start + local_t + 1].astype(np.float32, copy=False)

            water_patches = extract_patches(periodic_pad(water_t))
            y = water_next.reshape(step_pairs, 1).astype(np.float32, copy=False)
            step_keys_u32 = np.concatenate([water_patches, terrain_patches, y], axis=1).view(np.uint32)
            append_step_pairs_to_raw_buckets(step_keys_u32, raw_dir=raw_dir, num_buckets=num_buckets)

            if (local_t + 1) % max(1, local_pair_steps // max(1, progress_chunks)) == 0 or (local_t + 1) == local_pair_steps:
                done = local_t + 1
                total = local_pair_steps
                elapsed = time.time() - progress_start_time
                eta = (elapsed / done) * (total - done) if done > 0 else 0.0
                msg = f"spilled {done}/{total} step pairs | elapsed: {elapsed:.1f}s | eta: {eta:.1f}s"
                if log_rss:
                    rss = get_rss_bytes()
                    if rss is not None:
                        msg += f" | rss_max: {human_bytes(rss)}"
                log_msg(msg, mpi_ctx)

    log_msg("Starting local bucket dedup phase", mpi_ctx)
    local_unique = 0
    for bucket_id in range(num_buckets):
        raw_path = os.path.join(raw_dir, f"bucket_{bucket_id:05d}.raw")
        if not os.path.exists(raw_path):
            continue
        keys_u32, counts_i64 = dedup_raw_bucket_file(raw_path)
        local_unique += len(keys_u32)
        out_path = os.path.join(dedup_dir, f"bucket_{bucket_id:05d}.dedup")
        write_dedup_bucket(out_path, keys_u32, counts_i64)

    log_msg(f"Local dedup complete: unique={local_unique}", mpi_ctx)

    if mpi_ctx.enabled:
        mpi_ctx.comm.Barrier()

    log_msg("Starting owner-rank bucket merge", mpi_ctx)
    for bucket_id in range(num_buckets):
        owner = bucket_id % mpi_ctx.size
        if mpi_ctx.rank != owner:
            continue

        merged_keys = np.empty((0, KEY_FLOATS), dtype=np.uint32)
        merged_counts = np.empty((0,), dtype=np.int64)
        for src_rank in range(mpi_ctx.size):
            src_path = os.path.join(tmp_root, f"rank_{src_rank:05d}", "dedup", f"bucket_{bucket_id:05d}.dedup")
            if not os.path.exists(src_path):
                continue
            keys_u32, counts_i64 = read_dedup_bucket(src_path)
            merged_keys, merged_counts = merge_sorted_keys_counts(merged_keys, merged_counts, keys_u32, counts_i64)

        if len(merged_keys) > 0:
            owner_path = os.path.join(merged_dir, f"bucket_{bucket_id:05d}.dedup")
            write_dedup_bucket(owner_path, merged_keys, merged_counts)

        log_msg(
            f"owner merge bucket={bucket_id} complete | unique={len(merged_keys)} | "
            f"mem~{human_bytes(merged_keys.nbytes + merged_counts.nbytes)}",
            mpi_ctx,
        )

    if mpi_ctx.enabled:
        mpi_ctx.comm.Barrier()

    if mpi_ctx.is_root:
        global_bucket_paths = [
            os.path.join(merged_dir, f"bucket_{bucket_id:05d}.dedup")
            for bucket_id in range(num_buckets)
            if os.path.exists(os.path.join(merged_dir, f"bucket_{bucket_id:05d}.dedup"))
        ]
        stats = BuildStats(
            raw_pairs=raw_pairs,
            unique_pairs=0,
            h=h,
            w=w,
            used_steps=used_steps,
        )
        write_binary_from_global_buckets(
            global_bucket_paths=global_bucket_paths,
            output_dir=output_dir,
            num_batches=num_batches,
            shuffle_output=shuffle_output,
            topk=topk,
            source_data_path=data_path,
            max_steps=max_steps,
            stats=stats,
        )

    if mpi_ctx.enabled:
        mpi_ctx.comm.Barrier()

    if not keep_bucket_tmp:
        try:
            import shutil

            shutil.rmtree(rank_root, ignore_errors=True)
            if mpi_ctx.is_root:
                shutil.rmtree(merged_dir, ignore_errors=True)
                if os.path.isdir(tmp_root) and not os.listdir(tmp_root):
                    shutil.rmtree(tmp_root, ignore_errors=True)
        except Exception as error:
            log_msg(f"WARNING: failed to cleanup bucket tmp dirs: {error}", mpi_ctx)


def write_binary_batches(
    keys_u32: np.ndarray,
    counts_i64: np.ndarray,
    output_dir: str,
    num_batches: int,
    shuffle_output: bool,
    topk: int,
    source_data_path: str,
    max_steps: int | None,
    stats: BuildStats,
) -> None:
    os.makedirs(output_dir, exist_ok=True)

    n = len(keys_u32)
    if n == 0:
        raise ValueError("No records generated")

    counts_i32 = counts_i64.astype(np.int32, copy=False)
    if np.any(counts_i64 > np.iinfo(np.int32).max):
        raise OverflowError("Count exceeds int32 range")

    keys_f32 = keys_u32.view(np.float32)
    records = np.empty((n, RECORD_FLOATS), dtype=np.float32)
    records[:, :KEY_FLOATS] = keys_f32
    records[:, KEY_FLOATS] = counts_i32.view(np.float32)

    if shuffle_output:
        rng = np.random.default_rng(42)
        perm = rng.permutation(n)
        records = records[perm]
        counts_i64 = counts_i64[perm]

    num_batches = max(1, int(num_batches))
    boundaries = np.linspace(0, n, num_batches + 1, dtype=np.int64)
    batch_files = []
    for batch_idx in range(num_batches):
        start = int(boundaries[batch_idx])
        end = int(boundaries[batch_idx + 1])
        if end <= start:
            continue
        batch_path = os.path.join(output_dir, f"pairs_batch_{batch_idx:03d}.bin")
        records[start:end].astype("<f4", copy=False).tofile(batch_path)
        batch_files.append(os.path.basename(batch_path))

    if not batch_files:
        raise RuntimeError("No batch files written")

    counts_sorted_idx = np.argsort(counts_i64)[::-1]
    topk = max(1, topk)
    top_idx = counts_sorted_idx[: min(topk, len(counts_sorted_idx))]
    top_pairs = []
    for idx in top_idx:
        row = records[int(idx), :KEY_FLOATS].copy()
        top_pairs.append(
            {
                "count": int(counts_i64[int(idx)]),
                "water_patch": row[:9].tolist(),
                "terrain_patch": row[9:18].tolist(),
                "output": float(row[18]),
            }
        )

    raw_bytes = stats.raw_pairs * RECORD_FLOATS * 4
    dedup_bytes = stats.unique_pairs * RECORD_FLOATS * 4
    metadata = {
        "format": "water_pairs_v1",
        "record_layout": {
            "float32_values": 19,
            "count_int32": 1,
            "record_bytes": 80,
            "order": ["water_patch_3x3", "terrain_patch_3x3", "output", "count_i32"],
        },
        "source": {
            "data_path": source_data_path,
            "max_steps": max_steps,
            "used_steps": stats.used_steps,
            "h": stats.h,
            "w": stats.w,
        },
        "stats": {
            "raw_pairs": stats.raw_pairs,
            "unique_pairs": stats.unique_pairs,
            "dedup_saved_pairs": int(stats.raw_pairs - stats.unique_pairs),
            "dedup_ratio": float(stats.unique_pairs / stats.raw_pairs),
            "raw_size_bytes": int(raw_bytes),
            "dedup_size_bytes": int(dedup_bytes),
            "saved_size_bytes": int(raw_bytes - dedup_bytes),
        },
        "batching": {
            "num_batches": int(len(batch_files)),
            "shuffle_output": bool(shuffle_output),
            "files": batch_files,
        },
        "top_frequent_pairs": top_pairs,
    }

    meta_path = os.path.join(output_dir, "metadata.json")
    with open(meta_path, "w", encoding="utf-8") as mf:
        json.dump(metadata, mf, indent=2)

    print("\nBuild summary")
    print(f"  Raw pairs: {stats.raw_pairs}")
    print(f"  Unique pairs: {stats.unique_pairs}")
    print(f"  Saved pairs by dedup: {stats.raw_pairs - stats.unique_pairs}")
    print(f"  Raw data size:   {raw_bytes} bytes ({human_bytes(raw_bytes)})")
    print(f"  Dedup data size: {dedup_bytes} bytes ({human_bytes(dedup_bytes)})")
    print(f"  Saved size:      {raw_bytes - dedup_bytes} bytes ({human_bytes(raw_bytes - dedup_bytes)})")
    print(f"  Output directory: {output_dir}")
    print(f"  Batch files: {len(batch_files)}")
    print(f"  Metadata: {meta_path}")
    if top_pairs:
        most = top_pairs[0]
        print(f"  Most frequent pair count: {most['count']}")
        print(
            f"  Most frequent pair: water={most['water_patch']} | "
            f"terrain={most['terrain_patch']} | output={most['output']}",
            flush=True,
        )


def validate_inputs(data_path: str, output_dir: str, max_steps: int | None) -> None:
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Input HDF5 file not found: {data_path}")
    if not os.path.isfile(data_path):
        raise ValueError(f"Input path is not a file: {data_path}")

    if max_steps is not None and max_steps < 2:
        raise ValueError("--max-steps must be >= 2")

    out_parent = os.path.dirname(os.path.abspath(output_dir)) or "."
    if not os.path.isdir(out_parent):
        raise ValueError(f"Output parent directory does not exist: {out_parent}")

    if os.path.exists(output_dir) and not os.path.isdir(output_dir):
        raise ValueError(f"Output path exists but is not a directory: {output_dir}")

    import h5py

    try:
        with h5py.File(data_path, "r") as f:
            if "water" not in f or "terrain" not in f:
                raise ValueError("HDF5 file must contain datasets 'water' and 'terrain'")

            water = f["water"]
            terrain = f["terrain"]
            if water.ndim != 3:
                raise ValueError(f"Dataset 'water' must be 3D (T,H,W), got shape {water.shape}")
            if terrain.ndim != 2:
                raise ValueError(f"Dataset 'terrain' must be 2D (H,W), got shape {terrain.shape}")

            t, h, w = water.shape
            if terrain.shape != (h, w):
                raise ValueError(
                    f"Spatial mismatch: water has (H,W)=({h},{w}) but terrain has {terrain.shape}"
                )

            used_steps = t if max_steps is None else min(t, max_steps)
            if used_steps < 2:
                raise ValueError(
                    f"Not enough usable steps: total_steps={t}, max_steps={max_steps}, usable={used_steps}"
                )
    except OSError as err:
        raise ValueError(f"Could not open HDF5 file '{data_path}': {err}") from err


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Prepare deduplicated binary training pairs from world_trajectory.h5",
        formatter_class=argparse.RawTextHelpFormatter,
        epilog=(
            "Record format (80 bytes):\n"
            "  - 9x fp32 water patch (3x3)\n"
            "  - 9x fp32 terrain patch (3x3)\n"
            "  - 1x fp32 output\n"
            "  - 1x int32 count\n"
            "\n"
            "Examples:\n"
            "  python3 prepare_training_data.py ../../external_data/.../world_trajectory.h5 ./prepared_data --max-steps 100\n"
            "\n"
            "  python3 prepare_training_data.py ../../external_data/.../world_trajectory.h5 ./prepared_data_shuffled \\\n"
            "      --max-steps 1000 --num-batches 8 --shuffle-output\n"
            "\n"
            "MPI example:\n"
            "  srun --ntasks=4 python3 prepare_training_data.py ../../external_data/.../world_trajectory.h5 ./prepared_mpi --max-steps 20\n"
            "\n"
            "Disk-bucket mode example (lower RAM):\n"
            "  srun --ntasks=48 python3 -m mpi4py prepare_training_data.py ../../external_data/.../world_trajectory.h5 ./prepared_mpi \\\n"
            "      --max-steps 10000 --disk-buckets 1024 --bucket-tmp-dir /tmp/$USER/$SLURM_JOB_ID/buckets\n"
        ),
    )
    parser.add_argument("data_path", help="Input HDF5 file path")
    parser.add_argument("output_dir", help="Output directory for binary files")
    parser.add_argument(
        "--max-steps",
        type=int,
        default=None,
        help="Use at most this many time steps from input",
    )
    parser.add_argument(
        "--num-batches",
        type=int,
        default=1,
        help="Split final deduplicated output into K batch files",
    )
    parser.add_argument(
        "--shuffle-output",
        action="store_true",
        help="Shuffle final record order before writing batches",
    )
    parser.add_argument(
        "--topk",
        type=int,
        default=20,
        help="How many most frequent pairs to store in metadata",
    )
    parser.add_argument(
        "--progress-chunks",
        type=int,
        default=10,
        help="How many progress updates per rank during local step processing",
    )
    parser.add_argument(
        "--log-rss",
        action="store_true",
        help="Include process RSS high-water mark in progress logs (useful for OOM diagnostics)",
    )
    parser.add_argument(
        "--disk-buckets",
        type=int,
        default=0,
        help="Enable disk-backed dedup with this many hash buckets (0 disables)",
    )
    parser.add_argument(
        "--bucket-tmp-dir",
        default=None,
        help="Directory for temporary bucket files (defaults to <output_dir>/_bucket_tmp)",
    )
    parser.add_argument(
        "--keep-bucket-tmp",
        action="store_true",
        help="Keep temporary bucket files after completion (for debugging)",
    )
    return parser.parse_args()


def main() -> None:
    mpi_ctx = init_mpi_context()
    args = parse_args()
    slurm_ntasks = int(os.environ.get("SLURM_NTASKS", "1"))
    if slurm_ntasks > 1 and (not mpi_ctx.enabled or mpi_ctx.size == 1):
        raise RuntimeError(
            "SLURM requested multiple tasks but MPI is not initialized "
            f"(SLURM_NTASKS={slurm_ntasks}, mpi_enabled={mpi_ctx.enabled}, mpi_size={mpi_ctx.size}). "
            "Use an MPI launcher, e.g. 'srun --mpi=pmix_v3 python3 -m mpi4py prepare_training_data.py ...', "
            "and ensure mpi4py is installed in the active environment."
        )
    host = socket.gethostname()
    log_msg(
        f"Starting prepare_training_data.py on host={host} pid={os.getpid()} "
        f"mpi_enabled={mpi_ctx.enabled} size={mpi_ctx.size}",
        mpi_ctx,
    )
    log_msg(
        f"Args: data_path={args.data_path} output_dir={args.output_dir} "
        f"max_steps={args.max_steps} num_batches={args.num_batches} "
        f"shuffle_output={args.shuffle_output} topk={args.topk} "
        f"progress_chunks={args.progress_chunks} log_rss={args.log_rss} "
        f"disk_buckets={args.disk_buckets} bucket_tmp_dir={args.bucket_tmp_dir} "
        f"keep_bucket_tmp={args.keep_bucket_tmp}",
        mpi_ctx,
    )
    validate_inputs(args.data_path, args.output_dir, args.max_steps)
    if args.disk_buckets > 0:
        build_dedup_disk_buckets(
            data_path=args.data_path,
            output_dir=args.output_dir,
            max_steps=args.max_steps,
            mpi_ctx=mpi_ctx,
            progress_chunks=args.progress_chunks,
            log_rss=args.log_rss,
            num_buckets=args.disk_buckets,
            bucket_tmp_dir=args.bucket_tmp_dir,
            keep_bucket_tmp=args.keep_bucket_tmp,
            num_batches=args.num_batches,
            shuffle_output=args.shuffle_output,
            topk=args.topk,
        )
    else:
        keys_u32, counts_i64, stats = build_dedup_keys(
            args.data_path,
            args.max_steps,
            mpi_ctx,
            progress_chunks=args.progress_chunks,
            log_rss=args.log_rss,
        )

        if not mpi_ctx.enabled or mpi_ctx.is_root:
            write_binary_batches(
                keys_u32=keys_u32,
                counts_i64=counts_i64,
                output_dir=args.output_dir,
                num_batches=args.num_batches,
                shuffle_output=args.shuffle_output,
                topk=args.topk,
                source_data_path=args.data_path,
                max_steps=args.max_steps,
                stats=stats,
            )


if __name__ == "__main__":
    main()
