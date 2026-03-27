
# Top-down visualization of the world trajectory data

## Frame rendering

```bash
python3 render.py \
  --all-steps \
  --output-dir "external_data/rendered_frames" \
  --input-hdf5 "external_data/world_trajectory.h5" \
  --threads 96
```


## Video rendering

# Sliced

## 2:1 slices

Having a cross-section of the world trajectory data can be useful for understanding the terrain and how the water flows through it.

### Creating slices

Ensure that you are in the `smartsim/mini_app` directory and run the following command to create a slice of the world trajectory data at z=540. By default, the slices have a height of 1080px. The `--all-steps` flag indicates that you want to render all steps of the trajectory, and `--threads 96` specifies that you want to use 96 threads for rendering.

```bash
python3 render_slice.py \
  --input-hdf5 external_data/world_trajectory.h5 \
  --slice-z 540 \
  --output-dir external_data/rendered_slices/ \
  --all-steps \
  --threads 96
```

### Creating the combined 2:1 video


First, ensure that FFmpeg is available. On Claix just run `module load FFmpeg` to load the FFmpeg module.

Ensure that you are in the `smartsim/mini_app` directory and run the following command to create a combined video with the top view and the slice view.

```bash
./make_dual_view_video.sh \
  --top-pattern 'external_data/rendered_frames/frame_*.jpg' \
  --slice-pattern 'external_data/world_trajectory_slice_z0540_frames/frame_*.png' \
  --output external_data/10k_steps_dual.mp4 \
  --fps 60 \
  --threads 16
```


## mini-slices for combined 16:9

Previously, we combined two 2:1 images into a single 1:1 frame for the video ouput. As most screens are 16:9 (or close to it), we can instead create a 16:9 video by using a smaller slice of the world trajectory data. This will allow us to use the full width of the video frame for the slice, and only a portion of the height, which is more appropriate for a 16:9 aspect ratio.

### creating slices

Ensure that you are in the `smartsim/mini_app` directory and run the following command to create a slice of the world trajectory data at z=540, with a height range beginning at 40, and an image height of 135 (so range 40-175). The `--all-steps` flag indicates that you want to render all steps of the trajectory, and `--threads 96` specifies that you want to use 96 threads for rendering.

Ideally, use `salloc -N 1 --partition=devel --time=01:00:00 --exclusive` to allocate a node for this task, as it can be take long on a login node but isn't expensive enough to require a full job allocation to a compute node. Devel is just fine. The node has 96 cores, so you can use all of them for rendering.

```bash
python3 render_slice.py \
  --input-hdf5 external_data/world_trajectory.h5 \
  --slice-z 540 \
  --output-dir external_data/world_trajectory_slice_z0540_frames \
  --height-min 40 \
  --image-height 135 \
  --all-steps \
  --threads 96
```

### Creating the combined 16:9 video

First, ensure that FFmpeg is available. On Claix just run `module load FFmpeg` to load the FFmpeg module.

Ensure that you are in the `smartsim/mini_app` directory and run the following command to create a combined video with the top view and the slice view.

```bash
./make_dual_view_video.sh \
  --top-pattern 'external_data/rendered_frames/frame_*.jpg' \
  --slice-pattern 'external_data/world_trajectory_slice_z0540_frames/frame_*.png' \
  --output external_data/10k_steps_dual_16x9.mp4 \
  --width 2160 \
  --top-height 1080 \
  --bottom-height 135 \
  --fps 60 \
  --threads 16
```