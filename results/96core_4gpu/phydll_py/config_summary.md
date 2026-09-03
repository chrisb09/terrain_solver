# Configuration Analysis: PhyDLL Python

- **Framework / Provider**: `PHYDLL`
- **Slurm Job ID**: `3504590`
- **Hardware Environment**: CLAIX-23 Single GPU Node (1x c23g Node with 96 CPU Cores + 4x NVIDIA H100 GPUs)
- **Workload**: WaterCNN ($1920 \times 1080$, 22 timesteps)

## 1. Timestep Timing Statistics

| Metric | Duration (ms) |
|---|---|
| **Cold Start ML Step (Step 1)** | 5620.45 ms |
| **Warm ML Step Median** | **411.04 ms** |
| **Warm ML Step IQR** | 26.93 ms |
| **Warm ML Step Mean** | 408.52 ms |
| **Warm ML Step StdDev** | 27.05 ms |
| **Warm ML Step 95th Percentile** | 443.54 ms |
| **Regular Numerical Step Avg** | 1.35 ms |
| **Total Simulation Solve Time** | 19.0 s |

## 2. Resource Utilization

| Metric | Value |
|---|---|
| **Peak RSS (Max Rank)** | 286.6 MB |
| **Peak RSS (Sum over Ranks)** | 24857.4 MB |
| **InfiniBand Traffic (ib0 RX / TX)** | 0.04 MB / 0.03 MB |

## 3. Performance Visualizations

### Timestep Progression (Cold Start vs Steady State)
![Step Progression](fig_step_progression.png)

### Warm Step Latency Distribution & Boxplot
![Latency Distribution](fig_step_latency_distribution.png)

### Memory Footprint Evolution
![Memory Profile](fig_memory_profile.png)

