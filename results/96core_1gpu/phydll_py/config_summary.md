# Configuration Analysis: PhyDLL Python

- **Framework / Provider**: `PHYDLL`
- **Slurm Job ID**: `3504551`
- **Hardware Environment**: CLAIX-23 Single GPU Node (1x c23g Node with 96 CPU Cores, 1x NVIDIA H100 Active)
- **Workload**: WaterCNN ($1920 \times 1080$, 22 timesteps)

## 1. Timestep Timing Statistics

| Metric | Duration (ms) |
|---|---|
| **Cold Start ML Step (Step 1)** | 3937.72 ms |
| **Warm ML Step Median** | **814.88 ms** |
| **Warm ML Step IQR** | 60.18 ms |
| **Warm ML Step Mean** | 821.78 ms |
| **Warm ML Step StdDev** | 92.94 ms |
| **Warm ML Step 95th Percentile** | 959.56 ms |
| **Regular Numerical Step Avg** | 1.86 ms |
| **Total Simulation Solve Time** | 27.0 s |

## 2. Resource Utilization

| Metric | Value |
|---|---|
| **Peak RSS (Max Rank)** | 297.4 MB |
| **Peak RSS (Sum over Ranks)** | 25121.0 MB |
| **InfiniBand Traffic (ib0 RX / TX)** | 21.36 MB / 0.29 MB |

## 3. Performance Visualizations

### Timestep Progression (Cold Start vs Steady State)
![Step Progression](fig_step_progression.png)

### Warm Step Latency Distribution & Boxplot
![Latency Distribution](fig_step_latency_distribution.png)

### Memory Footprint Evolution
![Memory Profile](fig_memory_profile.png)

