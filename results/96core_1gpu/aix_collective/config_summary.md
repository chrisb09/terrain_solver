# Configuration Analysis: AIxelerator Collective

- **Framework / Provider**: `AIX`
- **Slurm Job ID**: `3504538`
- **Hardware Environment**: CLAIX-23 Single GPU Node (1x c23g Node with 96 CPU Cores, 1x NVIDIA H100 Active)
- **Workload**: WaterCNN ($1920 \times 1080$, 22 timesteps)

## 1. Timestep Timing Statistics

| Metric | Duration (ms) |
|---|---|
| **Cold Start ML Step (Step 1)** | 26089.40 ms |
| **Warm ML Step Median** | **30.61 ms** |
| **Warm ML Step IQR** | 1.43 ms |
| **Warm ML Step Mean** | 32.50 ms |
| **Warm ML Step StdDev** | 7.57 ms |
| **Warm ML Step 95th Percentile** | 43.68 ms |
| **Regular Numerical Step Avg** | 2.39 ms |
| **Total Simulation Solve Time** | 108.0 s |

## 2. Resource Utilization

| Metric | Value |
|---|---|
| **Peak RSS (Max Rank)** | 913.6 MB |
| **Peak RSS (Sum over Ranks)** | 37207.0 MB |
| **InfiniBand Traffic (ib0 RX / TX)** | 0.61 MB / 0.24 MB |

## 3. Performance Visualizations

### Timestep Progression (Cold Start vs Steady State)
![Step Progression](fig_step_progression.png)

### Warm Step Latency Distribution & Boxplot
![Latency Distribution](fig_step_latency_distribution.png)

### Memory Footprint Evolution
![Memory Profile](fig_memory_profile.png)

