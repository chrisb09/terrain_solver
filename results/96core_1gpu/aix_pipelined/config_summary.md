# Configuration Analysis: AIxelerator P2P

- **Framework / Provider**: `AIX`
- **Slurm Job ID**: `3504541`
- **Hardware Environment**: CLAIX-23 Single GPU Node (1x c23g Node with 96 CPU Cores, 1x NVIDIA H100 Active)
- **Workload**: WaterCNN ($1920 \times 1080$, 22 timesteps)

## 1. Timestep Timing Statistics

| Metric | Duration (ms) |
|---|---|
| **Cold Start ML Step (Step 1)** | 22381.60 ms |
| **Warm ML Step Median** | **32.42 ms** |
| **Warm ML Step IQR** | 2.00 ms |
| **Warm ML Step Mean** | 33.47 ms |
| **Warm ML Step StdDev** | 2.21 ms |
| **Warm ML Step 95th Percentile** | 37.37 ms |
| **Regular Numerical Step Avg** | 4.04 ms |
| **Total Simulation Solve Time** | 156.0 s |

## 2. Resource Utilization

| Metric | Value |
|---|---|
| **Peak RSS (Max Rank)** | 913.0 MB |
| **Peak RSS (Sum over Ranks)** | 37047.4 MB |
| **InfiniBand Traffic (ib0 RX / TX)** | 0.34 MB / 0.22 MB |

## 3. Performance Visualizations

### Timestep Progression (Cold Start vs Steady State)
![Step Progression](fig_step_progression.png)

### Warm Step Latency Distribution & Boxplot
![Latency Distribution](fig_step_latency_distribution.png)

### Memory Footprint Evolution
![Memory Profile](fig_memory_profile.png)

