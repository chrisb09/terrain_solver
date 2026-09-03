# Configuration Analysis: AIxelerator P2P

- **Framework / Provider**: `AIX`
- **Slurm Job ID**: `3504580`
- **Hardware Environment**: CLAIX-23 Single GPU Node (1x c23g Node with 96 CPU Cores + 4x NVIDIA H100 GPUs)
- **Workload**: WaterCNN ($1920 \times 1080$, 22 timesteps)

## 1. Timestep Timing Statistics

| Metric | Duration (ms) |
|---|---|
| **Cold Start ML Step (Step 1)** | 2264.23 ms |
| **Warm ML Step Median** | **29.69 ms** |
| **Warm ML Step IQR** | 2.29 ms |
| **Warm ML Step Mean** | 30.37 ms |
| **Warm ML Step StdDev** | 1.57 ms |
| **Warm ML Step 95th Percentile** | 32.74 ms |
| **Regular Numerical Step Avg** | 5.90 ms |
| **Total Simulation Solve Time** | 78.0 s |

## 2. Resource Utilization

| Metric | Value |
|---|---|
| **Peak RSS (Max Rank)** | 914.8 MB |
| **Peak RSS (Sum over Ranks)** | 37183.0 MB |
| **InfiniBand Traffic (ib0 RX / TX)** | 0.18 MB / 0.14 MB |

## 3. Performance Visualizations

### Timestep Progression (Cold Start vs Steady State)
![Step Progression](fig_step_progression.png)

### Warm Step Latency Distribution & Boxplot
![Latency Distribution](fig_step_latency_distribution.png)

### Memory Footprint Evolution
![Memory Profile](fig_memory_profile.png)

