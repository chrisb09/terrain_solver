# Configuration Analysis: AIxelerator Collective

- **Framework / Provider**: `AIX`
- **Slurm Job ID**: `3504577`
- **Hardware Environment**: CLAIX-23 Single GPU Node (1x c23g Node with 96 CPU Cores + 4x NVIDIA H100 GPUs)
- **Workload**: WaterCNN ($1920 \times 1080$, 22 timesteps)

## 1. Timestep Timing Statistics

| Metric | Duration (ms) |
|---|---|
| **Cold Start ML Step (Step 1)** | 18906.70 ms |
| **Warm ML Step Median** | **30.00 ms** |
| **Warm ML Step IQR** | 4.35 ms |
| **Warm ML Step Mean** | 29.78 ms |
| **Warm ML Step StdDev** | 2.57 ms |
| **Warm ML Step 95th Percentile** | 32.74 ms |
| **Regular Numerical Step Avg** | 3.13 ms |
| **Total Simulation Solve Time** | 102.0 s |

## 2. Resource Utilization

| Metric | Value |
|---|---|
| **Peak RSS (Max Rank)** | 913.2 MB |
| **Peak RSS (Sum over Ranks)** | 37209.5 MB |
| **InfiniBand Traffic (ib0 RX / TX)** | 0.60 MB / 0.22 MB |

## 3. Performance Visualizations

### Timestep Progression (Cold Start vs Steady State)
![Step Progression](fig_step_progression.png)

### Warm Step Latency Distribution & Boxplot
![Latency Distribution](fig_step_latency_distribution.png)

### Memory Footprint Evolution
![Memory Profile](fig_memory_profile.png)

