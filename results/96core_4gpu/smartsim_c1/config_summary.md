# Configuration Analysis: SmartSim Chain 1 (c=1)

- **Framework / Provider**: `SMARTSIM`
- **Slurm Job ID**: `3504567`
- **Hardware Environment**: CLAIX-23 Single GPU Node (1x c23g Node with 96 CPU Cores + 4x NVIDIA H100 GPUs)
- **Workload**: WaterCNN ($1920 \times 1080$, 22 timesteps)

## 1. Timestep Timing Statistics

| Metric | Duration (ms) |
|---|---|
| **Cold Start ML Step (Step 1)** | 5475.10 ms |
| **Warm ML Step Median** | **572.86 ms** |
| **Warm ML Step IQR** | 55.91 ms |
| **Warm ML Step Mean** | 598.00 ms |
| **Warm ML Step StdDev** | 57.36 ms |
| **Warm ML Step 95th Percentile** | 695.56 ms |
| **Regular Numerical Step Avg** | 4.95 ms |
| **Total Simulation Solve Time** | 113.0 s |

## 2. Resource Utilization

| Metric | Value |
|---|---|
| **Peak RSS (Max Rank)** | 418.7 MB |
| **Peak RSS (Sum over Ranks)** | 34919.6 MB |
| **InfiniBand Traffic (ib0 RX / TX)** | 402.94 MB / 1.09 MB |

## 3. Performance Visualizations

### Timestep Progression (Cold Start vs Steady State)
![Step Progression](fig_step_progression.png)

### Warm Step Latency Distribution & Boxplot
![Latency Distribution](fig_step_latency_distribution.png)

### Memory Footprint Evolution
![Memory Profile](fig_memory_profile.png)

