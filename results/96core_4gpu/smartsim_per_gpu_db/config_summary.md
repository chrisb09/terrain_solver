# Configuration Analysis: SmartSim Per-GPU DB

- **Framework / Provider**: `SMARTSIM`
- **Slurm Job ID**: `3504565`
- **Hardware Environment**: CLAIX-23 Single GPU Node (1x c23g Node with 96 CPU Cores + 4x NVIDIA H100 GPUs)
- **Workload**: WaterCNN ($1920 \times 1080$, 22 timesteps)

## 1. Timestep Timing Statistics

| Metric | Duration (ms) |
|---|---|
| **Cold Start ML Step (Step 1)** | 437.49 ms |
| **Warm ML Step Median** | **133.79 ms** |
| **Warm ML Step IQR** | 12.00 ms |
| **Warm ML Step Mean** | 132.09 ms |
| **Warm ML Step StdDev** | 10.40 ms |
| **Warm ML Step 95th Percentile** | 143.46 ms |
| **Regular Numerical Step Avg** | 4.18 ms |
| **Total Simulation Solve Time** | 89.0 s |

## 2. Resource Utilization

| Metric | Value |
|---|---|
| **Peak RSS (Max Rank)** | 421.7 MB |
| **Peak RSS (Sum over Ranks)** | 35061.0 MB |
| **InfiniBand Traffic (ib0 RX / TX)** | 0.04 MB / 0.02 MB |

## 3. Performance Visualizations

### Timestep Progression (Cold Start vs Steady State)
![Step Progression](fig_step_progression.png)

### Warm Step Latency Distribution & Boxplot
![Latency Distribution](fig_step_latency_distribution.png)

### Memory Footprint Evolution
![Memory Profile](fig_memory_profile.png)

