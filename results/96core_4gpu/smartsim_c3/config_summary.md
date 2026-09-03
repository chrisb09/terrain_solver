# Configuration Analysis: SmartSim Chain 3 (c=3)

- **Framework / Provider**: `SMARTSIM`
- **Slurm Job ID**: `3504573`
- **Hardware Environment**: CLAIX-23 Single GPU Node (1x c23g Node with 96 CPU Cores + 4x NVIDIA H100 GPUs)
- **Workload**: WaterCNN ($1920 \times 1080$, 22 timesteps)

## 1. Timestep Timing Statistics

| Metric | Duration (ms) |
|---|---|
| **Cold Start ML Step (Step 1)** | 5316.16 ms |
| **Warm ML Step Median** | **220.18 ms** |
| **Warm ML Step IQR** | 15.40 ms |
| **Warm ML Step Mean** | 227.06 ms |
| **Warm ML Step StdDev** | 25.72 ms |
| **Warm ML Step 95th Percentile** | 272.42 ms |
| **Regular Numerical Step Avg** | 4.76 ms |
| **Total Simulation Solve Time** | 111.0 s |

## 2. Resource Utilization

| Metric | Value |
|---|---|
| **Peak RSS (Max Rank)** | 418.0 MB |
| **Peak RSS (Sum over Ranks)** | 34922.0 MB |
| **InfiniBand Traffic (ib0 RX / TX)** | 403.08 MB / 1.25 MB |

## 3. Performance Visualizations

### Timestep Progression (Cold Start vs Steady State)
![Step Progression](fig_step_progression.png)

### Warm Step Latency Distribution & Boxplot
![Latency Distribution](fig_step_latency_distribution.png)

### Memory Footprint Evolution
![Memory Profile](fig_memory_profile.png)

