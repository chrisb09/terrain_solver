# Configuration Analysis: SmartSim Parallel (c=0)

- **Framework / Provider**: `SMARTSIM`
- **Slurm Job ID**: `3449953`
- **Hardware Environment**: CLAIX-23 Heterogeneous (1x c23mm CPU Node with 24 Ranks + 1x c23g GPU Node with 1x NVIDIA H100)
- **Workload**: WaterCNN ($1920 \times 1080$, 22 timesteps)

## 1. Timestep Timing Statistics

| Metric | Duration (ms) |
|---|---|
| **Cold Start ML Step (Step 1)** | 5395.12 ms |
| **Warm ML Step Median** | **98.57 ms** |
| **Warm ML Step IQR** | 8.91 ms |
| **Warm ML Step Mean** | 101.97 ms |
| **Warm ML Step StdDev** | 11.30 ms |
| **Warm ML Step 95th Percentile** | 119.76 ms |
| **Regular Numerical Step Avg** | 1.97 ms |
| **Total Simulation Solve Time** | 81.0 s |

## 2. Resource Utilization

| Metric | Value |
|---|---|
| **Peak RSS (Max Rank)** | 417.9 MB |
| **Peak RSS (Sum over Ranks)** | 9505.0 MB |
| **InfiniBand Traffic (ib0 RX / TX)** | 92.69 MB / 1626.56 MB |

## 3. Performance Visualizations

### Timestep Progression (Cold Start vs Steady State)
![Step Progression](fig_step_progression.png)

### Warm Step Latency Distribution & Boxplot
![Latency Distribution](fig_step_latency_distribution.png)

### Memory Footprint Evolution
![Memory Profile](fig_memory_profile.png)

