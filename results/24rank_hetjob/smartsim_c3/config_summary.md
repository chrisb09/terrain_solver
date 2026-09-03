# Configuration Analysis: SmartSim Chain 3 (c=3)

- **Framework / Provider**: `SMARTSIM`
- **Slurm Job ID**: `3449964`
- **Hardware Environment**: CLAIX-23 Heterogeneous (1x c23mm CPU Node with 24 Ranks + 1x c23g GPU Node with 1x NVIDIA H100)
- **Workload**: WaterCNN ($1920 \times 1080$, 22 timesteps)

## 1. Timestep Timing Statistics

| Metric | Duration (ms) |
|---|---|
| **Cold Start ML Step (Step 1)** | 271.38 ms |
| **Warm ML Step Median** | **114.20 ms** |
| **Warm ML Step IQR** | 6.78 ms |
| **Warm ML Step Mean** | 112.94 ms |
| **Warm ML Step StdDev** | 5.79 ms |
| **Warm ML Step 95th Percentile** | 120.58 ms |
| **Regular Numerical Step Avg** | 2.11 ms |
| **Total Simulation Solve Time** | 26.0 s |

## 2. Resource Utilization

| Metric | Value |
|---|---|
| **Peak RSS (Max Rank)** | 418.1 MB |
| **Peak RSS (Sum over Ranks)** | 9524.9 MB |
| **InfiniBand Traffic (ib0 RX / TX)** | 93.08 MB / 1619.88 MB |

## 3. Performance Visualizations

### Timestep Progression (Cold Start vs Steady State)
![Step Progression](fig_step_progression.png)

### Warm Step Latency Distribution & Boxplot
![Latency Distribution](fig_step_latency_distribution.png)

### Memory Footprint Evolution
![Memory Profile](fig_memory_profile.png)

