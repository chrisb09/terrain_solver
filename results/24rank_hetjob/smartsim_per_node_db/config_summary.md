# Configuration Analysis: SmartSim Per-Node DB

- **Framework / Provider**: `SMARTSIM`
- **Slurm Job ID**: `3449957`
- **Hardware Environment**: CLAIX-23 Heterogeneous (1x c23mm CPU Node with 24 Ranks + 1x c23g GPU Node with 1x NVIDIA H100)
- **Workload**: WaterCNN ($1920 \times 1080$, 22 timesteps)

## 1. Timestep Timing Statistics

| Metric | Duration (ms) |
|---|---|
| **Cold Start ML Step (Step 1)** | 327.24 ms |
| **Warm ML Step Median** | **94.88 ms** |
| **Warm ML Step IQR** | 22.96 ms |
| **Warm ML Step Mean** | 105.55 ms |
| **Warm ML Step StdDev** | 26.71 ms |
| **Warm ML Step 95th Percentile** | 147.18 ms |
| **Regular Numerical Step Avg** | 2.14 ms |
| **Total Simulation Solve Time** | 26.0 s |

## 2. Resource Utilization

| Metric | Value |
|---|---|
| **Peak RSS (Max Rank)** | 418.1 MB |
| **Peak RSS (Sum over Ranks)** | 9527.2 MB |
| **InfiniBand Traffic (ib0 RX / TX)** | 92.60 MB / 1630.27 MB |

## 3. Performance Visualizations

### Timestep Progression (Cold Start vs Steady State)
![Step Progression](fig_step_progression.png)

### Warm Step Latency Distribution & Boxplot
![Latency Distribution](fig_step_latency_distribution.png)

### Memory Footprint Evolution
![Memory Profile](fig_memory_profile.png)

