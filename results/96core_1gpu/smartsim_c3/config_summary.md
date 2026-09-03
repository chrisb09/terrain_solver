# Configuration Analysis: SmartSim Chain 3 (c=3)

- **Framework / Provider**: `SMARTSIM`
- **Slurm Job ID**: `3504535`
- **Hardware Environment**: CLAIX-23 Single GPU Node (1x c23g Node with 96 CPU Cores, 1x NVIDIA H100 Active)
- **Workload**: WaterCNN ($1920 \times 1080$, 22 timesteps)

## 1. Timestep Timing Statistics

| Metric | Duration (ms) |
|---|---|
| **Cold Start ML Step (Step 1)** | 292.04 ms |
| **Warm ML Step Median** | **229.39 ms** |
| **Warm ML Step IQR** | 25.56 ms |
| **Warm ML Step Mean** | 226.17 ms |
| **Warm ML Step StdDev** | 24.87 ms |
| **Warm ML Step 95th Percentile** | 260.52 ms |
| **Regular Numerical Step Avg** | 4.93 ms |
| **Total Simulation Solve Time** | 51.0 s |

## 2. Resource Utilization

| Metric | Value |
|---|---|
| **Peak RSS (Max Rank)** | 419.5 MB |
| **Peak RSS (Sum over Ranks)** | 35064.4 MB |
| **InfiniBand Traffic (ib0 RX / TX)** | 0.04 MB / 0.02 MB |

## 3. Performance Visualizations

### Timestep Progression (Cold Start vs Steady State)
![Step Progression](fig_step_progression.png)

### Warm Step Latency Distribution & Boxplot
![Latency Distribution](fig_step_latency_distribution.png)

### Memory Footprint Evolution
![Memory Profile](fig_memory_profile.png)

