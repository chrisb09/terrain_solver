# Configuration Analysis: SmartSim Parallel (c=0)

- **Framework / Provider**: `SMARTSIM`
- **Slurm Job ID**: `3504561`
- **Hardware Environment**: CLAIX-23 Single GPU Node (1x c23g Node with 96 CPU Cores + 4x NVIDIA H100 GPUs)
- **Workload**: WaterCNN ($1920 \times 1080$, 22 timesteps)

## 1. Timestep Timing Statistics

| Metric | Duration (ms) |
|---|---|
| **Cold Start ML Step (Step 1)** | 348.57 ms |
| **Warm ML Step Median** | **101.88 ms** |
| **Warm ML Step IQR** | 11.52 ms |
| **Warm ML Step Mean** | 108.46 ms |
| **Warm ML Step StdDev** | 19.08 ms |
| **Warm ML Step 95th Percentile** | 142.96 ms |
| **Regular Numerical Step Avg** | 4.14 ms |
| **Total Simulation Solve Time** | 96.0 s |

## 2. Resource Utilization

| Metric | Value |
|---|---|
| **Peak RSS (Max Rank)** | 418.7 MB |
| **Peak RSS (Sum over Ranks)** | 34927.5 MB |
| **InfiniBand Traffic (ib0 RX / TX)** | 0.02 MB / 0.01 MB |

## 3. Performance Visualizations

### Timestep Progression (Cold Start vs Steady State)
![Step Progression](fig_step_progression.png)

### Warm Step Latency Distribution & Boxplot
![Latency Distribution](fig_step_latency_distribution.png)

### Memory Footprint Evolution
![Memory Profile](fig_memory_profile.png)

