# Configuration Analysis: SmartSim Per-Node DB

- **Framework / Provider**: `SMARTSIM`
- **Slurm Job ID**: `3504563`
- **Hardware Environment**: CLAIX-23 Single GPU Node (1x c23g Node with 96 CPU Cores + 4x NVIDIA H100 GPUs)
- **Workload**: WaterCNN ($1920 \times 1080$, 22 timesteps)

## 1. Timestep Timing Statistics

| Metric | Duration (ms) |
|---|---|
| **Cold Start ML Step (Step 1)** | 363.74 ms |
| **Warm ML Step Median** | **108.24 ms** |
| **Warm ML Step IQR** | 20.86 ms |
| **Warm ML Step Mean** | 108.06 ms |
| **Warm ML Step StdDev** | 15.35 ms |
| **Warm ML Step 95th Percentile** | 128.75 ms |
| **Regular Numerical Step Avg** | 13.30 ms |
| **Total Simulation Solve Time** | 48.0 s |

## 2. Resource Utilization

| Metric | Value |
|---|---|
| **Peak RSS (Max Rank)** | 420.2 MB |
| **Peak RSS (Sum over Ranks)** | 35058.8 MB |
| **InfiniBand Traffic (ib0 RX / TX)** | 0.02 MB / 0.01 MB |

## 3. Performance Visualizations

### Timestep Progression (Cold Start vs Steady State)
![Step Progression](fig_step_progression.png)

### Warm Step Latency Distribution & Boxplot
![Latency Distribution](fig_step_latency_distribution.png)

### Memory Footprint Evolution
![Memory Profile](fig_memory_profile.png)

