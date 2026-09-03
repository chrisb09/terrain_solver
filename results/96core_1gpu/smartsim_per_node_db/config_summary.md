# Configuration Analysis: SmartSim Per-Node DB

- **Framework / Provider**: `SMARTSIM`
- **Slurm Job ID**: `3504529`
- **Hardware Environment**: CLAIX-23 Single GPU Node (1x c23g Node with 96 CPU Cores, 1x NVIDIA H100 Active)
- **Workload**: WaterCNN ($1920 \times 1080$, 22 timesteps)

## 1. Timestep Timing Statistics

| Metric | Duration (ms) |
|---|---|
| **Cold Start ML Step (Step 1)** | 625.06 ms |
| **Warm ML Step Median** | **101.42 ms** |
| **Warm ML Step IQR** | 7.08 ms |
| **Warm ML Step Mean** | 107.16 ms |
| **Warm ML Step StdDev** | 18.16 ms |
| **Warm ML Step 95th Percentile** | 140.57 ms |
| **Regular Numerical Step Avg** | 4.43 ms |
| **Total Simulation Solve Time** | 48.0 s |

## 2. Resource Utilization

| Metric | Value |
|---|---|
| **Peak RSS (Max Rank)** | 420.2 MB |
| **Peak RSS (Sum over Ranks)** | 35052.6 MB |
| **InfiniBand Traffic (ib0 RX / TX)** | 0.37 MB / 0.06 MB |

## 3. Performance Visualizations

### Timestep Progression (Cold Start vs Steady State)
![Step Progression](fig_step_progression.png)

### Warm Step Latency Distribution & Boxplot
![Latency Distribution](fig_step_latency_distribution.png)

### Memory Footprint Evolution
![Memory Profile](fig_memory_profile.png)

