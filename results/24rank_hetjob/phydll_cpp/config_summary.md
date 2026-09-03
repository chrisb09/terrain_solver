# Configuration Analysis: PhyDLL C++

- **Framework / Provider**: `PHYDLL`
- **Slurm Job ID**: `3449978`
- **Hardware Environment**: CLAIX-23 Heterogeneous (1x c23mm CPU Node with 24 Ranks + 1x c23g GPU Node with 1x NVIDIA H100)
- **Workload**: WaterCNN ($1920 \times 1080$, 22 timesteps)

## 1. Timestep Timing Statistics

| Metric | Duration (ms) |
|---|---|
| **Cold Start ML Step (Step 1)** | 18678.40 ms |
| **Warm ML Step Median** | **504.76 ms** |
| **Warm ML Step IQR** | 4.18 ms |
| **Warm ML Step Mean** | 516.98 ms |
| **Warm ML Step StdDev** | 42.53 ms |
| **Warm ML Step 95th Percentile** | 578.88 ms |
| **Regular Numerical Step Avg** | 2.03 ms |
| **Total Simulation Solve Time** | 66.0 s |

## 2. Resource Utilization

| Metric | Value |
|---|---|
| **Peak RSS (Max Rank)** | 319.2 MB |
| **Peak RSS (Sum over Ranks)** | 7001.1 MB |
| **InfiniBand Traffic (ib0 RX / TX)** | 185.69 MB / 3232.58 MB |

## 3. Performance Visualizations

### Timestep Progression (Cold Start vs Steady State)
![Step Progression](fig_step_progression.png)

### Warm Step Latency Distribution & Boxplot
![Latency Distribution](fig_step_latency_distribution.png)

### Memory Footprint Evolution
![Memory Profile](fig_memory_profile.png)

