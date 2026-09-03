# Configuration Analysis: PhyDLL Python

- **Framework / Provider**: `PHYDLL`
- **Slurm Job ID**: `3450422`
- **Hardware Environment**: CLAIX-23 Heterogeneous (1x c23mm CPU Node with 24 Ranks + 1x c23g GPU Node with 1x NVIDIA H100)
- **Workload**: WaterCNN ($1920 \times 1080$, 22 timesteps)

## 1. Timestep Timing Statistics

| Metric | Duration (ms) |
|---|---|
| **Cold Start ML Step (Step 1)** | 792.70 ms |
| **Warm ML Step Median** | **353.90 ms** |
| **Warm ML Step IQR** | 12.52 ms |
| **Warm ML Step Mean** | 356.44 ms |
| **Warm ML Step StdDev** | 10.07 ms |
| **Warm ML Step 95th Percentile** | 371.49 ms |
| **Regular Numerical Step Avg** | 1.99 ms |
| **Total Simulation Solve Time** | 50.0 s |

## 2. Resource Utilization

| Metric | Value |
|---|---|
| **Peak RSS (Max Rank)** | 318.5 MB |
| **Peak RSS (Sum over Ranks)** | 6972.0 MB |
| **InfiniBand Traffic (ib0 RX / TX)** | 185.55 MB / 3233.43 MB |

## 3. Performance Visualizations

### Timestep Progression (Cold Start vs Steady State)
![Step Progression](fig_step_progression.png)

### Warm Step Latency Distribution & Boxplot
![Latency Distribution](fig_step_latency_distribution.png)

### Memory Footprint Evolution
![Memory Profile](fig_memory_profile.png)

