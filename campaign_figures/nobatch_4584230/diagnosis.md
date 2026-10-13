# CPU Campaign Diagnosis

These findings concern the completed no-batching CPU campaign, not a new run.
No follow-up jobs were submitted as part of this investigation.

## SmartSim Batching

RedisAI BATCHSIZE is a leading-dimension aggregation threshold, not a hard
maximum. Whole requests may cross it. MINBATCHSIZE=0 does not wait for the
threshold to fill, and workers share the device queue rather than fixed rank
assignments. TOTAL_BS/TPQ is worth testing, but does not guarantee TPQ balanced
executions.

| Model | Total samples | TPQ=1 target | TPQ=12 target | TPQ=96 target |
|---|---:|---:|---:|---:|
| WaterCNN | 8294400 | 8294400 | 691200 | 86400 |
| Giant | 46080 | 46080 | 3840 | 480 |

Rank-local batches vary: WaterCNN 82368/85536/86112/89424; Giant 288/576.
At TPQ=96 the average-rank threshold can still aggregate smaller requests.
Use BATCHSIZE=0 for strictly one request per execution, or a threshold no
greater than the smallest local batch. Keep literal SmartSim default unbatched.

## AIx and PhyDLL CPU Batches

Both process a local tile as an existing tensor batch; neither does RedisAI
dynamic cross-request aggregation in this campaign. AIx's CPU service initializes
Torch's effective chunk size to its entire local host batch. The campaign's
--ml-batch-size 10000000 therefore does not induce CPU splitting.

PhyDLL retains TOML library.batch_size=500000, rather than the CLI's 10000000.
Its client supports splitting, but all local batches fit in one forward. There
are 96 solver processes and 96 DL processes with one-to-one source mapping.

## PhyDLL Output Send

Runtime logs and Cube payload metrics confirm active uniform_chunks transport:
18 input fields and one output field, with no symmetric output padding.
PhyDLL sends doubles: input payload 144*B bytes and output payload 8*B bytes.

Five-replicate mean milliseconds per DL rank per steady step:

| Component | WaterCNN | Giant |
|---|---:|---:|
| Forward | 210.677 | 6756.490 |
| Output-send phase | 109.958 | 4883.734 |
| Output-send child MPI_Barrier | 109.160 | 4883.576 |
| Output-send child MPI_Waitall | 0.467 | 0.00516 |

PhyDLL io_log has two optional barriers on its instance communicator. Output
field registration and send completion each log, giving four DL-communicator
barriers per steady frame. Faster DL ranks wait for slower forward ranks before
posting outputs. This is measured synchronization, not large output traffic or
double-counting. Disable PHYDLL_IO_LOG_BARRIERS=0 on both MPMD sides in a controlled
diagnostic. Removing it need not remove the slowest rank's inference time: waits
can move, and solver makespan still depends on stragglers.

The relevant code is extern/phydll/src/core/io.c:114-124 and kernel.c field/send
logging, under CPP-ML-Interface. The built library supports this existing switch.

## CPU Transfer Labels

dl_h2d wraps from_blob().clone(), view(), and to(CPU): clone is a real host
copy to owned tensor storage. dl_d2h wraps to(CPU).contiguous().view(): often
no bulk copy on CPU. Neither implies CUDA transfer. Float-to-double transport
conversion is measured separately in dl_output_reorder.

## AIx Slowdown and Monitoring

Old/new Giant pooled latency medians: 4.198/5.978 seconds. Mean rank-local
steady forward: 3.537526/3.566107 seconds. Typical rank forward times are nearly
unchanged, but the slow-rank tail increased. PSS scanning is a credible suspect:
Giant's ~227 GiB process set requires costly smaps_rollup sweeps. It is not proven
causal; old/new nodes and binaries are not identical controlled treatments.

The new jobs request exclusive nodes, but MPI ranks are unbound. Exclusive
allocation does not eliminate interference from this job's own sampler or ensure
Torch thread counts. PhyDLL explicitly uses one intra/inter-op thread; AIx does
not establish those counts in the inspected implementation.

Recommended next experiment: same-node, same-binary AIx monitor-off/on ABBA runs;
PhyDLL logging-barriers-off/on with monitoring held fixed; then a small SmartSim
TOTAL_BS/TPQ sweep. Do not start another full campaign until these controls pass.

## Phase Figures

Vertical simplified grouped and stacked figures include AIx and six configs.
They show per-rank steady-region means, not barrier-bracketed step makespan.
PhyDLL uses coherent DL-side receive/forward/send/other. SmartSim uses coherent
client-side put/model request/retrieval/other. Other is disjoint self-time, not
end-to-end makespan minus cross-process means. Such a subtraction can produce
large synchronization/imbalance residuals and is not a transport measurement.
