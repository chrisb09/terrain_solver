#!/usr/bin/env zsh

# 96-Core Exclusive CPU Benchmark Suite (10 Replicates):
# Evaluates 96-core exclusive node allocations across 3 scaling strategies:
# 1. Dynamic Balanced: TPQ = floor(96 / db_nodes), Intra = ceil(96 / TPQ)  (n = 1, 3, 4, 5, 6, 7, 8)
# 2. TPQ-Only Baseline: TPQ = 96, Intra = 1                               (n = 3, 4, 5, 6, 7, 8)
# 3. Intra-Only Baseline: TPQ = 1, Intra = 96                              (n = 3, 4, 5, 6, 7, 8)
#
# Grid: 1440x960 | Ranks: 96 (12x8 grid) | Allocation: Exclusive c23mm nodes
# All 190 jobs are chained with dependencies to prevent port collisions.

set -euo pipefail

SCRIPT_DIR="${0:A:h}"
RUNNER="${SCRIPT_DIR}/run_key_balancing_cpu_debug.sh"
MANIFEST_FILE="${SCRIPT_DIR}/cpu_96core_suite_jobs.txt"
CSV_MANIFEST="${SCRIPT_DIR}/cpu_96core_suite_manifest.csv"
BUILD_DIR="build_key_balance_direct"
MPI_RANKS=96
ALLOC_CPUS_PER_NODE=96
DB_CPU_CORES=96
TIME_LIMIT="${TIME_LIMIT_ENV:-00:30:00}"
RANK_GRID_X=12
RANK_GRID_Z=8
REPLICATES=10

if [[ ! -x "${SCRIPT_DIR}/solver_cpp/${BUILD_DIR}/terrain_solver" ]]; then
  print -u2 "Missing normal direct build: solver_cpp/${BUILD_DIR}/terrain_solver"
  print -u2 "Run ./build_key_balancing.sh direct first."
  exit 1
fi

typeset -A existing_submissions
previous_job_id=""

if [[ -f "${CSV_MANIFEST}" ]]; then
  print "Existing manifest found. Resume mode active."
  while IFS=, read -r rep db_nodes strat tpq intra job_id; do
    [[ "${rep}" == "replicate" ]] && continue
    [[ "${job_id}" =~ "^[0-9]+$" ]] || continue
    key="${rep}_${db_nodes}_${strat}_${tpq}_${intra}"
    existing_submissions[${key}]="${job_id}"
    previous_job_id="${job_id}"
  done < "${CSV_MANIFEST}"
else
  print "replicate,db_nodes,strategy,tpq,intra,job_id" > "${CSV_MANIFEST}"
fi

print "Submitting 96-Core Exclusive CPU ML Scaling Suite (190 jobs total)..."
print "Grid: 1440x960 | Ranks: 96 (12x8) | Node Allocation: Exclusive 96 cores/node | Wall time: ${TIME_LIMIT}"

job_count=${#existing_submissions}

for rep in $(seq 1 "${REPLICATES}"); do
  typeset -a strategies
  if (( rep % 3 == 1 )); then
    strategies=("dynamic" "tpq_only" "intra_only")
  elif (( rep % 3 == 2 )); then
    strategies=("intra_only" "dynamic" "tpq_only")
  else
    strategies=("tpq_only" "intra_only" "dynamic")
  fi

  for strat in "${strategies[@]}"; do
    case "${strat}" in
      dynamic)
        node_list=(1 3 4 5 6 7 8)
        ;;
      tpq_only|intra_only)
        node_list=(3 4 5 6 7 8)
        ;;
    esac

    for db_nodes in "${node_list[@]}"; do
      case "${strat}" in
        dynamic)
          tpq=$(( DB_CPU_CORES / db_nodes ))
          if (( tpq < 1 )); then tpq=1; fi
          intra=$(( (DB_CPU_CORES + tpq - 1) / tpq ))
          ;;
        tpq_only)
          tpq=96
          intra=1
          ;;
        intra_only)
          tpq=1
          intra=96
          ;;
      esac

      key="${rep}_${db_nodes}_${strat}_${tpq}_${intra}"
      if [[ -n "${existing_submissions[${key}]:-}" ]]; then
        previous_job_id="${existing_submissions[${key}]}"
        continue
      fi

      job_name="c96_r${rep}_n${db_nodes}_${strat}_t${tpq}_i${intra}"
      exports="ALL,USE_SCOREP_ENV=0,USE_SMARTSIM=1,ML_INTERFACE_ENV=smartsim,SKIP_COMPILE_ENV=1,COMPILE_OUTPUT_PATH_ENV=${BUILD_DIR},USE_LOCAL_RUNTIME_STAGE_ENV=0,USE_LOCAL_MODEL_CACHE_ENV=1,FORCE_FRESH_RUN_ENV=1,OVERWRITE_OUTPUT_ENV=1,SKIP_RENDERING_ENV=1,TOTAL_STEPS_ENV=10,SAVE_EVERY_ENV=10,MPI_RANKS_ENV=${MPI_RANKS},DB_NODES_ENV=${db_nodes},ML_INFERENCE_CPU_CORES_ENV=${DB_CPU_CORES},SMARTSIM_DEDICATED_DB_NODES_ENV=1,SMARTSIM_PIN_DB_NODELIST=1,RANK_GRID_X_ENV=${RANK_GRID_X},RANK_GRID_Z_ENV=${RANK_GRID_Z},TARGET_WIDTH_ENV=1440,TARGET_HEIGHT_ENV=960,CHUNK_SIZE_ENV=12,ML_BATCH_SIZE_ENV=50000,MODEL_NAME_ENV=benchmark_giant_mlp,MODEL_BACKEND_ENV=TORCH,SMARTSIM_BALANCED_KEYS=1,SMARTSIM_INTRA_OP_THREADS=${intra},SMARTSIM_INTER_OP_THREADS=1,SMARTSIM_THREADS_PER_QUEUE=${tpq},JOB_NAME_ENV=${job_name}"

      dependency_args=()
      if [[ -n "${previous_job_id}" ]]; then
        dependency_args=("--dependency=afterany:${previous_job_id}")
      fi

      submit_raw=""
      if ! submit_raw=$(sbatch --parsable \
        "${dependency_args[@]}" \
        --export="${exports}" \
        --partition=c23mm \
        --exclusive \
        --nodes=$((db_nodes + 1)) \
        --ntasks-per-node=1 \
        --cpus-per-task="${ALLOC_CPUS_PER_NODE}" \
        --mem=0 \
        --time="${TIME_LIMIT}" \
        "${RUNNER}" 2>&1); then
        print -u2 "Slurm submit limit reached (${submit_raw}). Submitted ${job_count}/190 jobs so far."
        print -u2 "Re-run ./submit_key_balancing_96core_cpu_suite.sh once queued jobs complete to submit remaining jobs."
        exit 0
      fi

      # Parse purely numeric job ID from sbatch output
      parsed_job_id=$(print -r -- "${submit_raw}" | grep -oE '[0-9]+' | tail -n 1)
      if [[ -z "${parsed_job_id}" ]]; then
        print -u2 "Failed to parse job ID from sbatch output: ${submit_raw}"
        exit 1
      fi

      previous_job_id="${parsed_job_id}"
      existing_submissions[${key}]="${previous_job_id}"
      job_count=$((job_count + 1))
      print "Submitted [${job_count}/190] Rep ${rep} n=${db_nodes} ${strat} (TPQ=${tpq}, Intra=${intra}): Job ID ${previous_job_id}"
      print "${db_nodes} ${strat} ${previous_job_id} tpq=${tpq} intra=${intra} rep=${rep}" >> "${MANIFEST_FILE}"
      print "${rep},${db_nodes},${strat},${tpq},${intra},${previous_job_id}" >> "${CSV_MANIFEST}"
    done
  done
done

print "All 190 jobs in the 96-core CPU suite submitted successfully."
print "Manifests saved to ${MANIFEST_FILE} and ${CSV_MANIFEST}."
