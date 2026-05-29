#!/bin/bash
set -euo pipefail

# Paper-style signal propagation:
# - p-dimensional unit signal at a random source node, zeros elsewhere
# - propagation is measured against total effective resistance
# - runs are repeated for classical MPNNs and NSD sheaf settings

PYTHON_BIN=${PYTHON_BIN:-python}
WANDB_PROJECT=${WANDB_PROJECT:-on-oversquashing-signal-propagation}
WANDB_ENTITY=${WANDB_ENTITY:-}
WANDB_MODE=${WANDB_MODE:-online}
WANDB_GROUP=${WANDB_GROUP:-signal-propagation-paper-classical-vs-sheaf}
RESULTS_FILE=${RESULTS_FILE:-simulation_stats/signal_propagation_results.csv}
MAX_PARALLEL_JOBS=${MAX_PARALLEL_JOBS:-1}
SKIP_COMPLETED=${SKIP_COMPLETED:-1}
CACHE_GRAPH_METRICS=${CACHE_GRAPH_METRICS:-1}
TORCH_NUM_THREADS=${TORCH_NUM_THREADS:-1}
TORCH_NUM_INTEROP_THREADS=${TORCH_NUM_INTEROP_THREADS:-1}

export OMP_NUM_THREADS=${OMP_NUM_THREADS:-${TORCH_NUM_THREADS}}
export MKL_NUM_THREADS=${MKL_NUM_THREADS:-${TORCH_NUM_THREADS}}
export OPENBLAS_NUM_THREADS=${OPENBLAS_NUM_THREADS:-${TORCH_NUM_THREADS}}
export NUMEXPR_NUM_THREADS=${NUMEXPR_NUM_THREADS:-${TORCH_NUM_THREADS}}

DATASETS=${DATASETS:-"PROTEINS NCI1 PTC ENZYMES"}
ROOT=${ROOT:-data/tu}
CLASSICAL_MODELS=${CLASSICAL_MODELS:-"gcn gat gin sage"}
CLASSICAL_HIDDEN_DIM=${CLASSICAL_HIDDEN_DIM:-64}
SHEAF_STALK_DIMS=${SHEAF_STALK_DIMS:-"1 2 3 5"}
SHEAF_HIDDEN_DIMS=${SHEAF_HIDDEN_DIMS:-"16 64 128 200"}
LAYERS=${LAYERS:-10}
SIGNAL_DIM=${SIGNAL_DIM:-5}
MAX_GRAPHS=${MAX_GRAPHS:-}
NUM_SOURCES=${NUM_SOURCES:-10}
SEEDS=${SEEDS:-"808"}
RUN_CLASSICAL=${RUN_CLASSICAL:-1}
RUN_SHEAF=${RUN_SHEAF:-1}
PIDS=()

ENTITY_ARG=""
if [[ -n "${WANDB_ENTITY}" && "${WANDB_ENTITY}" != "none" ]]; then
    ENTITY_ARG="--entity ${WANDB_ENTITY}"
fi

MAX_GRAPHS_ARG=""
if [[ -n "${MAX_GRAPHS}" ]]; then
    MAX_GRAPHS_ARG="--max_graphs ${MAX_GRAPHS}"
fi

completed_run() {
    local run_name=$1
    if [[ "${SKIP_COMPLETED}" != "1" || ! -s "${RESULTS_FILE}" ]]; then
        return 1
    fi
    grep -Fq ",${run_name}" "${RESULTS_FILE}"
}

wait_for_slot() {
    if (( MAX_PARALLEL_JOBS <= 1 )); then
        return 0
    fi

    local active
    active=$(jobs -pr | wc -l | tr -d '[:space:]')
    while (( active >= MAX_PARALLEL_JOBS )); do
        sleep 1
        active=$(jobs -pr | wc -l | tr -d '[:space:]')
    done
}

schedule_signal() {
    if (( MAX_PARALLEL_JOBS <= 1 )); then
        run_signal "$@"
    else
        wait_for_slot
        run_signal "$@" &
        PIDS+=("$!")
    fi
}

run_signal() {
    local dataset=$1
    local seed=$2
    local model=$3
    local hidden_dim=$4
    local experiment_label=$5
    local sheaf_variant=${6:-general}
    local stalk_dim=${7:-1}
    local normalize_output=${8:-1}
    local add_self_loops=${9:-1}

    local run_name="signal-${dataset}-${experiment_label}-p${SIGNAL_DIM}-layers${LAYERS}-seed${seed}"
    local raw_file="signal_propagation_results/${run_name}_raw.csv"
    if completed_run "${run_name}"; then
        echo "Skipping completed ${run_name}"
        return 0
    fi

    "${PYTHON_BIN}" exp/signal_propagation.py \
        --dataset "${dataset}" \
        --root "${ROOT}" \
        --models "${model}" \
        --layers "${LAYERS}" \
        --signal_dim "${SIGNAL_DIM}" \
        --hidden_dim "${hidden_dim}" \
        --num_sources "${NUM_SOURCES}" \
        --seed "${seed}" \
        --stalk_dim "${stalk_dim}" \
        --sheaf_variant "${sheaf_variant}" \
        --sheaf_normalize_output "${normalize_output}" \
        --sheaf_add_self_loops "${add_self_loops}" \
        --wandb_project "${WANDB_PROJECT}" \
        --wandb_group "${WANDB_GROUP}" \
        --wandb_mode "${WANDB_MODE}" \
        --wandb_tags "paper-signal-propagation,${dataset},${experiment_label}" \
        --wandb_run_name "${run_name}" \
        --results_file "${RESULTS_FILE}" \
        --raw_results_file "${raw_file}" \
        --experiment_label "${experiment_label}" \
        --cache_graph_metrics "${CACHE_GRAPH_METRICS}" \
        --torch_num_threads "${TORCH_NUM_THREADS}" \
        --torch_num_interop_threads "${TORCH_NUM_INTEROP_THREADS}" \
        ${ENTITY_ARG} \
        ${MAX_GRAPHS_ARG}
}

for dataset in ${DATASETS}; do
    for seed in ${SEEDS}; do
        if [[ "${RUN_CLASSICAL}" == "1" ]]; then
            for model in ${CLASSICAL_MODELS}; do
                schedule_signal \
                    "${dataset}" "${seed}" "${model}" "${CLASSICAL_HIDDEN_DIM}" \
                    "classic-${model}-h${CLASSICAL_HIDDEN_DIM}"
            done
        fi

        if [[ "${RUN_SHEAF}" == "1" ]]; then
            for stalk_dim in ${SHEAF_STALK_DIMS}; do
                for hidden_dim in ${SHEAF_HIDDEN_DIMS}; do
                    schedule_signal \
                        "${dataset}" "${seed}" nsd "${hidden_dim}" \
                        "nsd-orthogonal-nonnorm-noattn-stalk${stalk_dim}-h${hidden_dim}" \
                        orthogonal "${stalk_dim}" 0 0

                    schedule_signal \
                        "${dataset}" "${seed}" nsd "${hidden_dim}" \
                        "nsd-general-norm-stalk${stalk_dim}-h${hidden_dim}" \
                        general "${stalk_dim}" 1 1

                    schedule_signal \
                        "${dataset}" "${seed}" nsd "${hidden_dim}" \
                        "nsd-general-nonnorm-stalk${stalk_dim}-h${hidden_dim}" \
                        general "${stalk_dim}" 0 1
                done
            done
        fi
    done
done

if (( MAX_PARALLEL_JOBS > 1 )); then
    failed=0
    for pid in "${PIDS[@]}"; do
        if ! wait "${pid}"; then
            failed=1
        fi
    done
    if (( failed != 0 )); then
        exit 1
    fi
fi
