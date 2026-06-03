#!/bin/bash
set -euo pipefail

# Paper-style graph transfer:
# - 5-dimensional one-hot target label
# - unit features on auxiliary nodes and zero source node are generated in data/ring_transfer.py
# - MPNN depth equals source-target distance
# - topologies: Ring, CrossedRing, CliquePath (implemented as LOLLIPOP)

PYTHON_BIN=${PYTHON_BIN:-python}
WANDB_PROJECT=${WANDB_PROJECT:-on-oversquashing-transfer}
WANDB_ENTITY=${WANDB_ENTITY:-}
WANDB_MODE=${WANDB_MODE:-online}
WANDB_GROUP=${WANDB_GROUP:-graph-transfer-paper-classical-vs-sheaf-runpod}
RESULTS_FILE=${RESULTS_FILE:-simulation_stats/graph_transfer_results_runpod.csv}
RUN_LOG_DIR=${RUN_LOG_DIR:-logs/graph_transfer_runpod}
GPU_IDS=${GPU_IDS:-auto}
JOBS_PER_GPU=${JOBS_PER_GPU:-1}
MAX_PARALLEL_JOBS=${MAX_PARALLEL_JOBS:-auto}
SKIP_COMPLETED=${SKIP_COMPLETED:-1}
CACHE_SYNTHETIC_DATASET=${CACHE_SYNTHETIC_DATASET:-1}
TORCH_NUM_THREADS=${TORCH_NUM_THREADS:-1}
TORCH_NUM_INTEROP_THREADS=${TORCH_NUM_INTEROP_THREADS:-1}

export OMP_NUM_THREADS=${OMP_NUM_THREADS:-${TORCH_NUM_THREADS}}
export MKL_NUM_THREADS=${MKL_NUM_THREADS:-${TORCH_NUM_THREADS}}
export OPENBLAS_NUM_THREADS=${OPENBLAS_NUM_THREADS:-${TORCH_NUM_THREADS}}
export NUMEXPR_NUM_THREADS=${NUMEXPR_NUM_THREADS:-${TORCH_NUM_THREADS}}

CLASSICAL_MODELS=${CLASSICAL_MODELS:-"gcn gat gin sage"}
CLASSICAL_HIDDEN_DIM=${CLASSICAL_HIDDEN_DIM:-64}
SHEAF_STALK_DIMS=${SHEAF_STALK_DIMS:-"1 3 5"}
SHEAF_HIDDEN_DIMS=${SHEAF_HIDDEN_DIMS:-"10 50 100 200"}
SEEDS=${SEEDS:-"43"}
SIZES=${SIZES:-"2 6 10 20 30"}

EPOCHS=${EPOCHS:-100}
BATCH_SIZE=${BATCH_SIZE:-128}
TRAIN_SIZE=${TRAIN_SIZE:-5000}
TEST_SIZE=${TEST_SIZE:-500}
FEATURE_DIM=${FEATURE_DIM:-5}
RUN_CLASSICAL=${RUN_CLASSICAL:-1}
RUN_SHEAF=${RUN_SHEAF:-1}
PIDS=()
SLOT_GPUS=()
SLOT_PIDS=()
FAILED=0
GPU_SLOT_RESULT=""

ENTITY_ARG=""
if [[ -n "${WANDB_ENTITY}" && "${WANDB_ENTITY}" != "none" ]]; then
    ENTITY_ARG="--entity ${WANDB_ENTITY}"
fi

detect_gpu_ids() {
    if [[ "${GPU_IDS}" != "auto" ]]; then
        echo "${GPU_IDS}" | tr ',' ' '
        return 0
    fi

    if [[ -n "${CUDA_VISIBLE_DEVICES:-}" && "${CUDA_VISIBLE_DEVICES}" != "NoDevFiles" ]]; then
        echo "${CUDA_VISIBLE_DEVICES}" | tr ',' ' '
        return 0
    fi

    if command -v nvidia-smi >/dev/null 2>&1; then
        nvidia-smi --query-gpu=index --format=csv,noheader | tr -d ' ' | tr '\n' ' '
        return 0
    fi

    echo ""
}

initialise_gpu_slots() {
    local gpu_ids
    gpu_ids=$(detect_gpu_ids)

    if [[ -z "${gpu_ids// }" ]]; then
        if [[ "${MAX_PARALLEL_JOBS}" == "auto" ]]; then
            MAX_PARALLEL_JOBS=1
        fi
        echo "No GPUs detected. Running on CPU with MAX_PARALLEL_JOBS=${MAX_PARALLEL_JOBS}."
        return 0
    fi

    local gpu
    for gpu in ${gpu_ids}; do
        local replica
        for ((replica = 0; replica < JOBS_PER_GPU; replica++)); do
            SLOT_GPUS+=("${gpu}")
            SLOT_PIDS+=("")
        done
    done

    if [[ "${MAX_PARALLEL_JOBS}" == "auto" ]]; then
        MAX_PARALLEL_JOBS=${#SLOT_GPUS[@]}
    fi

    if (( MAX_PARALLEL_JOBS > ${#SLOT_GPUS[@]} )); then
        echo "MAX_PARALLEL_JOBS=${MAX_PARALLEL_JOBS} exceeds available GPU slots (${#SLOT_GPUS[@]}). Increase JOBS_PER_GPU to oversubscribe intentionally." >&2
        exit 1
    fi

    echo "RunPod GPU launcher: GPUs=(${gpu_ids}), JOBS_PER_GPU=${JOBS_PER_GPU}, MAX_PARALLEL_JOBS=${MAX_PARALLEL_JOBS}"
}

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

collect_finished_slot() {
    local slot=$1
    local pid=${SLOT_PIDS[$slot]}

    if [[ -z "${pid}" ]]; then
        return 0
    fi

    if jobs -pr | grep -qx "${pid}"; then
        return 1
    fi

    if ! wait "${pid}"; then
        FAILED=1
    fi
    SLOT_PIDS[$slot]=""
    return 0
}

wait_for_gpu_slot() {
    if (( ${#SLOT_GPUS[@]} == 0 )); then
        wait_for_slot
        GPU_SLOT_RESULT=""
        return 0
    fi

    while true; do
        local upper_bound=${MAX_PARALLEL_JOBS}
        local slot
        for ((slot = 0; slot < upper_bound; slot++)); do
            if collect_finished_slot "${slot}"; then
                GPU_SLOT_RESULT="${slot}"
                return 0
            fi
        done
        sleep 1
    done
}

schedule_transfer() {
    if (( MAX_PARALLEL_JOBS <= 1 )); then
        if (( ${#SLOT_GPUS[@]} > 0 )); then
            RUN_GPU_ID="${SLOT_GPUS[0]}" run_transfer "$@"
        else
            run_transfer "$@"
        fi
    else
        local slot
        wait_for_gpu_slot
        slot=${GPU_SLOT_RESULT}
        if [[ -n "${slot}" ]]; then
            RUN_GPU_ID="${SLOT_GPUS[$slot]}" run_transfer "$@" &
            SLOT_PIDS[$slot]="$!"
        else
            run_transfer "$@" &
        fi
        PIDS+=("$!")
    fi
}

run_transfer() {
    local dataset=$1
    local topology=$2
    local size=$3
    local seed=$4
    local model=$5
    local hidden_dim=$6
    local experiment_label=$7
    local sheaf_variant=${8:-general}
    local stalk_dim=${9:-1}
    local normalize_output=${10:-1}
    local add_self_loops=${11:-1}

    local add_crosses=0
    local layers

    case "${topology}" in
        ring)
            layers=$((size / 2))
            ;;
        crossed-ring)
            layers=$((size / 2))
            add_crosses=1
            ;;
        clique-path)
            layers=$((size / 2 + 1))
            ;;
        *)
            echo "Unknown topology ${topology}" >&2
            return 1
            ;;
    esac

    local run_name="transfer-${topology}-${experiment_label}-fdim${FEATURE_DIM}-dist${layers}-size${size}-seed${seed}"
    if completed_run "${run_name}"; then
        echo "Skipping completed ${run_name}"
        return 0
    fi

    mkdir -p "$(dirname "${RESULTS_FILE}")" "${RUN_LOG_DIR}"
    local log_file="${RUN_LOG_DIR}/${run_name}.log"
    local gpu_prefix=()
    if [[ -n "${RUN_GPU_ID:-}" ]]; then
        gpu_prefix=(env CUDA_VISIBLE_DEVICES="${RUN_GPU_ID}")
        echo "Starting ${run_name} on GPU ${RUN_GPU_ID}; log=${log_file}"
    else
        echo "Starting ${run_name} on CPU/default device; log=${log_file}"
    fi

    "${gpu_prefix[@]}" "${PYTHON_BIN}" exp/run.py \
        --dataset "${dataset}" \
        --model "${model}" \
        --cuda 0 \
        --synthetic_size "${size}" \
        --mpnn_layers "${layers}" \
        --hidden_dim "${hidden_dim}" \
        --stalk_dim "${stalk_dim}" \
        --sheaf_variant "${sheaf_variant}" \
        --sheaf_normalize_output "${normalize_output}" \
        --sheaf_add_self_loops "${add_self_loops}" \
        --bs "${BATCH_SIZE}" \
        --epochs "${EPOCHS}" \
        --seed "${seed}" \
        --add_crosses "${add_crosses}" \
        --num_class "${FEATURE_DIM}" \
        --input_dim "${FEATURE_DIM}" \
        --output_dim "${FEATURE_DIM}" \
        --synth_train_size "${TRAIN_SIZE}" \
        --synth_test_size "${TEST_SIZE}" \
        --wandb_project "${WANDB_PROJECT}" \
        --wandb_group "${WANDB_GROUP}" \
        --wandb_mode "${WANDB_MODE}" \
        --wandb_tags "paper-transfer,${topology},${experiment_label}" \
        --wandb_run_name "${run_name}" \
        --results_file "${RESULTS_FILE}" \
        --experiment_label "${experiment_label}" \
        --topology_label "${topology}" \
        --cache_synthetic_dataset "${CACHE_SYNTHETIC_DATASET}" \
        --torch_num_threads "${TORCH_NUM_THREADS}" \
        --torch_num_interop_threads "${TORCH_NUM_INTEROP_THREADS}" \
        ${ENTITY_ARG} \
        > "${log_file}" 2>&1
}

run_topology_grid() {
    local dataset=$1
    local topology=$2
    local size=$3
    local seed=$4

    if [[ "${RUN_CLASSICAL}" == "1" ]]; then
        for model in ${CLASSICAL_MODELS}; do
            schedule_transfer \
                "${dataset}" "${topology}" "${size}" "${seed}" \
                "${model}" "${CLASSICAL_HIDDEN_DIM}" \
                "classic-${model}-h${CLASSICAL_HIDDEN_DIM}"
        done
    fi

    if [[ "${RUN_SHEAF}" == "1" ]]; then
        for stalk_dim in ${SHEAF_STALK_DIMS}; do
            for hidden_dim in ${SHEAF_HIDDEN_DIMS}; do
                schedule_transfer \
                    "${dataset}" "${topology}" "${size}" "${seed}" \
                    nsd "${hidden_dim}" \
                    "nsd-orthogonal-nonnorm-noattn-stalk${stalk_dim}-h${hidden_dim}" \
                    orthogonal "${stalk_dim}" 0 0

                schedule_transfer \
                    "${dataset}" "${topology}" "${size}" "${seed}" \
                    nsd "${hidden_dim}" \
                    "nsd-general-norm-stalk${stalk_dim}-h${hidden_dim}" \
                    general "${stalk_dim}" 1 1
            done
        done
    fi
}

initialise_gpu_slots

for size in ${SIZES}; do
    for seed in ${SEEDS}; do
        run_topology_grid RING ring "${size}" "${seed}"
        run_topology_grid RING crossed-ring "${size}" "${seed}"
        run_topology_grid LOLLIPOP clique-path "${size}" "${seed}"
    done
done

if (( MAX_PARALLEL_JOBS > 1 )); then
    if (( ${#SLOT_GPUS[@]} > 0 )); then
        for slot in "${!SLOT_PIDS[@]}"; do
            pid=${SLOT_PIDS[$slot]}
            if [[ -n "${pid}" ]] && ! wait "${pid}"; then
                FAILED=1
            fi
        done
    else
        for pid in "${PIDS[@]}"; do
            if ! wait "${pid}"; then
                FAILED=1
            fi
        done
    fi
    if (( FAILED != 0 )); then
        exit 1
    fi
fi
