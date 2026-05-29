#!/bin/bash
set -euo pipefail

PYTHON_BIN=${PYTHON_BIN:-python}
WANDB_PROJECT=${WANDB_PROJECT:-on-oversquashing-transfer}
WANDB_ENTITY=${WANDB_ENTITY:-}
WANDB_MODE=${WANDB_MODE:-online}
WANDB_GROUP=${WANDB_GROUP:-graph-transfer-classical-vs-nsd}
MODELS=${MODELS:-"gcn gat nsd"}
SEEDS=${SEEDS:-"1 2 3"}
SIZES=${SIZES:-"2 4 6 8 10 12 14 16 18 20 22 24 26 28 30"}
EPOCHS=${EPOCHS:-100}
BATCH_SIZE=${BATCH_SIZE:-128}
HIDDEN_DIM=${HIDDEN_DIM:-32}
STALK_DIM=${STALK_DIM:-2}
SHEAF_VARIANT=${SHEAF_VARIANT:-diagonal}
TRAIN_SIZE=${TRAIN_SIZE:-5000}
TEST_SIZE=${TEST_SIZE:-500}
NUM_CLASS=${NUM_CLASS:-5}

ENTITY_ARG=""
if [[ -n "${WANDB_ENTITY}" && "${WANDB_ENTITY}" != "none" ]]; then
    ENTITY_ARG="--entity ${WANDB_ENTITY}"
fi

run_transfer() {
    local dataset=$1
    local topology=$2
    local size=$3
    local seed=$4
    local model=$5
    local add_crosses=0
    local layers
    local arity_args=""
    local run_name="${model}-${dataset}-${topology}-size${size}-seed${seed}"

    case "${topology}" in
        ring)
            layers=$((size / 2))
            ;;
        crossed-ring)
            layers=$((size / 2))
            add_crosses=1
            ;;
        lollipop)
            layers=$((size / 2 + 1))
            ;;
        tree)
            layers=$((size / 2))
            arity_args="--arity 2"
            ;;
        *)
            echo "Unknown topology ${topology}" >&2
            return 1
            ;;
    esac

    "${PYTHON_BIN}" exp/run.py \
        --dataset "${dataset}" \
        --model "${model}" \
        --synthetic_size "${size}" \
        --mpnn_layers "${layers}" \
        --hidden_dim "${HIDDEN_DIM}" \
        --stalk_dim "${STALK_DIM}" \
        --sheaf_variant "${SHEAF_VARIANT}" \
        --bs "${BATCH_SIZE}" \
        --epochs "${EPOCHS}" \
        --seed "${seed}" \
        --add_crosses "${add_crosses}" \
        --num_class "${NUM_CLASS}" \
        --input_dim "${NUM_CLASS}" \
        --output_dim "${NUM_CLASS}" \
        --synth_train_size "${TRAIN_SIZE}" \
        --synth_test_size "${TEST_SIZE}" \
        --wandb_project "${WANDB_PROJECT}" \
        --wandb_group "${WANDB_GROUP}" \
        --wandb_mode "${WANDB_MODE}" \
        --wandb_tags "paper-transfer,${topology},classical-vs-nsd" \
        --wandb_run_name "${run_name}" \
        ${ENTITY_ARG} \
        ${arity_args}
}

for size in ${SIZES}; do
    for seed in ${SEEDS}; do
        for model in ${MODELS}; do
            run_transfer RING ring "${size}" "${seed}" "${model}"
            run_transfer RING crossed-ring "${size}" "${seed}" "${model}"
            run_transfer LOLLIPOP lollipop "${size}" "${seed}" "${model}"
        done
    done
done
