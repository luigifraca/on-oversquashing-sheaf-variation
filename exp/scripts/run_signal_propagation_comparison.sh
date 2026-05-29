#!/bin/bash
set -euo pipefail

PYTHON_BIN=${PYTHON_BIN:-python}
WANDB_PROJECT=${WANDB_PROJECT:-on-oversquashing-signal-propagation}
WANDB_ENTITY=${WANDB_ENTITY:-}
WANDB_MODE=${WANDB_MODE:-online}
WANDB_GROUP=${WANDB_GROUP:-signal-propagation-classical-vs-nsd}
DATASET=${DATASET:-NCI1}
ROOT=${ROOT:-data/tu}
MODELS=${MODELS:-gcn,gat,nsd}
LAYERS=${LAYERS:-10}
HIDDEN_DIM=${HIDDEN_DIM:-5}
MAX_GRAPHS=${MAX_GRAPHS:-}
NUM_SOURCES=${NUM_SOURCES:-10}
SEED=${SEED:-808}
STALK_DIM=${STALK_DIM:-4}
SHEAF_VARIANT=${SHEAF_VARIANT:-general}

ENTITY_ARG=""
if [[ -n "${WANDB_ENTITY}" && "${WANDB_ENTITY}" != "none" ]]; then
    ENTITY_ARG="--entity ${WANDB_ENTITY}"
fi

MAX_GRAPHS_ARG=""
if [[ -n "${MAX_GRAPHS}" ]]; then
    MAX_GRAPHS_ARG="--max_graphs ${MAX_GRAPHS}"
fi

"${PYTHON_BIN}" exp/signal_propagation.py \
    --dataset "${DATASET}" \
    --root "${ROOT}" \
    --models "${MODELS}" \
    --layers "${LAYERS}" \
    --hidden_dim "${HIDDEN_DIM}" \
    --num_sources "${NUM_SOURCES}" \
    --seed "${SEED}" \
    --stalk_dim "${STALK_DIM}" \
    --sheaf_variant "${SHEAF_VARIANT}" \
    --wandb_project "${WANDB_PROJECT}" \
    --wandb_group "${WANDB_GROUP}" \
    --wandb_mode "${WANDB_MODE}" \
    --wandb_tags "paper-signal-propagation,classical-vs-nsd" \
    --wandb_name "signal-propagation-${DATASET}-${MODELS}-${LAYERS}" \
    ${ENTITY_ARG} \
    ${MAX_GRAPHS_ARG}
