#!/bin/bash

PYTHON=${PYTHON:-python}

is_done() {
    grep -q " on RING | SHA:" "$1" 2>/dev/null
}

mkdir -p synthetic_raw_results/crossed-ring-gin-hidden-8-bs-8
mkdir -p synthetic_raw_results/crossed-ring-sage-hidden-8-bs-8
mkdir -p synthetic_raw_results/crossed-ring-gcn-hidden-8-bs-8
mkdir -p synthetic_raw_results/crossed-ring-gat-hidden-8-bs-8

STALK_DIMS=(3 5)
HIDDEN_DIMS=(8)
SYNTHETIC_SIZES=(10 20 50)
SEEDS=(43)

for stalk_dim in "${STALK_DIMS[@]}"
do
    for hidden_dim in "${HIDDEN_DIMS[@]}"
    do
        mkdir -p synthetic_raw_results/crossed-ring-sheaf-d${stalk_dim}-hidden-${hidden_dim}-bs-8-nonorm
    done
done

for i in "${SYNTHETIC_SIZES[@]}"
    do
    for j in "${SEEDS[@]}"
    do
        L=$((i/2))
        # GIN CROSSED RING
        if ! is_done synthetic_raw_results/crossed-ring-gin-hidden-8-bs-8/size-$i-seed-$j
        then
            $PYTHON exp/run.py --dataset RING --bs 8 --epochs 100 --hidden_dim 8 --model gin --mpnn_layers $L --synthetic_size $i --add_crosses 1 --seed $j > synthetic_raw_results/crossed-ring-gin-hidden-8-bs-8/size-$i-seed-$j
        fi

        # SAGE CROSSED RING
        if ! is_done synthetic_raw_results/crossed-ring-sage-hidden-8-bs-8/size-$i-seed-$j
        then
            $PYTHON exp/run.py --dataset RING --bs 8 --epochs 100 --hidden_dim 8 --model sage --mpnn_layers $L --synthetic_size $i --add_crosses 1 --seed $j > synthetic_raw_results/crossed-ring-sage-hidden-8-bs-8/size-$i-seed-$j
        fi

        # GCN CROSSED RING
        if ! is_done synthetic_raw_results/crossed-ring-gcn-hidden-8-bs-8/size-$i-seed-$j
        then
            $PYTHON exp/run.py --dataset RING --bs 8 --epochs 100 --hidden_dim 8 --model gcn --mpnn_layers $L --synthetic_size $i --add_crosses 1 --seed $j > synthetic_raw_results/crossed-ring-gcn-hidden-8-bs-8/size-$i-seed-$j
        fi

        # GAT CROSSED RING
        if ! is_done synthetic_raw_results/crossed-ring-gat-hidden-8-bs-8/size-$i-seed-$j
        then
            $PYTHON exp/run.py --dataset RING --bs 8 --epochs 100 --hidden_dim 8 --model gat --mpnn_layers $L --synthetic_size $i --add_crosses 1 --seed $j > synthetic_raw_results/crossed-ring-gat-hidden-8-bs-8/size-$i-seed-$j
        fi

        for stalk_dim in "${STALK_DIMS[@]}"
        do
            for hidden_dim in "${HIDDEN_DIMS[@]}"
            do
                result_dir=synthetic_raw_results/crossed-ring-sheaf-d${stalk_dim}-hidden-${hidden_dim}-bs-8-nonorm
                if ! is_done $result_dir/size-$i-seed-$j
                then
                    $PYTHON exp/run.py --dataset RING --bs 8 --epochs 100 --hidden_dim $hidden_dim --model sheaf --d $stalk_dim --sheaf_normalised false --mpnn_layers $L --synthetic_size $i --add_crosses 1 --seed $j > $result_dir/size-$i-seed-$j
                fi
            done
        done
    done
done
