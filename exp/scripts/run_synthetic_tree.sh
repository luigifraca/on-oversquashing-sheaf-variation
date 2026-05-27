#!/bin/bash

mkdir -p synthetic_raw_results/tree-arity-2-gin
mkdir -p synthetic_raw_results/tree-arity-2-sage
mkdir -p synthetic_raw_results/tree-arity-2-gcn
mkdir -p synthetic_raw_results/tree-arity-2-gat

STALK_DIMS=(1 2 3 5)
HIDDEN_DIMS=(8 16 32 64)
SYNTHETIC_SIZES=(2 6 10 20 50)
SEEDS=(43)

for stalk_dim in "${STALK_DIMS[@]}"
do
    for hidden_dim in "${HIDDEN_DIMS[@]}"
    do
        mkdir -p synthetic_raw_results/tree-arity-2-sheaf-d${stalk_dim}-hidden-${hidden_dim}
    done
done

for i in "${SYNTHETIC_SIZES[@]}"
    do
    for j in "${SEEDS[@]}"
    do
        L=$((i/2))
        # GIN TREE
        if [ ! -f synthetic_raw_results/tree-arity-2-gin/size-$i-seed-$j ]
        then
            python exp/run.py --dataset TREE --bs 128 --epochs 100 --hidden_dim 64 --model gin --arity 2 --mpnn_layers $L --synthetic_size $i --add_crosses 0 --seed $j > synthetic_raw_results/tree-arity-2-gin/size-$i-seed-$j
        fi

        # SAGE TREE
        if [ ! -f synthetic_raw_results/tree-arity-2-sage/size-$i-seed-$j ]
        then
            python exp/run.py --dataset TREE --bs 128 --epochs 100 --hidden_dim 64 --model sage --arity 2 --mpnn_layers $L --synthetic_size $i --add_crosses 0 --seed $j > synthetic_raw_results/tree-arity-2-sage/size-$i-seed-$j
        fi

        # GCN TREE
        if [ ! -f synthetic_raw_results/tree-arity-2-gcn/size-$i-seed-$j ]
        then
            python exp/run.py --dataset TREE --bs 128 --epochs 100 --hidden_dim 64 --model gcn --arity 2 --mpnn_layers $L --synthetic_size $i --add_crosses 0 --seed $j > synthetic_raw_results/tree-arity-2-gcn/size-$i-seed-$j
        fi

        # GAT TREE
        if [ ! -f synthetic_raw_results/tree-arity-2-gat/size-$i-seed-$j ]
        then
            python exp/run.py --dataset TREE --bs 128 --epochs 100 --hidden_dim 64 --model gat --arity 2 --mpnn_layers $L --synthetic_size $i --add_crosses 0 --seed $j > synthetic_raw_results/tree-arity-2-gat/size-$i-seed-$j
        fi

        for stalk_dim in "${STALK_DIMS[@]}"
        do
            for hidden_dim in "${HIDDEN_DIMS[@]}"
            do
                result_dir=synthetic_raw_results/tree-arity-2-sheaf-d${stalk_dim}-hidden-${hidden_dim}
                if [ ! -f $result_dir/size-$i-seed-$j ]
                then
                    python exp/run.py --dataset TREE --bs 128 --epochs 100 --hidden_dim $hidden_dim --model sheaf --d $stalk_dim --arity 2 --mpnn_layers $L --synthetic_size $i --add_crosses 0 --seed $j > $result_dir/size-$i-seed-$j
                fi
            done
        done
    done
done
