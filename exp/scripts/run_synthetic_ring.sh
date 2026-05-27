#!/bin/bash

mkdir -p synthetic_raw_results/ring-gin
mkdir -p synthetic_raw_results/ring-sage
mkdir -p synthetic_raw_results/ring-gcn
mkdir -p synthetic_raw_results/ring-gat

STALK_DIMS=(1 2 3 5)
HIDDEN_DIMS=(8 16 32)
SYNTHETIC_SIZES=(10 20 50)
SEEDS=(43)

for stalk_dim in "${STALK_DIMS[@]}"
do
    for hidden_dim in "${HIDDEN_DIMS[@]}"
    do
        mkdir -p synthetic_raw_results/ring-sheaf-d${stalk_dim}-hidden-${hidden_dim}
    done
done

for i in "${SYNTHETIC_SIZES[@]}"
    do
    for j in "${SEEDS[@]}"
    do
        L=$((i/2))
        # GIN RING
        if [ ! -f synthetic_raw_results/ring-gin/size-$i-seed-$j ]
        then
            python exp/run.py --dataset RING --bs 128 --epochs 100 --hidden_dim 64 --model gin --mpnn_layers $L --synthetic_size $i --add_crosses 0 --seed $j > synthetic_raw_results/ring-gin/size-$i-seed-$j
        fi

        # SAGE RING
        if [ ! -f synthetic_raw_results/ring-sage/size-$i-seed-$j ]
        then
            python exp/run.py --dataset RING --bs 128 --epochs 100 --hidden_dim 64 --model sage --mpnn_layers $L --synthetic_size $i --add_crosses 0 --seed $j > synthetic_raw_results/ring-sage/size-$i-seed-$j
        fi

        # GCN RING
        if [ ! -f synthetic_raw_results/ring-gcn/size-$i-seed-$j ]
        then
            python exp/run.py --dataset RING --bs 128 --epochs 100 --hidden_dim 64 --model gcn --mpnn_layers $L --synthetic_size $i --add_crosses 0 --seed $j > synthetic_raw_results/ring-gcn/size-$i-seed-$j
        fi

        # GAT RING
        if [ ! -f synthetic_raw_results/ring-gat/size-$i-seed-$j ]
        then
            python exp/run.py --dataset RING --bs 128 --epochs 100 --hidden_dim 64 --model gat --mpnn_layers $L --synthetic_size $i --add_crosses 0 --seed $j > synthetic_raw_results/ring-gat/size-$i-seed-$j
        fi

        for stalk_dim in "${STALK_DIMS[@]}"
        do
            for hidden_dim in "${HIDDEN_DIMS[@]}"
            do
                result_dir=synthetic_raw_results/ring-sheaf-d${stalk_dim}-hidden-${hidden_dim}
                if [ ! -f $result_dir/size-$i-seed-$j ]
                then
                    python exp/run.py --dataset RING --bs 128 --epochs 100 --hidden_dim $hidden_dim --model sheaf --d $stalk_dim --mpnn_layers $L --synthetic_size $i --add_crosses 0 --seed $j > $result_dir/size-$i-seed-$j
                fi
            done
        done
    done
done
