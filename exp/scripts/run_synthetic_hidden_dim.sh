#!/bin/bash


HIDDEN_DIMS=(1 2 4 8 16 32)
SHEAF_STALK_DIMS=(1 2 3 5)
SHEAF_HIDDEN_DIMS=(8 16 32 64)
SYNTHETIC_SIZES=(2 6 10 20 50)
SEEDS=(43)

for hidden_dim in "${HIDDEN_DIMS[@]}"
do
    mkdir -p synthetic_raw_results/ring-gcn-hidden-dim-$hidden_dim
    mkdir -p synthetic_raw_results/crossed-ring-gcn-hidden-dim-$hidden_dim
    mkdir -p synthetic_raw_results/lollipop-gcn-hidden-dim-$hidden_dim
    for i in "${SYNTHETIC_SIZES[@]}"
    do
        for j in "${SEEDS[@]}"
        do
            # GCN RING
            L=$((i/2))
            if [ ! -f synthetic_raw_results/ring-gcn-hidden-dim-$hidden_dim/size-$i-seed-$j ]
            then
                python exp/run.py --dataset RING --bs 128 --epochs 100 --hidden_dim $hidden_dim --model gcn --mpnn_layers $L --synthetic_size $i --add_crosses 0 --seed $j > synthetic_raw_results/ring-gcn-hidden-dim-$hidden_dim/size-$i-seed-$j
            fi

            # GCN CROSSED RING
            if [ ! -f synthetic_raw_results/crossed-ring-gcn-hidden-dim-$hidden_dim/size-$i-seed-$j ]
            then
                python exp/run.py --dataset RING --bs 128 --epochs 100 --hidden_dim $hidden_dim --model gcn --mpnn_layers $L --synthetic_size $i --add_crosses 1 --seed $j > synthetic_raw_results/crossed-ring-gcn-hidden-dim-$hidden_dim/size-$i-seed-$j
            fi

            # GCN LOLLIPOP
            L=$((i/2 + 1))
            if [ ! -f synthetic_raw_results/lollipop-gcn-hidden-dim-$hidden_dim/size-$i-seed-$j ]
            then
                python exp/run.py --dataset LOLLIPOP --bs 128 --epochs 100 --hidden_dim $hidden_dim --model gcn --mpnn_layers $L --synthetic_size $i --add_crosses 0 --seed $j > synthetic_raw_results/lollipop-gcn-hidden-dim-$hidden_dim/size-$i-seed-$j
            fi
        done
    done
done

for stalk_dim in "${SHEAF_STALK_DIMS[@]}"
do
    for hidden_dim in "${SHEAF_HIDDEN_DIMS[@]}"
    do
        mkdir -p synthetic_raw_results/ring-sheaf-d${stalk_dim}-hidden-dim-${hidden_dim}
        mkdir -p synthetic_raw_results/crossed-ring-sheaf-d${stalk_dim}-hidden-dim-${hidden_dim}
        mkdir -p synthetic_raw_results/lollipop-sheaf-d${stalk_dim}-hidden-dim-${hidden_dim}
        for i in "${SYNTHETIC_SIZES[@]}"
        do
            for j in "${SEEDS[@]}"
            do
                L=$((i/2))

                if [ ! -f synthetic_raw_results/ring-sheaf-d${stalk_dim}-hidden-dim-${hidden_dim}/size-$i-seed-$j ]
                then
                    python exp/run.py --dataset RING --bs 128 --epochs 100 --hidden_dim $hidden_dim --model sheaf --d $stalk_dim --mpnn_layers $L --synthetic_size $i --add_crosses 0 --seed $j > synthetic_raw_results/ring-sheaf-d${stalk_dim}-hidden-dim-${hidden_dim}/size-$i-seed-$j
                fi

                if [ ! -f synthetic_raw_results/crossed-ring-sheaf-d${stalk_dim}-hidden-dim-${hidden_dim}/size-$i-seed-$j ]
                then
                    python exp/run.py --dataset RING --bs 128 --epochs 100 --hidden_dim $hidden_dim --model sheaf --d $stalk_dim --mpnn_layers $L --synthetic_size $i --add_crosses 1 --seed $j > synthetic_raw_results/crossed-ring-sheaf-d${stalk_dim}-hidden-dim-${hidden_dim}/size-$i-seed-$j
                fi

                L=$((i/2 + 1))
                if [ ! -f synthetic_raw_results/lollipop-sheaf-d${stalk_dim}-hidden-dim-${hidden_dim}/size-$i-seed-$j ]
                then
                    python exp/run.py --dataset LOLLIPOP --bs 128 --epochs 100 --hidden_dim $hidden_dim --model sheaf --d $stalk_dim --mpnn_layers $L --synthetic_size $i --add_crosses 0 --seed $j > synthetic_raw_results/lollipop-sheaf-d${stalk_dim}-hidden-dim-${hidden_dim}/size-$i-seed-$j
                fi
            done
        done
    done
done
