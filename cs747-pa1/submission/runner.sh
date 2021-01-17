#!/bin/bash

BANDIT_SCRIPT="./bandit.py"

execute() {
    python3 ${BANDIT_SCRIPT} --instance $1 --algorithm $2 --randomSeed $3 --epsilon $4 --horizon $5
}

task() {
    ep=0.02
    local algorithms="$1"
    for ins in "../instances/i-1.txt"; do
        for algo in $algorithms; do
            for horizon in 102400; do
                for seed in {0..49}; do
                    execute $ins $algo $seed $ep $horizon
                done
            done
        done
    done
}


algo="$*"
echo "Executing Algorithms: $algo"
task "$algo"
