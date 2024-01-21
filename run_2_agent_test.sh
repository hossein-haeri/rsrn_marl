#!/bin/bash
cd maddpg/experiments/


num_runs=3
num_episodes=1000

for i in $(seq 1 $num_runs); do
    loadingdir=./saved_policy/2-agent/WSM/fully-connected/slow_$i/
    python train_v3.py --test-mode --num-episodes $num_episodes --load-dir $loadingdir &
done


for i in $(seq 1 $num_runs); do
    loadingdir=./saved_policy/2-agent/WPM/fully-connected/slow_$i/
    python train_v3.py --test-mode --num-episodes $num_episodes --load-dir $loadingdir &
done
# python train_v3.py --test-mode --num-episodes 100 --load-dir ./saved_policy/2-agent/WSM/fully-connected/stuck_1/

for i in $(seq 1 $num_runs); do
    loadingdir=./saved_policy/2-agent/MinMax/fully-connected/slow_$i/
    python train_v3.py --test-mode --num-episodes $num_episodes --load-dir $loadingdir &
done


for i in $(seq 1 $num_runs); do
    loadingdir=./saved_policy/2-agent/WSM/fully-connected/stuck_$i/
    python train_v3.py --test-mode --num-episodes $num_episodes --load-dir $loadingdir&
done


for i in $(seq 1 $num_runs); do
    loadingdir=./saved_policy/2-agent/WPM/fully-connected/stuck_$i/
    python train_v3.py --test-mode --num-episodes $num_episodes --load-dir $loadingdir &
done


for i in $(seq 1 $num_runs); do
    loadingdir=./saved_policy/2-agent/MinMax/fully-connected/stuck_$i/
    python train_v3.py --test-mode --num-episodes $num_episodes --load-dir $loadingdir &
done

# python train_v3.py --num-agents 2 --num-landmarks 2 --rsrn-type MinMax --network fully-connected --num-episodes $num_episodes --agent-limitation test_stuck --restore --load-dir $loadingdir  --exp-name test_stuck_$i  &
wait