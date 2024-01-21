#!/bin/bash
cd maddpg/experiments/


num_runs=3
num_episodes=150000

for i in $(seq 1 $num_runs); do
    python train_v3.py --num-agents 2 --num-landmarks 2 --rsrn-type WSM --network fully-connected --num-episodes $num_episodes --exp-name slow_$i  &
done


for i in $(seq 1 $num_runs); do
    python train_v3.py --num-agents 2 --num-landmarks 2 --rsrn-type WPM --network fully-connected --num-episodes $num_episodes --exp-name slow_$i  &
done


for i in $(seq 1 $num_runs); do
    python train_v3.py --num-agents 2 --num-landmarks 2 --rsrn-type MinMax --network fully-connected --num-episodes $num_episodes --exp-name slow_$i  &
done


for i in $(seq 1 $num_runs); do
    python train_v3.py --num-agents 2 --num-landmarks 2 --rsrn-type WSM --network fully-connected --num-episodes $num_episodes --agent-limitation stuck --exp-name stuck_$i  &
done


for i in $(seq 1 $num_runs); do
    python train_v3.py --num-agents 2 --num-landmarks 2 --rsrn-type WPM --network fully-connected --num-episodes $num_episodes --agent-limitation stuck --exp-name stuck_$i  &
done


for i in $(seq 1 $num_runs); do
    python train_v3.py --num-agents 2 --num-landmarks 2 --rsrn-type MinMax --network fully-connected --num-episodes $num_episodes --agent-limitation stuck --exp-name stuck_$i  &
done


wait