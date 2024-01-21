@echo off
cd C:\Users\Hossein_Haeri\reward-sharing-relational-networks\venv\Scripts
call activate
cd ..\..\maddpg\experiments
start python train_v0.py --exp-name fully-connected_bounded_01
start python train_v0.py --exp-name fully-connected_bounded_02
start python train_v0.py --exp-name fully-connected_bounded_03
start python train_v0.py --exp-name fully-connected_bounded_04
start python train_v0.py --exp-name fully-connected_bounded_05
start python train_v0.py --exp-name fully-connected_bounded_06
start python train_v0.py --exp-name fully-connected_bounded_07
start python train_v0.py --exp-name fully-connected_bounded_08
start python train_v0.py --exp-name fully-connected_bounded_09
start python train_v0.py --exp-name fully-connected_bounded_10
cmd /k