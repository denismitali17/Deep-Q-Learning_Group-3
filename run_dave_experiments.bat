@echo off
cd /d "%~dp0"
set PYTHON=C:\Users\user\PycharmProjects\LoRA\lora-venv\Scripts\python.exe
set NAME=Dave

echo Resuming experiments for %NAME% (exp 1-5 already done)
echo Experiment 6: resuming from 200k checkpoint
echo Experiments 7-10: fresh 1M runs
echo.

echo ---- Experiment 6 of 10 (1M, resuming from 200k) ----
%PYTHON% train.py --member_name "%NAME%" --experiment_name "dave_exp6" --lr 1e-4 --gamma 0.90 --batch_size 128 --exploration_initial_eps 1.0 --exploration_final_eps 0.05 --exploration_fraction 0.10 --policy CnnPolicy --total_timesteps 1000000 --resume "checkpoints/dave_exp6/dqn_checkpoint_200000_steps.zip"

echo ---- Experiment 7 of 10 (1M) ----
%PYTHON% train.py --member_name "%NAME%" --experiment_name "dave_exp7" --lr 5e-5 --gamma 0.99 --batch_size 32 --exploration_initial_eps 1.0 --exploration_final_eps 0.001 --exploration_fraction 0.30 --policy CnnPolicy --total_timesteps 1000000

echo ---- Experiment 8 of 10 (1M) ----
%PYTHON% train.py --member_name "%NAME%" --experiment_name "dave_exp8" --lr 1e-3 --gamma 0.95 --batch_size 64 --exploration_initial_eps 1.0 --exploration_final_eps 0.01 --exploration_fraction 0.10 --policy CnnPolicy --total_timesteps 1000000

echo ---- Experiment 9 of 10 (1M) ----
%PYTHON% train.py --member_name "%NAME%" --experiment_name "dave_exp9" --lr 1e-4 --gamma 0.999 --batch_size 32 --exploration_initial_eps 1.0 --exploration_final_eps 0.001 --exploration_fraction 0.05 --policy CnnPolicy --total_timesteps 1000000

echo ---- Experiment 10 of 10 (1M) ----
%PYTHON% train.py --member_name "%NAME%" --experiment_name "dave_exp10" --lr 1e-4 --gamma 0.99 --batch_size 32 --exploration_initial_eps 1.0 --exploration_final_eps 0.01 --exploration_fraction 0.10 --policy MlpPolicy --total_timesteps 1000000

echo.
echo All experiments complete!
echo Results: results\experiments.csv
echo Final model: dqn_model.zip
pause
