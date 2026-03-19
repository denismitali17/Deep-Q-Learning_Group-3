@echo off
cd /d "%~dp0"
set PYTHON=C:\Users\user\PycharmProjects\LoRA\lora-venv\Scripts\python.exe
set NAME=Dave

echo Running experiments 2-10 for %NAME% (exp1 already done)
echo Experiments 2-5: 300k timesteps
echo Experiments 6-10: 1M timesteps
echo Results will be saved to results\experiments.csv
echo Checkpoints saved every 50k steps to checkpoints\
echo.

echo ---- Experiment 2 of 10 (300k) ----
%PYTHON% train.py --member_name "%NAME%" --experiment_name "dave_exp2" --lr 5e-5 --gamma 0.99 --batch_size 64 --exploration_initial_eps 1.0 --exploration_final_eps 0.01 --exploration_fraction 0.10 --policy CnnPolicy --total_timesteps 300000

echo ---- Experiment 3 of 10 (300k) ----
%PYTHON% train.py --member_name "%NAME%" --experiment_name "dave_exp3" --lr 5e-4 --gamma 0.99 --batch_size 32 --exploration_initial_eps 1.0 --exploration_final_eps 0.05 --exploration_fraction 0.05 --policy CnnPolicy --total_timesteps 300000

echo ---- Experiment 4 of 10 (300k) ----
%PYTHON% train.py --member_name "%NAME%" --experiment_name "dave_exp4" --lr 1e-4 --gamma 0.95 --batch_size 64 --exploration_initial_eps 1.0 --exploration_final_eps 0.01 --exploration_fraction 0.15 --policy CnnPolicy --total_timesteps 300000

echo ---- Experiment 5 of 10 (300k) ----
%PYTHON% train.py --member_name "%NAME%" --experiment_name "dave_exp5" --lr 2e-4 --gamma 0.99 --batch_size 32 --exploration_initial_eps 1.0 --exploration_final_eps 0.01 --exploration_fraction 0.20 --policy CnnPolicy --total_timesteps 300000

echo ---- Experiment 6 of 10 (1M) ----
%PYTHON% train.py --member_name "%NAME%" --experiment_name "dave_exp6" --lr 1e-4 --gamma 0.90 --batch_size 128 --exploration_initial_eps 1.0 --exploration_final_eps 0.05 --exploration_fraction 0.10 --policy CnnPolicy --total_timesteps 1000000

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
