# Deep Q-Learning: Atari Pong

DQN agent trained on `PongNoFrameskip-v4` using Stable Baselines3 and Gymnasium.

## Gameplay Demo

[![Gameplay Demo](https://img.youtube.com/vi/Jkjw_3mBC4c/0.jpg)](https://youtu.be/Jkjw_3mBC4c)

## How to Run

### Training
```bash
python train.py --member_name "YourName" --experiment_name "exp1" --lr 1e-4 --gamma 0.99 --batch_size 32 --total_timesteps 300000
```

### Playing
```bash
python play.py --model_path dqn_model.zip --episodes 5
```

## Hyperparameter Tuning Results

### David

| # | lr | gamma | batch | eps_start | eps_end | expl_fraction | policy | timesteps | mean_reward | noted_behavior |
|---|------|-------|-------|-----------|---------|---------------|-----------|-----------|-------------|----------------|
| 1 | 1e-4 | 0.99 | 32 | 1.0 | 0.01 | 0.10 | CnnPolicy | 300k | -21.0 | Baseline. Small batch + 300k steps = no learning at all. Agent lost every game 21-0. |
| 2 | 5e-5 | 0.99 | 64 | 1.0 | 0.01 | 0.10 | CnnPolicy | 300k | -16.2 | Bumping batch to 64 made a huge difference, agent started scoring even at 300k. |
| 3 | 5e-4 | 0.99 | 32 | 1.0 | 0.05 | 0.05 | CnnPolicy | 300k | -21.0 | Higher lr + fast exploration decay. Exploration ended at 15k steps, way too early. No learning. |
| 4 | 1e-4 | 0.95 | 64 | 1.0 | 0.01 | 0.15 | CnnPolicy | 300k | -13.7 | Best 300k run. Batch=64 + lower gamma + longer exploration all helped. Scored ~7 points per game. |
| 5 | 2e-4 | 0.99 | 32 | 1.0 | 0.01 | 0.20 | CnnPolicy | 300k | -21.0 | Longer exploration didn't help with batch=32. Small batch is just not enough at 300k. |
| 6 | 1e-4 | 0.90 | 128 | 1.0 | 0.05 | 0.10 | CnnPolicy | 1M | +14.2 | Batch=128 was a game changer. Agent started winning games around 700k steps, ended up at ~18-4 wins. |
| 7 | 5e-5 | 0.99 | 32 | 1.0 | 0.001 | 0.30 | CnnPolicy | 1M | -1.5 | Very slow lr + long exploration. Agent barely broke even by 1M steps. Needed more training time. |
| 8 | 1e-3 | 0.95 | 64 | 1.0 | 0.01 | 0.10 | CnnPolicy | 1M | -21.0 | lr=1e-3 killed it. Even with batch=64 and 1M steps, the high lr just destabilized everything. |
| 9 | 1e-4 | 0.999 | 32 | 1.0 | 0.001 | 0.05 | CnnPolicy | 1M | +17.4 | Best result overall. High gamma + very low final epsilon. Agent was winning games 19-2 on average. |
| 10 | 1e-4 | 0.99 | 32 | 1.0 | 0.01 | 0.10 | MlpPolicy | 1M | -21.0 | MLP just can't do pixel input. No learning at all, proves CNN is needed for Atari. |

### Denis

| # | lr | gamma | batch | eps_start | eps_end | expl_fraction | policy | timesteps | mean_reward | noted_behavior |
|---|------|-------|-------|-----------|---------|---------------|-----------|-----------|-------------|----------------|
| 1 | 1e-4 | 0.99 | 32 | 1.0 | 0.05 | 0.10 | CnnPolicy | 1M | -21.0 | Baseline config, no improvement even at 1M steps with batch=32. |
| 2 | 5e-4 | 0.99 | 32 | 1.0 | 0.05 | 0.10 | CnnPolicy | 1M | -21.0 | Higher lr did not help, same -21.0 result as baseline. |
| 3 | 1e-5 | 0.99 | 32 | 1.0 | 0.05 | 0.10 | CnnPolicy | 500k | -20.0 | Very low lr, slight improvement to -20.0 but learning too slow to get far in 500k. |
| 4 | 1e-4 | 0.80 | 32 | 1.0 | 0.05 | 0.10 | CnnPolicy | 500k | -21.0 | Low gamma meant the agent only cared about short term rewards, no learning. |
| 5 | 1e-4 | 0.999 | 32 | 1.0 | 0.05 | 0.10 | CnnPolicy | 1M | -13.7 | Best result. High gamma helped the agent plan ahead during rallies. High std (7.8) shows inconsistent games though. |
| 6 | 1e-4 | 0.99 | 64 | 1.0 | 0.05 | 0.10 | CnnPolicy | 1M | -14.5 | Larger batch helped, second best result. More stable than exp5 with lower std (1.4). |
| 7 | 2.5e-4 | 0.99 | 64 | 1.0 | 0.01 | 0.10 | CnnPolicy | 2M | -21.0 | Even 2M steps didn't help. Slightly high lr with batch=64 failed to converge. |
| 8 | 1e-4 | 0.99 | 32 | 1.0 | 0.05 | 0.05 | CnnPolicy | 500k | -20.8 | Fast exploration decay, agent stopped exploring at 25k steps. Barely any improvement. |
| 9 | 1e-4 | 0.99 | 32 | 1.0 | 0.05 | 0.50 | CnnPolicy | 500k | -14.7 | Slow decay kept exploring for 250k steps. Third best result, exploration helped despite short training. |
| 10 | 1e-4 | 0.99 | 32 | 1.0 | 0.05 | 0.10 | MlpPolicy | 500k | -21.0 | MLP cannot process pixel input, no learning at all. |

### Queen

| # | lr | gamma | batch | eps_start | eps_end | expl_fraction | policy | mean_reward | noted_behavior |
|---|------|-------|-------|-----------|---------|---------------|-----------|-------------|----------------|
| 1 | 1e-4 | 0.99 | 32 | 1.0 | 0.01 | 0.10 | CnnPolicy | | |
| 2 | 1e-4 | 0.99 | 16 | 1.0 | 0.01 | 0.10 | CnnPolicy | | |
| 3 | 1e-4 | 0.99 | 128 | 1.0 | 0.01 | 0.10 | CnnPolicy | | |
| 4 | 1e-4 | 0.99 | 256 | 1.0 | 0.01 | 0.10 | CnnPolicy | | |
| 5 | 5e-5 | 0.99 | 64 | 1.0 | 0.05 | 0.15 | CnnPolicy | | |
| 6 | 2e-4 | 0.95 | 32 | 1.0 | 0.01 | 0.20 | CnnPolicy | | |
| 7 | 1e-4 | 0.99 | 64 | 1.0 | 0.01 | 0.25 | CnnPolicy | | |
| 8 | 5e-4 | 0.90 | 128 | 1.0 | 0.05 | 0.10 | CnnPolicy | | |
| 9 | 1e-4 | 0.999 | 16 | 1.0 | 0.001 | 0.10 | CnnPolicy | | |
| 10 | 2e-4 | 0.99 | 256 | 1.0 | 0.01 | 0.05 | CnnPolicy | | |

### Roxanne

| # | lr | gamma | batch | eps_start | eps_end | expl_fraction | policy | mean_reward | noted_behavior |
|---|------|-------|-------|-----------|---------|---------------|-----------|-------------|----------------|
| 1 | 1e-4 | 0.99 | 32 | 1.0 | 0.01 | 0.10 | CnnPolicy | | |
| 2 | 5e-5 | 0.95 | 64 | 1.0 | 0.05 | 0.15 | CnnPolicy | | |
| 3 | 2e-4 | 0.99 | 16 | 1.0 | 0.01 | 0.20 | CnnPolicy | | |
| 4 | 1e-4 | 0.999 | 32 | 1.0 | 0.001 | 0.05 | CnnPolicy | | |
| 5 | 5e-4 | 0.90 | 128 | 1.0 | 0.10 | 0.10 | CnnPolicy | | |
| 6 | 1e-3 | 0.99 | 32 | 1.0 | 0.01 | 0.10 | CnnPolicy | | |
| 7 | 1e-4 | 0.95 | 64 | 1.0 | 0.05 | 0.30 | CnnPolicy | | |
| 8 | 5e-5 | 0.999 | 32 | 1.0 | 0.001 | 0.10 | CnnPolicy | | |
| 9 | 2e-4 | 0.90 | 64 | 1.0 | 0.01 | 0.20 | CnnPolicy | | |
| 10 | 1e-4 | 0.99 | 128 | 1.0 | 0.05 | 0.15 | CnnPolicy | | |

## Hyperparameter Tuning Discussion

The biggest takeaway across our experiments is that batch size really matters. Runs with batch_size=32 at 300k steps consistently got -21.0, but bumping to 64 immediately showed improvement (-16.2 and -13.7). At 1M steps with batch_size=128, the agent actually started winning games (+14.2). Multiple members saw the same pattern, larger batches consistently outperformed smaller ones.

Learning rate is tricky. 1e-4 is the sweet spot. lr=1e-3 got -21.0 even at 1M steps, the high lr just blew up training completely. Similarly lr=5e-4 also failed at 1M steps. Going too low (1e-5) also hurts since the agent learns too slowly to get anywhere in limited steps.

High gamma worked surprisingly well. Our best result (+17.4) used gamma=0.999, and other members saw the same. gamma=0.999 consistently produced better results than lower values. Makes sense for Pong since the rallies are long and the agent needs to think ahead. On the flip side, very low gamma like 0.8 killed performance entirely.

Exploration fraction had interesting effects. Using exploration_fraction=0.5 (exploring for half the training) still got a decent -14.7, showing that more exploration can help if the agent has enough time to also exploit what it learned. But fast exploration decay (fraction=0.05) gave -20.8 since the agent stopped exploring before it had enough experience.

MLP was tested and got -21.0 regardless of timesteps. It just can't handle pixel input. No convolutional layers means it can't pick up on spatial patterns like ball position or paddle movement. CNN is the only option for Atari.
