import argparse
import os
import sys
import csv
from datetime import datetime
from pathlib import Path

from stable_baselines3 import DQN
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback, CallbackList
from stable_baselines3.common.evaluation import evaluate_policy

from env_utils import make_env, check_requirements


def setup_logging(script_name):
    logs_dir = Path("logs")
    logs_dir.mkdir(exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = logs_dir / f"{script_name}_{timestamp}.log"
    log_file = open(log_path, "w")

    class Tee:
        def __init__(self, *streams):
            self.streams = streams

        def write(self, data):
            for s in self.streams:
                s.write(data)
                s.flush()

        def flush(self):
            for s in self.streams:
                s.flush()

    tee_out = Tee(sys.stdout, log_file)
    tee_err = Tee(sys.stderr, log_file)
    sys.stdout = tee_out
    sys.stderr = tee_err
    return log_file, log_path


def parse_args():
    parser = argparse.ArgumentParser(description="Train a DQN agent on Atari Pong")
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--exploration_fraction", type=float, default=0.1)
    parser.add_argument("--exploration_initial_eps", type=float, default=1.0)
    parser.add_argument("--exploration_final_eps", type=float, default=0.01)
    parser.add_argument("--total_timesteps", type=int, default=300000)
    parser.add_argument("--policy", type=str, default="CnnPolicy", choices=["CnnPolicy", "MlpPolicy"])
    parser.add_argument("--buffer_size", type=int, default=50000)
    parser.add_argument("--resume", type=str, default=None, help="Path to checkpoint zip to resume training from")
    parser.add_argument("--experiment_name", type=str, default="default")
    parser.add_argument("--member_name", type=str, default="unnamed")
    return parser.parse_args()


def main():
    args = parse_args()
    log_file, log_path = setup_logging("train")

    try:
        check_requirements()

        env = make_env(seed=0)
        use_cnn = args.policy == "CnnPolicy"
        eval_env = make_env(seed=42, transpose_image=use_cnn)

        checkpoint_dir = os.path.join("checkpoints", args.experiment_name)
        os.makedirs(checkpoint_dir, exist_ok=True)

        checkpoint_callback = CheckpointCallback(
            save_freq=50000,
            save_path=checkpoint_dir,
            name_prefix="dqn_checkpoint",
        )

        best_model_path = os.path.join(".", f"best_model_{args.experiment_name}")
        eval_callback = EvalCallback(
            eval_env,
            best_model_save_path=best_model_path,
            log_path=best_model_path,
            eval_freq=25000,
            deterministic=True,
            render=False,
        )

        callback = CallbackList([checkpoint_callback, eval_callback])

        os.makedirs("runs", exist_ok=True)

        if args.resume:
            print(f"Resuming from checkpoint: {args.resume}")
            model = DQN.load(
                args.resume,
                env=env,
                learning_rate=args.lr,
                gamma=args.gamma,
                batch_size=args.batch_size,
                exploration_fraction=args.exploration_fraction,
                exploration_initial_eps=args.exploration_initial_eps,
                exploration_final_eps=args.exploration_final_eps,
                tensorboard_log="runs/",
                verbose=1,
                buffer_size=args.buffer_size,
            )
        else:
            model = DQN(
                args.policy,
                env,
                learning_rate=args.lr,
                gamma=args.gamma,
                batch_size=args.batch_size,
                exploration_fraction=args.exploration_fraction,
                exploration_initial_eps=args.exploration_initial_eps,
                exploration_final_eps=args.exploration_final_eps,
                tensorboard_log="runs/",
                verbose=1,
                buffer_size=args.buffer_size,
            )

        print(f"Training with experiment: {args.experiment_name}")
        print(f"Member: {args.member_name}")
        print(f"Policy: {args.policy}")
        print(f"Total timesteps: {args.total_timesteps}")
        if args.resume:
            print(f"Resumed from: {args.resume}")

        model.learn(
            total_timesteps=args.total_timesteps,
            callback=callback,
            tb_log_name=args.experiment_name,
        )

        model.save("dqn_model")

        mean_reward, std_reward = evaluate_policy(model, eval_env, n_eval_episodes=10, deterministic=True)

        results_dir = Path("results")
        results_dir.mkdir(exist_ok=True)
        csv_path = results_dir / "experiments.csv"

        fieldnames = [
            "member_name", "experiment_name", "policy", "lr", "gamma",
            "batch_size", "exploration_fraction", "exploration_initial_eps",
            "exploration_final_eps", "total_timesteps", "mean_reward",
            "std_reward", "timestamp",
        ]

        write_header = not csv_path.exists()
        with open(csv_path, "a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            if write_header:
                writer.writeheader()
            writer.writerow({
                "member_name": args.member_name,
                "experiment_name": args.experiment_name,
                "policy": args.policy,
                "lr": args.lr,
                "gamma": args.gamma,
                "batch_size": args.batch_size,
                "exploration_fraction": args.exploration_fraction,
                "exploration_initial_eps": args.exploration_initial_eps,
                "exploration_final_eps": args.exploration_final_eps,
                "total_timesteps": args.total_timesteps,
                "mean_reward": mean_reward,
                "std_reward": std_reward,
                "timestamp": datetime.now().isoformat(),
            })

        print(f"\nTraining complete")
        print(f"Member: {args.member_name}")
        print(f"Experiment: {args.experiment_name}")
        print(f"Policy: {args.policy}")
        print(f"Learning rate: {args.lr}")
        print(f"Gamma: {args.gamma}")
        print(f"Batch size: {args.batch_size}")
        print(f"Exploration fraction: {args.exploration_fraction}")
        print(f"Exploration initial eps: {args.exploration_initial_eps}")
        print(f"Exploration final eps: {args.exploration_final_eps}")
        print(f"Total timesteps: {args.total_timesteps}")
        print(f"Mean reward: {mean_reward:.2f}")
        print(f"Std reward: {std_reward:.2f}")
        print(f"Model saved to dqn_model.zip")
        print(f"Results appended to {csv_path}")

        env.close()
        eval_env.close()

    except Exception as e:
        print(f"Error during training: {e}", file=sys.stderr)
        print(f"Log file: {log_path}", file=sys.stderr)
        raise
    finally:
        log_file.close()


if __name__ == "__main__":
    main()
