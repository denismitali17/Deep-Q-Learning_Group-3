import argparse
import sys
from datetime import datetime
from pathlib import Path

from stable_baselines3 import DQN

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
    parser = argparse.ArgumentParser(description="Play Atari Pong with a trained DQN agent")
    parser.add_argument("--model_path", type=str, default="dqn_model.zip")
    parser.add_argument("--episodes", type=int, default=5)
    parser.add_argument("--no-render", action="store_true", help="Disable rendering (for headless environments like Colab)")
    return parser.parse_args()


def main():
    args = parse_args()
    log_file, log_path = setup_logging("play")

    try:
        check_requirements()

        render_mode = "rgb_array" if args.no_render else "human"
        env = make_env(seed=0, render_mode=render_mode)

        model = DQN.load(args.model_path, env=env)

        episode_rewards = []
        episode_count = 0
        obs = env.reset()

        while episode_count < args.episodes:
            action, _ = model.predict(obs, deterministic=True)
            obs, rewards, dones, infos = env.step(action)

            for info in infos:
                if "episode" in info:
                    ep_reward = info["episode"]["r"]
                    episode_count += 1
                    episode_rewards.append(ep_reward)
                    print(f"Episode {episode_count}: reward = {ep_reward:.1f}")
                    if episode_count >= args.episodes:
                        break

        mean_reward = sum(episode_rewards) / len(episode_rewards) if episode_rewards else 0
        print(f"\nPlayed {len(episode_rewards)} episodes")
        print(f"Mean reward: {mean_reward:.2f}")

        env.close()

    except Exception as e:
        print(f"Error during playback: {e}", file=sys.stderr)
        print(f"Log file: {log_path}", file=sys.stderr)
        raise
    finally:
        log_file.close()


if __name__ == "__main__":
    main()
